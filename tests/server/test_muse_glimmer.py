# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer acceptance against an exported package served by ``flare run``.

Set FLARE_RUN_CUDA_TESTS=1, FLARE_MUSE_GLIMMER_TEST_URL and
FLARE_MUSE_GLIMMER_SNAPSHOT. Export with --prompt-logprobs, the default
131072-token context, concurrency four and a 4096-token prefill chunk.
"""

import json
import math
import os
from pathlib import Path
import time

import pytest

from tests.server.test_muse_spark import (NativeMuse, content, test_concurrency_four_and_prefix_reuse,
                                          test_unsupported_media_is_rejected)

pytestmark = pytest.mark.skipif(
    os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not os.environ.get('FLARE_MUSE_GLIMMER_TEST_URL'),
    reason='requires an exported Muse Glimmer package running through flare run',
)

LONG_INPUTS = (4097, 8193, 16385, 32769, 65537, 131008)
CASES = ([('text', i) for i in range(20)] + [('tool', i) for i in range(12)]
         + [('json', i) for i in range(12)] + [('long', i) for i in range(6)])


class NativeGlimmer(NativeMuse):
    def __init__(self):
        from transformers import PreTrainedTokenizerFast

        self.url = os.environ['FLARE_MUSE_GLIMMER_TEST_URL'].rstrip('/')
        self.key = os.environ.get('FLARE_API_TEST_KEY', 'local')
        self.snapshot = Path(os.environ['FLARE_MUSE_GLIMMER_SNAPSHOT'])
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.snapshot)
        self.template = (self.snapshot / 'chat_template.jinja').read_text()
        card, = self.request('/v1/models')['data']
        self.model = card['id']
        assert 'glimmer' in self.model.lower(), self.model
        assert card['max_model_len'] == 131072
        assert card['flare']['max_concurrency'] == 4
        assert card['flare']['input_modalities'] == ['text']
        defaults = card['flare']['sampling_defaults']
        assert defaults['temperature'] == 1
        assert defaults['top_p'] == pytest.approx(.95)
        assert defaults['top_k'] == 64

    def body(self, prompt, **options):
        body = dict(model=self.model, messages=[dict(role='user', content=prompt)],
            temperature=1., top_p=.95, top_k=64, seed=41873, max_tokens=64, return_token_ids=True,
            chat_template_kwargs=dict(reasoning_strength='low', current_date='2026-11-03',
                                      add_generation_prompt=True))
        body.update(options)
        return body


@pytest.fixture(scope='module')
def service():
    return NativeGlimmer()


@pytest.mark.parametrize('kind,index', CASES, ids=[f'{kind}-{i}' for kind, i in CASES])
def test_fifty_prompt_cases(service, kind, index):
    if kind == 'text':
        if index < 10:
            a, b = 17 + index, 23 - index
            result = service.ask(f'What is {a} times {b}? Reply with only the integer.', compare_prompt=True)
            assert content(result) == str(a * b)
        else:
            value = ['GLIMMER-53197', 'café', '世界', 'naïve', 'á', 'Δοκιμή',
                     'مرحبا', '日本語', 'Straße', '한국어'][index - 10]
            result = service.ask(f'Repeat exactly this text, without quotation marks or explanation: {value}',
                                 compare_prompt=True)
            assert content(result) == value
    elif kind == 'tool':
        value = dict(value=index + 2)
        function = dict(name='record_value', description='Record the supplied values exactly.',
            parameters=dict(type='object', properties=dict(value=dict(type='integer')),
                            required=['value'], additionalProperties=False))
        choice = 'required' if index % 2 else dict(type='function', function=dict(name='record_value'))
        result = service.ask('Call record_value once with exactly these argument values:\n' + json.dumps(value),
            tools=[dict(type='function', function=function)], tool_choice=choice)
        call, = result['choices'][0]['message']['tool_calls']
        assert call['function']['name'] == 'record_value'
        assert json.loads(call['function']['arguments']) == value
        assert result['choices'][0]['finish_reason'] == 'tool_calls'
    elif kind == 'json':
        value = dict(index=index, text='世界 café', enabled=bool(index % 2))
        result = service.ask('Return the requested JSON object.', response_format=dict(type='json_schema',
            json_schema=dict(name='exact_value', strict=True, schema=dict(const=value))))
        assert json.loads(content(result)) == value
    else:
        target = LONG_INPUTS[index]
        result = service.ask(service.long_prompt(target), compare_prompt=True)
        assert target - 8 <= result['usage']['prompt_tokens'] <= target
        assert content(result) == 'MUSE53197'


def test_streamed_and_buffered_replay_agree(service):
    prompt = 'Reply with only GLIMMER-STREAM.'
    buffered = service.ask(prompt)
    events = []
    with service.open('/v1/chat/completions', service.body(prompt, stream=True,
            stream_options=dict(include_usage=True))) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            if line.strip() == b'data: [DONE]':
                break
            events.append(json.loads(line[6:]))
        else:
            pytest.fail('stream ended without [DONE]')
    choices = [choice for event in events for choice in event['choices']]
    assert ''.join(choice.get('delta', {}).get('content') or '' for choice in choices).strip() == content(buffered)
    assert [token for choice in choices for token in choice.get('token_ids', [])] == buffered['choices'][0]['token_ids']
    assert choices[-1]['finish_reason'] == 'stop'
    assert any(event['usage']['completion_tokens'] > 0 for event in events if event.get('usage'))


def test_prompt_logprobs_match_checkpoint_token_ids(service):
    tokens = service.expected_prompt(service.body('A cache preserves context. café 世界 17 times 23 equals 391.'))
    result = service.request('/v1/completions', dict(model=service.model, prompt=tokens,
        max_tokens=0, prompt_logprobs=2))
    assert result['usage']['prompt_tokens'] == len(tokens)
    assert result['usage']['completion_tokens'] == 0
    scores = result['choices'][0]['prompt_logprobs']
    assert len(scores) == len(tokens) and scores[0] is None
    for token, values in zip(tokens[1:], scores[1:], strict=True):
        assert str(token) in values
        assert 2 <= len(values) <= 3
        for token_id, value in values.items():
            assert 0 <= int(token_id) < len(service.tokenizer)
            assert math.isfinite(value['logprob']) and value['logprob'] <= 0
            assert value['rank'] >= 1


def test_cancel_drains_and_recovers(service):
    before = service.metrics()
    saw_token = False
    with service.open('/v1/chat/completions', service.body(
            'List the integers from 1 through 2000, one per line.', stream=True, max_tokens=2048)) as response:
        for line in response:
            if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                if any(choice.get('token_ids') for choice in json.loads(line[6:]).get('choices', [])):
                    saw_token = True
                    break
    assert saw_token
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        after = service.metrics()
        if after['flare_active_requests'] == after['flare_http_inflight_requests'] == 0:
            break
        time.sleep(.02)
    assert after['flare_active_requests'] == after['flare_http_inflight_requests'] == 0
    assert after['flare_requests_cancelled_total'] > before['flare_requests_cancelled_total']
    assert content(service.ask('Reply with only GLIMMER-RECOVER.')) == 'GLIMMER-RECOVER'
