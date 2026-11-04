# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer acceptance against an exported package served by ``flare run``.

Set FLARE_RUN_CUDA_TESTS=1, FLARE_MUSE_GLIMMER_TEST_URL and
FLARE_MUSE_GLIMMER_SNAPSHOT. Export with --prompt-logprobs, the default
131072-token context, concurrency four and a 4096-token prefill chunk.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not os.environ.get('FLARE_MUSE_GLIMMER_TEST_URL'),
    reason='requires an exported Muse Glimmer package running through flare run',
)

LONG_INPUTS = (4097, 8193, 16385, 32769, 65537, 131008)
CASES = ([('text', i) for i in range(20)] + [('tool', i) for i in range(12)]
         + [('json', i) for i in range(12)] + [('long', i) for i in range(6)])


class NativeGlimmer:
    def __init__(self):
        from transformers import PreTrainedTokenizerFast

        self.url = os.environ['FLARE_MUSE_GLIMMER_TEST_URL'].rstrip('/')
        self.key = os.environ.get('FLARE_API_TEST_KEY', 'local')
        snapshot = Path(os.environ['FLARE_MUSE_GLIMMER_SNAPSHOT'])
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(snapshot)
        self.template = (snapshot / 'chat_template.jinja').read_text()
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

    def open(self, path, body=None):
        request = urllib.request.Request(self.url + path, None if body is None else json.dumps(body).encode(), {
            'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}',
            'anthropic-version': '2023-06-01',
        })
        return urllib.request.urlopen(request, timeout=1800)

    def request(self, path, body=None):
        try:
            with self.open(path, body) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            pytest.fail(f'{path}: HTTP {error.code}: {error.read().decode()}')

    def body(self, prompt, **options):
        body = dict(model=self.model, messages=[dict(role='user', content=prompt)],
            temperature=1., top_p=.95, top_k=64, seed=41873, max_tokens=64, return_token_ids=True,
            chat_template_kwargs=dict(reasoning_strength='low', current_date='2026-11-03',
                                      add_generation_prompt=True))
        body.update(options)
        return body

    def expected_prompt(self, body):
        options = dict(body['chat_template_kwargs'])
        if body.get('tools'):
            options['tools'] = body['tools']
        messages = [dict(message) for message in body['messages']]
        instructions = []
        if body.get('tool_choice') == 'required':
            instructions.append('You must call at least one of the provided tools in your next response. '
                                'Choose the appropriate tool and arguments.')
            options['tool_choice'] = 'required'
        if body.get('response_format', {}).get('type') == 'json_schema':
            schema = json.dumps(body['response_format']['json_schema']['schema'], ensure_ascii=False, separators=(',', ':'))
            instructions.append('Return the final answer as a JSON value matching this schema. Do not wrap it in Markdown '
                'fences or add text outside the JSON value. Tool calls may precede the final answer.\nJSON schema:\n' + schema)
        if instructions:
            instruction = '\n\n'.join(instructions)
            if messages and messages[0]['role'] == 'system':
                messages[0]['content'] += '\n\n' + instruction
            else:
                messages.insert(0, dict(role='system', content=instruction))
        return self.tokenizer.apply_chat_template(messages, chat_template=self.template,
            tokenize=True, return_dict=False, **options)

    def ask(self, prompt, **options):
        body = self.body(prompt, **options)
        result = self.request('/v1/chat/completions', body)
        assert result['choices'][0]['finish_reason'] != 'length', result
        assert result['usage']['completion_tokens'] == len(result['choices'][0]['token_ids'])
        assert result['prompt_token_ids'] == self.expected_prompt(body)
        return result

    def long_prompt(self, target):
        head = 'The secret code is GLIMMER53197. Preserve it while reading the filler.\n'
        tail = '\nWhat is the secret code? Reply with only the code.'
        prompt = lambda count: head + ' filler' * count + tail
        length = lambda count: len(self.expected_prompt(self.body(prompt(count))))
        low, high = 0, target
        while low < high:
            middle = (low + high + 1) // 2
            if length(middle) <= target:
                low = middle
            else:
                high = middle - 1
        assert target - 8 <= length(low) <= target
        return prompt(low)

    def metrics(self):
        with self.open('/metrics') as response:
            return {key: float(value) for line in response.read().decode().splitlines()
                    if line and not line.startswith('#') for key, value in [line.rsplit(' ', 1)]}


@pytest.fixture(scope='module')
def service():
    return NativeGlimmer()


def content(result):
    return (result['choices'][0]['message'].get('content') or '').strip()


@pytest.mark.parametrize('kind,index', CASES, ids=[f'{kind}-{i}' for kind, i in CASES])
def test_fifty_prompt_cases(service, kind, index):
    if kind == 'text':
        if index < 10:
            a, b = 17 + index, 23 - index
            result = service.ask(f'What is {a} times {b}? Reply with only the integer.')
            assert content(result) == str(a * b)
        else:
            value = ['GLIMMER-53197', 'café', '世界', 'naïve', 'á', 'Δοκιμή',
                     'مرحبا', '日本語', 'Straße', '한국어'][index - 10]
            result = service.ask(f'Repeat exactly this text, without quotation marks or explanation: {value}')
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
        result = service.ask(service.long_prompt(target))
        assert target - 8 <= result['usage']['prompt_tokens'] <= target
        assert content(result) == 'GLIMMER53197'


def test_concurrency_four_and_prefix_reuse(service):
    prefix = f'Cache test {time.time_ns()}.\n' + 'A repeated prefix preserves the conversation. ' * 1000
    history = [dict(role='user', content=prefix), dict(role='assistant', content='I have read the context.')]

    def ask(code):
        return service.ask('', messages=[*history, dict(role='user', content=f'Reply with only this code: {code}')])

    first, second = ask('GLIMMER-REPEAT'), ask('GLIMMER-REPEAT')
    assert content(first) == content(second) == 'GLIMMER-REPEAT'
    assert first['choices'][0]['token_ids'] == second['choices'][0]['token_ids']
    assert second['usage']['prompt_tokens_details']['cached_tokens'] > 4096
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(ask, [f'GLIMMER-BRANCH-{i}' for i in range(4)]))
    for index, result in enumerate(results):
        assert content(result) == f'GLIMMER-BRANCH-{index}'
        assert result['usage']['prompt_tokens_details']['cached_tokens'] > 4096


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
    assert any(event.get('usage', {}).get('completion_tokens', 0) > 0 for event in events if event.get('usage'))


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


@pytest.mark.parametrize('block', [
    dict(type='image_url', image_url=dict(url='data:image/png;base64,AA==')),
    dict(type='input_audio', input_audio=dict(data='AA==', format='wav')),
    dict(type='video_url', video_url=dict(url='https://example.invalid/video.mp4')),
])
def test_unsupported_media_is_rejected(service, block):
    with pytest.raises(urllib.error.HTTPError) as rejected:
        service.open('/v1/chat/completions', service.body('', messages=[dict(role='user', content=[block])]))
    assert rejected.value.code == 400
