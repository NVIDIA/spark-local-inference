# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark text acceptance against an exported core served by ``flare run``.

Set FLARE_RUN_CUDA_TESTS=1, FLARE_MUSE_TEST_URL, and FLARE_MUSE_SPARK_SNAPSHOT.
Export with --prompt-logprobs. The core must prepare TP8 SM120, context
262144, concurrency four, and a 4096-token prefill chunk. These tests never launch a diagnostic Python server.
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
    os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not os.environ.get('FLARE_MUSE_TEST_URL'),
    reason='requires an exported Muse Spark core running through flare run',
)

EFFORTS = ('low', 'medium', 'high', 'xhigh')
LONG_INPUTS = (4097, 8193, 16385, 65537, 131073, 262080)
CASES = ([('text', i) for i in range(20)] + [('reasoning', i) for i in range(8)]
         + [('tool', i) for i in range(8)] + [('json', i) for i in range(8)]
         + [('long', i) for i in range(6)])


class NativeMuse:
    def __init__(self):
        from transformers import PreTrainedTokenizerFast

        self.url = os.environ['FLARE_MUSE_TEST_URL'].rstrip('/')
        self.key = os.environ.get('FLARE_API_TEST_KEY', 'local')
        self.snapshot = Path(os.environ['FLARE_MUSE_SPARK_SNAPSHOT'])
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.snapshot)
        self.template = (self.snapshot / 'chat_template.jinja').read_text()
        cards = self.request('/v1/models')['data']
        assert len(cards) == 1
        self.card = cards[0]
        self.model = self.card['id']
        assert 'muse' in self.model.lower(), self.model
        assert self.card['max_model_len'] == 262144
        assert self.card['flare']['max_concurrency'] == 4
        assert self.card['flare']['input_modalities'] == ['text']
        defaults = self.card['flare']['sampling_defaults']
        assert defaults['temperature'] == defaults['top_p'] == 1
        assert defaults['top_k'] == -1

    def open(self, path, body=None):
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, payload, {
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

    def metrics(self):
        with self.open('/metrics') as response:
            return {key: float(value) for line in response.read().decode().splitlines()
                    if line and not line.startswith('#') for key, value in [line.rsplit(' ', 1)]}

    def body(self, prompt, **options):
        body = dict(model=self.model, messages=[dict(role='user', content=prompt)],
                    temperature=1., top_p=1., top_k=-1, seed=41873,
                    max_tokens=2048, return_token_ids=True,
                    chat_template_kwargs=dict(reasoning_effort='low', current_date='2026-10-27',
                                              add_generation_prompt=True))
        body.update(options)
        return body

    def expected_prompt(self, body):
        options = dict(body['chat_template_kwargs'])
        if body.get('tools'):
            options['tools'] = body['tools']
        return self.tokenizer.apply_chat_template(body['messages'], chat_template=self.template,
            tokenize=True, return_dict=False, **options)

    def ask(self, prompt, *, compare_prompt=False, **options):
        body = self.body(prompt, **options)
        result = self.request('/v1/chat/completions', body)
        choice = result['choices'][0]
        assert choice['finish_reason'] != 'length', result
        assert result['usage']['completion_tokens'] == len(choice['token_ids'])
        if compare_prompt:
            assert result['prompt_token_ids'] == self.expected_prompt(body)
        return result

    def long_prompt(self, target):
        head = 'The secret access code is MUSE53197. Preserve it while reading the filler.\n'
        tail = '\nWhat is the secret access code? Reply with only the code.'
        prompt = lambda count: head + ' filler' * count + tail
        length = lambda count: len(self.expected_prompt(self.body(prompt(count), max_tokens=64)))
        low, high = 0, target
        while low < high:
            middle = (low + high + 1) // 2
            if length(middle) <= target:
                low = middle
            else:
                high = middle - 1
        assert target - 8 <= length(low) <= target
        return prompt(low)


@pytest.fixture(scope='module')
def service():
    return NativeMuse()


def content(result):
    return (result['choices'][0]['message'].get('content') or '').strip()


@pytest.mark.parametrize('kind,index', CASES, ids=[f'{kind}-{i}' for kind, i in CASES])
def test_fifty_prompt_cases(service, kind, index):
    def ask(prompt, **options):
        return service.ask(prompt, max_tokens=64, **options)

    if kind == 'text':
        if index < 12:
            a, b = 17 + index, 23 - index
            result = ask(f'What is {a} times {b}? Reply with only the integer.', compare_prompt=True)
            assert content(result) == str(a * b)
        else:
            value = ['MUSE-53197', 'café', '世界', 'naïve', 'á', 'Δοκιμή', 'مرحبا', '日本語'][index - 12]
            result = ask(f'Repeat exactly this text, without quotation marks or explanation: {value}', compare_prompt=True)
            assert content(result) == value
    elif kind == 'reasoning':
        effort = EFFORTS[index // 2]
        boxes, per_box, removed = 7 + index, 13 + index, 19 + index
        result = ask(f'A store has {boxes} boxes of {per_box} bolts. It sells {removed} bolts '
            'and receives two more boxes with the same number per box. How many bolts does it have? '
            'Put only the integer in the final answer.', compare_prompt=True,
            chat_template_kwargs=dict(reasoning_effort=effort, current_date='2026-10-27', add_generation_prompt=True))
        assert content(result) == str((boxes + 2) * per_box - removed)
        assert result['choices'][0]['message'].get('reasoning_content'), result
    elif kind == 'tool':
        value = dict(label=f'café 世界 {index}', count=index + 2, enabled=index % 2 == 0,
                     optional=None, items=[index, index + 1], options=dict(mode='exact'))
        properties = dict(label=dict(type='string'), count=dict(type='integer'), enabled=dict(type='boolean'),
                          optional=dict(type=['string', 'null']), items=dict(type='array', items=dict(type='integer')),
                          options=dict(type='object', properties=dict(mode=dict(type='string'))))
        function = dict(name='record_values', description='Record the supplied values exactly.',
            parameters=dict(type='object', properties=properties, required=list(properties)))
        choice = 'required' if index % 2 else dict(type='function', function=dict(name='record_values'))
        result = ask('Call record_values once with exactly these argument values:\n'
            + json.dumps(value, ensure_ascii=False), tools=[dict(type='function', function=function)], tool_choice=choice)
        calls = result['choices'][0]['message']['tool_calls']
        assert len(calls) == 1
        assert calls[0]['function']['name'] == 'record_values'
        assert json.loads(calls[0]['function']['arguments']) == value
        assert result['choices'][0]['finish_reason'] == 'tool_calls'
    elif kind == 'json':
        value = dict(index=index, text='世界 café <atem:function_calls> literal', enabled=bool(index % 2))
        result = ask('Return the requested JSON object.', response_format=dict(type='json_schema',
            json_schema=dict(name='exact_value', strict=True, schema=dict(const=value))))
        assert json.loads(content(result)) == value
    else:
        target = LONG_INPUTS[index]
        result = ask(service.long_prompt(target), compare_prompt=True)
        assert target - 8 <= result['usage']['prompt_tokens'] <= target
        assert content(result) == 'MUSE53197'


def test_concurrency_four_and_prefix_reuse(service):
    nonce = str(time.time_ns())
    prefix = f'Cache test {nonce}.\n' + 'A repeated prefix preserves the complete conversation. ' * 1800
    history = [dict(role='user', content=prefix), dict(role='assistant', content='I have read the context.')]

    def ask(code):
        return service.ask('', messages=[*history, dict(role='user', content=f'Reply with only this code: {code}')])

    first, second = ask('MUSE-REPEAT'), ask('MUSE-REPEAT')
    assert content(first) == content(second) == 'MUSE-REPEAT'
    assert first['choices'][0]['token_ids'] == second['choices'][0]['token_ids']
    assert second['usage']['prompt_tokens_details']['cached_tokens'] > 4096
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(ask, [f'MUSE-BRANCH-{i}' for i in range(4)]))
    for index, result in enumerate(results):
        assert content(result) == f'MUSE-BRANCH-{index}'
        assert result['usage']['prompt_tokens_details']['cached_tokens'] > 4096


def test_stream_cancel_drains_and_recovers(service):
    before = service.metrics()
    body = service.body('List the integers from 1 through 2000, one per line.', stream=True, max_tokens=2048)
    saw_generated = False
    with service.open('/v1/chat/completions', body) as response:
        for line in response:
            if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                event = json.loads(line[6:])
                choice = event.get('choices', [{}])[0]
                if choice.get('token_ids'):
                    saw_generated = True
                    break
    assert saw_generated
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        after = service.metrics()
        if after['flare_active_requests'] == after['flare_http_inflight_requests'] == 0:
            break
        time.sleep(.02)
    assert after['flare_active_requests'] == after['flare_http_inflight_requests'] == 0
    assert after['flare_requests_cancelled_total'] > before['flare_requests_cancelled_total']
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: service.ask(f'Reply with only RECOVER{i}.'), range(4)))
    assert [content(result) for result in results] == [f'RECOVER{i}' for i in range(4)]


@pytest.mark.parametrize('path,body', [
    ('/v1/chat/completions', dict(messages=[dict(role='user', content=[dict(type='image_url', image_url=dict(url='data:image/png;base64,AA=='))])])),
    ('/v1/chat/completions', dict(messages=[dict(role='user', content=[dict(type='input_audio', input_audio=dict(data='AA==', format='wav'))])])),
    ('/v1/chat/completions', dict(messages=[dict(role='user', content=[dict(type='video_url', video_url=dict(url='https://example.invalid/video.mp4'))])])),
])
def test_unsupported_media_is_rejected(service, path, body):
    with pytest.raises(urllib.error.HTTPError) as rejected:
        service.open(path, dict(body, model=service.model))
    assert rejected.value.code == 400


def test_context_capacity_rejects_oversized_requests(service):
    prompt = service.long_prompt(262080) + ' filler' * 128
    with pytest.raises(urllib.error.HTTPError) as rejected:
        service.open('/v1/chat/completions', service.body(prompt, max_tokens=64))
    assert rejected.value.code == 400


@pytest.mark.parametrize('path,body', [
    ('/v1/messages', dict(max_tokens=2048, messages=[dict(role='user', content='Reply with only PROTOCOL-MESSAGES.')])),
    ('/v1/responses', dict(max_output_tokens=2048, input='Reply with only PROTOCOL-RESPONSES.')),
])
def test_native_protocol_endpoints(service, path, body):
    response = service.request(path, dict(body, model=service.model, temperature=1, top_p=1, top_k=-1,
        chat_template_kwargs=dict(reasoning_effort='low', current_date='2026-10-27')))
    if path == '/v1/messages':
        assert response['stop_reason'] == 'end_turn'
        assert ''.join(block['text'] for block in response['content'] if block['type'] == 'text').strip() == 'PROTOCOL-MESSAGES'
    else:
        assert response['status'] == 'completed'
        text = ''.join(part['text'] for item in response['output'] if item['type'] == 'message'
                       for part in item['content'] if part['type'] == 'output_text')
        assert text.strip() == 'PROTOCOL-RESPONSES'


@pytest.mark.parametrize('protocol', ['completions', 'chat'])
def test_prompt_logprob_scores_align_with_actual_checkpoint_tokens(service, protocol):
    prompt = 'A cache preserves context. café 世界 17 times 23 equals 391.'
    if protocol == 'completions':
        tokens = service.expected_prompt(service.body(prompt))
        body = dict(model=service.model, prompt=tokens, max_tokens=0,
                    prompt_logprobs=2)
        result = service.request('/v1/completions', body)
        scores = result['choices'][0]['prompt_logprobs']
        assert result['usage']['completion_tokens'] == 0
        assert result['choices'][0]['finish_reason'] == 'length'
    else:
        body = service.body(prompt, max_tokens=1, prompt_logprobs=2)
        tokens = service.expected_prompt(body)
        result = service.request('/v1/chat/completions', body)
        assert result['prompt_token_ids'] == tokens
        scores = result['prompt_logprobs']
        assert result['usage']['completion_tokens'] == 1
    assert result['usage']['prompt_tokens'] == len(tokens)
    assert len(scores) == len(tokens)
    assert scores[0] is None
    for token, values in zip(tokens[1:], scores[1:], strict=True):
        assert str(token) in values
        assert 2 <= len(values) <= 3
        for token_id, value in values.items():
            assert 0 <= int(token_id) < len(service.tokenizer)
            assert math.isfinite(value['logprob']) and value['logprob'] <= 0
            assert value['rank'] >= 1
            assert isinstance(value['decoded_token'], str)
        ordered = sorted(values.values(), key=lambda entry: entry['logprob'], reverse=True)
        for first, second in zip(ordered, ordered[1:]):
            assert first['rank'] <= second['rank']
