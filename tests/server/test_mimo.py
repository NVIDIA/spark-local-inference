# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live MiMo text, image and prefix-cache requests through the native TP service.

Set FLARE_RUN_CUDA_TESTS=1 and FLARE_MIMO_TEST_URL to a MiMo service with
concurrency four and a 4096-token prefill chunk. Exercise both DFlash and
--mtp-draft-tokens 0 when validating a prepared core.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('FLARE_RUN_CUDA_TESTS') != '1' or not os.environ.get('FLARE_MIMO_TEST_URL'),
    reason='requires a native MiMo service',
)


@pytest.fixture(scope='module')
def service():
    url = os.environ['FLARE_MIMO_TEST_URL'].rstrip('/')
    with urllib.request.urlopen(url + '/v1/models', timeout=10) as response:
        model = json.load(response)['data'][0]['id']
    assert 'mimo' in model.lower(), model

    def ask(content, **options):
        body = dict(model=model, messages=[dict(role='user', content=content)], temperature=1,
                    seed=1234, max_tokens=128, enable_thinking=False)
        body.update(options)
        request = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            pytest.fail(f"HTTP {error.code}: {error.read().decode()}")
        assert result['choices'][0]['finish_reason'] != 'length', result
        return result

    return ask


def answer(result):
    return result['choices'][0]['message']['content'].strip()


def image(color):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new('RGB', (192, 128), color).save(buffer, format='PNG')
    return dict(type='image_url', image_url=dict(url='data:image/png;base64,'
                + base64.b64encode(buffer.getvalue()).decode()))


def test_text_and_concurrent_requests(service):
    assert answer(service('What is 17 times 23? Reply with only the integer.')) == '391'
    prompts = [f'Repeat exactly this code and nothing else: FLARE{i}128' for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(service, prompts))
    for index, result in enumerate(results):
        assert f'FLARE{index}128' in answer(result)


def test_images_keep_order_across_chunked_prefill(service):
    question = dict(type='text', text='Name the color of each image, in order. Output only the two colors.')
    content = [image('red'), image('blue'), question]
    result = answer(service(content)).lower()
    assert 'red' in result and 'blue' in result and result.index('red') < result.index('blue')
    content.insert(0, dict(type='text', text='filler ' * 4100))
    response = service(content)
    assert response['usage']['prompt_tokens'] > 4096
    result = answer(response).lower()
    assert 'red' in result and 'blue' in result and result.index('red') < result.index('blue')


def test_long_context_and_reused_prefix(service):
    prefix = ' '.join(f'Item {i} records code {i * 17 % 997}.' for i in range(1200))
    prompt = prefix + '\nWhat code does item 211 record? Reply with only the integer.'
    first, second = service(prompt), service(prompt)
    assert first['usage']['prompt_tokens'] > 4096
    assert answer(first) == answer(second) == '596'


def test_decode_across_sliding_window_boundary(service):
    import re

    result = service('Count from 1 through 96 inclusive, separated by commas. Output only the numbers, '
                     'with no omissions.', max_tokens=512)
    assert result['usage']['completion_tokens'] > 256
    assert [int(value) for value in re.findall(r'\d+', answer(result))] == list(range(1, 97))
