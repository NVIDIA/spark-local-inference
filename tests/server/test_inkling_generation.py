# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fifty Inkling generation cases with a fixed 64-token output budget.

Use the exported-package endpoint and checkpoint variables documented in
``test_inkling.py``. The package retains its default 262144-token context.
"""

import json
import os
from pathlib import Path

import pytest

from tests.server.test_inkling import pytestmark, service, text


LONG_INPUTS = (4097, 8193, 16385, 32769, 65537, 262080)
CASES = ([('text', i) for i in range(32)] + [('tool', i) for i in range(12)]
         + [('long', i) for i in range(6)])


@pytest.fixture(scope='module')
def checkpoint_frontend():
    from transformers import PreTrainedTokenizerFast

    snapshot = Path(os.environ['FLARE_INKLING_SNAPSHOT'])
    return (PreTrainedTokenizerFast.from_pretrained(snapshot),
            (snapshot / 'chat_template.jinja').read_text())


def long_prompt(frontend, target):
    tokenizer, template = frontend
    head = 'The secret code is INKLING-84127. Remember it while reading the filler.\n'
    tail = '\nWhat is the secret code? Reply with only the code.'

    def length(prompt):
        return len(tokenizer.apply_chat_template([dict(role='user', content=prompt)],
            chat_template=template, tokenize=True, return_dict=False,
            add_generation_prompt=True, reasoning_effort='none'))

    # Each additional filler contributes one token in the pinned checkpoint tokenizer.
    count = target - length(head + ' filler' + tail) + 1
    prompt = head + ' filler' * count + tail
    assert length(prompt) == target
    return prompt


@pytest.mark.parametrize('kind,index', CASES, ids=[f'{kind}-{i}' for kind, i in CASES])
def test_fifty_generation_cases(service, checkpoint_frontend, kind, index):
    options = dict(temperature=1., top_p=1., top_k=-1, min_p=0.,
        repetition_penalty=1., presence_penalty=0., frequency_penalty=0.,
        seed=58123, max_tokens=64, chat_template_kwargs=dict(reasoning_effort='none'))
    if kind == 'text':
        if index < 20:
            a, b = 31 + index, 47 - index
            prompt = f'What is {a} times {b}? Reply with only the integer.'
            expected = str(a * b)
        else:
            expected = ['INKLING-64283', 'café noir', '世界和平', 'naïve', 'á', 'Δοκιμή',
                        'مرحبا', '日本語', 'Straße', '한국어', 'mañana', 'résumé'][index - 20]
            prompt = 'Repeat exactly this text, without quotation marks or explanation: ' + expected
    elif kind == 'tool':
        expected = dict(value=index + 11)
        function = dict(name='record_value', description='Record the supplied value exactly.',
            parameters=dict(type='object', properties=dict(value=dict(type='integer')),
                            required=['value'], additionalProperties=False))
        options['tools'] = [dict(type='function', function=function)]
        options['tool_choice'] = ('required' if index % 2 else
                                 dict(type='function', function=dict(name='record_value')))
        prompt = 'Call record_value once with exactly these arguments:\n' + json.dumps(expected)
    else:
        prompt = long_prompt(checkpoint_frontend, LONG_INPUTS[index])
        expected = 'INKLING-84127'

    result = service[0]([dict(role='user', content=prompt)], **options)
    choice = result['choices'][0]
    assert 0 < result['usage']['completion_tokens'] <= 64
    assert result['usage']['prompt_tokens'] == len(result['prompt_token_ids'])
    if kind == 'tool':
        call, = choice['message']['tool_calls']
        assert call['function']['name'] == 'record_value'
        assert json.loads(call['function']['arguments']) == expected
        assert choice['finish_reason'] == 'tool_calls'
    else:
        assert text(result) == expected
        assert choice['finish_reason'] == 'stop'
    if kind == 'long':
        assert result['usage']['prompt_tokens'] == LONG_INPUTS[index]
