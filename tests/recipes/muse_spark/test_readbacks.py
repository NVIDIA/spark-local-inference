# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled-forward diagnostics compare FP8 caches exactly and reject nonfinite data."""

import copy

import pytest
import torch

from qualify.muse_spark_logits import TextReadback, compare_readbacks


def readback():
    return dict(prompt_tokens=[1, 2], teacher_tokens=[3],
        logits=torch.zeros(3, 8, dtype=torch.bfloat16),
        hidden=torch.zeros(3, 4, dtype=torch.bfloat16),
        prefill_logits=torch.zeros(8, dtype=torch.bfloat16),
        prefill_hidden=torch.zeros(4, dtype=torch.bfloat16),
        cache_outputs=[{'endpoint1': torch.zeros(2, 2, 1, 2, 64, dtype=torch.float8_e4m3fn)}])


def test_fp8_readback_reports_exact_cache_difference():
    compiled = readback()
    raw = copy.deepcopy(compiled)
    assert compare_readbacks(compiled, raw) == dict(equal=True, differences={})
    raw['cache_outputs'][0]['endpoint1'].view(torch.uint8)[0, 1, 0, 1, 3] = 1
    result = compare_readbacks(compiled, raw)
    assert not result['equal']
    difference, = result['differences']['cache_outputs']
    assert difference['first_difference'] == dict(layer=0, kind='value', token=0, head=1, channel=3)
    assert difference['changed_elements'] == 1
    assert difference['max_absolute_error'] == 2 ** -9


@pytest.mark.parametrize('side', ['compiled', 'raw'])
def test_fp8_readback_rejects_nonfinite_cache(side):
    values = dict(compiled=readback(), raw=readback())
    values[side]['cache_outputs'][0]['endpoint1'].view(torch.uint8)[0, 0, 0, 0, 0] = 127
    with pytest.raises(ValueError, match='cache contains nonfinite'):
        compare_readbacks(values['compiled'], values['raw'])


def test_text_readback_consumes_returned_values_without_decoder_scratch():
    tokens = torch.tensor([3, 5, 8])

    class Forward:
        def observe(self):
            return {'target.token_ids': tokens + 1, 'target.positions': tokens - 1,
                    'target.hidden': tokens[:, None].to(torch.bfloat16),
                    'target.logits': (tokens[:, None] * 2).to(torch.bfloat16)}

    outputs = dict(token_ids=torch.empty_like(tokens), positions=torch.empty_like(tokens),
        hidden=torch.empty((3, 1), dtype=torch.bfloat16), logits=torch.empty((3, 1), dtype=torch.bfloat16))
    operation = TextReadback(Forward(), outputs)
    for _ in range(2):
        for value in outputs.values():
            value.fill_(-99)
        operation()
        assert torch.equal(outputs['token_ids'], tokens + 1)
        assert torch.equal(outputs['positions'], tokens - 1)
        assert torch.equal(outputs['hidden'], tokens[:, None].bfloat16())
        assert torch.equal(outputs['logits'], (tokens[:, None] * 2).bfloat16())
        tokens.add_(7)
