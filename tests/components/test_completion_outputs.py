# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Completion storage remains shared across serial mixed graph variants."""

import os

import pytest
import torch

from ignition.components.completion import (
    CompletionOutputs, LOGPROB_FIELDS, LOGPROB_PAIR_FIELDS,
    validate_completion_buffers,
)


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='CUDA opt-in')
@pytest.mark.parametrize("tokens", [1, 8, 13, 16, 32])
def test_mixed_roles_merge_only_the_active_frame(tokens):
    outputs = CompletionOutputs.allocate((2, 4, 16 + tokens), device='cuda')
    prompt, decode = outputs.private(), outputs.private()
    assert prompt.requested is decode.requested is outputs.requested
    prompt.records.fill_(11)
    prompt.logprobs.fill_(12)
    decode.records.fill_(21)
    decode.logprobs.fill_(22)
    outputs.records.fill_(-7)
    outputs.logprobs.fill_(-8)
    if tokens > 1:
        prompt.speculative_survival.fill_(.9)
        decode.speculative_survival.fill_(.5)
    outputs.merge(prompt, decode, torch.tensor(2, device='cuda'), torch.tensor([1], dtype=torch.int32, device='cuda'))
    assert torch.all(outputs.records[0] == -7)
    assert torch.all(outputs.logprobs[0] == -8)
    assert torch.all(outputs.records[1, 2] == 11)
    assert torch.all(outputs.logprobs[1, 2, :, :42] == 12)
    assert torch.all(outputs.records[1, [0, 1, 3]] == 21)
    assert torch.all(outputs.logprobs[1, [0, 1, 3], :, :42] == 22)
    assert torch.all(outputs.logprobs[1, :, :, 42:] == -8)
    if tokens > 1:
        assert torch.all(outputs.speculative_survival[0] == -1)
        assert torch.all(outputs.speculative_survival[1, 2] == -1)
        assert torch.all(outputs.speculative_survival[1, [0, 1, 3]] == .5)
    assert outputs.zero_ranges()


@pytest.mark.parametrize('change', ['missing_output', 'missing_flags', 'wrong_dtype', 'wrong_frame', 'wrong_slots', 'wrong_k'])
def test_malformed_logprob_declarations_are_rejected(change):
    outputs = CompletionOutputs.allocate((2, 4, 32), device='meta')
    buffers = outputs.buffers()
    validate_completion_buffers(buffers, 4)
    if change == 'missing_output':
        del buffers['completion_logprobs']
    elif change == 'missing_flags':
        del buffers['logprob_requested']
    elif change == 'wrong_dtype':
        buffers['completion_logprobs'] = buffers['completion_logprobs'].float()
    else:
        shape = {
            'wrong_frame': (1, 4, outputs.token_capacity, LOGPROB_FIELDS),
            'wrong_slots': (2, 2, outputs.token_capacity, LOGPROB_FIELDS),
            'wrong_k': (2, 4, outputs.token_capacity, LOGPROB_FIELDS + LOGPROB_PAIR_FIELDS),
        }[change]
        buffers['completion_logprobs'] = torch.empty(shape, dtype=torch.int32, device='meta')
    with pytest.raises(ValueError) as raised:
        validate_completion_buffers(buffers, 4)
    message = str(raised.value)
    if change.startswith('missing_'):
        missing = 'completion_logprobs' if change == 'missing_output' else 'logprob_requested'
        assert missing in message
    else:
        assert 'completion_logprobs' in message
        assert 'expected shape' in message and 'dtype torch.int32' in message
        assert 'got shape' in message and 'dtype torch.' in message
    validate_completion_buffers({'completion_records': outputs.records}, 4)


@pytest.mark.parametrize('shape,dtype', [((1, 4, 15), torch.float32),
    ((2, 3, 15), torch.float32), ((2, 4, 8), torch.float32), ((2, 4, 15), torch.float64)])
def test_optional_survival_requires_two_float32_frames(shape, dtype):
    outputs = CompletionOutputs.allocate((2, 4, 32), device='meta')
    validate_completion_buffers(outputs.buffers(), 4)
    buffers = {'completion_records': outputs.records,
               'completion_speculative_survival': torch.empty(shape, dtype=dtype, device='meta')}
    with pytest.raises(ValueError, match='completion_speculative_survival'):
        validate_completion_buffers(buffers, 4)
    with pytest.raises(ValueError, match='token_capacity'):
        CompletionOutputs.allocate((2, 4, 8), device='meta')


def test_completion_storage_covers_sixteen_tokens_and_fifteen_survival_values():
    outputs = CompletionOutputs.allocate((2, 4, 32), device="meta")
    assert outputs.records.shape == (2, 4, 32)
    assert outputs.logprobs.shape == (2, 4, 16, LOGPROB_FIELDS)
    assert outputs.speculative_survival.shape == (2, 4, 15)
    validate_completion_buffers(outputs.buffers(), 4)
    with pytest.raises(ValueError, match="token_capacity"):
        CompletionOutputs.allocate((2, 4, 16), device="meta")


def test_private_completion_outputs_are_disjoint_and_accounted():
    outputs = CompletionOutputs.allocate((2, 4, 24), device="cpu")
    base = outputs.nbytes
    prompt, decode = outputs.private(), outputs.private()
    assert prompt.records.data_ptr() != decode.records.data_ptr()
    assert prompt.logprobs.data_ptr() != decode.logprobs.data_ptr()
    assert prompt.requested is decode.requested is outputs.requested
    private_bytes = sum(t.nbytes for role in (prompt, decode)
                        for t in (role.records, role.logprobs, role.speculative_survival))
    assert outputs.nbytes == base + private_bytes
    ranges = outputs.zero_ranges()
    assert any(start <= prompt.records.data_ptr() < start + size for start, size in ranges)
    assert any(start <= decode.logprobs.data_ptr() < start + size for start, size in ranges)
