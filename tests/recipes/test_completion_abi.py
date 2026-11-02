# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ABI6 completion writers preserve physical lanes and frame ownership."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ignition.components.sampling import _completion_fields


@pytest.mark.parametrize("fields", [32])
def test_completion_shape_contract(fields):
    assert _completion_fields(torch.zeros(2, 4, fields, dtype=torch.int64), torch.device("cpu")) == fields


@pytest.mark.parametrize("shape,dtype", [((2, 4, 8), torch.int64), ((2, 4, 16), torch.int64), ((2, 3, 32), torch.int64), ((2, 4, 24), torch.int64),
    ((4, 32), torch.int64), ((2, 4, 32), torch.int32)])
def test_completion_shape_rejects_malformed_records(shape, dtype):
    with pytest.raises(ValueError, match="completion_records"):
        _completion_fields(torch.zeros(shape, dtype=dtype), torch.device("cpu"))


@pytest.mark.parametrize("fields", [32])
def test_actual_completion_kernels_with_cpu_interpreter(fields):
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(fields)],
        env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")
@pytest.mark.parametrize("fields", [32])
def test_actual_completion_kernels_cuda(fields):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    _kernel_cases(fields, "cuda")


def _kernel_cases(fields, device="cpu"):
    from ignition.components.sampling import AdmissionBuffers, _admission_finish, _complete
    from ignition.recipes.qwen38 import vision_state
    from ignition.recipes.qwen38.maintenance import _snapshot_complete
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from ignition.recipes.qwen38.vision_state import _vision_finish
    if os.environ.get("TRITON_INTERPRET") == "1":
        # The interpreter cannot add a global tl.constexpr to its tensor wrapper.
        vision_state._JIT_VISION_ERROR_BASE = vision_state.VISION_ERROR_BASE

    def tensor(values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device=device)

    # Strided live counts exercise the hybrid-state view contract.
    backing = torch.zeros((8, 19), dtype=torch.int64, device=device)
    state = DeviceRequestState.allocate(device, vocab_size=32, consumed_tokens=backing[:4, 0])
    state.admit(2, RequestAdmission(7, 11, temperature=1., max_new_tokens=2, eos_token_ids=()), [3] * 64 + [4],
        consumed_tokens=65, ple_history=(3, 4))
    records = torch.full((2, 4, fields), -99, dtype=torch.int64, device=device)
    frame, sequence = tensor([1], torch.int32), tensor([91])
    slots, epochs, live = tensor([2], torch.int32), tensor([7]), tensor([1], torch.int32)
    enabled = tensor([1], torch.int32)
    maximum, sampled = tensor([3.], torch.float32), tensor([5])
    sample_valid = tensor([True], torch.bool)
    args = (maximum, sampled, sample_valid, maximum, slots, epochs, live, enabled,
        state.generation, state.consumed_tokens, state.generated_tokens, state.pending_token, state.status,
        state.error_code, state.rng_counter, state.histogram, state.max_new_tokens,
        state.eos_token_ids, state.eos_count, records, frame, sequence,
        32, 1, state.consumed_tokens.stride(0), fields, True)

    def check(slot, base):
        row = records[1, slot].cpu().tolist()
        assert row[:8] == base, row
        if fields == 32:
            token = base[4]
            assert row[8:] == [int(token >= 0), 0, 0, 0, -1,
                max(base[2] - 1, 0), 0, 1, token] + [-1] * 15, row
        assert records[0].eq(-99).all(), "other completion frame was overwritten"

    _complete[(4,)](*args)
    check(2, [7, 91, 65, 1, 5, 1, 0, 0])
    check(0, [0, 0, 0, 0, -1, 0, 0, 0])
    assert state.histogram[2, 5] == 1 and state.rng_counter[2] == 1
    # A queued second sample fails rather than duplicating the pending token.
    _complete[(4,)](*args)
    check(2, [7, 91, 65, 1, -1, 3, 2, 0])
    state.status[2], state.error_code[2] = 1, 0
    # Consumption-only records contain no token, even with pending state.
    _complete[(4,)](*args[:-1], False)
    check(2, [7, 91, 65, 1, -1, 1, 0, 0])
    _complete[(4,)](*args[:-1], False)
    check(2, [7, 91, 65, 1, -1, 1, 0, 0])
    epochs.fill_(6)
    _complete[(4,)](*args[:-1], False)
    check(2, [0, 0, 0, 0, -1, 0, 0, 0])

    admission = AdmissionBuffers.allocate(device, vocab_size=32)
    admission.stage(0, 2, RequestAdmission(8, 19, temperature=1., eos_token_ids=()), [3, 4])
    admission.admission_count.fill_(1)
    errors = torch.zeros(4, dtype=torch.int32, device=device)
    a = admission
    finish_args = (a.admission_count, a.admission_slots, a.admission_generations, a.admission_seed_bits,
        a.admission_parameters, a.admission_top_k, a.admission_max_new_tokens, a.admission_eos_token_ids,
        a.admission_eos_count, a.admission_grammar_enabled, errors, state.generation, state.seed, state.rng_counter,
        state.parameters, state.top_k, state.max_new_tokens, state.eos_token_ids, state.eos_count, state.grammar_enabled,
        state.generated_tokens, state.pending_token, state.status, state.error_code, state.consumed_tokens,
        records, frame, sequence, 4, state.consumed_tokens.stride(0), fields)
    _admission_finish[(4,)](*finish_args)
    check(2, [8, 91, 65, 0, -1, 1, 0, 0])
    errors[0] = 2
    _admission_finish[(4,)](*finish_args)
    check(2, [8, 91, 65, 0, -1, 3, 4, 0])
    errors.zero_()
    _snapshot_complete[(4,)](a.admission_slots, a.admission_generations, a.admission_count, errors,
        state.generation, state.consumed_tokens, state.generated_tokens, state.status, state.error_code,
        records, frame, sequence, 4, state.consumed_tokens.stride(0), fields)
    check(2, [8, 91, 65, 0, -1, 1, 0, 0])
    errors[0] = 9
    _snapshot_complete[(4,)](a.admission_slots, a.admission_generations, a.admission_count, errors,
        state.generation, state.consumed_tokens, state.generated_tokens, state.status, state.error_code,
        records, frame, sequence, 4, state.consumed_tokens.stride(0), fields)
    check(2, [8, 91, 65, 0, -1, 3, 73, 0])

    errors.zero_()
    feature_generation, feature_count = tensor([0] * 4), tensor([0] * 4, torch.int32)
    vision_args = (tensor([2], torch.int32), tensor([8]), tensor([256], torch.int32), tensor([64], torch.int32),
        errors, feature_generation, feature_count, state.generation, state.consumed_tokens, state.generated_tokens,
        state.status, state.error_code, records, frame, sequence, state.consumed_tokens.stride(0), fields)
    _vision_finish[(4,)](*vision_args)
    check(2, [8, 91, 65, 0, -1, 1, 0, 0])
    assert feature_count.tolist() == [0, 0, 128, 0]
    errors[0] = 4
    _vision_finish[(4,)](*vision_args)
    check(2, [8, 91, 65, 0, -1, 3, 132, 0])
    check(0, [0, 0, 0, 0, -1, 0, 0, 0])


if __name__ == "__main__":
    _kernel_cases(int(sys.argv[1]))
    assert not torch.cuda.is_initialized()
