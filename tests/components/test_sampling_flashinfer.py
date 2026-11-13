# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from contextlib import ExitStack, contextmanager

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.components.sampling import (
    DUPLICATE_SLOT,
    ERROR,
    FINISHED,
    INVALID_DISTRIBUTION,
    RUNNING,
    DeviceRequestState,
    PreparedSampling,
    RequestAdmission,
)
from ignition.preparation.graphs import PreparedGraphBundle

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)
@pytest.fixture(params=[154880, 248320], ids=["glm", "qwen"])
def vocab_size(request):
    return request.param


class CapturedSampler:
    def __init__(self, rows, *, vocab_size, compile_forward=False, workspace=None):
        assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
        self.rows = rows
        self.state = DeviceRequestState.allocate("cuda", vocab_size=vocab_size)
        self.logits = torch.zeros((rows, vocab_size), device="cuda", dtype=torch.float32)
        self.slots = torch.arange(rows, device="cuda", dtype=torch.int32)
        self.epochs = torch.ones(rows, device="cuda", dtype=torch.int64)
        self.live = torch.zeros(1, device="cuda", dtype=torch.int32)
        self.records = torch.full((2, 4, 32), -99, device="cuda", dtype=torch.int64)
        self.frame = torch.zeros(1, device="cuda", dtype=torch.int32)
        self.sequence = torch.ones(1, device="cuda", dtype=torch.int64)
        self.enabled = torch.ones(rows, device="cuda", dtype=torch.int32)
        for slot in range(4):
            self.admit(slot)
        self.sampler = PreparedSampling(self.logits, self.state, self.slots, self.epochs,
            self.live, completion_outputs(self.records), self.frame, self.sequence, sample_enabled=self.enabled, workspace=workspace)
        self.sampler.prepare()
        self.call = self.sampler
        if compile_forward == 'aot':
            import io
            from ignition.preparation.aot import aot_forward

            artifact = aot_forward(self.sampler)
            artifact(self.sampler)
            restored = torch.compiler.load_compiled_function(
                io.BytesIO(type(artifact).serialize(artifact).serialized_data))
            self.call = lambda: restored(self.sampler)
        elif compile_forward:
            from ignition.preparation.compilation import compile_forward as compile_model

            self.call = compile_model(self.sampler)
            self.call()
        self.probabilities_pointer = self.sampler.probabilities.data_ptr()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.call()
        self.bundle = PreparedGraphBundle(device=0, graphs={"sample": self.graph},
            buffers={"logits": self.logits, "records": self.records,
                     "probabilities": self.sampler.probabilities},
            resources=(self.state, self.sampler, self.call, self.slots, self.epochs, self.live,
                       self.frame, self.sequence, self.enabled))

    def admit(self, slot, *, counter=0, seed=2**64 - 97, **parameters):
        values = {"generation": 1, "seed": seed, "max_new_tokens": 100,
                  "temperature": 1., "top_k": 0, "top_p": 1., "eos_token_ids": ()}
        values.update(parameters)
        self.state.admit(slot, RequestAdmission(**values), [])
        self.state.rng_counter[slot] = counter

    def replay(self):
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        ticket = self.worker.submit(0)
        result = self.worker.poll(timeout_ms=30000)
        assert result is not None and result.ticket == ticket and result.error is None
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        assert self.sampler.probabilities.data_ptr() == self.probabilities_pointer
        return self.records[self.frame.item()].cpu()


@contextmanager
def captured(rows, *, vocab_size, compile_forward=False):
    owner = CapturedSampler(rows, vocab_size=vocab_size, compile_forward=compile_forward)
    owner.worker = owner.bundle.start_worker()
    try:
        yield owner
    finally:
        owner.worker.close()


def snapshot(state):
    return {name: getattr(state, name).cpu().clone() for name in state.__dataclass_fields__}


def test_native_sampling_reuses_operands_across_row_capacities():
    from ignition.preparation.scratch import bind_scratch

    workspace = bind_scratch(PreparedSampling.scratch_layout(4, 10000), device="cuda:0")
    owners = [(CapturedSampler(rows, vocab_size=10000),
               CapturedSampler(rows, vocab_size=10000, workspace=workspace)) for rows in (1, 4)]
    try:
        for pair in owners:
            for owner in pair:
                owner.worker = owner.bundle.start_worker()
        for index in (1, 0, 1, 0):
            results = []
            for owner in owners[index]:
                for slot in range(4):
                    owner.admit(slot, top_k=20, top_p=.9, repetition_penalty=1.1)
                owner.live.fill_(owner.rows)
                owner.slots.copy_(torch.tensor([3, 0, 2, 1][:owner.rows], device="cuda", dtype=torch.int32))
                owner.epochs[-1] = 99 if owner.rows > 1 else 1
                owner.logits.copy_(torch.arange(10000, device="cuda").remainder(13).float()[None, :])
                for tensor in workspace.values():
                    tensor.fill_(True if tensor.dtype == torch.bool else 255)
                records = owner.replay()
                results.append((records, snapshot(owner.state)))
            assert torch.equal(results[0][0], results[1][0])
            for name, value in results[0][1].items():
                assert torch.equal(value, results[1][1][name]), name
    finally:
        for pair in owners:
            for owner in pair:
                if hasattr(owner, "worker"):
                    owner.worker.close()


@pytest.mark.parametrize("compile_forward", [False, True, 'aot'])
def test_native_flashinfer_cutoff_ties_and_device_parameter_changes(compile_forward, vocab_size):
    cases = (
        # Threshold top-k retains every tie, including more than k tokens.
        ([4., 3., 3., 1.], 2, 1., 0., [0, 1, 2]),
        # The top-p cutoff retains both equal-probability boundary tokens.
        ([math.log(2.), 0., 0.], 0, .6, 0., [0, 1, 2]),
        ([math.log(2.), 0., 0.], -1, 1., .6, [0]),
        ([0., 0., 0.], 0, 1., 1., [0, 1, 2]),
        ([0., 0., 0.], 1, 1., 0., [0, 1, 2]),
        ([4., 3., 3., 1.], vocab_size, 1., 0., [0, 1, 2, 3]),
        ([4., 3., 3., 1.], 2, 1., 0., [0, 1, 2]),
    )
    with captured(1, vocab_size=vocab_size, compile_forward=compile_forward) as owner:
        owner.slots.fill_(2)
        owner.live.fill_(1)
        for iteration, (values, top_k, top_p, min_p, support) in enumerate(cases):
            owner.admit(2, counter=2**32 + iteration, seed=2**64 - 97 - iteration,
                        top_k=top_k, top_p=top_p, min_p=min_p)
            owner.logits.fill_(-torch.inf)
            owner.logits[0, :len(values)].copy_(torch.tensor(values))
            owner.frame.fill_(iteration % 2)
            owner.sequence.fill_(101 + iteration)
            records = owner.replay()
            probabilities = owner.sampler.probabilities[0].cpu()
            assert probabilities.isfinite().all()
            assert torch.nonzero(probabilities > 0).flatten().tolist() == support
            torch.testing.assert_close(probabilities.sum(), torch.tensor(1.), rtol=0, atol=2e-6)
            token = records[2, 4].item()
            assert token in support
            assert records[2][..., :8].tolist() == [1, 101 + iteration, 0, 1, token, RUNNING, 0, 0]
            assert owner.state.rng_counter[2].item() == 2**32 + iteration + 1
            counts = owner.state.histogram.cpu()
            assert counts.sum() == 1 and counts[2, token] == 1
            assert (records[[0, 1, 3], 4] == -1).all()
            assert records[[0, 1, 3], 5].sum() == 0


def test_native_flashinfer_rng_is_invariant_to_capacity_packing_and_physical_slot(vocab_size):
    with ExitStack() as stack:
        owners = [stack.enter_context(captured(rows, vocab_size=vocab_size, compile_forward=compiled))
                  for compiled in (False, True) for rows in (1, 2, 4)]
        for seed, counter in ((17, 0), (2**64 - 79, 2**32 - 1), (2**63 + 19, 2**32 + 7)):
            sampled = []
            for owner in owners:
                for rotation in range(4):
                    order = [(rotation + row) % 4 for row in range(owner.rows)]
                    for slot in range(4):
                        owner.admit(slot, seed=seed, counter=counter)
                    owner.slots.copy_(torch.tensor(order, dtype=torch.int32))
                    owner.live.fill_(owner.rows)
                    owner.logits.fill_(-torch.inf)
                    owner.logits[:, :16].zero_()
                    records = owner.replay()
                    sampled.extend(records[order, 4].tolist())
                    assert owner.state.rng_counter.cpu().tolist() == [
                        counter + int(slot in order) for slot in range(4)]
                    assert owner.state.generated_tokens.cpu().tolist() == [
                        int(slot in order) for slot in range(4)]
            assert len(set(sampled)) == 1
            assert 0 <= sampled[0] < 16


def test_native_flashinfer_rejects_invalid_rows_and_ignores_masked_nan(vocab_size):
    with captured(4, vocab_size=vocab_size) as owner:
        owner.live.fill_(4)
        owner.logits.zero_()
        owner.logits[0, 5] = torch.nan
        owner.logits[1, 5] = torch.inf
        owner.logits[3, 5] = torch.nan
        owner.state.grammar_enabled[2:] = 1
        owner.state.grammar_mask[2:].zero_()
        owner.state.grammar_mask[3, 0] = 1 << 7
        records = owner.replay()
        for slot in (0, 1, 2):
            assert records[slot][..., :8].tolist() == [1, 1, 0, 0, -1, ERROR, INVALID_DISTRIBUTION, 0]
        assert records[3][..., :8].tolist() == [1, 1, 0, 1, 7, RUNNING, 0, 0]
        assert owner.state.rng_counter.cpu().tolist() == [0, 0, 0, 1]
        assert owner.state.histogram.sum().item() == 1
        assert owner.state.histogram[3, 7].item() == 1
        before = snapshot(owner.state)
        owner.live.zero_()
        owner.replay()
        after = snapshot(owner.state)
        for name in before:
            assert torch.equal(before[name], after[name]), name

        owner.admit(2)
        owner.slots[0] = 2
        owner.live.fill_(1)
        owner.logits[0].fill_(-torch.inf)
        records = owner.replay()
        assert records[2][..., :8].tolist() == [1, 1, 0, 0, -1, ERROR, INVALID_DISTRIBUTION, 0]
        assert owner.state.rng_counter[2].item() == 0


def test_native_flashinfer_stale_duplicate_and_disabled_rows_do_not_sample(vocab_size):
    with captured(4, vocab_size=vocab_size) as owner:
        owner.slots.copy_(torch.tensor([0, 0, 2, 3], dtype=torch.int32))
        owner.epochs[2] = 99
        owner.enabled[3] = 0
        owner.live.fill_(4)
        owner.logits[3].fill_(torch.nan)
        before = snapshot(owner.state)
        records = owner.replay()
        assert records[0][..., :8].tolist() == [1, 1, 0, 0, -1, ERROR, DUPLICATE_SLOT, 0]
        assert records[3][..., :8].tolist() == [1, 1, 0, 0, -1, RUNNING, 0, 0]
        assert (records[[1, 2], 4] == -1).all() and records[[1, 2], 5].sum() == 0
        after = snapshot(owner.state)
        for name in before:
            expected = before[name]
            if name == "status":
                expected[0] = ERROR
            elif name == "error_code":
                expected[0] = DUPLICATE_SLOT
            assert torch.equal(expected, after[name]), name


@pytest.mark.parametrize("compile_forward", [False, True])
def test_native_flashinfer_mixed_temperatures_eos_and_budget_stop_are_terminal(vocab_size, compile_forward):
    with captured(4, vocab_size=vocab_size, compile_forward=compile_forward) as owner:
        owner.admit(0, temperature=0.7, max_new_tokens=1)
        owner.admit(1, eos_token_ids=(7,))
        owner.admit(2, max_new_tokens=1)
        owner.live.fill_(4)
        owner.enabled[3] = 0
        owner.logits.fill_(-torch.inf)
        owner.logits[0, 5] = 3.
        owner.logits[1, 7] = 0.
        owner.logits[2, 9] = 0.
        records = owner.replay()
        assert records[:3, 4].tolist() == [5, 7, 9]
        assert owner.sampler.probabilities[0, 5] == 1
        assert owner.sampler.probabilities[0].count_nonzero() == 1
        assert owner.sampler.sampled_tokens[:3].cpu().tolist() == [5, 7, 9]
        assert owner.sampler.sample_valid[:3].all()
        assert records[:3, 5].tolist() == [FINISHED] * 3
        assert owner.state.rng_counter.cpu().tolist() == [1, 1, 1, 0]
        assert owner.state.generated_tokens.cpu().tolist() == [1, 1, 1, 0]
        assert owner.state.pending_token.cpu().tolist() == [5, 7, 9, -1]
        before = snapshot(owner.state)
        owner.logits.fill_(torch.nan)
        records = owner.replay()
        assert records[:, 4].tolist() == [-1] * 4
        after = snapshot(owner.state)
        for name in before:
            assert torch.equal(before[name], after[name]), name
