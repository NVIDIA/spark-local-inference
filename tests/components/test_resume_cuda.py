# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled cached-prefix gathering preserves continuation and request sampling state."""

import os

import pytest
import torch

from ignition.components.resume import PreparedResumeState
from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.state import StateLayout, StateStorage
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 4])
@torch.no_grad()
def test_resume_gathers_exact_hidden_without_consumption(rows):
    layout = StateLayout.from_specs((("hidden", (4096,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    pool = StateStorage.allocate(layout, 6, "cuda:0")
    state = DeviceRequestState.allocate("cuda:0", vocab_size=1024, consumed_tokens=pool.region("consumed_tokens")[1:5, 0])
    saved = pool.region("hidden")[1:5]
    saved.normal_()
    slots = torch.tensor([2, 0, 3, 1][:rows], dtype=torch.int32, device="cuda:0")
    epochs = torch.full((rows,), 7, dtype=torch.int64, device="cuda:0")
    count = torch.full((1,), rows, dtype=torch.int32, device="cuda:0")
    operation = PreparedResumeState(state, saved, slots, epochs, count, world_size=1, state_rows=torch.arange(state.capacity, device="cuda")[:, None].expand(-1, 8).contiguous())

    def forward():
        operation.validation_sum.copy_(operation.validate())
        operation.gather()

    call = compile_forward(forward)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    bundle = PreparedGraphBundle(device=0, graphs={"resume": graph}, buffers={"slots": slots},
        resources=(pool, state, slots, epochs, count, operation, call))
    worker = bundle.start_worker()
    try:
        for fault in (None, "pending", "empty", "stale", "duplicate"):
            if rows == 1 and fault == "duplicate":
                continue
            slots.copy_(torch.tensor([2, 0, 3, 1][:rows]))
            epochs.fill_(7)
            for slot in range(4):
                state.admit(slot, RequestAdmission(7, 13), [5, 6, 7], consumed_tokens=3)
            if fault == "pending":
                state.pending_token[2] = 17
            elif fault == "empty":
                state.consumed_tokens[2] = 0
            elif fault == "stale":
                epochs[0] = 6
            elif fault == "duplicate":
                slots[1] = 2
            backing = pool.backing.clone()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(0)
            result = worker.poll(timeout_ms=30000)
            assert result is not None and result.ticket == ticket and result.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key], key
            assert torch.equal(pool.backing, backing)
            assert not state.rng_counter.count_nonzero() and not state.generated_tokens.count_nonzero()
            assert state.histogram.sum() == 12
            for row, slot in enumerate(slots.tolist()):
                active = not (slot == 2 and fault is not None)
                expected = saved[slot] if active else torch.zeros_like(saved[slot])
                torch.testing.assert_close(operation.hidden[row], expected, rtol=0, atol=0)
            assert state.error_code[2] == (33 if fault in ("pending", "empty", "duplicate") else 0)
            assert state.pending_token[2] == (17 if fault == "pending" else -1)
    finally:
        worker.close()
        graph.reset()
