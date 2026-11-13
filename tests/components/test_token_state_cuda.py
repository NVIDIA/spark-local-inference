# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native token consumption with and without recipe-owned previous-token history."""

import os

import pytest
import torch

from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.token_state import PreparedTokenState
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_packed_consumption_and_pending_guards(with_history, compiled):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    state = DeviceRequestState.allocate("cuda", vocab_size=154880)
    history = torch.zeros((4, 2), dtype=torch.int64, device="cuda") if with_history else None
    tokens = torch.arange(65, dtype=torch.int64, device="cuda")
    spans = torch.tensor([0, 63, 64, 64, 65], dtype=torch.int32, device="cuda")
    slots = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device="cuda")
    epochs = torch.full((4,), 7, dtype=torch.int64, device="cuda")
    live = torch.full((1,), 4, dtype=torch.int32, device="cuda")
    operation = PreparedTokenState(state, tokens, spans, slots, epochs, live,
                                   history=history, history_pad_id=99)
    operation.prepare()

    def forward():
        operation.gather()
        operation.consume()

    call = compile_forward(forward) if compiled else forward
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    bundle = PreparedGraphBundle(device=0, graphs={"consume": graph}, buffers={"tokens": tokens},
        resources=(state, history, operation, call, spans, slots, epochs, live))
    worker = bundle.start_worker()
    try:
        for fault in (None, "pending", "duplicate", "stale"):
            slots.copy_(torch.tensor([2, 0, 3, 1], dtype=torch.int32))
            epochs.fill_(7)
            for slot in range(4):
                state.admit(slot, RequestAdmission(7, 13), [5, 6, 7], consumed_tokens=3)
            state.pending_token[1] = 64
            if history is not None:
                history.copy_(torch.tensor([[5, 6]] * 4))
            expected_counts = [4, 4, 66, 3]
            expected_status = [1] * 4
            expected_errors = [0] * 4
            expected_history = [[6, 63], [6, 64], [61, 62], [5, 6]]
            if fault == "pending":
                state.pending_token[2] = 0
                expected_counts[2], expected_status[2], expected_errors[2] = 3, 3, 3
                expected_history[2] = [5, 6]
            elif fault == "duplicate":
                slots[1] = 2
                expected_counts[0], expected_counts[2] = 3, 3
                expected_status[2], expected_errors[2] = 3, 4
                expected_history[0] = expected_history[2] = [5, 6]
            elif fault == "stale":
                epochs[0] = 6
                expected_counts[2] = 3
                expected_history[2] = [5, 6]
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(0)
            completion = worker.poll(timeout_ms=30000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key], key
            assert state.consumed_tokens.cpu().tolist() == expected_counts
            assert state.status.cpu().tolist() == expected_status
            assert state.error_code.cpu().tolist() == expected_errors
            assert state.pending_token[1] == -1
            assert state.histogram.sum() == 12 and not state.rng_counter.count_nonzero()
            if history is None:
                assert operation.history is None
            else:
                assert history.cpu().tolist() == expected_history
                gathered = [[99, 99] if fault == "stale" and row == 0 else [5, 6] for row in range(4)]
                assert operation.history.cpu().tolist() == gathered
    finally:
        worker.close()
        graph.reset()
