# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Request capacity through admission, sampling, consumption, and native replay."""

import os

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.components.capacity import batch_capacities, request_capacity
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState, PreparedAdmission, PreparedSampling, RequestAdmission
from ignition.components.token_state import PreparedTokenState
from ignition.preparation.graphs import PreparedGraphBundle


@pytest.mark.parametrize("limit", range(1, 65))
def test_admission_limit_and_padded_storage(limit):
    capacity = request_capacity(limit)
    assert limit <= capacity < 2 * limit
    assert batch_capacities(limit)[-1] == capacity
    state = DeviceRequestState.allocate("cpu", vocab_size=32, capacity=capacity)
    state.admit(limit - 1, RequestAdmission(1, 7), [3, 3, 8])
    assert state.histogram[limit - 1, 3] == 2
    assert state.histogram.sum() == 3
    assert state.capacity == capacity


@pytest.mark.parametrize("limit", [0, 65, -1, True, 4.0])
def test_invalid_admission_limits(limit):
    with pytest.raises(ValueError):
        request_capacity(limit)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")
@pytest.mark.parametrize("capacity", [1, 2, 8, 64])
@torch.no_grad()
def test_native_replay_reorders_every_slot_without_allocation(capacity):
    state = DeviceRequestState.allocate("cuda", vocab_size=32, capacity=capacity)
    admission = AdmissionBuffers.allocate("cuda", vocab_size=32, capacity=capacity)
    slots = torch.arange(capacity - 1, -1, -1, dtype=torch.int32, device="cuda")
    epochs = torch.ones(capacity, dtype=torch.int64, device="cuda")
    live = torch.tensor([capacity], dtype=torch.int32, device="cuda")
    records = torch.full((2, capacity, 32), -99, dtype=torch.int64, device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.ones(1, dtype=torch.int64, device="cuda")
    for row in range(capacity):
        admission.stage(row, capacity - row - 1, RequestAdmission(1, row, top_k=0, top_p=1), [3])
    admission.admission_count.fill_(capacity)
    initialize = PreparedAdmission(state, admission, completion_outputs(records), frame, sequence, capacity=capacity)
    initialize.prepare()
    logits = torch.full((capacity, 32), -torch.inf, device="cuda")
    expected = slots.long() % 32
    logits.scatter_(1, expected[:, None], 0)
    sampler = PreparedSampling(logits, state, slots, epochs, live, completion_outputs(records), frame, sequence)
    sampler.prepare()
    tokens = torch.empty(capacity, dtype=torch.int64, device="cuda")
    spans = torch.arange(capacity + 1, dtype=torch.int32, device="cuda")
    consume = PreparedTokenState(state, tokens, spans, slots, epochs, live)
    consume.prepare()
    graphs = {}
    torch.cuda.synchronize()
    with torch.cuda.graph(graphs.setdefault("admit", torch.cuda.CUDAGraph())):
        initialize()
    with torch.cuda.graph(graphs.setdefault("sample", torch.cuda.CUDAGraph())):
        sampler()
        consume.gather()
        tokens.copy_(consume.pending_tokens)
        consume.consume()
    owner = PreparedGraphBundle(device=0, graphs=graphs, buffers={"records": records},
        resources=(state, admission, initialize, sampler, consume, tokens, spans, slots, epochs, live, frame, sequence))
    worker = owner.start_worker()
    try:
        for index in (0, 1):
            if index == 1:
                frame.fill_(1)
                sequence.fill_(2)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            ticket = worker.submit(index)
            result = worker.poll(timeout_ms=30000)
            assert result.ticket == ticket and result.error is None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        assert tokens.tolist() == expected.tolist()
        assert state.consumed_tokens.tolist() == [1] * capacity
        assert records[0, :, 1].tolist() == [1] * capacity
        assert records[1, :, 1].tolist() == [2] * capacity
        assert records[1, :, 4].tolist() == [slot % 32 for slot in range(capacity)]
    finally:
        worker.close()
