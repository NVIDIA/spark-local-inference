# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native GLM request packing, physical page mapping, and mutation guards."""

import os

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.glm53.batch import BatchInputs, PreparedBatch
from ignition.recipes.glm53.state import DecoderMetadata

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("slots", [1, 2])
@torch.no_grad()
def test_small_request_capacities_pad_rank_agreement(slots):
    from ignition.components.resume import PreparedResumeState

    inputs = BatchInputs.allocate(128, 1, "cuda", capacity=slots)
    inputs.request_state_rows[:, 0] = torch.arange(1, inputs.row_slot_ids.numel() + 1, device="cuda")
    requests = DeviceRequestState.allocate("cuda", vocab_size=154880, capacity=slots)
    metadata = DecoderMetadata.allocate(1, 1, 1, "cuda")
    batch = PreparedBatch(inputs, requests, metadata, mode="decode", shard=TensorParallelShard(0, 2), pages=torch.tensor([1], device="cuda:0"))
    requests.admit(0, RequestAdmission(7, 13), [3], consumed_tokens=1)
    requests.pending_token[0] = 5
    inputs.row_slot_ids[0], inputs.row_generations[0], inputs.live_rows[0] = 0, 7, 1
    inputs.token_ids[0] = 5
    inputs.page_tables[0, 0] = 0
    batch.functional()
    validation, present = batch.validate()
    metadata = batch.begin(validation * 2, present)
    assert validation.nbytes == 16
    assert metadata.indexed.positions.tolist() == [1]
    assert requests.error_code[0] == 0

    requests.pending_token[0] = -1
    hidden = torch.randn((slots + 1, 16), dtype=torch.bfloat16, device="cuda")
    resume = PreparedResumeState(requests, hidden, inputs.row_slot_ids[:1],
                                 inputs.row_generations[:1], inputs.live_rows, world_size=2, state_rows=inputs.request_state_rows)
    resume.functional()
    validation, present = resume.validate()
    assert validation.nbytes == 16
    torch.testing.assert_close(resume.gather(validation * 2, present=present), hidden[1:2], rtol=0, atol=0)


@pytest.mark.parametrize("rank,world", [(0, 2), (3, 4)])
@pytest.mark.parametrize("mode,capacity", [("prefill", 65), ("prefill", 4096), ("decode", 4)])
@torch.no_grad()
def test_native_glm_metadata_and_consumption(rank, world, mode, capacity):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    rows = 1 if mode == "prefill" else capacity
    inputs = BatchInputs.allocate(capacity, 32, "cuda")
    inputs.request_state_rows[:, 0] = torch.arange(1, inputs.row_slot_ids.numel() + 1, device="cuda")
    requests = DeviceRequestState.allocate("cuda", vocab_size=154880)
    metadata = DecoderMetadata.allocate(capacity, rows, 32, "cuda")
    batch = PreparedBatch(inputs, requests, metadata, mode=mode,
        shard=TensorParallelShard(rank, world), pages=torch.tensor([64], device="cuda"))

    batch.functional()
    def forward():
        validation, present = batch.validate()
        packed = batch.begin(validation * world, present)
        metadata.token_ids.copy_(packed.token_ids)
        for name in metadata.recurrent.__dataclass_fields__:
            getattr(metadata.recurrent, name).copy_(getattr(packed.recurrent, name))
        for name in metadata.indexed.__dataclass_fields__:
            getattr(metadata.indexed, name).copy_(getattr(packed.indexed, name))
        batch.finish(packed)

    call = compile_forward(forward)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    bundle = PreparedGraphBundle(device=0, graphs={"batch": graph},
        buffers={"tokens": metadata.token_ids, "slots": metadata.indexed.token_slots},
        resources=(batch, inputs, requests, metadata, call))
    worker = bundle.start_worker()
    try:
        faults = (None, "partial", "stale", "page", "token", "pending", "context")
        if mode == "decode":
            faults += ("duplicate",)
        for fault in faults:
            slots = [3] if mode == "prefill" else [3, 1, 0, 2]
            lengths = [capacity if fault != "partial" else 13] if mode == "prefill" else [1] * rows
            positions = [0, 63, 255, 256]
            if mode == "prefill" and fault is None:
                positions[3] = 0
            if fault == "context":
                positions[3] = 8192
            for slot in range(4):
                requests.admit(slot, RequestAdmission(7, 13, max_new_tokens=10000),
                    [3] * positions[slot], consumed_tokens=positions[slot])
            page_table = torch.stack([(torch.arange(32) + slot * 7) % 64 for slot in range(4)]).int()
            if fault == "page":
                page_table[3, 0] = 64
            token_ids = list(range(10, 10 + capacity))
            if fault == "token":
                token_ids[0] = 154880
            active = [True] * rows
            if mode == "decode":
                for row, slot in enumerate(slots):
                    requests.pending_token[slot] = token_ids[row]
            if fault == "pending":
                requests.pending_token[3] = -1 if mode == "decode" else 99
            if fault == "duplicate":
                slots[1] = 3
                active[1] = False
            epochs = [7] * rows
            if fault == "stale":
                epochs[0] = 6
            if fault not in (None, "partial"):
                active[0] = False
            inputs.row_slot_ids.fill_(-1)
            inputs.row_slot_ids[:rows].copy_(torch.tensor(slots, dtype=torch.int32))
            inputs.row_generations.zero_()
            inputs.row_generations[:rows].copy_(torch.tensor(epochs))
            inputs.query_lengths[:rows].copy_(torch.tensor(lengths, dtype=torch.int32))
            inputs.live_rows.fill_(rows)
            inputs.token_ids.copy_(torch.tensor(token_ids))
            inputs.page_tables.copy_(page_table)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(0)
            completion = worker.poll(timeout_ms=30000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key], key
            starts, expected_tokens, expected_positions, expected_requests, expected_slots = [0], [], [], [], []
            expected_consumed = positions.copy()
            for row, slot in enumerate(slots):
                count = lengths[row] if active[row] else 0
                starts.append(starts[-1] + count)
                expected_consumed[slot] += count
                expected_tokens.extend(token_ids[:count] if mode == "prefill" else token_ids[row:row + count])
                for position in range(positions[slot], positions[slot] + count):
                    expected_positions.append(position)
                    expected_requests.append(row)
                    expected_slots.append(int(page_table[slot, position // 256]) * 256 + position % 256)
            count = len(expected_tokens)
            g, index = metadata.recurrent, metadata.indexed
            assert g.query_start_loc.cpu().tolist() == starts
            assert index.query_start_loc.cpu().tolist() == starts
            assert metadata.token_ids[:count].cpu().tolist() == expected_tokens
            assert not metadata.token_ids[count:].count_nonzero()
            assert index.positions[:count].cpu().tolist() == expected_positions
            assert (index.positions[count:] == -1).all()
            assert index.token_slots[:count].cpu().tolist() == expected_slots
            assert (index.token_slots[count:] == -1).all()
            assert index.requests[:count].cpu().tolist() == expected_requests
            assert g.convolution_slots.cpu().tolist() == [slot + 1 if live else -1 for slot, live in zip(slots, active, strict=True)]
            assert g.initial_state_indices.cpu().tolist() == [slot + 1 if live and positions[slot] > 0 else 0 for slot, live in zip(slots, active, strict=True)]
            assert metadata.sample_indices.cpu().tolist() == [max(end - 1, 0) for end in starts[1:]]
            assert g.num_tokens.item() == count and g.num_seqs.item() == rows
            assert requests.consumed_tokens.cpu().tolist() == expected_consumed
            assert requests.status[3] == (3 if fault not in (None, "partial", "stale") else 1)
            assert requests.error_code[3] == (32 if fault not in (None, "partial", "stale") else 0)
            assert batch.token_state.history is None
    finally:
        worker.close()
        graph.reset()
