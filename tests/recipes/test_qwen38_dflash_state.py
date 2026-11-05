# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed DFlash2 verification metadata and accepted-prefix bookkeeping."""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned CUDA GPU")


@torch.no_grad()
def test_packed_target_metadata_capture_and_width_one_transition():
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.qwen38.decoder_state import DecoderInputs
    from ignition.recipes.qwen38.dflash import PreparedTargetState
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from ignition.recipes.qwen38.state import HybridStatePool

    device = torch.device("cuda:0")
    requests = DeviceRequestState.allocate(device, vocab_size=248320, capacity=4)
    pool = HybridStatePool.allocate(6, device, speculative_tokens=7)
    inputs = DecoderInputs.allocate(device, context=256, tokens=32, capacity=4)
    inputs.row_slot_ids.copy_(torch.tensor([2, 1, 0, -1], device=device, dtype=torch.int32))
    inputs.row_generations.fill_(7)
    inputs.live_rows.fill_(3)
    for slot in (0, 2):
        requests.admit(slot, RequestAdmission(7, 19), [11, 12] * 64, consumed_tokens=62 if slot == 2 else 128)
        requests.pending_token[slot] = 90 + slot
    pool.rope_delta[:, 0].copy_(torch.arange(6, device=device) * 10)
    previous = torch.tensor([2, 4, 6, 7], device=device, dtype=torch.int32)
    owner = SimpleNamespace(inputs=inputs, requests=requests, pool=pool,
        previous_accepted=previous, context=256)
    metadata = Metadata.allocate(12, 4, 4, device, samples=32, max_query_tokens=8)
    starts = torch.tensor([62, 70, 128, 0], device=device)
    lengths = torch.tensor([3, 0, 2, 0], device=device, dtype=torch.int32)
    rows = torch.tensor([3, 2, 1, 0], device=device)
    pages = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [-1] * 4],
                         device=device, dtype=torch.int32)
    tokens = torch.arange(32, device=device).reshape(4, 8) + 100
    metadata.set(starts, lengths, rows, pages, tokens, inputs.live_rows)
    state = PreparedTargetState(owner, metadata)
    hidden = torch.arange(12, device=device, dtype=torch.bfloat16)[:, None].expand(12, 2560).contiguous()
    sample = torch.empty((32, 2560), dtype=torch.bfloat16, device=device)
    state.prepare_hidden(hidden, sample)
    counts = torch.tensor([2, 0, 1, 0], device=device, dtype=torch.int32)

    def execute():
        state.begin()
        state.finish()
        state.retain_accepted(counts)

    execute()
    previous.copy_(torch.tensor([2, 4, 6, 7], device=device, dtype=torch.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    previous.copy_(torch.tensor([2, 4, 6, 7], device=device, dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    assert state.gdn.query_start_loc.tolist() == [0, 3, 3, 5, 5]
    assert state.gdn.convolution_slots.tolist() == [3, -1, 1, -1]
    assert state.gdn.accepted_tokens.tolist() == [6, 1, 2, 1]
    assert state.gdn.state_indices[:, 0].tolist() == [3, 0, 1, 0]
    assert state.qsa.raw_state_slot_ids.tolist() == [3, -1, 1, -1]
    assert state.qsa.request_ids.tolist() == [0, 0, 0, 2, 2, -1, -1, -1, -1, -1, -1, -1]
    assert state.qsa.query_positions[:5].tolist() == [62, 63, 64, 128, 129]
    assert state.qsa.rope_positions[:5, 0].tolist() == [92, 93, 94, 138, 139]
    assert state.qsa.slot_mapping[:5].tolist() == [126, 127, 128, 704, 705]
    assert state.qsa.sequence_lengths.tolist() == [65, 0, 130, 0]
    assert previous.tolist() == [1, 4, 2, 7]
    torch.testing.assert_close(sample, hidden[metadata.sample_indices], rtol=0, atol=0)

    counts.copy_(torch.tensor([1, 0, 2, 0], device=device, dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    assert previous.tolist() == [2, 4, 1, 7]
    assert state.gdn.accepted_tokens.tolist() == [2, 1, 1, 1]
    graph.reset()

    single = Metadata.allocate(4, 4, 4, device, max_query_tokens=1)
    single.set(starts, torch.tensor([1, 0, 1, 0], device=device, dtype=torch.int32),
        rows, pages, tokens[:, :1], inputs.live_rows)
    narrow = PreparedTargetState(owner, single)
    narrow.prepare_hidden(hidden[:4], sample[:4])
    narrow.begin()
    assert narrow.gdn.accepted_tokens.tolist() == [1, 1, 2, 1]
    narrow.finish()
    assert previous.tolist() == [1, 4, 1, 7]


@torch.no_grad()
def test_dflash_snapshot_restore_and_slot_reuse_normalize_selected_tail():
    from ignition.components.dflash2_geometry import DraftAttentionGeometry
    from ignition.components.sampling import AdmissionBuffers
    from ignition.recipes.qwen38.attention import QSAState
    from ignition.recipes.qwen38.decoder_state import DecoderInputs
    from ignition.recipes.qwen38.maintenance import PreparedMaintenance
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from ignition.recipes.qwen38.state import HybridStatePool, StateTransferMetadata
    from tests.components.completion_support import completion_outputs

    device = torch.device("cuda:0")
    config = SimpleNamespace(layers=5,
        attention=DraftAttentionGeometry(32, 8, 128, 128, 128, 10000., 2048))
    pool = HybridStatePool.allocate(4, device, speculative_tokens=7, draft_config=config)
    requests = DeviceRequestState.allocate(device, vocab_size=128, capacity=2,
        consumed_tokens=pool.consumed_tokens[1:3, 0], ple_history=pool.ple_history[1:3])
    inputs = DecoderInputs.allocate(device, context=256, tokens=16, capacity=2)
    inputs.request_state_rows.copy_(torch.tensor([[1] * 8, [2] * 8], device=device))
    inputs.page_tables.copy_(torch.tensor([[0, 1, 2, 3]] * 2, device=device, dtype=torch.int32))
    inputs.row_slot_ids[0] = 0
    inputs.row_generations[0] = 7
    inputs.live_rows.fill_(1)
    qsa = tuple(QSAState.allocate(pages=4, slots=3, device=device, draft_tokens=7) for _ in range(12))
    previous = torch.tensor([6, 7], device=device, dtype=torch.int32)
    owner = SimpleNamespace(previous_accepted=previous, draft_tables=None)
    metadata = StateTransferMetadata.allocate(2, device)
    admissions = AdmissionBuffers.allocate(device, vocab_size=128, capacity=2)
    records = torch.zeros((2, 2, 32), dtype=torch.int64, device=device)
    common = (requests, pool, metadata, completion_outputs(records),
        torch.zeros(1, dtype=torch.int32, device=device), torch.ones(1, dtype=torch.int64, device=device))
    graphs = {}
    for mode in ("snapshot", "restore", "init"):
        operation = PreparedMaintenance(mode, *common, admission_buffers=admissions,
            forward_inputs=inputs, qsa_states=qsa, capacity=2, speculation=owner)
        operation.prepare()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        graphs[mode] = (graph, operation)
    requests.admit(0, RequestAdmission(7, 19, eos_token_ids=()), [1, 2] * 33, consumed_tokens=65)
    convolution = pool.gdn[0].convolution
    convolution[1].copy_(torch.arange(10, device=device, dtype=torch.bfloat16)[None, :].expand(10240, 10))
    pool.ple_convolution[1].copy_(torch.arange(16, device=device, dtype=torch.bfloat16)[None, :].expand(10240, 16))
    ring = pool.region("draft0.key")
    ring[1].fill_(0.5)
    metadata.request_slots[0] = 0
    metadata.source_slots[0] = 1
    metadata.destination_slots[0] = 3
    metadata.operation[0] = 0
    metadata.qsa_slots[0] = -1
    metadata.accepted_tokens[0] = 0
    metadata.expected_consumed[0] = 65
    metadata.num_ops.fill_(1)
    graphs["snapshot"][0].replay()
    torch.cuda.synchronize()
    assert metadata.errors[0].item() == 0
    assert convolution[3, 0].tolist() == [5, 6, 7, 0, 0, 0, 0, 0, 0, 0]
    assert pool.ple_convolution[3, 0].tolist() == list(range(5, 14)) + [0] * 7
    assert torch.equal(ring[3].view(torch.uint8), ring[1].view(torch.uint8))
    assert previous.tolist() == [6, 7]
    admissions.stage(0, 1, RequestAdmission(8, 20, eos_token_ids=()), [1, 2] * 33)
    admissions.admission_count.fill_(1)
    metadata.request_slots[0] = 1
    metadata.source_slots[0] = 3
    metadata.destination_slots[0] = 2
    metadata.qsa_slots[0] = 2
    metadata.accepted_tokens[0] = 1
    graphs["restore"][0].replay()
    torch.cuda.synchronize()
    assert metadata.errors[0].item() == 0
    assert previous.tolist() == [6, 1]
    assert torch.equal(convolution[2], convolution[3])
    assert torch.equal(ring[2].view(torch.uint8), ring[3].view(torch.uint8))
    previous[1] = 4
    admissions.stage(0, 1, RequestAdmission(9, 21, eos_token_ids=()), [1, 2])
    metadata.source_slots[0] = 2
    metadata.operation[0] = 1
    metadata.expected_consumed[0] = 0
    graphs["init"][0].replay()
    torch.cuda.synchronize()
    assert metadata.errors[0].item() == 0
    assert previous.tolist() == [6, 1]
    assert convolution[2].count_nonzero().item() == 0
    assert ring[2].view(torch.uint8).count_nonzero().item() == 0
    for graph, _ in graphs.values():
        graph.reset()
