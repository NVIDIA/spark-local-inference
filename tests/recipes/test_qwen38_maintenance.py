# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Complete hybrid-row transitions and admission ordering through native replay."""

import os
from types import SimpleNamespace

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.components.sampling import AdmissionBuffers
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.qwen38.attention import QSAState
from ignition.recipes.qwen38.maintenance import PreparedMaintenance
from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
from ignition.recipes.qwen38.state import HybridStatePool, StateTransferMetadata


def test_maintenance_rejects_unknown_operation_before_allocation():
    with pytest.raises(ValueError, match="init/restore/snapshot"):
        PreparedMaintenance("copy-anywhere", None, None, None, None, None, None)


CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and explicitly assigned CUDA_VISIBLE_DEVICES")


@pytest.fixture(scope="module")
def backing():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    pool = HybridStatePool.allocate(7, "cuda")
    requests = DeviceRequestState.allocate("cuda", vocab_size=128,
        consumed_tokens=pool.consumed_tokens[1:5, 0], ple_history=pool.ple_history[1:5])
    qsa = tuple(QSAState.allocate(pages=4, slots=5, device=torch.device("cuda:0")) for _ in range(12))
    return pool, requests, qsa


def _staging():
    metadata = StateTransferMetadata.allocate(4, "cuda")
    admissions = AdmissionBuffers.allocate("cuda", vocab_size=128)
    inputs = SimpleNamespace(request_state_rows=torch.arange(1, 5, device="cuda")[:, None].expand(4, 8).contiguous(), row_slot_ids=torch.full((4,), -1, dtype=torch.int32, device="cuda"),
        row_generations=torch.zeros(4, dtype=torch.int64, device="cuda"),
        live_rows=torch.zeros(1, dtype=torch.int32, device="cuda"),
        page_tables=torch.tensor([[0, 1, 2, 3]] * 4, dtype=torch.int32, device="cuda"))
    records = torch.full((2, 4, 32), -22, dtype=torch.int64, device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.ones(1, dtype=torch.int64, device="cuda")
    return metadata, admissions, inputs, records, frame, sequence


def _command(metadata, *, source, destination, operation, expected, qsa_slot=-1, row=0):
    metadata.request_slots[row] = (qsa_slot if qsa_slot >= 0 else source) - 1
    metadata.source_slots[row] = source
    metadata.destination_slots[row] = destination
    metadata.operation[row] = operation
    metadata.qsa_slots[row] = qsa_slot
    metadata.accepted_tokens[row] = 1 if qsa_slot >= 0 else 0
    metadata.expected_consumed[row] = expected


def _replay(worker, index=0):
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    ticket = worker.submit(index)
    result = worker.poll(timeout_ms=30000)
    assert result.ticket == ticket and result.error is None
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations


def _worker(operations, records, resources):
    graphs = {}
    for operation in operations:
        operation.prepare()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        graphs[operation.mode] = graph
    return PreparedGraphBundle(device=0, graphs=graphs, buffers={"records": records},
                               resources=tuple(operations) + resources).start_worker()


@CUDA
def test_native_initialize_publish_restore_preserves_exact_hybrid_bytes(backing):
    pool, requests, qsa = backing
    metadata, admissions, inputs, records, frame, sequence = _staging()
    for slot in range(4):
        requests.admit(slot, RequestAdmission(1, 51 + slot, eos_token_ids=()), [1, 2, 3],
                       consumed_tokens=3, ple_history=(2, 3))
    common = (requests, pool, metadata, completion_outputs(records), frame, sequence)
    init = PreparedMaintenance("init", *common, admission_buffers=admissions, forward_inputs=inputs, qsa_states=qsa)
    restore = PreparedMaintenance("restore", *common, admission_buffers=admissions, forward_inputs=inputs, qsa_states=qsa)
    snapshot = PreparedMaintenance("snapshot", *common, forward_inputs=inputs, qsa_states=qsa)
    worker = _worker((init, restore, snapshot), records, (pool, requests, qsa, metadata, admissions, inputs))
    try:
        pool.backing[3].fill_(0x5A)
        for state in qsa:
            state.raw_logical_positions[3].fill_(123)
            state.raw_rope_positions[3].fill_(456)
            state.raw_interval_start_positions[3] = 55
        unrelated = pool.backing[1].clone()
        requests.pending_token[0] = 77
        admissions.stage(0, 2, RequestAdmission(10, 2**64 - 1, eos_token_ids=()), list(range(65)))
        admissions.admission_count.fill_(1)
        _command(metadata, source=3, destination=3, operation=1, expected=0, qsa_slot=3)
        metadata.num_ops.fill_(1)
        _replay(worker, 0)
        assert records[0, 2].cpu()[..., :8].tolist() == [10, 1, 0, 0, -1, 1, 0, 0]
        assert requests.ple_history[2].cpu().tolist() == [248044, 248044]
        assert requests.seed[2].item() == 2**64 - 1
        assert requests.histogram[2].sum() == 65
        assert pool.last_hidden[3].count_nonzero() == 0
        assert pool.gdn[0].recurrent[3].count_nonzero() == 0
        for state in qsa:
            assert state.raw_logical_positions[3].eq(-1).all()
            assert state.raw_rope_positions[3].eq(-1).all()
            assert state.raw_interval_start_positions[3] == -1
        assert torch.equal(pool.backing[1], unrelated) and requests.pending_token[0] == 77

        # The live row retains the exact consumed endpoint used by all36 recurrent layers.
        pool.backing[3].fill_(5)
        pool.consumed_tokens[3] = 65
        pool.ple_history[3].copy_(torch.tensor([63, 64]))
        pool.last_hidden[3].fill_(.5)
        pool.rope_delta[3] = 7
        requests.consumed_tokens[2] = 65
        inputs.row_slot_ids[0] = 2
        inputs.row_generations[0] = 10
        inputs.live_rows.fill_(1)
        _command(metadata, source=3, destination=5, operation=0, expected=65)
        sequence.fill_(2)
        _replay(worker, 2)
        prefix_bytes = pool.layout.region("qsa.3.raw_keys").offset
        assert torch.equal(pool.backing[3, :prefix_bytes], pool.backing[5, :prefix_bytes])
        assert records[0, 2].cpu()[..., :8].tolist() == [10, 2, 65, 0, -1, 1, 0, 0]
        pool.backing[3, 0] = 19
        assert pool.backing[5, 0] == 5

        admissions.stage(0, 1, RequestAdmission(20, 33, eos_token_ids=()), list(range(65)))
        _command(metadata, source=5, destination=2, operation=0, expected=65, qsa_slot=2)
        frame.fill_(1)
        sequence.fill_(3)
        _replay(worker, 1)
        assert torch.equal(pool.backing[2], pool.backing[5])
        assert records[1, 1].cpu()[..., :8].tolist() == [20, 3, 65, 0, -1, 1, 0, 0]
        assert requests.histogram[1].sum() == 65 and requests.generated_tokens[1] == 0
        assert requests.pending_token[1] == -1
        for state in qsa:
            assert state.raw_interval_start_positions[2] == 64
            assert state.raw_logical_positions[2].eq(-1).all()
        assert torch.equal(pool.backing[1], unrelated) and requests.pending_token[0] == 77
    finally:
        worker.close()


@CUDA
@pytest.mark.parametrize("fault,mode,code", [
    ("bad_admission", "init", 5), ("wrong_destination", "init", 72),
    ("wrong_restore_endpoint", "restore", 68), ("wrong_snapshot_endpoint", "snapshot", 73),
    ("stale_generation", "snapshot", 72), ("duplicate_destination", "snapshot", 70),
])
def test_maintenance_failure_precedes_mutation_and_preserves_other_generations(backing, fault, mode, code):
    pool, requests, qsa = backing
    metadata, admissions, inputs, records, frame, sequence = _staging()
    for slot in (1, 2):
        requests.admit(slot, RequestAdmission(10 + slot, 51, eos_token_ids=()), list(range(128)),
                       consumed_tokens=128, ple_history=(126, 127))
        requests.pending_token[slot] = 7
    admissions.stage(0, 2, RequestAdmission(22, 44, eos_token_ids=()), list(range(65)))
    admissions.admission_count.fill_(1)
    inputs.row_slot_ids[0], inputs.row_generations[0] = 2, 12
    inputs.live_rows.fill_(1)
    if mode == "init":
        _command(metadata, source=3, destination=3, operation=1, expected=0, qsa_slot=3)
    elif mode == "restore":
        _command(metadata, source=5, destination=3, operation=0, expected=64, qsa_slot=3)
    else:
        _command(metadata, source=3, destination=5, operation=0, expected=128)
    if fault == "bad_admission":
        admissions.admission_parameters[0, 0] = float("nan")
    elif fault == "wrong_destination":
        metadata.destination_slots[0] = 2
    elif fault == "wrong_restore_endpoint":
        pool.consumed_tokens[5] = 65
    elif fault == "wrong_snapshot_endpoint":
        metadata.expected_consumed[0] = 127
    elif fault == "stale_generation":
        inputs.row_generations[0] = 11
    elif fault == "duplicate_destination":
        inputs.live_rows.fill_(2)
        inputs.row_slot_ids[1], inputs.row_generations[1] = 1, 11
        _command(metadata, source=2, destination=5, operation=0, expected=128, row=1)
    operation = PreparedMaintenance(mode, requests, pool, metadata, completion_outputs(records), frame, sequence,
        admission_buffers=admissions if mode != "snapshot" else None,
        forward_inputs=inputs, qsa_states=qsa)
    worker = _worker((operation,), records, (pool, requests, qsa, admissions, inputs))
    before_rows = pool.backing[[2, 3, 5]].clone()
    before_counts = requests.histogram.clone()
    before_epochs = requests.generation.clone()
    before_pending = requests.pending_token.clone()
    before_status = requests.status.clone()
    before_tags = qsa[0].raw_logical_positions.clone()
    try:
        metadata.num_ops.fill_(2 if fault == "duplicate_destination" else 1)
        _replay(worker)
        assert torch.equal(pool.backing[[2, 3, 5]], before_rows)
        assert torch.equal(requests.histogram, before_counts)
        assert torch.equal(requests.generation, before_epochs)
        assert torch.equal(requests.pending_token, before_pending)
        assert torch.equal(requests.status, before_status)
        assert torch.equal(qsa[0].raw_logical_positions, before_tags)
        assert records[0, 2, 5] == 3 and records[0, 2, 6] == code
        assert records[0, 2, 4] == -1
        if fault == "stale_generation":
            assert records[0, 2, 0] == 11 and records[0, 2, 2:4].sum() == 0
    finally:
        worker.close()
