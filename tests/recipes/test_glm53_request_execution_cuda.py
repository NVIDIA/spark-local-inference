# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM request metadata and completions through native uploads and transport frames.

The graphs exercise request processing with fixed test logits; model layers are
qualified by the full-checkpoint tensor-parallel tests.
"""

import os
import time

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import (
    AdmissionBuffers,
    DeviceRequestState,
    PreparedAdmission,
    PreparedCompletion,
    PreparedSampling,
    RequestAdmission,
)
from ignition.preparation.compilation import compile_forward
from ignition.preparation.execution import GraphSignature, PreparedExecution
from ignition.recipes.glm53.batch import BatchInputs, PreparedBatch
from ignition.recipes.glm53.state import DecoderMetadata

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("fields", [32])
@torch.no_grad()
def test_native_glm_admission_prefill_decode_and_rank_completion(rank, fields):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    requests = DeviceRequestState.allocate("cuda", vocab_size=154880)
    inputs = BatchInputs.allocate(128, 32, "cuda")
    admission = AdmissionBuffers.allocate("cuda", vocab_size=requests.vocab_size)
    records = torch.zeros((2, 4, fields), dtype=torch.int64, device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.zeros(1, dtype=torch.int64, device="cuda")
    init = PreparedAdmission(requests, admission, completion_outputs(records), frame, sequence, capacity=1)
    init.prepare()
    operations, graphs, signatures, owners = {"init": init}, {}, {}, [init]
    signatures["init"] = GraphSignature("init", 1, 1, tuple(admission.__dataclass_fields__))
    for mode, capacity in (("prefill", 128), ("decode", 1)):
        metadata = DecoderMetadata.allocate(capacity, 1, 32, "cuda")
        batch = PreparedBatch(inputs, requests, metadata, mode=mode,
            shard=TensorParallelShard(rank, 4), pages=torch.tensor([64], device="cuda"))
        arguments = (requests, inputs.row_slot_ids[:1], inputs.row_generations[:1],
                     inputs.live_rows, records, frame, sequence)
        if rank == 0:
            logits = torch.full((1, requests.vocab_size), -torch.inf, device="cuda", dtype=torch.bfloat16)
            logits[0, 11:14] = 0
            completion = PreparedSampling(logits, *arguments, sample_enabled=inputs.sample_enabled[:1])
        else:
            completion = PreparedCompletion(*arguments)
        completion.prepare()

        batch.functional()
        def forward(batch=batch, completion=completion):
            validation, present = batch.validate()
            metadata = batch.begin(validation * batch.shard.world_size, present)
            batch.finish(metadata)
            completion()

        operations[mode] = compile_forward(forward)
        signatures[mode] = GraphSignature(mode, 1, capacity, tuple(inputs.buffers()))
        owners.extend((batch, metadata, completion))
    for name, operation in operations.items():
        operation()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        graphs[name] = graph
    buffers = inputs.buffers()
    buffers.update({name: getattr(admission, name) for name in admission.__dataclass_fields__})
    buffers.update(completion_records=records, frame_index=frame, submission_sequence=sequence)
    prepared = PreparedExecution(device=0, graphs=graphs, buffers=buffers, signatures=signatures,
        resources=(requests, inputs, admission, operations, owners))
    executor = prepared.start_executor()

    def replay(name, fields):
        uploads = [(name, 0, value.numpy().tobytes()) for name, value in fields.items()]
        before = torch.cuda.memory_stats()
        fence = executor.submit(prepared.graph_indices[name], uploads)
        assert fence is not None
        deadline = time.monotonic() + 30
        result = None
        while result is None and time.monotonic() < deadline:
            result = executor.poll()
            if result is None:
                time.sleep(.001)
        assert result is not None and result.fence == fence
        after = torch.cuda.memory_stats()
        for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
            assert before[key] == after[key], key
        return result.records[2]

    try:
        host_admission = AdmissionBuffers.allocate("cpu", vocab_size=requests.vocab_size)
        host_admission.stage(0, 2, RequestAdmission(7, 13, max_new_tokens=2), list(range(65)))
        host_admission.admission_count.fill_(1)
        record = replay("init", {name: getattr(host_admission, name) for name in host_admission.__dataclass_fields__})
        assert record[:8] == [7, 1, 0, 0, -1, 1, 0, 0]
        host = BatchInputs.allocate(128, 32, "cpu")
        host.row_slot_ids[0] = 2
        host.row_generations[0] = 7
        host.live_rows.fill_(1)
        host.query_lengths[0] = 65
        host.token_ids[:65].copy_(torch.arange(65))
        host.page_tables[2].copy_(torch.arange(32) + 7)
        host.sample_enabled[0] = 1
        record = replay("prefill", host.buffers())
        assert record[:4] == [7, 2, 65, int(rank == 0)]
        assert record[5:8] == [1, 0, 0]
        if rank == 0:
            assert record[4] in (11, 12, 13)
        else:
            assert record[4] == -1
        host.query_lengths[0] = 1
        host.token_ids[0] = record[4] if rank == 0 else 13
        record = replay("decode", host.buffers())
        assert record[:4] == [7, 3, 66, 2 if rank == 0 else 0]
        assert record[5:8] == [2 if rank == 0 else 1, 0, 0]
        assert requests.rng_counter[2].item() == (2 if rank == 0 else 0)
        assert requests.histogram[2].sum().item() == 65 + (2 if rank == 0 else 0)
        if fields == 32:
            assert record[8] == int(rank == 0)
            assert record[16] == record[4]
        assert executor.outstanding == 0
    finally:
        executor.close()
        assert executor.drained
        for graph in graphs.values():
            graph.reset()
