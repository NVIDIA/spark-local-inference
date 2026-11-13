# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""b12x agreement keeps packed GLM rows identical across TP ranks."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four assigned GPUs and FLARE_RUN_TP_CUDA_TESTS=1")


def _rank(rank, world, rendezvous):
    from ignition.components.collectives import PreparedAllReduce
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_reduce_runtime,
    )
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.batch import BatchInputs, PreparedBatch
    from ignition.recipes.glm53.state import DecoderMetadata

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=2))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    runtime = create_all_reduce_runtime(group, device, max_tokens=1, hidden_size=8)
    graph, worker = torch.cuda.CUDAGraph(), None
    try:
        inputs = BatchInputs.allocate(4, 32, device)
        requests = DeviceRequestState.allocate(device, vocab_size=154880)
        metadata = DecoderMetadata.allocate(4, 4, 32, device)
        batch = PreparedBatch(inputs, requests, metadata, mode="decode",
            shard=TensorParallelShard(rank, world), pages=torch.tensor([64], device=device))
        with RecordedPreparationSession(device=device, checkpoint="glm-request-validation",
                workload_schema="glm-batch-tp-v1", numerical_policy="default", coordinator=coordinator) as session:
            validation = torch.empty((1, 8), dtype=torch.bfloat16, device=device)
            totals = torch.empty_like(validation)
            agreement = PreparedAllReduce(validation, totals,
                runtime=runtime, coordinator=coordinator, session=session, name="batch.validation")
            batch.functional()
            agreement.functional(session=session)
            session.tune()

            def forward():
                validation, present = batch.validate()
                reduced = agreement(validation)
                totals.copy_(reduced)
                packed = batch.begin(reduced, present)
                metadata.token_ids.copy_(packed.token_ids)
                for name in metadata.recurrent.__dataclass_fields__:
                    getattr(metadata.recurrent, name).copy_(getattr(packed.recurrent, name))
                for name in metadata.indexed.__dataclass_fields__:
                    getattr(metadata.indexed, name).copy_(getattr(packed.indexed, name))
                batch.finish(packed)

            call = compile_forward(forward)
            call()
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                call()
            torch.cuda.synchronize(device)
            session.freeze()
            bundle = PreparedGraphBundle(device=rank, graphs={"batch": graph},
                buffers={"tokens": metadata.token_ids, "flags": totals},
                resources=(requests, inputs, metadata, batch, agreement, call, runtime, session, stream))
            worker = bundle.start_worker()
            for fault in (None, "pending", "epoch"):
                for slot in range(4):
                    requests.admit(slot, RequestAdmission(7, 13), [3] * 255, consumed_tokens=255)
                inputs.row_slot_ids.copy_(torch.tensor([3, 1, 0, -1], dtype=torch.int32))
                inputs.row_generations.fill_(7)
                inputs.live_rows.fill_(3)
                inputs.token_ids.copy_(torch.tensor([7, 8, 9, 0]))
                inputs.page_tables.copy_(torch.arange(32, dtype=torch.int32)[None, :].expand(4, -1))
                requests.pending_token.copy_(torch.tensor([9, 8, -1, 7]))
                if fault == "pending" and rank == 0:
                    requests.pending_token[3] = -1
                if fault == "epoch" and rank == world - 1:
                    requests.generation[1] = 8
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                dist.barrier(group=group)
                ticket = worker.submit(0)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key], key
                expected_tokens = [7, 8, 9, 0] if fault is None else [8, 9, 0, 0] if fault == "pending" else [7, 9, 0, 0]
                assert metadata.token_ids.cpu().tolist() == expected_tokens
                assert requests.consumed_tokens[0] == 256
                if fault == "pending":
                    assert requests.status[3] == 3 and requests.error_code[3] == 32
                    assert requests.consumed_tokens[3] == 255
                if fault == "epoch":
                    assert requests.status[1] == (1 if rank == world - 1 else 3)
                    assert requests.consumed_tokens[1] == 255
                values = torch.cat((metadata.recurrent.query_start_loc.long(), metadata.indexed.positions,
                                    metadata.indexed.token_slots, metadata.token_ids)).cpu()
                gathered = [None] * world
                dist.all_gather_object(gathered, values, group=group)
                for peer in gathered:
                    torch.testing.assert_close(peer, values, rtol=0, atol=0)
            worker.close()
            worker = None
            graph.reset()
    finally:
        if worker is not None:
            worker.close()
        graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_native_glm_request_agreement(world, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 4
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)
