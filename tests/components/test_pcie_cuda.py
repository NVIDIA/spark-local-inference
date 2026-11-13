# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiled b12x tensor-parallel graphs replayed by native workers on every rank."""

import os
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires explicitly assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1")


def _rank(rank, world_size, rendezvous, rows, uneven=False):
    from b12x.comm.pcie import AllReduce
    from b12x.preparation import PreparationSession

    from ignition.components.collectives import PreparedAllReduce
    from ignition.components.linear import UnquantizedProjection
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=world_size, timeout=timedelta(minutes=2))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    source = torch.zeros(rows, 4096, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    seed = torch.full_like(source, rank + 1)
    result = torch.empty_like(source)
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    runtime = AllReduce.from_exchange_group(exchange_group=group, device=device, single_channel=True)
    graph = torch.cuda.CUDAGraph()
    worker = None
    try:
        preparation = RecordedPreparationSession(device=device, checkpoint="collective-test",
            workload_schema="uneven-rank-preparation-v1", numerical_policy="default",
            coordinator=coordinator) if uneven else PreparationSession(device=device, autotune=True,
                compile_workers=0, collective_barrier=coordinator.barrier)
        with preparation as session:
            reduction = PreparedAllReduce(source, output, runtime=runtime, session=session,
                coordinator=coordinator, name=f"reduce{rows}")
            projection, second = None, None
            if uneven:
                if rank == 0:
                    projection_output = torch.empty((rows, 64), dtype=torch.bfloat16, device=device)
                    projection = UnquantizedProjection(source,
                        torch.full((64, 4096), .125, dtype=torch.bfloat16, device=device), projection_output,
                        session=session, name="vocabulary")
                second_output = torch.empty_like(source)
                second = PreparedAllReduce(output, second_output, runtime=runtime, session=session,
                    coordinator=coordinator, name=f"second{rows}")
                session.tune()

            def forward():
                source.copy_(seed + counter.float() * .125)
                value = reduction(source)
                if projection is not None:
                    projection(source)
                if second is not None:
                    value = second(value)
                result.copy_(value * 2)
                counter.add_(1)

            dist.barrier(group=group)
            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                compiled()
            torch.cuda.synchronize(device)
            session.freeze()
            bundle = PreparedGraphBundle(device=rank, graphs={"reduce": graph},
                buffers={"source": source, "output": output, "seed": seed,
                         "result": result, "counter": counter},
                resources=(compiled, reduction, second, projection, runtime, session, stream))
            worker = bundle.start_worker()
            for repetitions, offset in ((8, 0.), (31, .5)):
                counter.zero_()
                seed.fill_(rank + 1 + offset)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                dist.barrier(group=group)
                ticket = worker.submit(0, repetitions=repetitions)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket
                assert completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                expected = 2 * (world_size * (world_size + 1) / 2
                                + world_size * (offset + (repetitions - 1) * .125))
                if uneven:
                    expected *= world_size
                torch.testing.assert_close(result, torch.full_like(result, expected), rtol=0, atol=0)
                assert counter.item() == repetitions
                dist.barrier(group=group)
            worker.close()
            worker = None
            graph.reset()
    finally:
        if worker is not None:
            worker.close()
        graph.reset()
        if sys.exception() is None:
            runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
@pytest.mark.parametrize("rows", [1, 8])
def test_native_compiled_all_reduce(world_size, rows, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} assigned GPUs")
    mp.spawn(_rank, args=(world_size, (tmp_path / "rendezvous").as_uri(), rows),
             nprocs=world_size, join=True)


@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_rank_local_projection_between_collectives(world_size, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} assigned GPUs")
    mp.spawn(_rank, args=(world_size, (tmp_path / "rendezvous").as_uri(), 1, True),
             nprocs=world_size, join=True)


def _chunked_rank(rank, world_size, rendezvous, rows, capacity):
    from b12x.comm.pcie import AllReduce

    from ignition.components.collectives import PreparedChunkedAllReduce
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=world_size, timeout=timedelta(minutes=2))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    source = torch.arange(rows, device=device, dtype=torch.float32)[:, None].expand(rows, 4096).bfloat16() + rank
    output = torch.empty_like(source)
    runtime = AllReduce.from_exchange_group(exchange_group=group, device=device, single_channel=True,
                                            max_size=capacity * 4096 * 2)
    try:
        with RecordedPreparationSession(device=device, checkpoint="collective-test",
                workload_schema="chunked-reduction-v1", numerical_policy="default", coordinator=coordinator) as session:
            reduction = PreparedChunkedAllReduce(source, output, runtime=runtime, session=session,
                                                 coordinator=coordinator, name=f"chunked{rows}")
            session.tune()
        # A channel for `capacity` rows reduces the larger source in row chunks it accepts.
        assert len(reduction.parts) == -(-rows // capacity)
        reduction(source)
        torch.cuda.synchronize(device)
        expected = (source.float() - rank) * world_size + world_size * (world_size - 1) / 2
        torch.testing.assert_close(output.float(), expected, rtol=0, atol=0)
    finally:
        if sys.exception() is None:
            runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_chunked_all_reduce_reduces_rows_beyond_the_channel_capacity(world_size, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} assigned GPUs")
    mp.spawn(_chunked_rank, args=(world_size, (tmp_path / "rendezvous").as_uri(), 64, 16),
             nprocs=world_size, join=True)


def _chunked_gather_rank(rank, world_size, rendezvous, rows, capacity):
    from ignition.components.collectives import PreparedChunkedAllGather
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=world_size, timeout=timedelta(minutes=2))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    columns = 64 * world_size
    source = (torch.arange(rows, device=device, dtype=torch.float32)[:, None] * 8 + rank).expand(rows, 64).bfloat16()
    output = torch.empty((rows, columns), dtype=torch.bfloat16, device=device)
    runtime = create_all_gather_runtime(group, device, max_rows=capacity, columns=columns)
    try:
        with RecordedPreparationSession(device=device, checkpoint="collective-test",
                workload_schema="chunked-gather-v1", numerical_policy="default", coordinator=coordinator) as session:
            gather = PreparedChunkedAllGather(source.contiguous(), output, runtime=runtime, session=session,
                                              coordinator=coordinator, name=f"gather{rows}")
            session.tune()
        # A channel for `capacity` rows gathers the larger source in row chunks it accepts.
        assert len(gather.parts) == -(-rows // capacity)
        gather(source.contiguous())
        torch.cuda.synchronize(device)
        for peer in range(world_size):
            expected = (torch.arange(rows, device=device, dtype=torch.float32)[:, None] * 8 + peer).expand(rows, 64)
            torch.testing.assert_close(output[:, peer * 64:(peer + 1) * 64], expected.bfloat16(), rtol=0, atol=0)
    finally:
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_chunked_all_gather_gathers_rows_beyond_the_channel_capacity(world_size, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} assigned GPUs")
    mp.spawn(_chunked_gather_rank, args=(world_size, (tmp_path / "rendezvous").as_uri(), 64, 16),
             nprocs=world_size, join=True)
