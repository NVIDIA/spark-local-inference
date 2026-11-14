# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TP readiness timing separates delivery stalls from peer computation and reduction time."""

from datetime import timedelta
import os
import threading
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl
from triton.language.extra.cuda import globaltimer


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires explicitly assigned tensor-parallel GPUs")


@triton.jit
def _work(NANOSECONDS: tl.constexpr):
    started = globaltimer()
    while globaltimer() - started < NANOSECONDS:
        pass


def _rank(rank, world, rendezvous):
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    from cuda.bindings import driver
    from ignition.components.collectives import _all_reduce
    from ignition.components.readiness import PreparedReadinessWait, ReadinessTimings, record_timestamp, wait_for_readiness
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    flag = MappedHostAllocation((1,), torch.int32, device)
    progress = MappedHostAllocation((1,), torch.int32, device)
    flag.host_view.zero_()
    progress.host_view.zero_()
    generation = 0
    try:
        with CoreStorage(rank=rank, world=world, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=1,
                                               hidden_size=8, ipc=storage.peer_storage)
            try:
                with RecordedPreparationSession(device=device, checkpoint="readiness-timing",
                        workload_schema="readiness-intervals", numerical_policy="default",
                        coordinator=coordinator, compile_workers=4) as session:
                    frame = torch.zeros(1, dtype=torch.int32, device=device)
                    timings = ReadinessTimings.allocate(frame, 1)
                    origin = torch.zeros((2, 1), dtype=torch.int64, device=device)
                    wait = PreparedReadinessWait(timings, 0, runtime=runtime, session=session,
                                                 coordinator=coordinator, name="readiness")
                    session.tune(show_progress=False)
                    ready = torch.cuda.Event(external=True)
                    ready.record()
                    compute, delivery = torch.cuda.Stream(), torch.cuda.Stream()
                    graphs = []
                    for work_ns in (0, 40_000_000, 10_000_000):
                        local_work = work_ns if rank == 1 else 0

                        def forward():
                            _all_reduce(wait.input, wait.output, wait.handle - 1)
                            record_timestamp(origin, frame, 0)
                            progress.device_view.fill_(1)
                            _work[(1,)](local_work, num_warps=1)
                            wait_for_readiness(frame, ready.cuda_event, timings.timestamps, frame, 0,
                                               wait.input, wait.output, wait.handle)

                        forward()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with runtime.capture(stream=compute), session.capture(), torch.cuda.graph(graph, stream=compute):
                            forward()
                        graphs.append(graph)
                    session.freeze()
                    torch.cuda.synchronize()
                    for generation, (graph, delay) in enumerate(zip(graphs, (.02, .02, .03), strict=True), 1):
                        index = generation % 2
                        frame.fill_(index)
                        progress.host_view.zero_()
                        if rank != 0:
                            flag.host_view.fill_(generation)
                        torch.cuda.synchronize()
                        with torch.cuda.stream(delivery):
                            status, = driver.cuStreamWaitValue32(driver.CUstream(delivery.cuda_stream),
                                flag.device_view.data_ptr(), generation,
                                driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ)
                            assert int(status) == 0
                            ready.record(delivery)

                        def release():
                            deadline = time.monotonic() + 10
                            while progress.host_view.item() != 1 and time.monotonic() < deadline:
                                time.sleep(.0001)
                            time.sleep(delay)
                            flag.host_view.fill_(generation)

                        worker = threading.Thread(target=release) if rank == 0 else None
                        dist.barrier()
                        if worker is not None:
                            worker.start()
                        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                        with torch.cuda.stream(compute):
                            graph.replay()
                        compute.synchronize()
                        if worker is not None:
                            worker.join()
                        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                        stamps = timings.timestamps.cpu()[index].tolist()
                        offset = origin.cpu()[index, 0].item()
                        samples = [None] * world
                        dist.all_gather_object(samples, [value - offset for value in stamps])
                        through = min(end - begin for begin, _, end in samples)
                        barrier = min(end - ready_at for _, ready_at, end in samples)
                        exposed = through - barrier
                        reference = max(value[1] for value in samples) - max(value[0] for value in samples)
                        assert abs(exposed - reference) < 500_000
                        if generation == 2:
                            assert exposed < 500_000
                        else:
                            assert 12_000_000 < exposed < 80_000_000
                        if rank == 0:
                            print(f"case={generation} exposed_ns={exposed} barrier_ns={barrier} rank_timestamps={samples}", flush=True)
                    for graph in graphs:
                        graph.reset()
            finally:
                runtime.close()
    finally:
        flag.host_view.fill_(generation)
        torch.cuda.synchronize()
        flag.close()
        progress.close()
        dist.destroy_process_group()


def test_readiness_wait_with_stalled_and_computing_peers(tmp_path):
    assert torch.cuda.device_count() == 4
    mp.spawn(_rank, args=(4, (tmp_path / "rendezvous").as_uri()), nprocs=4, join=True)
