# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured readiness timestamps retain exposed host delay in each output frame."""

import os
import threading
import time

import pytest
import torch
from b12x.sequence._shared.disk_table import MappedHostAllocation
from cuda.bindings import driver

from ignition.components.readiness import ReadinessTimings, record_timestamp, wait_for_readiness


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned CUDA device")


@torch.no_grad()
def test_captured_wait_separates_readiness_from_graph_work_and_preserves_frames():
    frame = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    timings = ReadinessTimings.allocate(frame, 1)
    flag = MappedHostAllocation((1,), torch.int32, frame.device)
    flag.host_view.zero_()
    compute, delivery = torch.cuda.Stream(), torch.cuda.Stream()
    ready = torch.cuda.Event(external=True)
    ready.record()
    for slot in range(3):
        record_timestamp(timings.timestamps, frame, slot)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=compute):
        wait_for_readiness(frame, ready.cuda_event, timings.timestamps, frame, 0,
                           frame, frame, 0)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    previous = None
    try:
        for generation in range(1, 5):
            index = generation % 2
            frame.fill_(index)
            torch.cuda.synchronize()
            with torch.cuda.stream(delivery):
                status, = driver.cuStreamWaitValue32(driver.CUstream(delivery.cuda_stream),
                    flag.device_view.data_ptr(), generation,
                    driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ)
                assert int(status) == 0
                ready.record(delivery)

            def release():
                time.sleep(.02)
                flag.host_view.fill_(generation)

            worker = threading.Thread(target=release)
            worker.start()
            with torch.cuda.stream(compute):
                graph.replay()
            compute.synchronize()
            worker.join()
            assert torch.cuda.memory_allocated() == allocated
            observed = timings.timestamps.cpu()
            before, ready_at, synchronized = observed[index].tolist()
            assert before < ready_at <= synchronized
            assert 10_000_000 <= ready_at - before < 1_000_000_000
            assert synchronized - ready_at < 10_000_000
            if previous is not None:
                assert torch.equal(observed[1 - index], previous[1 - index])
            previous = observed
    finally:
        flag.host_view.fill_(generation)
        torch.cuda.synchronize()
        graph.reset()
        flag.close()
