# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame-owned GPU timestamps around captured external readiness waits."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import globaltimer


@triton.jit
def _timestamp(timestamps, frame, STRIDE: tl.constexpr, SLOT: tl.constexpr):
    index = tl.load(frame)
    tl.store(timestamps + index * STRIDE + SLOT, globaltimer())


def record_timestamp(timestamps, frame, slot):
    _timestamp[(1,)](timestamps, frame, timestamps.shape[1], slot, num_warps=1)


def wait_for_readiness(source, event, timings, frame, interval, barrier_input, barrier_output, barrier_handle):
    from cuda.bindings import driver
    from ignition.components.collectives import _all_reduce

    record_timestamp(timings, frame, 3 * interval)
    stream = torch.cuda.current_stream(source.device)
    flags = int(torch.cuda.is_current_stream_capturing())
    status, = driver.cuStreamWaitEvent(driver.CUstream(stream.cuda_stream), driver.CUevent(event), flags)
    if int(status):
        raise RuntimeError(f"embedding readiness wait failed: {status}")
    record_timestamp(timings, frame, 3 * interval + 1)
    if barrier_handle > 0:
        _all_reduce(barrier_input, barrier_output, barrier_handle - 1)
    record_timestamp(timings, frame, 3 * interval + 2)


class ReadinessTimings:
    def __init__(self, frame, intervals):
        if (frame.shape != (1,) or frame.dtype != torch.int32 or frame.device.type != "cuda"
                or type(intervals) is not int or not 0 < intervals <= 64):
            raise ValueError("readiness timings require a CUDA int32 frame index and 1 to 64 intervals")
        self.frame = frame
        self.timestamps = torch.zeros((2, 3 * intervals), dtype=torch.int64, device=frame.device)

    @classmethod
    def allocate(cls, frame, intervals):
        return cls(frame, intervals)

    def buffers(self):
        return {"readiness_timestamps": self.timestamps}


class PreparedReadinessWait:
    def __init__(self, timings, interval, *, runtime, session, coordinator, name):
        from ignition.components.collectives import PreparedAllReduce

        if type(interval) is not int or not 0 <= interval < timings.timestamps.shape[1] // 3:
            raise ValueError("readiness interval is outside its timestamp buffer")
        self.timings, self.interval = timings, interval
        self.input = torch.zeros((1, 8), dtype=torch.bfloat16, device=timings.frame.device)
        self.output = torch.zeros_like(self.input)
        self.barrier = PreparedAllReduce(self.input, self.output, runtime=runtime, session=session,
                                        coordinator=coordinator, name=name)
        self.handle = 0 if self.barrier.single else self.barrier.handle + 1

    def functional(self, *, session):
        self.barrier.functional(session=session)
        self.input = self.output = None
        return self

    def operands(self, source):
        if self.input is not None:
            return self.input, self.output
        barrier_input = torch.zeros((1, 8), dtype=torch.bfloat16, device=source.device)
        return barrier_input, torch.empty_like(barrier_input)
