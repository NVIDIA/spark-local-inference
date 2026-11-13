# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlap streams remain distinct from pooled capture and tuning streams."""

import os

import pytest
import torch

from ignition.preparation.scratch import ScratchArena, scratch_views


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


def test_auxiliary_stream_remains_independent_after_pool_wraparound():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    arenas = [ScratchArena("cuda:0") for _ in range(2)]
    streams = [arena.auxiliary_stream for arena in arenas]
    pooled = [torch.cuda.Stream(device="cuda:0") for _ in range(96)]
    assert len({stream.cuda_stream for stream in streams}) == 2
    assert not {stream.cuda_stream for stream in streams} & {stream.cuda_stream for stream in pooled}
    values = [torch.zeros(128, device="cuda:0") for _ in streams]
    for index, (stream, value) in enumerate(zip(streams, values, strict=True)):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            value.fill_(index + 1)
        torch.cuda.current_stream().wait_stream(stream)
    for index, value in enumerate(values):
        torch.testing.assert_close(value, torch.full_like(value, index + 1))


class _PartitionForward:
    def __init__(self, backing):
        self.left, self.right = scratch_views(backing, ((128, 64),) * 2)
        self.source = torch.randn_like(self.left)
        self.output = torch.empty_like(self.source)

    def __call__(self):
        self.left.copy_(self.source.sin())
        torch.ops.aten.cos.out(self.left, out=self.right)
        self.output.copy_(self.left + self.right)


def test_partitioned_scratch_rebinds_and_alternates_with_whole_buffer_graph(tmp_path):
    from ignition.preparation.aot import aot_forward

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    backing = torch.zeros(32768, device=device)
    first = _PartitionForward(backing)
    compiled = aot_forward(first)
    artifact = tmp_path / 'partitioned.bin'
    compiled.save_compiled_function(str(artifact))
    backing = torch.empty_like(backing)
    forward = _PartitionForward(backing)
    with artifact.open('rb') as source:
        restored = torch.compiler.load_compiled_function(source)
    restored(forward)
    whole, partitioned = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(whole):
        backing.fill_(float('nan'))
    with torch.cuda.graph(partitioned):
        restored(forward)
    allocated = torch.cuda.memory_allocated()
    for value in (0., .25, -2.):
        forward.source.fill_(value)
        whole.replay()
        partitioned.replay()
        expected = forward.source.sin()
        torch.testing.assert_close(forward.output, expected + expected.cos())
        del expected
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    whole.reset()
    partitioned.reset()
