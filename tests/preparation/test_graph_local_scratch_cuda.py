# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-local prepared operands reuse storage and survive variant replay."""

import os

import pytest
import torch
import triton
import triton.language as tl

from ignition.preparation.aot import aot_forward


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device",
)

_ADDRESSES = []


@triton.jit
def _stage(source, offset, workspace, count: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + index, index < count, other=0) + tl.load(offset)
    tl.store(workspace + index, value, index < count)


@triton.jit
def _finish(workspace, output, count: tl.constexpr, scale: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(workspace + index, index < count, other=0)
    tl.store(output + index, value * scale, index < count)


@torch.library.custom_op(
    "test_graph_local_scratch::prepared_out", mutates_args=("output", "workspace")
)
def _prepared_out(
    source: torch.Tensor,
    offset: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    handle: int,
) -> None:
    assert output.data_ptr() != workspace.data_ptr()
    _ADDRESSES.append((output.data_ptr(), workspace.data_ptr()))
    grid = (triton.cdiv(source.numel(), 256),)
    _stage[grid](source, offset, workspace, source.numel(), 256)
    _finish[grid](workspace, output, source.numel(), handle, 256)


@_prepared_out.register_fake
def _fake_prepared_out(source, offset, output, workspace, handle):
    pass


class Forward:
    def __init__(self, count, *, external):
        self.source = torch.empty(count, device="cuda", dtype=torch.float32)
        self.zero = torch.zeros(1, device="cuda", dtype=torch.float32)
        self.output = torch.empty((2, count), device="cuda", dtype=torch.float32)
        self.first = torch.empty_like(self.source) if external else None
        self.first_workspace = torch.empty_like(self.source) if external else None
        self.second = torch.empty_like(self.source) if external else None
        self.second_workspace = torch.empty_like(self.source) if external else None

    def __call__(self):
        first = torch.empty_like(self.source) if self.first is None else self.first
        first_workspace = (
            torch.empty_like(self.source) if self.first_workspace is None else self.first_workspace
        )
        _prepared_out(self.source, self.zero, first, first_workspace, 2)
        self.output[0].copy_(first)
        second = torch.empty_like(self.source) if self.second is None else self.second
        second_workspace = (
            torch.empty_like(self.source) if self.second_workspace is None else self.second_workspace
        )
        _prepared_out(self.source, self.output[0, :1], second, second_workspace, 4)
        self.output[1].copy_(second)


@torch.no_grad()
def test_storage_backend_reuses_graph_local_prepared_outputs_and_workspace():
    for external in (True, False):
        forward = Forward(1 << 18, external=external)
        forward.source.fill_(.25)
        artifact = aot_forward(forward)
        _ADDRESSES.clear()
        artifact(forward)
        torch.cuda.synchronize()
        assert len(_ADDRESSES) == 2
        first, second = map(set, _ADDRESSES)
        if external:
            assert first.isdisjoint(second)
        else:
            assert first == second
        torch.testing.assert_close(forward.output[0], forward.source * 2, rtol=0, atol=0)
        torch.testing.assert_close(forward.output[1], forward.source * 12, rtol=0, atol=0)


@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_graph_local_prepared_scratch_survives_alternating_capture_variants(compiled):
    forwards = [Forward(count, external=False) for count in (1 << 16, 1 << 12)]
    calls = [aot_forward(forward) if compiled else forward for forward in forwards]
    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle()
    graphs = []
    try:
        for forward, call in zip(forwards, calls, strict=True):
            forward.source.fill_(.25)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                if compiled:
                    call(forward)
                else:
                    call()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                if compiled:
                    call(forward)
                else:
                    call()
            graphs.append(graph)
        for index, value in ((0, .5), (1, -.25), (1, .125), (0, -1.), (1, .75), (0, 0.)):
            forward = forwards[index]
            forward.source.fill_(value)
            forward.output.fill_(float("nan"))
            before = torch.cuda.memory_stats()
            graphs[index].replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for name in (
                "allocation.all.allocated", "allocation.all.freed",
                "segment.all.allocated", "segment.all.freed",
            ):
                assert before[name] == after[name]
            torch.testing.assert_close(forward.output[0], forward.source * 2, rtol=0, atol=0)
            torch.testing.assert_close(forward.output[1], forward.source * 12, rtol=0, atol=0)
    finally:
        for graph in graphs:
            graph.reset()
