# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-local tensors remain live until an auxiliary-stream projection joins."""

import os

import pytest
import torch
import triton

from tests.preparation import test_graph_local_scratch_cuda as serial
from ignition.components.projection_overlap import _join
from ignition.preparation.aot import aot_forward


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device",
)

_ASYNC_ADDRESSES = []


@torch.library.custom_op(
    "test_graph_local_overlap::start", mutates_args=("output", "workspace")
)
def _start(
    source: torch.Tensor,
    offset: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    handle: int,
    stream: int,
) -> None:
    main = torch.cuda.current_stream(source.device)
    assert main.cuda_stream != stream
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(main)
    _ASYNC_ADDRESSES.append((source.data_ptr(), output.data_ptr(), workspace.data_ptr()))
    with torch.cuda.stream(auxiliary):
        grid = (triton.cdiv(source.numel(), 256),)
        serial._stage[grid](source, offset, workspace, source.numel(), 256)
        serial._finish[grid](workspace, output, source.numel(), handle, 256)


@_start.register_fake
def _fake_start(source, offset, output, workspace, handle, stream):
    pass


class Forward:
    def __init__(self, count):
        self.source = torch.empty(count, device="cuda", dtype=torch.float32)
        self.zero = torch.zeros(1, device="cuda", dtype=torch.float32)
        self.output = torch.empty((3, count), device="cuda", dtype=torch.float32)
        self.stream = torch.cuda.Stream()
        self.stream_handle = self.stream.cuda_stream

    def __call__(self):
        source = torch.empty_like(self.source)
        source_workspace = torch.empty_like(self.source)
        serial._prepared_out(self.source, self.zero, source, source_workspace, 1)
        auxiliary = torch.empty_like(source)
        auxiliary_workspace = torch.empty_like(source)
        _start(source, self.zero, auxiliary, auxiliary_workspace, 2, self.stream_handle)
        main = torch.empty_like(source)
        main_workspace = torch.empty_like(source)
        serial._prepared_out(self.source, self.zero, main, main_workspace, 4)
        _join(source, main, [auxiliary], auxiliary_workspace, self.stream_handle)
        self.output[0].copy_(auxiliary)
        self.output[1].copy_(main)
        tail = torch.empty_like(source)
        tail_workspace = torch.empty_like(source)
        serial._prepared_out(self.source, self.output[0, :1], tail, tail_workspace, 8)
        self.output[2].copy_(tail)


@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_local_overlap_preserves_async_operands_through_join_and_capture(compiled):
    forward = Forward(1 << 18)
    forward.source.fill_(.25)
    call = aot_forward(forward) if compiled else forward

    def run():
        if compiled:
            call(forward)
        else:
            call()

    serial._ADDRESSES.clear()
    _ASYNC_ADDRESSES.clear()
    run()
    torch.cuda.synchronize()
    assert len(serial._ADDRESSES) == 3
    assert len(_ASYNC_ADDRESSES) == 1
    pending = set(_ASYNC_ADDRESSES[0])
    assert len(pending) == 3
    assert pending.isdisjoint(serial._ADDRESSES[1])
    if compiled:
        assert pending.intersection(serial._ADDRESSES[2])

    stream = torch.cuda.Stream()
    assert stream.cuda_stream != forward.stream_handle
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            run()
        for value in (.125, -.25, .5, 0.):
            forward.source.fill_(value)
            forward.output.fill_(float("nan"))
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for name in (
                "allocation.all.allocated", "allocation.all.freed",
                "segment.all.allocated", "segment.all.freed",
            ):
                assert before[name] == after[name]
            for index, scale in enumerate((2, 4, 24)):
                torch.testing.assert_close(
                    forward.output[index], forward.source * scale, rtol=0, atol=0
                )
    finally:
        graph.reset()
