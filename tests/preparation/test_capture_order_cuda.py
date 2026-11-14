# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiled graph variants reuse the largest captured allocation after reload."""

from contextlib import nullcontext
import os
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@torch.library.custom_op("flare_test_capture_order::write", mutates_args=("scratch", "output"))
def _write(scratch: torch.Tensor, source: torch.Tensor, output: torch.Tensor) -> None:
    scratch.copy_(source.expand_as(scratch))
    output.copy_(scratch[-1:])


@_write.register_fake
def _fake_write(scratch, source, output):
    pass


class Forward:
    def __init__(self, size, source, output):
        self.size, self.source, self.output = size, source, output

    def __call__(self):
        scratch = torch.empty((self.size,), dtype=torch.uint8, device=self.source.device)
        _write(scratch, self.source, self.output)


@torch.no_grad()
def test_capture_pool_reuses_larger_compiled_request_on_fresh_and_cached_load(tmp_path):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    mib = 1 << 20
    source = torch.ones(1, dtype=torch.uint8, device=device)
    functions = {name: Forward(size, source, torch.empty_like(source))
                 for name, size in (("small", 32 * mib + 65), ("large", 35 * mib))}
    signatures = {name: GraphSignature("decode", 1, 1) for name in functions}
    session = SimpleNamespace(capture=nullcontext, namespace=SimpleNamespace(to_dict=lambda: {}),
                              requested_compile_workers=1)
    for restart in (False, True):
        pool = torch.cuda.graph_pool_handle()
        graphs, compiled = capture_forwards(functions, signatures, session=session, device=device,
            stream=torch.cuda.Stream(device=device), pool=pool, cache_root=tmp_path)
        try:
            assert tuple(graphs) == ("large", "small")
            assert all(forward.cache_hit is restart for forward in compiled.values())
            assert compiled["small"].capture_allocation_sizes == (32 * mib + 65,)
            assert compiled["large"].capture_allocation_sizes == (35 * mib,)
            segments = [segment for segment in torch.cuda.memory_snapshot(include_traces=False)
                        if tuple(segment["segment_pool_id"]) == tuple(pool)]
            assert sum(segment["total_size"] for segment in segments) == 36 * mib
            for value in (7, 19, 3):
                for name in ("small", "large", "small"):
                    source.fill_(value)
                    functions[name].output.zero_()
                    torch.cuda.synchronize(device)
                    before = torch.cuda.memory_stats(device)
                    graphs[name].replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    for key in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key]
                    assert functions[name].output.item() == value
        finally:
            for graph in graphs.values():
                graph.reset()
            graphs.clear()
            compiled.clear()
