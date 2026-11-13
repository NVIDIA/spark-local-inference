# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured fork/join and shared-workspace ordering for Qwen input projections."""

import os
from pathlib import Path

import pytest
import torch
from b12x.preparation import PreparationSession
from cuda.bindings import driver as cuda

from ignition.components.linear import ParallelProjections, QuantizedProjection
from ignition.components.parallel import TensorParallelShard
from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.qwen38 import TEXT_PREFIX
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.weights import QwenWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


@pytest.fixture(scope="module", params=[1, 2])
def weights(request):
    snapshot = os.environ.get("FLARE_QWEN38_SNAPSHOT")
    if snapshot is None:
        pytest.skip("FLARE_QWEN38_SNAPSHOT must identify the checkpoint")
    manifest = inspect_qwen38_checkpoint(Path(snapshot))
    prefix = f"{TEXT_PREFIX}.layers.0.linear_attn"
    with QwenWeightLoader(manifest, "cuda", prefixes=(prefix,),
                          shard=TensorParallelShard(0, request.param)) as loader:
        return loader.load_gdn_projections(prefix)


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("activation_mode", ["quantized", "auto"])
@torch.no_grad()
def test_projection_fork_join_preserves_source_and_workspace_lifetimes(weights, rows, compiled, activation_mode):
    arena = ScratchArena(torch.device("cuda", 0))
    source = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda")
    input_copy = torch.empty_like(source)
    output = [torch.empty((rows, weights[name].out_features), dtype=torch.bfloat16, device="cuda")
              for name in ("in_proj_qkvz", "in_proj_ba")]
    snapshots = [torch.empty_like(value) for value in output]
    with PreparationSession(device="cuda", autotune=False, compile_workers=0) as session:
        projections = [QuantizedProjection(input_copy, weights[name], value, session=session,
                        name=name, activation_mode=activation_mode, workspace=arena)
                       for name, value in zip(("in_proj_qkvz", "in_proj_ba"), output, strict=True)]
        parallel = ParallelProjections(*projections, session=session, workspace=arena)
        arena.freeze()
        arena.finalize()
        session.freeze()

        def forward():
            input_copy.copy_(source)
            parallel(input_copy)
            for value, snapshot in zip(output, snapshots, strict=True):
                snapshot.copy_(value)
            arena.tensor.fill_(255)
            input_copy.zero_()

        call = compile_forward(forward) if compiled else forward
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            call()
        graph.instantiate()
        raw = cuda.CUgraph(graph.raw_cuda_graph())
        status, _, _, _, count = cuda.cuGraphGetEdges(raw)
        assert status == cuda.CUresult.CUDA_SUCCESS
        status, sources, destinations, _, _ = cuda.cuGraphGetEdges(raw, count)
        assert status == cuda.CUresult.CUDA_SUCCESS
        edges = [(int(source), int(destination)) for source, destination in zip(sources, destinations, strict=True)]
        assert any(sum(parent == node for parent, _ in edges) >= 2 for node, _ in edges)
        assert any(sum(child == node for _, child in edges) >= 2 for _, node in edges)
        for _ in range(3):
            source.normal_()
            input_copy.copy_(source)
            for projection in projections:
                projection(input_copy)
            expected = [value.clone() for value in output]
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            for actual, reference in zip(snapshots, expected, strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        graph.reset()
