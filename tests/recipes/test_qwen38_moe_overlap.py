# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Concurrent expert branches retain graph dependencies and scratch lifetimes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from cuda.bindings import driver as cuda

from ignition.components.parallel import TensorParallelShard
from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.qwen38 import TEXT_PREFIX
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.qwen38.moe import PreparedQwenMoe, QwenMoeWeights, QwenMoeScratch
from tests.recipes.test_qwen38_moe_cuda import observe_moe
from ignition.recipes.qwen38.weights import QwenWeightLoader, allocate_expert_scale_scratch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


def runs_concurrently(graph):
    """Whether two nodes of a retained captured graph can run at once: unordered by any dependency path.

    Redundant dependencies, such as an event wait on the stream that recorded it, are not concurrency.
    """
    raw = cuda.CUgraph(graph.raw_cuda_graph())
    status, _, _, _, count = cuda.cuGraphGetEdges(raw)
    assert status == cuda.CUresult.CUDA_SUCCESS
    status, parents, children, _, _ = cuda.cuGraphGetEdges(raw, count)
    assert status == cuda.CUresult.CUDA_SUCCESS
    status, nodes, _ = cuda.cuGraphGetNodes(raw, cuda.cuGraphGetNodes(raw)[2])
    assert status == cuda.CUresult.CUDA_SUCCESS
    edges = [(int(parent), int(child)) for parent, child in zip(parents, children, strict=True)]
    waiting = {int(node): 0 for node in nodes}
    for _, child in edges:
        waiting[child] += 1
    ready = [node for node, count in waiting.items() if not count]
    while len(ready) == 1:
        node = ready.pop()
        for parent, child in edges:
            if parent == node:
                waiting[child] -= 1
                if not waiting[child]:
                    ready.append(child)
    return bool(ready)


@pytest.fixture(scope="module", params=[1, 2], ids=["tp1", "tp2"])
def weights(request):
    snapshot = os.environ.get("FLARE_QWEN38_SNAPSHOT")
    if snapshot is None:
        pytest.skip("FLARE_QWEN38_SNAPSHOT must identify the checkpoint")
    manifest = inspect_qwen38_checkpoint(Path(snapshot))
    shard = TensorParallelShard(0, request.param)
    scales = allocate_expert_scale_scratch("cuda:0", shard=shard)
    with QwenWeightLoader(manifest, "cuda:0", prefixes=(f"{TEXT_PREFIX}.layers.0.mlp.",), shard=shard) as loader:
        return QwenMoeWeights.load(loader, 0, scale_scratch=scales), scales


# Without stream overlap the shared expert precedes routing on the main stream at every row count.
@pytest.mark.parametrize("rows,overlap", [*((rows, True) for rows in (1, 2, 4, 6, 8, 24, 256, 512, 4096)),
                                          *((rows, False) for rows in (1, 8, 256))])
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_shared_routed_composition_and_arena_reuse(weights, rows, overlap, compiled):
    weights, _scale_owners = weights
    arena = ScratchArena("cuda:0", overlap=overlap)
    source = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda") * .1
    input_copy = torch.empty_like(source)
    output_copy = torch.empty_like(source)
    shared_copy = torch.empty_like(source)
    with RecordedPreparationSession(device="cuda:0", checkpoint="qwen38-moe-overlap",
            workload_schema="qwen38-moe-overlap", numerical_policy="b12x") as session:
        session.register_scratch(arena)
        moe = PreparedQwenMoe(input_copy, weights, session=session, name="overlap",
                              workspace=arena, projection_workspace=arena)
        moe.functional(session=session)
        session.tune()
        arena.release_preparation()
        readback = QwenMoeScratch.allocate(rows, source.device)
        observe_moe(moe, readback)
        assert (moe.concurrent_shared is not None) == (overlap and rows <= 256)

        def forward():
            input_copy.copy_(source)
            output_copy.copy_(moe(input_copy))
            shared_copy.copy_(readback.shared.output)
            input_copy.zero_()

        call = compile_forward(forward) if compiled else forward
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            call()
        graph.instantiate()
        session.freeze()
        raw = cuda.CUgraph(graph.raw_cuda_graph())
        status, _, _, _, count = cuda.cuGraphGetEdges(raw)
        assert status == cuda.CUresult.CUDA_SUCCESS
        status, sources, destinations, _, _ = cuda.cuGraphGetEdges(raw, count)
        assert status == cuda.CUresult.CUDA_SUCCESS
        edges = [(int(a), int(b)) for a, b in zip(sources, destinations, strict=True)]
        if moe.concurrent_shared is not None:
            assert any(sum(parent == node for parent, _ in edges) >= 2 for node, _ in edges)
            assert any(sum(child == node for _, child in edges) >= 2 for _, node in edges)
        elif not overlap:
            assert not runs_concurrently(graph)
        for _ in range(3):
            source.normal_(std=.1)
            input_copy.copy_(source)
            projected, gate = moe.shared.project_runtime(input_copy)
            expected_shared = moe.shared.finish_runtime(projected, gate)
            selected = moe.routing(moe.router_projection(input_copy))
            expected = moe.experts(input_copy, selected.ids, selected.weights) + expected_shared
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            torch.testing.assert_close(shared_copy, expected_shared, rtol=0, atol=0)
            # Native routed experts permit atomic accumulation in either order.
            torch.testing.assert_close(output_copy, expected, rtol=.01, atol=.001)
        graph.reset()


def test_aot_moe_restores_fresh_buffers_plans_and_streams_without_tracing(tmp_path):
    if os.environ.get("FLARE_QWEN38_SNAPSHOT") is None:
        pytest.skip("FLARE_QWEN38_SNAPSHOT must identify the checkpoint")
    artifact = tmp_path / "moe.bin"
    script = r'''
from types import SimpleNamespace
import importlib.util
import sys
from pathlib import Path
import torch
from ignition.preparation import aot

mode, module_path, artifact = sys.argv[1:]
spec = importlib.util.spec_from_file_location("test_qwen38_moe_overlap", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

def forbidden(*args, **kwargs):
    raise AssertionError("restoring a forward must not trace or compile")

def compile_forward(forward):
    if mode == "save":
        compiled = aot.aot_forward(forward)
        compiled(forward)
        compiled.save_compiled_function(artifact)
    else:
        aot.make_fx = forbidden
        aot.StorageBackend.__init__ = forbidden
        torch._dynamo.convert_frame.fullgraph_capture = forbidden
        with open(artifact, "rb") as file:
            compiled = torch.compiler.load_compiled_function(file)
    return lambda: compiled(forward)

module.compile_forward = compile_forward
with torch.no_grad():
    weights = module.weights.__wrapped__(SimpleNamespace(param=2))
    if mode == "load":
        module.test_shared_routed_composition_and_arena_reuse(weights, rows=1, overlap=True, compiled=False)
    module.test_shared_routed_composition_and_arena_reuse(weights, rows=1, overlap=True, compiled=True)
    if mode == "load":
        module.test_shared_routed_composition_and_arena_reuse(weights, rows=1, overlap=True, compiled=True)
'''
    for mode in ("save", "load"):
        subprocess.run([sys.executable, "-c", script, mode, str(Path(__file__).resolve()),
                        str(artifact)], check=True, timeout=180)
    assert artifact.stat().st_size < 2 * 1024 * 1024


@torch.no_grad()
def test_aot_shared_expert_order_rebinds_without_recompilation(weights, tmp_path, monkeypatch):
    from ignition.preparation import aot

    weights, _scale_owners = weights
    arena = ScratchArena("cuda:0")
    source = torch.randn(1, 2560, dtype=torch.bfloat16, device="cuda") * .1
    output = torch.empty_like(source)
    shared = torch.empty_like(source)
    with RecordedPreparationSession(device="cuda:0", checkpoint="qwen38-moe-order",
            workload_schema="qwen38-moe-order", numerical_policy="b12x") as session:
        session.register_scratch(arena)
        moe = PreparedQwenMoe(source, weights, session=session, name="overlap",
                              workspace=arena, projection_workspace=arena)
        moe.functional(session=session)
        session.tune()
        arena.release_preparation()
        readback = QwenMoeScratch.allocate(1, source.device)
        observe_moe(moe, readback)
        assert moe.concurrent_shared is not None
        moe.concurrent_shared.defer_launch = False

        def forward():
            output.copy_(moe(source))
            shared.copy_(readback.shared.output)

        forward()
        session.freeze()

        artifact = aot.aot_forward(forward)
        artifact(forward)
        path = tmp_path / "shared-order.bin"
        artifact.save_compiled_function(str(path))

        def forbidden(*args, **kwargs):
            raise AssertionError("rebinding shared-expert launch order must not trace or compile")

        monkeypatch.setattr(aot, "make_fx", forbidden)
        monkeypatch.setattr(aot.StorageBackend, "__init__", forbidden)
        monkeypatch.setattr(torch._dynamo.convert_frame, "fullgraph_capture", forbidden)
        with path.open("rb") as file:
            restored = torch.compiler.load_compiled_function(file)
        for deferred in (True, False, True):
            moe.concurrent_shared.defer_launch = deferred
            assert restored.guard_check(forward)
            restored(forward)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                restored(forward)
            for _ in range(2):
                source.normal_(std=.1)
                projected, gate = moe.shared.project_runtime(source)
                expected_shared = moe.shared.finish_runtime(projected, gate)
                selected = moe.routing(moe.router_projection(source))
                expected = moe.experts(source, selected.ids, selected.weights) + expected_shared
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                torch.testing.assert_close(shared, expected_shared, rtol=0, atol=0)
                torch.testing.assert_close(output, expected, rtol=.01, atol=.001)
            graph.reset()
