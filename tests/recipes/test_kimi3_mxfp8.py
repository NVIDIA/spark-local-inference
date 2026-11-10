# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi checkpoint projections and concurrent SiTU experts with online MXFP8 storage."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.kimi3.decoder import PreparedMLP
from ignition.recipes.kimi3.model import situ
from ignition.recipes.kimi3.projections import prepare_projection
from ignition.recipes.kimi3.weights import KimiWeightLoader

CHECKPOINT = Path("/data/models/Kimi-K3-MXFP4-CSF")
pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned CUDA device and Kimi checkpoint")


def expanded(weight):
    from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch

    if isinstance(weight, torch.Tensor):
        return weight
    packed = weight.weight
    return dequantize_mxfp8_rows_torch(packed.values, packed.scale_rows)[:, :weight.in_features].bfloat16()


def relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def runs_concurrently(graph):
    from cuda.bindings import driver as cuda

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


@pytest.fixture(scope="module")
def weights():
    if not CHECKPOINT.is_dir():
        pytest.skip("Kimi checkpoint is unavailable")
    torch.cuda.set_device(0)
    manifest = inspect_indexed_checkpoint(CHECKPOINT)
    result = []
    for online in (False, True):
        with KimiWeightLoader(manifest, torch.device("cuda:0"), shard=TensorParallelShard(0, 16),
                prefixes=("language_model.model.layers.1.",), online_mxfp8=online) as loader:
            result.append(loader.load_layer(1))
    return result


def test_weight_selection_and_error(weights):
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    baseline, packed = weights
    for key in ("latent_down", "latent_down_rows", "latent_up"):
        assert isinstance(packed[key], MXFP8LinearWeight)
        assert relative_error(expanded(packed[key]), baseline[key]) < .035
    for key in ("gate_up", "down"):
        assert isinstance(packed["shared"][key], MXFP8LinearWeight)
        assert relative_error(expanded(packed["shared"][key]), baseline["shared"][key]) < .035
    for key in ("router", "correction", "latent_norm"):
        torch.testing.assert_close(packed[key], baseline[key], rtol=0, atol=0)
    assert packed["router"].dtype == torch.bfloat16
    assert packed["experts"].has_csf_blocks


@pytest.mark.parametrize("layer", [0, 3])
@pytest.mark.parametrize("rank", [0, 15])
def test_attention_conversion(layer, rank):
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    if not CHECKPOINT.is_dir():
        pytest.skip("Kimi checkpoint is unavailable")
    manifest = inspect_indexed_checkpoint(CHECKPOINT)
    loaded = []
    for online in (False, True):
        with KimiWeightLoader(manifest, torch.device("cuda:0"), shard=TensorParallelShard(rank, 16),
                prefixes=(f"language_model.model.layers.{layer}.self_attn.",), online_mxfp8=online) as loader:
            loaded.append(loader.load_attention(layer))
    baseline, packed = loaded
    projections = ("input", "forget", "output") if layer == 0 else ("latent_gate", "q_b", "output")
    for key in projections:
        assert isinstance(packed[key], MXFP8LinearWeight)
        assert relative_error(expanded(packed[key]), baseline[key]) < .035
    for key in baseline.keys() - set(projections):
        torch.testing.assert_close(packed[key], baseline[key], rtol=0, atol=0)


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 65, 4096])
@torch.no_grad()
def test_latent_projections(weights, online, rows):
    selected = weights[int(online)]
    arena = ScratchArena("cuda:0")
    torch.manual_seed(41)
    with RecordedPreparationSession(device="cuda:0", checkpoint=str(CHECKPOINT),
            workload_schema="kimi3-latent", numerical_policy="default", compile_workers=4) as session:
        session.register_scratch(arena)
        projections = []
        for key in ("latent_down", "latent_down_rows", "latent_up"):
            reference = expanded(selected[key])
            source = torch.randn(rows, reference.shape[1], device="cuda:0", dtype=torch.bfloat16) * .1
            output = torch.empty(rows, reference.shape[0], device="cuda:0", dtype=torch.bfloat16)
            op = prepare_projection(source, selected[key], output, session=session, name=key,
                                    workspace=arena).functional(session=session)
            projections.append((op, source, reference, torch.empty_like(output)))
        session.tune(show_progress=False)
        arena.release_preparation()

        def forward():
            for op, source, _, observed in projections:
                observed.copy_(op(source))

        compiled = compile_forward(forward)
        compiled()
        expected = []
        for _, source, reference, observed in projections:
            assert relative_error(observed, torch.nn.functional.linear(source, reference)) < .01
            expected.append(observed.clone())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        for _ in range(4):
            for _, _, _, observed in projections:
                observed.fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        for (_, _, _, observed), reference in zip(projections, expected, strict=True):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("rows,compiled,overlap", [(1, False, True), (8, True, True),
                                                  (65, False, False), (4096, False, False)])
@torch.no_grad()
def test_shared_expert(weights, online, rows, compiled, overlap):
    from ignition.components.moe import PreparedRoutedExperts

    w = weights[int(online)]
    torch.manual_seed(41)
    arena = ScratchArena("cuda:0", overlap=overlap)
    source = torch.randn(rows, 7168, device="cuda:0", dtype=torch.bfloat16) * .1
    input_copy = source.clone()
    result = torch.empty_like(source)
    latent = torch.randn(rows, 3584, device="cuda:0", dtype=torch.bfloat16) * .1
    routed_output = torch.empty_like(latent)
    ids = torch.randint(0, 896, (rows, 16), device="cuda:0", dtype=torch.int32)
    scores = torch.softmax(torch.randn(rows, 16, device="cuda:0"), dim=-1)
    with RecordedPreparationSession(device="cuda:0", checkpoint=str(CHECKPOINT),
            workload_schema="kimi3-shared", numerical_policy="default", compile_workers=4) as session:
        session.register_scratch(arena)
        routed = (PreparedRoutedExperts(latent, w["experts"], ids, scores, routed_output,
            session=session, name="routed", workspace=arena) if overlap else None)
        mlp = PreparedMLP(w["shared"], input_copy, session=session, name="shared", workspace=arena,
                          parallel_plans=(routed.plan,) if overlap else ())
        if routed is not None:
            routed.functional(session=session)
        session.tune(show_progress=False)
        arena.release_preparation()
        reference_weights = {key: expanded(value) for key, value in w["shared"].items()}

        def forward():
            input_copy.copy_(source)
            if overlap:
                pending = mlp.start(input_copy)
                routed_result = routed(latent, ids, scores)
                output = mlp.finish(pending, routed_result)
                routed_output.copy_(routed_result)
            else:
                output = mlp(input_copy)
            result.copy_(output)
            input_copy.zero_()

        call = compile_forward(forward) if compiled else forward
        call()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            call()
        graph.instantiate()
        assert runs_concurrently(graph) is overlap
        for _ in range(4):
            source.normal_(std=.1)
            expected = torch.nn.functional.linear(situ(torch.nn.functional.linear(
                source, reference_weights["gate_up"])), reference_weights["down"])
            input_copy.copy_(source)
            serial = mlp(input_copy).clone()
            routed_expected = routed(latent, ids, scores).clone() if overlap else None
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert relative_error(result, expected) < .015
            torch.testing.assert_close(result, serial, rtol=0, atol=0)
            if overlap:
                torch.testing.assert_close(routed_output, routed_expected, rtol=0, atol=0)
