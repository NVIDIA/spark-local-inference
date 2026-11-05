# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""QSA stream ordering across compiled graphs, KV rebinding, and scratch reuse."""

import json
import os
from itertools import pairwise
from pathlib import Path

import pytest
import torch
from b12x.preparation import PreparationSession
from cuda.bindings import driver as cuda

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.qwen38 import TEXT_PREFIX
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.attention import (
    PreparedAttention,
    QSAMetadata,
    QSAScratch,
    QSAState,
    RotaryTables,
)
from ignition.recipes.qwen38.weights import QwenWeightLoader

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


@pytest.fixture(scope="module")
def weights():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention

    snapshot = os.environ.get("FLARE_QWEN38_SNAPSHOT")
    if snapshot is None:
        pytest.skip("FLARE_QWEN38_SNAPSHOT must identify the checkpoint")
    manifest = inspect_qwen38_checkpoint(Path(snapshot))
    config = Qwen4ExpTextConfig.from_dict(json.loads((manifest.snapshot / "config.json").read_text())["text_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        module = Qwen4ExpTextAttention(config, layer_idx=3)
    module.eval().requires_grad_(False)
    prefix = f"{TEXT_PREFIX}.layers.3.self_attn"
    with QwenWeightLoader(manifest, "cuda:0", prefixes=(prefix + ".",)) as loader:
        for name in ("q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm"):
            module.get_submodule(name).weight = torch.nn.Parameter(
                loader.load_tensor(f"{prefix}.{name}.weight"), requires_grad=False)
        return module, loader.load_qsa_projections(prefix)


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("compiled", [False, True, "aot"])
# Without stream overlap the projections and selection run serially with the same b12x attention.
@pytest.mark.parametrize("overlap", [True, False])
@torch.no_grad()
def test_selector_qkv_fork_join_and_rebound_cache(weights, rows, compiled, overlap):
    def compile_call(fn):
        if compiled == "aot":
            import io
            from ignition.preparation.aot import aot_forward

            artifact = aot_forward(fn)
            artifact(fn)
            data = type(artifact).serialize(artifact).serialized_data
            restored = torch.compiler.load_compiled_function(io.BytesIO(data))
            return lambda: restored(fn)
        return compile_forward(fn)

    module, weights = weights
    device = torch.device("cuda:0")
    arena = ScratchArena(device, overlap=overlap)
    source = torch.randn(rows, 2560, dtype=torch.bfloat16, device=device) * .1
    input_copy, output_copy = torch.zeros_like(source), torch.empty_like(source)
    interval_start = torch.zeros(1, dtype=torch.int64, device=device)
    scratch = QSAScratch.allocate(rows, device)
    state = QSAState.allocate(pages=1, slots=1, device=device)
    metadata = QSAMetadata.allocate(tokens=rows, batch=1, context=128, device=device)
    rotary = RotaryTables.allocate(module.config, 128, device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        attention = PreparedAttention(module, weights, input_copy, scratch, state, metadata, rotary,
            session=session, name="overlap", workspace=arena, projection_workspace=arena, max_cache_pages=2)
        arena.freeze()
        arena.finalize()
        session.freeze()
        state = state.resize_pages_for_preparation(2)
        attention.rebind_pages_for_preparation(state)
        assert (attention.concurrent is not None) is overlap and attention.sparse is None
        if overlap:
            views = [attention.concurrent.workspace, *attention.concurrent.projection_workspaces]
            assert all(a.data_ptr() + a.nbytes <= b.data_ptr() for a, b in pairwise(views))
        concurrent, attention.concurrent = attention.concurrent, None
        try:
            serial = compile_call(lambda: attention(input_copy)) if compiled else lambda: attention(input_copy)
            serial()
        finally:
            attention.concurrent = concurrent

        def forward():
            input_copy.copy_(source)
            positions = interval_start + torch.arange(rows, device=device)
            metadata.query_positions.copy_(positions)
            metadata.rope_positions.copy_(positions[:, None].expand(-1, 3))
            metadata.sequence_lengths.copy_((interval_start + rows).to(torch.int32))
            metadata.slot_mapping.copy_(metadata.main_block_table[0, positions // 64] * 64 + positions % 64)
            attention(input_copy)
            output_copy.copy_(scratch.output)
            arena.tensor.fill_(255)
            input_copy.zero_()

        call = compile_call(forward) if compiled else forward
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            call()
        graph.instantiate()
        raw = cuda.CUgraph(graph.raw_cuda_graph())
        status, _, _, _, count = cuda.cuGraphGetEdges(raw)
        assert status == cuda.CUresult.CUDA_SUCCESS
        status, parents, children, _, _ = cuda.cuGraphGetEdges(raw, count)
        assert status == cuda.CUresult.CUDA_SUCCESS
        edges = [(int(a), int(b)) for a, b in zip(parents, children, strict=True)]
        if overlap:
            assert any(sum(parent == node for parent, _ in edges) >= 2 for node, _ in edges)
            assert any(sum(child == node for _, child in edges) >= 2 for _, node in edges)
        assert runs_concurrently(graph) is overlap
        metadata.main_block_table.copy_(torch.tensor([[1, 0]], dtype=torch.int32, device=device))
        metadata.compressed_block_table.copy_(metadata.main_block_table)
        metadata.raw_state_slot_ids.zero_()
        metadata.request_ids.zero_()
        metadata.query_start_loc.copy_(torch.tensor([0, rows], dtype=torch.int32, device=device))
        metadata.num_accepted_tokens.fill_(1)
        metadata.is_prefilling.fill_(rows > 1)
        for start in (0, 3, 63):
            source.normal_(std=.1)
            interval_start.fill_(start)
            positions = torch.arange(start, start + rows, device=device)
            metadata.query_positions.copy_(positions)
            metadata.rope_positions.copy_(positions[:, None].expand(-1, 3))
            metadata.sequence_lengths.fill_(start + rows)
            metadata.slot_mapping.copy_(metadata.main_block_table[0, positions // 64] * 64 + positions % 64)
            state.raw_interval_start_positions.fill_(start - 1)
            initial = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
            input_copy.copy_(source)
            concurrent, attention.concurrent = attention.concurrent, None
            try:
                serial()
            finally:
                attention.concurrent = concurrent
            expected_output = scratch.output.clone()
            expected_positions = scratch.selected_positions.clone()
            expected_state = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
            for name, value in initial.items():
                getattr(state, name).copy_(value)
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            torch.testing.assert_close(output_copy, expected_output, rtol=0, atol=0)
            torch.testing.assert_close(scratch.selected_positions, expected_positions, rtol=0, atol=0)
            for name, value in expected_state.items():
                torch.testing.assert_close(getattr(state, name), value, rtol=0, atol=0)
        graph.reset()
