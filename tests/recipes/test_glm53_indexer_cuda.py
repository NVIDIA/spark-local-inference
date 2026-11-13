# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint GLM index production and state advance through native graph replay."""

import os
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from ignition.components.linear import QuantizedProjection

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53.indexer import (
    IndexMetadata,
    IndexScratch,
    IndexState,
    PreparedGlmIndexer,
)
from ignition.recipes.glm53.model import create_model_tree
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@pytest.fixture(scope="module")
def model_weights():
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    tree = create_model_tree(snapshot)
    module = tree.model.language_model.layers[3].self_attn.indexer
    manifest = inspect_indexed_checkpoint(snapshot)
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 2),
            prefixes=(f"{TEXT_PREFIX}.layers.3.self_attn.",)) as loader:
        weights = loader.load_attention(3)
    return module, weights


@pytest.mark.parametrize("mode,rows", [("decode", 2), ("decode", 32), ("prefill", 65), ("prefill", 4096)])
@torch.no_grad()
def test_glm_indexer_checkpoint_native_replay(model_weights, mode, rows):
    module, weights = model_weights
    device = torch.device("cuda:0")
    sequences = 2 if mode == "decode" else 1
    # Decode rows belong to two requests, as speculative verification packs them; above 16 rows b12x
    # selects through its tiled route.
    per_request = rows // sequences
    source = torch.zeros((rows, 4096), dtype=torch.bfloat16, device=device)
    query_residual = torch.zeros((rows, 1536), dtype=torch.bfloat16, device=device)
    scratch = IndexScratch.allocate(rows, 64, mode, device)
    state = IndexState.allocate(128, 4, device)
    metadata = IndexMetadata.allocate(rows, sequences, 64, device)
    arena = ScratchArena(device)
    graph, worker = torch.cuda.CUDAGraph(), None
    with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
            workload_schema="glm-indexer-layer-v1", numerical_policy="default") as session:
        try:
            session.register_scratch(arena)
            # The index key and head weights are columns of the layer's fused input projection.
            projected = torch.zeros((rows, 2208), dtype=torch.bfloat16, device=device)
            projection = QuantizedProjection(source, weights["qkv_a_proj"], projected, session=session,
                name="qkv_a_proj", workspace=arena, activation_mode="auto")
            indexer = PreparedGlmIndexer(module, weights, source, query_residual, projected[:, 2048:2176],
                projected[:, 2176:2208], scratch, state, metadata,
                mode=mode, session=session, name="indexer", workspace=arena)
            projection.functional(session=session)
            indexer.functional(session=session)
            session.tune()
            arena.release_preparation()

            def forward():
                projected = projection(source)
                _, local = indexer.forward(source, query_residual, projected[:, 2048:2176],
                    projected[:, 2176:2208], metadata)
                for field in fields(scratch):
                    getattr(scratch, field.name).copy_(getattr(local, field.name))

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            session.freeze()
            bundle = PreparedGraphBundle(device=0, graphs={mode: graph},
                buffers={"source": source, "query": query_residual, "cache": state.cache, "tail": state.tail},
                resources=(compiled, indexer, session, arena))
            worker = bundle.start_worker()
            # Valid FP8 index pages (64 pools' keys, then their 64 scales) give candidate pools distinct scores.
            pages = state.cache.view(state.cache.shape[0], -1)
            pages[:, :64 * 128].copy_(torch.randn((pages.shape[0], 64 * 128), device=device)
                                      .to(torch.float8_e4m3fn).view(torch.uint8))
            pages[:, 64 * 128:].view(torch.float32).copy_(torch.rand((pages.shape[0], 64), device=device) * .2 + .05)
            state.tail.normal_(std=.1)
            metadata.block_table.copy_(torch.randperm(128, device=device, dtype=torch.int32)
                                       .reshape(2, 64)[:sequences])
            metadata.state_slots.copy_(torch.tensor([2, 1][:sequences], dtype=torch.int64, device=device))
            if mode == "decode":
                metadata.query_start_loc.copy_(torch.arange(3, dtype=torch.int32, device=device) * per_request)
                metadata.requests.copy_(torch.arange(rows, dtype=torch.int32, device=device) // per_request)
            else:
                metadata.query_start_loc[1] = rows
            intermediates = ("query", "query_fp8", "query_scales", "normalized_key", "gate", "weights")
            for step in range(3):
                source.normal_(std=.1)
                query_residual.normal_(std=.1)
                if mode == "decode":
                    # Past 2,048 tokens each row has more pools than the 512 it selects.
                    starts = torch.tensor([4351, 2303], dtype=torch.int64, device=device) + step * per_request
                    metadata.positions.copy_(starts[metadata.requests.long()]
                                             + torch.arange(rows, device=device) % per_request)
                else:
                    metadata.positions.copy_(torch.arange(rows, device=device, dtype=torch.int64) + 3 + step * rows)
                metadata.token_slots.copy_(metadata.block_table[metadata.requests.long(), metadata.positions // 256].long()
                                          * 256 + metadata.positions % 256)
                if mode == "decode" and step == 2:
                    metadata.state_slots[-1] = -1
                    metadata.positions[-per_request:] = -1
                    metadata.token_slots[-per_request:] = -1
                before_cache, before_tail = state.cache.clone(), state.tail.clone()
                forward()
                expected = tuple(getattr(scratch, name).clone() for name in intermediates)
                expected_cache, expected_tail = state.cache.clone(), state.tail.clone()
                expected_indices, expected_lengths = scratch.physical.clone(), scratch.selected_lengths.clone()
                state.cache.copy_(before_cache)
                state.tail.copy_(before_tail)
                # Poison readbacks so every compared value must be written by replay.
                for field in fields(scratch):
                    getattr(scratch, field.name).view(torch.uint8).zero_()
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                ticket = worker.submit(0)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                for name, reference in zip(intermediates, expected, strict=True):
                    value = getattr(scratch, name)
                    if value.dtype == torch.float8_e4m3fn:
                        value, reference = value.view(torch.uint8), reference.view(torch.uint8)
                    torch.testing.assert_close(value, reference, rtol=0, atol=0,
                                               msg=lambda message, name=name: f"{name}: {message}")
                torch.testing.assert_close(state.cache, expected_cache, rtol=0, atol=0)
                torch.testing.assert_close(state.tail, expected_tail, rtol=0, atol=0)
                torch.testing.assert_close(scratch.selected_lengths, expected_lengths, rtol=0, atol=0)
                torch.testing.assert_close(scratch.physical.sort().values, expected_indices.sort().values, rtol=0, atol=0)
        finally:
            if worker is not None:
                worker.close()
            graph.reset()
