# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured Qwen GDN prefill with independent token and sequence counts."""

import os
from pathlib import Path

import pytest
import torch

from ignition.recipes import resolve

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU")


@pytest.mark.parametrize("checkpoint,online_mxfp8", [("qwen3.8-27b", False), ("qwen3.8-27b-qad", False), ("qwen3.8-27b-qad", True), ("hybrid", True)])
@torch.no_grad()
def test_checkpoint_captured_chunked_prefill_preserves_counts_and_state(checkpoint, online_mxfp8):
    from b12x.preparation import PreparationSession
    from ignition.components.dflash2 import PreparedContext
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38_dense.decoder import PreparedDecoder, Scratch
    from ignition.recipes.qwen38_dense.pool import ContinuationPool
    from ignition.recipes.qwen38_dense.weights import ModelWeights

    device = torch.device("cuda:0")
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    target = (Path(os.environ.get("FLARE_QWEN38_DENSE_CHECKPOINT",
        "/home/luke/models/Qwen3.8-27B-NVIDIA-NVFP4-QAD-MXFP8")) if checkpoint == "hybrid"
        else resolve(checkpoint).snapshot(hub))
    draft = hub / "models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5"
    weights = ModelWeights.load(inspect_indexed_checkpoint(target), device, mtp=True, draft_snapshot=draft, online_mxfp8=online_mxfp8)
    pool = ContinuationPool.for_model(weights.config, 3, 2, device, mtp=True)
    metadata = Metadata.allocate(128, 1, 1, device)
    starts = torch.zeros(1, dtype=torch.int64, device=device)
    lengths = torch.zeros(1, dtype=torch.int32, device=device)
    rows = torch.ones(1, dtype=torch.int64, device=device)
    pages = torch.zeros((1, 1), dtype=torch.int32, device=device)
    tokens = torch.arange(11, 139, device=device).reshape(1, 128)
    live_rows = torch.ones(1, dtype=torch.int32, device=device)

    def inputs(start, count):
        starts.fill_(start)
        lengths.fill_(count)
        metadata.set(starts, lengths, rows, pages, tokens + start, live_rows)

    with PreparationSession(device=device, autotune=False) as session:
        arena = ScratchArena(device)
        scratch = Scratch.allocate(128, 1, weights.config, device)
        decoder = PreparedDecoder(weights, scratch, metadata, pool, mode="prefill", vocabulary_runtime=None,
            coordinator=None, session=session, name="qwen.actual.prefill128", workspace=arena)
        context = PreparedContext(weights, decoder.auxiliary, metadata, pool, session=session,
            name="qwen.actual.prefill.context", workspace=arena, runtime=None, coordinator=None).functional()
        decoder.release_outputs()
        observed_hidden = torch.empty((1, weights.config.hidden), dtype=torch.bfloat16, device=device)
        observed_features = torch.empty((128, 25600), dtype=torch.bfloat16, device=device)
        observed_counts = torch.empty((2,), dtype=torch.int32, device=device)
        arena.finalize()

        def forward():
            hidden, _, auxiliary, _ = decoder(metadata)
            delta = decoder._delta_metadata(metadata)
            observed_counts.copy_(torch.cat((delta.num_seqs, delta.num_tokens)))
            observed_hidden.copy_(hidden)
            observed_features.copy_(auxiliary)
            context(auxiliary, metadata.lengths, metadata=metadata)

        inputs(0, 17)
        compiled = compile_forward(forward)
        pool.clear_for_preparation()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        pool.clear_for_preparation()
        errors = []
        for start, count in ((0, 17), (17, 13)):
            inputs(start, count)
            before = pool.backing.clone()
            before_pages = {name: value.clone() for name, value in pool.pages.items()}
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert observed_counts[0].item() == 1
            assert observed_counts[1].item() == count
            actual = pool.backing.clone()
            actual_pages = {name: value.clone() for name, value in pool.pages.items()}
            actual_hidden = observed_hidden.clone()
            actual_features = observed_features[:count].clone()
            assert torch.equal(actual[0], before[0])
            assert torch.equal(actual[2], before[2])
            pool.backing.copy_(before)
            for name, value in before_pages.items():
                pool.pages[name].copy_(value)
            forward()
            torch.cuda.synchronize()
            relative = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
            errors.append(dict(start=start, tokens=count, hidden_relative_l2=relative(actual_hidden, observed_hidden),
                feature_relative_l2=relative(actual_features, observed_features[:count])))
            pool.backing.copy_(actual)
            for name, value in actual_pages.items():
                pool.pages[name].copy_(value)
        print(dict(chunked_prefill=errors, sequence_count=1, capacity=128), flush=True)
        graph.reset()
