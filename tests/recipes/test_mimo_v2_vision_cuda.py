# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pooled MiMo image scratch preserves compiled encoder and publication results."""

import json
import math
import os
from pathlib import Path

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.components.paged_batch import BatchInputs
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import DeviceRequestState
from ignition.components.vision_state import PrefillVisionInputs, VisionState
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.recipes.mimo_v2.vision import FEATURES, HIDDEN, PreparedVision, VisionInputs, VisionScratch
from ignition.recipes.mimo_v2.vision_state import PreparedVisionGeneration
from ignition.recipes.mimo_v2.vision_weights import VisionWeights

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@pytest.fixture(scope="module")
def weights():
    path = Path(os.environ.get("FLARE_MIMO_CHECKPOINT", str(Path.home() /
        ".cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Pro-MOPD/snapshots/adea8e2c5373181e5a973fa1ecb343cb31af214b")))
    if not (path / "config.json").is_file():
        pytest.skip("MiMo V2.6 Pro MOPD checkpoint is unavailable")
    config = json.loads((path / "config.json").read_text())
    shard = TensorParallelShard(0, 1)
    return VisionWeights.load(inspect_indexed_checkpoint(path), config, "cuda:0", shard), config


@pytest.mark.parametrize("capacity", [256, 16384])
@torch.no_grad()
def test_pooled_encoder_publication_matches_unpooled_after_poisoning(weights, capacity):
    from b12x.preparation import PreparationSession

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    weights, config = weights
    device = torch.device("cuda:0")
    pool, arena = GraphBufferPool(device, chunk_bytes=2 << 20), ScratchArena(device)
    inputs = VisionInputs.allocate(device)
    side = math.isqrt(capacity)
    positions = torch.stack(torch.meshgrid(torch.arange(side, device=device),
        torch.arange(side, device=device), indexing="ij"), -1)
    positions = positions.reshape(side // 2, 2, side // 2, 2, 2).permute(0, 2, 1, 3, 4).reshape(capacity, 2)
    columns = torch.arange(capacity // 4, device=device).reshape(side // 2, side // 2).T.flatten()
    inputs.vision_position_ids[:capacity].copy_(positions)
    inputs.vision_column_indices[:capacity // 4].copy_(columns)
    inputs.vision_row_indices[:capacity // 4].copy_(columns.argsort())
    inputs.vision_cu_seqlens[1] = capacity
    inputs.vision_num_patches.fill_(capacity)
    inputs.vision_num_features.fill_(capacity // 4)
    inputs.vision_slot.zero_()
    inputs.vision_generation.fill_(7)
    requests = DeviceRequestState.allocate(device, vocab_size=config["vocab_size"], capacity=2)
    requests.generation[0], requests.status[0] = 7, 1
    batch = BatchInputs.allocate(capacity, 1, device, capacity=2)
    records = torch.zeros((2, 2, 32), dtype=torch.int64, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    graphs, owners, states = {}, [], []
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        for pooled in (False, True):
            buffers = pool.cursor() if pooled else None
            scratch = VisionScratch.allocate(capacity, weights.shard, device, buffers=buffers)
            encoder = PreparedVision(weights, scratch, inputs, runtime=None, coordinator=None,
                session=session, workspace=arena, name=f"vision-{pooled}", buffers=buffers)
            state = VisionState(batch, PrefillVisionInputs.allocate(capacity, device), requests,
                image_token=config["image_token_id"], features=FEATURES, hidden=HIDDEN)
            state.feature_slot.fill_(0)
            operation = PreparedVisionGeneration(encoder, state, completion_outputs(records), frame, sequence, buffers=buffers)
            states.append(state)
            owners.append(operation)
        arena.finalize()
        calls = [compile_forward(operation) for operation in owners]
        for call in calls:
            call()
        session.freeze()
        for index, call in enumerate(calls):
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                call()
            graphs[str(index)] = graph
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={},
            resources=(owners, calls, states, weights, inputs, requests, pool, arena, session))
        worker = bundle.start_worker()
        try:
            for seed in (31, 72):
                torch.manual_seed(seed)
                inputs.vision_pixel_values[:capacity].normal_()
                for chunks in pool.chunks.values():
                    for tensor in chunks:
                        tensor.fill_(float("nan") if tensor.is_floating_point() else -37)
                for state in states:
                    state.arena.fill_(float("nan"))
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                for index in (0, 1):
                    ticket = worker.submit(index)
                    completion = worker.poll(timeout_ms=30000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                expected = states[0].arena[0, :capacity // 4]
                actual = states[1].arena[0, :capacity // 4]
                assert torch.isfinite(expected).all()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for state in states:
                    assert state.count.tolist() == [capacity // 4, 0]
                    assert state.generation.tolist() == [7, 0]
                    assert torch.isnan(state.arena[1]).all()
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
