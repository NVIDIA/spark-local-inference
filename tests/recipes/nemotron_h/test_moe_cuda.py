# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bolt latent MoE blocks against decoded GGML weights and explicit routing."""

import gc
import os
from functools import partial
from types import SimpleNamespace
from importlib import import_module
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


def _forward_with_reuse(block, source, staged, arena, output, metadata):
    staged.copy_(source)
    output.copy_(block(staged, metadata))
    arena.tensor.fill_(255)
    staged.fill_(float("nan"))


class RouterReadback:
    def __init__(self, operation, ids, weights):
        self.operation, self.ids, self.weights = operation, ids, weights

    def __call__(self, logits, **kwargs):
        routes = self.operation(logits, **kwargs)
        self.ids.copy_(routes.ids)
        self.weights.copy_(routes.weights)
        return routes


@pytest.mark.parametrize("tokens", [1, 17, 64])
@pytest.mark.parametrize("schedule", ["serial", "shared_first", "routed_first"])
@torch.no_grad()
def test_checkpoint_latent_moe(tokens, schedule, tmp_path):
    from b12x.preparation import PreparationSession
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.nemotron_h.moe import PreparedLatentMoE
    from ignition.recipes.nemotron_h.weights import NemotronWeightLoader
    from ignition.reference.qwen38_quant import CheckpointTensors

    path = Path(os.environ.get("FLARE_NEMOTRON_CHECKPOINT",
        "/data/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"))
    if not path.is_dir():
        pytest.skip("set FLARE_NEMOTRON_CHECKPOINT to the Bolt checkpoint")
    manifest = inspect_indexed_checkpoint(path)
    prefix = "language_model.backbone.layers.1.mixer"
    with NemotronWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                              prefixes=("language_model.backbone.layers.1.",)) as loader:
        weights, config = loader.load_layer(1), loader.config
    torch.manual_seed(81)
    source = torch.randn((tokens, config.hidden), device="cuda", dtype=torch.bfloat16) * .2
    staged = torch.empty_like(source)
    count = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0", overlap=schedule != "serial")
        block = PreparedLatentMoE(weights, staged, config, 1, session=session, name="moe",
                                 workspace=arena, num_valid_tokens=count)
        if schedule != "serial":
            block.concurrent_shared.defer_launch = schedule == "routed_first"
        block.functional(session=session)
        arena.finalize()
        metadata = SimpleNamespace(live_counts=count)
        ids_readback = torch.empty((tokens, 9), device="cuda", dtype=torch.int32)
        weights_readback = torch.empty((tokens, 9), device="cuda", dtype=torch.float32)
        block.router = RouterReadback(block.router, ids_readback, weights_readback)
        actual = torch.empty_like(source)
        operation = partial(_forward_with_reuse, block, source, staged, arena, actual, metadata)
        artifact_path = tmp_path / "moe.bin"
        with CompilePool(source.device) as compiler:
            compiler.submit(operation, artifact_path).result()
        with artifact_path.open("rb") as file:
            artifact = torch.compiler.load_compiled_function(file)
        forward = lambda: artifact(operation)
        forward()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        with CheckpointTensors(manifest, byte_budget=512 << 20) as reader:
            def linear(value, name, codec):
                raw = reader.read(prefix + "." + name + ".weight")
                decoded = import_module(f"b12x.testing.{codec}_reference").dequantize_blocks(raw).bfloat16().cuda().float()
                return (value.float() @ decoded.T).bfloat16()

            scores = (source.float() @ weights["router"].float().T).sigmoid()
            ids = (scores + weights["correction"].float()).topk(9, dim=-1).indices
            route_weights = scores.gather(1, ids)
            route_weights = route_weights / route_weights.sum(-1, keepdim=True) * config.text["routed_scaling_factor"]
            latent = linear(source, "fc1_latent_proj", "q8_0")
            total = torch.zeros_like(latent, dtype=torch.float32)
            for expert in ids.unique().tolist():
                rows, columns = torch.where(ids == expert)
                hidden = linear(latent[rows], f"experts.{expert}.up_proj", "iq2_xxs").relu().square()
                down = linear(hidden, f"experts.{expert}.down_proj", "iq2_xxs")
                total.index_add_(0, rows, down.float() * route_weights[rows, columns, None])
            routed = linear(total.bfloat16(), "fc2_latent_proj", "q8_0")
            shared = linear(source, "shared_experts.up_proj", "iq2_xxs").relu().square()
            shared = linear(shared, "shared_experts.down_proj", "iq2_xxs")
            expected = routed + shared
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert error < .015, error.item()
        assert (actual.float() - expected.float()).abs().max() < .05 * expected.float().abs().max()
        complete = actual.clone()
        for live in (tokens // 2, 0, tokens - 1, tokens):
            count.fill_(live)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual[:live], complete[:live], rtol=.01, atol=.001)
            assert torch.all(ids_readback[live:] == -1)
            assert torch.count_nonzero(weights_readback[live:]) == 0
