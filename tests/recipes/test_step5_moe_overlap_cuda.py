# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step-5 expert branches preserve input and scratch lifetimes in captured graphs."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.step5.decoder import PreparedFeedForward, Scratch
from ignition.recipes.step5.weights import Step5WeightLoader

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@pytest.fixture(scope="module")
def weights():
    default = (Path.home() / ".cache/huggingface/hub/models--stepfun-ai--Step-5-Preview-NVFP4"
               / "snapshots/bc246cc2848e17b37f377eda8fd825c5b3018d1f")
    snapshot = Path(os.getenv("FLARE_STEP5_SNAPSHOT", default))
    if not (snapshot / "config.json").exists():
        pytest.skip("requires the Step-5 checkpoint")
    shard = TensorParallelShard(0, 4)
    with Step5WeightLoader(inspect_indexed_checkpoint(snapshot), "cuda:0", shard=shard,
            prefixes=("model.layers.3.",), expert_scale_compression="csf") as loader:
        return loader.load_layer(3), loader.config, shard


@torch.no_grad()
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8])
def test_shared_routed_graph_order_and_scratch_reuse(weights, rows):
    layer, config, shard = weights
    scratch = Scratch.allocate(config, shard, rows, 1, 128, "cuda:0")
    arena = ScratchArena("cuda:0")
    source = torch.randn_like(scratch.normalized)
    output = torch.empty_like(scratch.partial)
    shared = torch.empty_like(scratch.shared.output)
    graphs = []
    with RecordedPreparationSession(device="cuda:0", checkpoint="step5-moe-overlap",
            workload_schema="expert-branch-order", numerical_policy="checkpoint") as session:
        session.register_scratch(arena)
        moe = PreparedFeedForward(3, layer, config, scratch, runtime=None, coordinator=None,
            session=session, name=f"moe:{rows}", workspace=arena, overlap=True, reduce_output=False)
        session.tune(show_progress=False)

        def forward():
            output.copy_(moe(source))
            shared.copy_(scratch.shared.output)
            arena.tensor.fill_(255)
            scratch.normalized.zero_()

        try:
            for deferred in (False, True):
                moe.concurrent.defer_launch = deferred
                call = compile_forward(forward)
                call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    call()
                graphs.append(graph)
            session.freeze()
            for iteration in range(12):
                source.normal_(generator=torch.Generator(device="cuda").manual_seed(5011 + iteration))
                scratch.normalized.copy_(source)
                expected_shared = moe.mlp(scratch.normalized).clone()
                moe.router(moe.router_projection(scratch.normalized))
                expected_ids = scratch.router.ids.clone()
                expected_weights = scratch.router.weights.clone()
                expected_routed = moe.experts().clone()
                # Dynamic MoE uses BF16 atomic scatter; bound comparisons by serial repeatability.
                repeats = torch.stack([moe.experts().float().clone() for _ in range(8)])
                repeat_errors = repeats - expected_routed.float()
                repeat_max = repeat_errors.abs().max().item()
                repeat_norm = repeat_errors.flatten(1).norm(dim=1).max().item()
                reference_norm = expected_routed.float().norm().item()
                assert repeat_norm <= .01 * reference_norm
                for graph in graphs:
                    output.fill_(float("nan"))
                    shared.fill_(float("nan"))
                    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                    torch.testing.assert_close(shared, expected_shared, rtol=0, atol=0)
                    torch.testing.assert_close(scratch.router.ids, expected_ids, rtol=0, atol=0)
                    torch.testing.assert_close(scratch.router.weights, expected_weights, rtol=0, atol=0)
                    torch.testing.assert_close(output, shared.float() + scratch.experts.float(), rtol=0, atol=0)
                    error = scratch.experts.float() - expected_routed.float()
                    assert error.norm().item() <= min(.01 * reference_norm, 2 * repeat_norm)
                    torch.testing.assert_close(scratch.experts, expected_routed, rtol=0, atol=2 * repeat_max)
        finally:
            for graph in graphs:
                graph.reset()
