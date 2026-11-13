# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint GLM routing agrees with the canonical FP32 sigmoid router."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.linear import UnquantizedProjection
from ignition.components.moe import PreparedRouter, RouterScratch
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53.model import create_model_tree
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("tokens", [1, 65, 4096])
@torch.no_grad()
def test_actual_sigmoid_router_compiled_graph(tokens):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    model = create_model_tree(snapshot)
    module = model.model.language_model.layers[4].mlp.gate
    manifest = inspect_indexed_checkpoint(snapshot)
    prefix = f"{TEXT_PREFIX}.layers.4.mlp.gate"
    device = torch.device("cuda", 0)
    with GlmWeightLoader(manifest, device, shard=TensorParallelShard(0, 2), prefixes=(prefix + ".",)) as loader:
        module.weight = torch.nn.Parameter(loader.load_tensor(prefix + ".weight"), requires_grad=False)
        module.e_score_correction_bias = loader.load_tensor(prefix + ".e_score_correction_bias")
    assert module.weight.dtype == torch.bfloat16
    source = torch.zeros((tokens, 4096), device=device, dtype=torch.bfloat16)
    logits = torch.empty((tokens, 288), device=device)
    scratch = RouterScratch.allocate(tokens, device, experts=288, top_k=8)
    with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
                                    workload_schema="glm-router-v1", numerical_policy="default") as session:
        projection = UnquantizedProjection(source, module.weight, logits, session=session, name="router")
        routing = PreparedRouter(logits, scratch, session=session, name="sigmoid", score_func="sigmoid",
                                  correction_bias=module.e_score_correction_bias, routed_scaling_factor=2.5)
        projection.functional(session=session)
        routing.functional(session=session)
        session.tune()

        def forward():
            values = projection(source)
            selected = routing(values)
            logits.copy_(values)
            scratch.ids.copy_(selected.ids)
            scratch.weights.copy_(selected.weights)

        compiled = compile_forward(forward)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for scale in (.1, .5):
            source.normal_(std=scale)
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            assert before["allocation.all.allocated"] == after["allocation.all.allocated"]
            expected_logits, expected_weights, expected_ids = module(source)
            torch.testing.assert_close(logits, expected_logits, rtol=2e-4, atol=2e-5)
            actual_order = scratch.ids.argsort(-1)
            expected_order = expected_ids.argsort(-1)
            torch.testing.assert_close(scratch.ids.gather(1, actual_order),
                expected_ids.gather(1, expected_order).to(torch.int32), rtol=0, atol=0)
            torch.testing.assert_close(scratch.weights.gather(1, actual_order),
                expected_weights.gather(1, expected_order), rtol=2e-5, atol=2e-6)
        graph.reset()
