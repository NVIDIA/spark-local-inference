# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-owned residual injection preserves BF16 values and graph storage."""

import os

import pytest
import torch
from b12x.norm import hyperconnection as hc
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall

from ignition.preparation.compilation import compile_forward
from tests.tensor_examples import TensorExamples

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned CUDA device")


@pytest.mark.parametrize("tokens", [1, 8, 64])
@torch.no_grad()
def test_combine_provided_output_matches_functional_with_exact_alias(tokens):
    device = torch.device("cuda", 0)
    buffers = TensorExamples(device)
    residual = buffers.zeros((tokens, 10240), torch.bfloat16)
    block = buffers.zeros((tokens, 2560), torch.bfloat16)
    logits = buffers.zeros((tokens, 336), torch.bfloat16)[:, 320:324]
    out = buffers.zeros((tokens, 10240), torch.bfloat16)
    saved = torch.empty_like(residual)
    caps = hc.Caps(device=device, max_tokens=tokens, hidden_size=2560, streams=4, lowrank=320)
    plans = {mode: hc.plan(caps, invocation=FrozenMapping({"operation": "combine", "output_mode": mode}))
             for mode in ("provided", "functional")}
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare(tuple(plan.request(name=mode,
            prepare_call=lambda state: PreparedCall(run=lambda: state.launch(residual, block, logits, out)))
            for mode, plan in plans.items()))

        def run():
            saved.copy_(residual)
            hc.run_combine(residual, block, logits, out=residual, plan=plans["provided"])

        forward = compile_forward(run)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        for _ in range(3):
            residual.normal_()
            block.normal_()
            logits.normal_()
            before = residual.clone()
            expected = hc.run_combine(residual, block, logits, plan=plans["functional"])
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            torch.testing.assert_close(residual, expected, rtol=0, atol=0)
            torch.testing.assert_close(saved, before, rtol=0, atol=0)
        graph.reset()
