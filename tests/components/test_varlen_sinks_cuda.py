# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision GQA uses null sink logits with symmetric local attention."""

import os

import pytest
import torch

from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("window", [None, (64, 64)])
def test_vision_gqa_sinks_under_compiled_replay(window, functional):
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(21)
    q = torch.randn((256, 4, 64), device=device, generator=generator).bfloat16()
    k = torch.randn((256, 1, 64), device=device, generator=generator).bfloat16()
    v = torch.randn((256, 1, 64), device=device, generator=generator).bfloat16()
    out = torch.empty_like(q)
    cu = torch.tensor([0, 256], dtype=torch.int32, device=device)
    sink = torch.arange(4, dtype=torch.float32, device=device)
    with RecordedPreparationSession(device=device, checkpoint="vision-gqa", workload_schema="vision-gqa",
                                     numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        op = PreparedVarlenAttention(q, k, v, cu, out, sink=sink, window=window,
                                     session=session, workspace=arena, name="attention")
        if functional:
            op.functional(session=session)
        session.tune()
        if functional:
            arena.release_preparation()
            attention = op
            def op():
                return out.copy_(attention(q, k, v, cu))
        compiled = compile_forward(op)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for length in [256, 80]:
            cu[1] = length
            out.fill_(float("nan"))
            graph.replay()
            for token in [0, length // 2, length - 1]:
                first, last = (0, length) if window is None else (max(0, token - 64), min(length, token + 65))
                scores = q[token].float() @ k[first:last, 0].float().T / 8
                probs = torch.cat((scores, sink[:, None]), -1).softmax(-1)[:, :-1]
                torch.testing.assert_close(out[token].float(), probs @ v[first:last, 0].float(), rtol=.02, atol=.004)
        graph.reset()
