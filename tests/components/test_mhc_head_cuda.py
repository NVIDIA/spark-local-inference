# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learned hyperconnection reduction against dense FP32 checkpoint arithmetic."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.mhc import PreparedMHCHead
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device"
)


@pytest.mark.parametrize("tokens", (1, 6, 17))
@torch.no_grad()
def test_learned_head_replays_fp32_mixing_and_bf16_output(tokens):
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(4104)
    residual = torch.randn((tokens, 4, 4096), dtype=torch.bfloat16, device=device)
    weights = {
        "fn": torch.randn((4, 16384), device=device) * 0.01,
        "scale": torch.tensor([0.7], device=device),
        "base": torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device),
    }
    output = torch.empty((tokens, 4096), dtype=torch.bfloat16, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        head = PreparedMHCHead(residual, weights, output, rms_eps=1e-20, hc_eps=1e-6,
                               session=session, name="head")
        compiled = compile_forward(lambda: head(residual))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for gain in (0.25, 1.0, 4.0):
            residual.normal_(std=gain)
            output.fill_(float("nan"))
            graph.replay()
            flat = residual.float().flatten(1)
            projection = torch.nn.functional.linear(flat, weights["fn"])
            normalized = projection * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-20)
            mix = (normalized * weights["scale"] + weights["base"]).sigmoid() + 1e-6
            expected = torch.einsum("th,thd->td", mix, residual.float()).bfloat16()
            torch.testing.assert_close(output, expected, rtol=0.008, atol=0.008)
        graph.reset()
