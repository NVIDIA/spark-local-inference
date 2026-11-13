# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped draft convolution rounding, block boundaries, and graph replay."""

import os

import pytest
import torch

cuda = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")


@cuda
@torch.no_grad()
def test_dynamic_convolution_preserves_bf16_rounding_and_block_boundaries():
    from ignition.components.grouped_convolution import _convolve
    from ignition.preparation.compilation import compile_forward

    torch.cuda.set_device(0)
    torch.manual_seed(2605)
    rows, hidden, group, width = 24, 7168, 16, 8
    x = torch.randn((rows, hidden), dtype=torch.bfloat16, device="cuda")
    delta = torch.randn((rows, 2, 2, hidden // group), dtype=x.dtype, device=x.device)
    base = torch.randn((2, 2, hidden), dtype=x.dtype, device=x.device)
    output = torch.empty_like(x)
    for side in (0, 1):
        coeff = base[side].view(1, 2, hidden // group, group) + delta[:, side, :, :, None]
        shifted = torch.zeros_like(x).view(-1, width, hidden // group, group)
        shifted[:, 1:] = x.view_as(shifted)[:, :-1]
        expected = ((coeff[:, 0] * x.view(rows, hidden // group, group)) +
                    (coeff[:, 1] * shifted.view(rows, hidden // group, group))).flatten(1)
        def forward():
            torch.library.wrap_triton(_convolve)[((x.numel() + 255) // 256,)](
                x, delta, base, output, rows, hidden, group, width, 2, delta.stride(0), side, 256)
        compiled = compile_forward(forward)
        compiled()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        output.zero_()
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
