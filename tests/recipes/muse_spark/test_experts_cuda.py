# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 expert stages and post-reduction expert normalization against canonical arithmetic."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned CUDA_VISIBLE_DEVICES")


@pytest.mark.parametrize("tokens,valid", [(1, 1), (17, 13), (4096, 4079)])
def test_route_finalizer_canonical_rounding(tokens, valid):
    from transformers.models.muse_spark.modeling_muse_spark import MuseSparkTextInputScaledRMSNorm
    from ignition.components.route_finalizer import PreparedRouteFinalizer

    torch.manual_seed(1801)
    routes = torch.randn((tokens, 8, 4096), device="cuda", dtype=torch.bfloat16)
    gain = (torch.randn(4096, device="cuda") * .2).bfloat16()
    weights = torch.softmax(torch.randn(tokens, 8, device="cuda"), -1)
    ids = torch.rand((tokens, 256), device="cuda").topk(8, dim=-1).indices.int()
    ids[valid:] = -1
    ids[0, -1] = 256
    weights[valid:] = 0
    output = torch.empty((tokens, 4096), dtype=torch.bfloat16, device="cuda")
    count = torch.tensor([valid], device="cuda", dtype=torch.int32)
    prepared = PreparedRouteFinalizer(routes, gain, weights, ids, count, output, eps=1e-8, num_experts=256)
    norm = MuseSparkTextInputScaledRMSNorm(4096, eps=1e-8).cuda()
    norm.weight = torch.nn.Parameter(gain, requires_grad=False)
    values = (norm(routes) * weights[..., None]).bfloat16()
    values[0, -1] = 0
    expected = torch.zeros_like(output)
    order = ids.argsort(dim=-1)
    for slot in range(8):
        index = order[:, slot, None, None].expand(-1, 1, 4096)
        expected.add_(values.gather(1, index).squeeze(1))
    prepared()
    # Approximate reciprocal square root can differ at a BF16 midpoint.
    error = (output.float() - expected.float()).abs()
    assert error.max().item() <= .015625
    assert error.norm().item() / expected.float().norm().item() < .001
    assert torch.count_nonzero(output[valid:]) == 0
    run = prepared
    if tokens == 17:
        from ignition.preparation.compilation import compile_forward
        run = compile_forward(prepared)
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    output.zero_()
    graph.replay()
    torch.testing.assert_close(output, expected, rtol=.016, atol=.015625)


def test_bf16_experts_raw_silu():
    from ignition.components.bf16_experts import PreparedBF16Experts
    from ignition.preparation.scratch import GraphBufferPool

    torch.manual_seed(171)
    source = torch.randn((17, 128), device="cuda", dtype=torch.bfloat16)
    gate_up = (torch.randn((4, 128, 128), device="cuda") * .05).bfloat16()
    down = (torch.randn((4, 128, 64), device="cuda") * .05).bfloat16()
    ids = torch.rand((17, 4), device="cuda").topk(2, dim=-1).indices.int()
    weights = torch.rand((17, 2), device="cuda")
    ids[13:] = -1
    valid = torch.tensor([13], device="cuda", dtype=torch.int32)
    output = torch.empty((17, 2, 128), device="cuda", dtype=torch.bfloat16)
    prepared = PreparedBF16Experts(source, gate_up, down, ids, weights, output, valid,
                                  buffers=GraphBufferPool("cuda").cursor())
    expected = torch.zeros_like(output)
    for token in range(13):
        for route in range(2):
            expert = ids[token, route].item()
            gate, up = torch.nn.functional.linear(source[token], gate_up[expert]).chunk(2)
            expected[token, route] = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down[expert])
    prepared()
    torch.testing.assert_close(output, expected, rtol=.025, atol=.002)
    assert torch.count_nonzero(output[13:]) == 0
    from ignition.preparation.compilation import compile_forward
    run = compile_forward(prepared)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.testing.assert_close(output, expected, rtol=.025, atol=.002)
