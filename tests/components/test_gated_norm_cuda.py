# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused sigmoid-gated RMSNorm against the canonical FP32 computation."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("heads", [16, 32])
def test_gated_norm_matches_canonical_math_and_zeroes_inactive_rows(heads):
    from ignition.components.gated_norm import PreparedGatedRMSNorm

    generator = torch.Generator(device="cuda").manual_seed(heads)
    rows, dim = 300, 128
    hidden, gate = (torch.randn((rows, heads, dim), generator=generator, device="cuda").bfloat16() for _ in range(2))
    weight = torch.randn(dim, generator=generator, device="cuda").bfloat16()
    hidden[260:] = float("nan")
    output = torch.full_like(hidden, float("nan"))
    PreparedGatedRMSNorm(hidden, gate, weight, output, torch.tensor([250], dtype=torch.int32, device="cuda"),
                         eps=1e-6)()
    x = hidden[:250].float()
    expected = weight.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)) * torch.sigmoid(gate[:250].float())
    # One BF16 ulp of rounding difference is allowed at any magnitude.
    torch.testing.assert_close(output[:250], expected.bfloat16(), rtol=2**-7, atol=4e-3)
    assert (output[:250] == expected.bfloat16()).float().mean() > 0.999
    assert not output[250:].any()


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_rounded_gated_norm_matches_a_bf16_module_with_a_strided_gate(activation, weight_dtype):
    from ignition.components.gated_norm import PreparedGatedRMSNorm

    generator = torch.Generator(device="cuda").manual_seed(48)
    rows, heads, dim = 300, 48, 128
    hidden = torch.randn((rows, heads, dim), generator=generator, device="cuda").bfloat16()
    packed = torch.randn((rows, 16384), generator=generator, device="cuda").bfloat16()
    gate = packed[:, 10240:].view(rows, heads, dim)
    weight = (1 + 0.3 * torch.randn(dim, generator=generator, device="cuda")).to(weight_dtype)
    output = torch.full_like(hidden, float("nan"))
    PreparedGatedRMSNorm(hidden, gate, weight, output, torch.tensor([250], dtype=torch.int32, device="cuda"),
                         eps=1e-6, round_normalized=True, activation=activation)()
    # Qwen's RMSNormGated: normalize in FP32, round to the BF16 input dtype, weight, then gate in FP32.
    x = hidden[:250].float()
    weighted = weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16()
    g = gate[:250].float()
    expected = (weighted * (g * torch.sigmoid(g) if activation == "silu" else torch.sigmoid(g))).bfloat16()
    # The rounded intermediate can carry one ulp into the weighted product: allow two.
    torch.testing.assert_close(output[:250], expected, rtol=2**-6, atol=4e-3)
    assert (output[:250] == expected).float().mean() > 0.9999
    assert not output[250:].any()


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_prepared_gate_before_grouped_norm_graph(weight_dtype):
    from b12x.preparation import PreparationSession
    from ignition.components.gated_norm import PreparedGatedRMSNorm

    torch.manual_seed(38)
    rows, groups, dim = 33, 8, 1024
    hidden = torch.randn((rows, groups, dim), device="cuda", dtype=torch.bfloat16)
    packed = torch.randn((rows, 18560), device="cuda", dtype=torch.bfloat16)
    gate = packed[:, :groups * dim].view(rows, groups, dim)
    weight = torch.randn((groups, dim), device="cuda", dtype=weight_dtype)
    output = torch.empty_like(hidden)
    live = torch.tensor([31], device="cuda", dtype=torch.int32)
    with PreparationSession(device="cuda:0", autotune=True) as session:
        norm = PreparedGatedRMSNorm(hidden, gate, weight, output, live, eps=1e-5,
            activation="silu", norm_before_gate=False, session=session, name="mamba.norm")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            norm()
        allocated = torch.cuda.memory_allocated()
        for count in (31, 1, 0, 33):
            hidden.normal_()
            live.fill_(count)
            graph.replay()
            x = hidden[:count].float() * torch.nn.functional.silu(gate[:count].float())
            expected = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weight.float()).bfloat16()
            torch.testing.assert_close(output[:count], expected, rtol=2**-7, atol=.004)
            assert not output[count:].any()
            del x, expected
            assert torch.cuda.memory_allocated() == allocated


@pytest.mark.parametrize("encoded_only", [False, True])
def test_gated_norm_mxfp8_prefill_graph_live_rows_and_storage(encoded_only):
    from b12x.quantization import mxfp8
    from ignition.components.gated_norm import PreparedGatedRMSNorm

    torch.manual_seed(0)
    rows, heads, dim = 4096, 48, 128
    hidden = torch.randn((rows, heads, dim), device="cuda", dtype=torch.bfloat16)
    packed = torch.randn((rows, 16384), device="cuda", dtype=torch.bfloat16)
    gate = packed[:, 10240:].view(rows, heads, dim)
    weight = torch.ones(dim, device="cuda", dtype=torch.float32)
    output = torch.empty_like(hidden)
    live = torch.tensor([rows], device="cuda", dtype=torch.int32)
    bases = tuple(torch.zeros(shape, dtype=dtype, device="cuda") for shape, dtype in zip(
        mxfp8.dense_rows_storage_shapes(rows, heads * dim),
        (torch.float8_e4m3fn, torch.uint8, torch.uint8), strict=True))
    encoded = mxfp8.dense_rows_from_bases(*bases)
    output.fill_(17)
    norm = PreparedGatedRMSNorm(hidden, gate, weight, None if encoded_only else output, live, eps=1e-6,
        round_normalized=True, activation="silu", encoded=encoded)
    norm()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        norm()
    graph.replay()
    torch.cuda.synchronize()
    prior = encoded.values.clone()
    hidden.normal_()
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(prior.view(torch.uint8), encoded.values.view(torch.uint8))
    assert torch.isfinite(encoded.values.float()).all()
    del prior
    allocated = torch.cuda.memory_allocated()
    for count in (rows // 2, 0, rows):
        live.fill_(count)
        graph.replay()
        torch.cuda.synchronize()
        if encoded_only:
            assert (output == 17).all()
        else:
            assert not output[count:].any()
        assert not encoded.values.view(torch.uint8)[count:].any()
        assert (encoded.scale_rows.view(torch.uint8).reshape(rows, -1)[count:] == 127).all()
        assert torch.cuda.memory_allocated() == allocated
