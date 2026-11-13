# SPDX-License-Identifier: Apache-2.0

"""Fused rank-unpack and Q/KV RMSNorm matches the original kernels bitwise."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 4, 8, 32])
@torch.no_grad()
def test_unpack_and_rms_norm_qkv_matches_original(rows):
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.kimi3.pointwise import rms_norm_columns, unpack_and_rms_norm_qkv

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    world, local_q, local_kv = 16, 96, 36
    generator = torch.Generator().manual_seed(4200 + rows)
    gathered = (torch.randn((rows, world, local_q + local_kv), generator=generator) * 0.2).to(torch.bfloat16).to(device)
    special = torch.tensor([torch.nan, torch.inf, -torch.inf, -0.0, 0.0, 1., -1., 0.25, -0.25, 32., -32., 127.],
                           dtype=torch.bfloat16, device=device)
    gathered.view(-1)[:special.numel()] = special
    q_weight = torch.linspace(0.75, 1.25, 1536, device=device).bfloat16()
    kv_weight = torch.linspace(0.5, 1.5, 512, device=device).bfloat16()
    q_output = torch.full((rows, 1536), 19, dtype=torch.bfloat16, device=device)
    latent_output = torch.full((rows, 576), -23, dtype=torch.bfloat16, device=device)
    eps = 1e-5

    def forward():
        unpack_and_rms_norm_qkv(gathered, local_q, local_kv, q_weight, kv_weight, q_output, latent_output, eps)

    def check():
        query = gathered[:, :, :local_q].reshape(rows, 1536)
        latent = gathered[:, :, local_q:].reshape(rows, 576)
        expected_q = rms_norm_columns(query, q_weight, torch.empty_like(q_output), eps)
        expected_latent = torch.empty_like(latent_output)
        expected_latent[:, :512] = rms_norm_columns(latent[:, :512], kv_weight, expected_latent[:, :512], eps)
        expected_latent[:, 512:] = latent[:, 512:]
        torch.cuda.synchronize(device)
        torch.testing.assert_close(q_output, expected_q, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(latent_output, expected_latent, rtol=0, atol=0, equal_nan=True)

    compiled = compile_forward(forward)
    compiled()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    gathered.copy_((torch.randn(gathered.shape, generator=generator) * 0.2).to(torch.bfloat16).to(device))
    gathered.view(-1)[:special.numel()] = special
    q_output.fill_(19)
    latent_output.fill_(-23)
    before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
    check()
    graph.reset()


@pytest.mark.parametrize("rows", [4, 32])
@torch.no_grad()
def test_unpack_and_rms_norm_qkv_dynamic_outputs_match_original(rows):
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.kimi3.pointwise import rms_norm_columns, unpack_and_rms_norm_qkv

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    world, local_q, local_kv = 16, 96, 36
    generator = torch.Generator().manual_seed(4300 + rows)
    gathered = (torch.randn((rows, world, local_q + local_kv), generator=generator) * 0.2).to(torch.bfloat16).to(device)
    special = torch.tensor([torch.nan, torch.inf, -torch.inf, -0.0, 0.0, 1., -1., 0.25, -0.25, 32., -32., 127.],
                           dtype=torch.bfloat16, device=device)
    gathered.view(-1)[:special.numel()] = special
    q_weight = torch.linspace(0.75, 1.25, 1536, device=device).bfloat16()
    kv_weight = torch.linspace(0.5, 1.5, 512, device=device).bfloat16()
    eps = 1e-5

    def forward():
        query_norm = torch.empty((rows, 1536), dtype=torch.bfloat16, device=device)
        latent = torch.empty((rows, 576), dtype=torch.bfloat16, device=device)
        unpack_and_rms_norm_qkv(gathered, local_q, local_kv, q_weight, kv_weight, query_norm, latent, eps)
        return query_norm, latent

    def check(actual):
        query = gathered[:, :, :local_q].reshape(rows, 1536)
        latent = gathered[:, :, local_q:].reshape(rows, 576)
        expected_q = rms_norm_columns(query, q_weight, torch.empty_like(actual[0]), eps)
        expected_latent = torch.empty_like(actual[1])
        expected_latent[:, :512] = rms_norm_columns(latent[:, :512], kv_weight, expected_latent[:, :512], eps)
        expected_latent[:, 512:] = latent[:, 512:]
        torch.cuda.synchronize(device)
        for value, wanted in zip(actual, (expected_q, expected_latent), strict=True):
            torch.testing.assert_close(value, wanted, rtol=0, atol=0, equal_nan=True)

    compiled = compile_forward(forward)
    check(compiled())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled()
    gathered.copy_((torch.randn(gathered.shape, generator=generator) * 0.2).to(torch.bfloat16).to(device))
    gathered.view(-1)[:special.numel()] = special
    for value in actual:
        value.fill_(19)
    before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
    check(actual)
    graph.reset()
