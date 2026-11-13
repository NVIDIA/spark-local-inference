# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed projection fusion preserves normalization, rotary rounding, and KV slots."""

import os

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.qk_norm_rope import PreparedQKNormRoPE
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.step5.model import RMSNorm, rotary, rotary_inverse

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


@pytest.mark.parametrize("indexer", [False, True])
@pytest.mark.parametrize("tokens", [1, 7, 128])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@torch.no_grad()
def test_qk_normalization_rotary_and_append(indexer, tokens, kv_dtype):
    torch.manual_seed(719)
    device = torch.device("cuda:0")
    heads, groups, dim, rotated = (4, 1, 256, 32) if indexer else (8, 1, 192, 192)
    width = (heads + 2 * groups) * dim + heads
    parent = torch.randn(tokens, width + 32, device=device).bfloat16()
    projected = parent[:, 16:16 + width]
    qw, kw = (torch.randn(dim, device=device) * .1 for _ in range(2))
    bias = None
    if indexer:
        qw, kw = qw.bfloat16(), (1 + kw).bfloat16()
        bias = (torch.randn(dim, device=device) * .1).bfloat16()
    inverse = rotary_inverse(rotated, 1_000_000, None, device)
    m = Metadata.allocate(tokens, 1, 8, device)
    m.positions.copy_(torch.arange(tokens, device=device) + 127)
    m.page_table.copy_(torch.arange(7, -1, -1, device=device))
    if tokens > 1:
        m.positions[-1] = -1
    q = torch.empty(tokens, heads, dim, device=device, dtype=torch.float8_e4m3fn if indexer else torch.bfloat16)
    kv_dtype = torch.bfloat16 if indexer else kv_dtype
    k = torch.empty(tokens, groups, dim, device=device, dtype=kv_dtype)
    v = torch.empty_like(k)
    kc = torch.full((8, 128, groups, dim), 17, device=device, dtype=kv_dtype)
    vc = torch.full_like(kc, 23)
    with RecordedPreparationSession(device=device, checkpoint="qk-rope", workload_schema="qk-rope",
                                    numerical_policy="default") as session:
        operation = PreparedQKNormRoPE(projected, qw, kw, inverse, m, q, k, epsilon=1e-6,
            session=session, name="qk", bias=bias, value=None if indexer else v,
            key_cache=None if indexer else kc, value_cache=None if indexer else vc)
        saved = tuple(t.clone() for t in (m.requests, m.positions, m.page_table))
        for tensor in (m.requests, m.positions, m.page_table):
            tensor.fill_(1 << 30)
        session.tune()
        for tensor, initial in zip((m.requests, m.positions, m.page_table), saved, strict=True):
            tensor.copy_(initial)
        forward = compile_forward(operation)
        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        parent.mul_(.75)
        k.fill_(float("nan"))
        kc.fill_(17)
        vc.fill_(23)
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        iq = projected[:, :heads * dim].view(tokens, heads, dim)
        ik = projected[:, heads * dim:(heads + groups) * dim].view(tokens, groups, dim)
        eq = rotary(RMSNorm(qw, 1e-6)(iq), m.positions, inverse).to(q.dtype)
        if indexer:
            x = ik.float()
            mean = x.mean(-1, keepdim=True)
            variance = x.square().mean(-1, keepdim=True) - mean.square()
            ek = ((x - mean) * torch.rsqrt(variance + 1e-6) * kw.float() + bias.float()).bfloat16()
        else:
            ek = RMSNorm(kw, 1e-6)(ik)
        ek = rotary(ek, m.positions, inverse)
        torch.testing.assert_close(q.float(), eq.float(), atol=.016 if indexer else .008, rtol=.01)
        torch.testing.assert_close(k.float(), ek.to(kv_dtype).float(), atol=.008, rtol=.01)
        if not indexer:
            ev = projected[:, (heads + groups) * dim:(heads + 2 * groups) * dim].view_as(v)
            torch.testing.assert_close(v.float(), ev.to(kv_dtype).float(), atol=0, rtol=0)
            expected_key, expected_value = torch.full_like(kc, 17), torch.full_like(vc, 23)
            for row in range(tokens):
                position = int(m.positions[row])
                if position >= 0:
                    page = int(m.page_table[0, position // 128])
                    expected_key[page, position % 128] = k[row]
                    expected_value[page, position % 128] = ev[row]
            torch.testing.assert_close(kc.float(), expected_key.float(), atol=0, rtol=0)
            torch.testing.assert_close(vc.float(), expected_value.float(), atol=0, rtol=0)
        graph.reset()


@torch.no_grad()
def test_qk_tuning_memory_is_independent_of_cache_capacity(tmp_path):
    device = torch.device("cuda:0")
    tokens, heads, groups, dim = 7, 8, 1, 192
    projected = torch.randn(tokens, (heads + 2 * groups) * dim, device=device, dtype=torch.bfloat16)
    qw, kw = (torch.zeros(dim, device=device) for _ in range(2))
    inverse = rotary_inverse(dim, 1_000_000, None, device)
    metadata = Metadata.allocate(tokens, 1, 1024, device)
    query = torch.empty(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    key = torch.empty(tokens, groups, dim, device=device, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    page_elements = 128 * groups * dim
    caches = tuple(torch.empty_strided((1024, 128, groups, dim),
        (2 * page_elements, groups * dim, dim, 1), device=device, dtype=torch.bfloat16) for _ in range(2))
    with RecordedPreparationSession(device=device, checkpoint=str(tmp_path), workload_schema="qk-cache-capacity",
                                    numerical_policy="default") as session:
        PreparedQKNormRoPE(projected, qw, kw, inverse, metadata, query, key, value=value,
            key_cache=caches[0], value_cache=caches[1], epsilon=1e-6, session=session, name="qk")
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        report = session.tune()
        torch.cuda.synchronize()
        assert report["benchmarked_candidates"] > 0
        growth = torch.cuda.max_memory_allocated(device) - before
        eviction_bytes = max(2 * torch.cuda.get_device_properties(device).L2_cache_size, 64 << 20)
        assert growth < eviction_bytes + (16 << 20)
        assert torch.cuda.memory_allocated(device) - before < 1 << 20
