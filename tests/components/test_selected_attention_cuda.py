# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared sparse GQA and region selection retain capture-stable b12x programs."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.paged_batch import Metadata
from ignition.components.region_selection import PreparedRegionSelection
from ignition.components.selected_attention import PreparedSelectedAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


def check_graph(operation, output, arena, expected, **tolerance):
    operation()
    torch.testing.assert_close(output, expected, **tolerance)
    compiled = compile_forward(operation)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    baseline = output.clone()
    output.zero_()
    arena.tensor.fill_(0x7f)
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
    torch.testing.assert_close(output, baseline, atol=0, rtol=0)
    torch.testing.assert_close(output, expected, **tolerance)


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("tokens", [3, 129])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prepared_selected_attention(tokens, gated, kv_dtype, functional):
    torch.manual_seed(92)
    device = torch.device("cuda:0")
    q = torch.randn(tokens, 8, 192, device=device, dtype=torch.bfloat16)
    k = torch.randn(16, 128, 2, 192, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k, v = k.to(kv_dtype), v.to(kv_dtype)
    pages = torch.arange(16, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(tokens, device=device, dtype=torch.int32)
    positions = torch.full((tokens,), 2047, device=device, dtype=torch.int64)
    positions[-1] = -1
    selected = torch.full((tokens, 2, 4104), -1, device=device, dtype=torch.int32)
    selected[:, 0, :1024] = torch.arange(0, 2048, 2, device=device)
    selected[:, 1, :1024] = torch.arange(1, 2048, 2, device=device)
    output = torch.empty_like(q)
    gate = torch.randn(tokens, 24, device=device, dtype=torch.bfloat16)[:, 8:16] if gated else None
    expected = torch.zeros_like(output)
    for group in range(2):
        idx = selected[0, group, :1024].long()
        keys, values = (x.float().flatten(0, 1)[idx, group] for x in (k, v))
        scores = torch.einsum("thd,kd->thk", q[:, group * 4:(group + 1) * 4].float(), keys) * 192 ** -.5
        expected[:, group * 4:(group + 1) * 4] = torch.softmax(scores, -1) @ values
    expected[-1].zero_()
    if gate is not None:
        expected *= gate.sigmoid()[..., None]
    arena = ScratchArena(device)
    with PreparationSession(device=device, autotune=True, compile_workers=0) as session:
        operation = PreparedSelectedAttention(q, k, v, pages, requests, positions, selected, output,
            session=session, workspace=arena, name="selected", output_gate=gate)
        session.freeze()
        arena.finalize()
        arena.freeze()
        if functional:
            attention = operation.functional(session=session)
            arena.release_preparation()
            def operation():
                return output.copy_(attention(q, pages, requests, positions, selected, output_gate=gate))
        check_graph(operation, output, arena, expected, rtol=.02, atol=.004)


@pytest.mark.parametrize("tokens", [1, 12, 67])
@pytest.mark.parametrize("first_region", [0, 511, 512, 600])
def test_prepared_region_selection(tokens, first_region):
    torch.manual_seed(621)
    device = torch.device("cuda:0")
    groups, heads = 2, 4
    query = torch.randn(tokens, groups, heads, 256, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(tokens, groups, heads, device=device)
    weights[:, 0].zero_()
    cache = torch.randn(64, 16, 256, device=device).to(torch.float8_e4m3fn)
    m = Metadata.allocate(tokens, 1, 64, device)
    m.positions.copy_(torch.arange(tokens, device=device) * 8 + first_region * 8)
    m.lengths.fill_(8192)
    m.page_table.copy_(torch.arange(64, device=device))
    if tokens > 1:
        m.positions[-1] = -1
    output = torch.empty(tokens, groups, 512, dtype=torch.int32, device=device)
    expected = torch.full_like(output, -1)
    scores = (torch.einsum("tghd,kd->tghk", query.float(), cache.flatten(0, 1).float()).relu()
              * weights[..., None]).sum(2)
    for row in range(tokens):
        length = max(int(m.positions[row]) // 8, 0)
        for group in range(groups):
            ids = scores[row, group, :length].argsort(descending=True, stable=True)[:512].sort().values
            expected[row, group, :ids.numel()] = ids.int()
    arena = ScratchArena(device)
    with PreparationSession(device=device, autotune=True, compile_workers=0) as session:
        operation = PreparedRegionSelection(query, weights, cache, m, output, ratio=8,
            session=session, workspace=arena, name="select")
        session.freeze()
        arena.finalize()
        arena.freeze()
        check_graph(operation, output, arena, expected, rtol=0, atol=0)
