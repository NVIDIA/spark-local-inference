# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Circular history attention preserves candidate causality and private proposals."""

import os

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.window_attention import PreparedWindowAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("batch,width,first,step", [(1, 4096, 0, None), (1, 640, 0, None), (1, 4, 0, None), (4, 7, 3, None),
                                                 (4, 4, 3, None), (4, 1, 3, 2)])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_window_attention(batch, width, first, step, gated, kv_dtype, functional):
    torch.manual_seed(721)
    device = torch.device("cuda:0")
    window, heads, groups, dim = 512, 8, 1, 192
    tokens = batch * width
    q = torch.randn(tokens, heads, dim, device=device).bfloat16()
    k = torch.randn(tokens, groups, dim, device=device).bfloat16()
    v = torch.randn_like(k)
    hk = torch.randn(batch + 1, window, groups, dim, device=device).bfloat16()
    hv = torch.randn_like(hk)
    hk, hv = hk.to(kv_dtype), hv.to(kv_dtype)
    original = hk.clone(), hv.clone()
    m = Metadata.allocate(tokens, batch, 1, device)
    counts = torch.full((batch,), width, dtype=torch.int32, device=device)
    starts = torch.tensor([0, 510, 1027, 1024][:batch], device=device)
    m.set(starts, counts, torch.arange(1, batch + 1, device=device), m.page_table,
          torch.zeros(batch, width, device=device, dtype=torch.int64), torch.tensor([batch], device=device))
    proposal = None
    if step is not None:
        pk = torch.randn(batch, 3, groups, dim, device=device).bfloat16()
        pv = torch.randn_like(pk)
        proposal = pk, pv, step
        m.query_start_loc.copy_(torch.arange(batch + 1, device=device) * 3)
        m.positions.add_(step)
    m.positions.masked_fill_(m.positions < first, -1)
    if batch > 1:
        m.lengths[-1] = 0
        m.positions[-width:] = -1
    out = torch.empty_like(q)
    gate = torch.randn(tokens, heads * 3, device=device, dtype=torch.bfloat16)[:, heads:2 * heads] if gated else None
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="window-gqa", workload_schema="window-gqa",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        operation = PreparedWindowAttention(q, k, v, hk, hv, m, out,
            first_position=first, proposal=proposal, output_gate=gate, session=session, workspace=arena, name="window")
        if functional:
            operation.functional(session=session)
        session.tune()
        if functional:
            arena.release_preparation()
            attention = operation
            def operation():
                return out.copy_(attention(q, k, v, m, output_gate=gate, proposal=proposal))
        compiled = compile_forward(operation)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        out.fill_(float("nan"))
        if gate is not None:
            gate.neg_()
        arena.tensor.fill_(0x7f)
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        for row in range(batch):
            for col in sorted({0, width // 2, width - 1}):
                token = row * width + col
                position = int(m.positions[token])
                if position < 0:
                    assert not out[token].any()
                    continue
                begin = int(starts[row])
                positions = torch.arange(max(first, position - window + 1), position + 1, device=device)
                old = positions < begin
                current_k, current_v = (k[row * width:(row + 1) * width], v[row * width:(row + 1) * width]) if step is None else (pk[row], pv[row])
                keys, values = [] , []
                for history, current, result in ((hk, current_k, keys), (hv, current_v, values)):
                    result.append(torch.cat((history.float()[row + 1, positions[old] % window], current.float()[positions[~old] - begin])))
                logits = q[token].float() @ keys[0][:, 0].float().T / dim ** .5
                expected = logits.softmax(-1) @ values[0][:, 0].float()
                if gate is not None:
                    expected = (expected.bfloat16() * gate[token].sigmoid()[:, None]).float()
                torch.testing.assert_close(out[token].float(), expected, atol=.004, rtol=.02)
        torch.testing.assert_close(hk.float(), original[0].float(), atol=0, rtol=0)
        torch.testing.assert_close(hv.float(), original[1].float(), atol=0, rtol=0)
        graph.reset()


@pytest.mark.parametrize("extent", [8, 512])
@pytest.mark.parametrize("capacity,count", [(1, 0), (1, 1), (4, 2), (641, 641)])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_ring_commit_updates_only_accepted_slots(extent, capacity, count, kv_dtype):
    from ignition.components.gated_compression import commit_gated

    device = torch.device("cuda:0")
    values = torch.arange(capacity * 192, device=device).reshape(capacity, 192).remainder(128).bfloat16()
    scores = -values
    values, scores = values.to(kv_dtype), scores.to(kv_dtype)
    state_values = torch.full((3, extent, 1, 192), 17, device=device, dtype=kv_dtype)
    state_scores = torch.full_like(state_values, 23)
    m = Metadata.allocate(capacity, 1, 1, device)
    start = 2 * extent - 2
    m.starts.fill_(start)
    m.state_rows.fill_(1)
    m.query_start_loc[-1] = capacity
    counts = torch.tensor([count], device=device, dtype=torch.int32)
    expected_values, expected_scores = state_values.clone(), state_scores.clone()
    for i in range(count):
        expected_values[1, (start + i) % extent, 0] = values[i]
        expected_scores[1, (start + i) % extent, 0] = scores[i]
    operation = compile_forward(lambda: commit_gated(values, scores, state_values, state_scores, m, counts))
    operation()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    state_values.fill_(17)
    state_scores.fill_(23)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(state_values.float(), expected_values.float(), atol=0, rtol=0)
    torch.testing.assert_close(state_scores.float(), expected_scores.float(), atol=0, rtol=0)
    graph.reset()
