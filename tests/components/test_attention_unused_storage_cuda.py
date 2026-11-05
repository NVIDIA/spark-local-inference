# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared attention ignores unused cache records and tensor padding."""

import os

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.paged_attention import PreparedPagedAttention
from ignition.components.selected_attention import PreparedSelectedAttention
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.components.window_attention import PreparedWindowAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU",
)


def _capture(operation, session):
    session.tune()
    forward = compile_forward(operation)
    forward()
    graph = torch.cuda.CUDAGraph()
    with session.capture(), torch.cuda.graph(graph):
        forward()
    session.freeze()
    return graph


def _replay(graph):
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations


def _poison(tensor, mask):
    if tensor.dtype == torch.float8_e4m3fn:
        tensor.view(torch.uint8).masked_fill_(mask.expand_as(tensor), 0x7F)
    else:
        tensor.masked_fill_(mask.expand_as(tensor), float("nan"))


@pytest.mark.parametrize("rows", [3, 129])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("gated", [False, True])
@torch.no_grad()
def test_selected_attention_ignores_unselected_cache(rows, kv_dtype, gated):
    device = torch.device("cuda", 0)
    torch.manual_seed(924)
    q = torch.randn(rows, 8, 192, device=device).bfloat16()
    k = torch.randn(6, 128, 2, 192, device=device).to(kv_dtype)
    v = torch.randn_like(k.bfloat16()).to(kv_dtype)
    clean_k, clean_v = k.clone(), v.clone()
    pages = torch.tensor([[3, 1, 5, 0]], dtype=torch.int32, device=device)
    requests = torch.zeros(rows, dtype=torch.int32, device=device)
    positions = torch.full((rows,), 64, dtype=torch.int64, device=device)
    positions[-1] = -1
    selected = torch.full((rows, 2, 4104), -1, dtype=torch.int32, device=device)
    selected[:, :, :65] = torch.arange(65, device=device)
    selected[:, :, 17] = -1
    live = selected[0, 0][selected[0, 0] >= 0].long()
    output = torch.empty_like(q)
    gate = torch.randn(rows, 8, device=device).bfloat16() if gated else None
    expected = torch.zeros_like(output)
    for group in range(2):
        scores = q[:-1, group * 4:(group + 1) * 4].float() @ k[3, live, group].float().T / 192 ** .5
        expected[:-1, group * 4:(group + 1) * 4] = scores.softmax(-1) @ v[3, live, group].float()
    if gate is not None:
        expected *= gate.sigmoid()[..., None]
    unused = torch.ones((*k.shape[:-1], 1), dtype=torch.bool, device=device)
    unused[3, live] = False
    with RecordedPreparationSession(device=device, checkpoint="selected-gqa", workload_schema="selected-gqa",
                                    numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        operation = PreparedSelectedAttention(q, k, v, pages, requests, positions, selected, output,
            output_gate=gate, session=session, workspace=arena, name="selected")
        graph = _capture(operation, session)
        try:
            _replay(graph)
            baseline = output.clone()
            torch.testing.assert_close(baseline, expected, atol=.004, rtol=.02)
            for operand in ("k", "v", "both"):
                k.copy_(clean_k)
                v.copy_(clean_v)
                if operand in ("k", "both"):
                    _poison(k, unused)
                if operand in ("v", "both"):
                    _poison(v, unused)
                saved_k, saved_v = k.view(torch.uint8).clone(), v.view(torch.uint8).clone()
                for _ in range(3):
                    _replay(graph)
                    assert torch.isfinite(output).all(), operand
                    torch.testing.assert_close(output, baseline, atol=0, rtol=0)
                torch.testing.assert_close(k.view(torch.uint8), saved_k)
                torch.testing.assert_close(v.view(torch.uint8), saved_v)
            for target in (k, v):
                k.copy_(clean_k)
                v.copy_(clean_v)
                mask = torch.zeros_like(unused)
                mask[3, 0] = True
                _poison(target, mask)
                _replay(graph)
                assert not torch.isfinite(output[0]).all()
        finally:
            graph.reset()


@pytest.mark.parametrize("rows", [3, 129])
@pytest.mark.parametrize("start", [17, 517])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("gated", [False, True])
@torch.no_grad()
def test_window_attention_ignores_unused_history(rows, start, kv_dtype, gated):
    device = torch.device("cuda", 0)
    torch.manual_seed(725)
    q = torch.randn(rows, 8, 192, device=device).bfloat16()
    k = torch.randn(rows, 1, 192, device=device).bfloat16()
    v = torch.randn_like(k)
    hk = torch.randn(2, 512, 1, 192, device=device).to(kv_dtype)
    hv = torch.randn_like(hk.bfloat16()).to(kv_dtype)
    clean_hk, clean_hv = hk.clone(), hv.clone()
    m = Metadata.allocate(rows, 1, 1, device)
    m.set(torch.tensor([start], device=device), torch.tensor([rows - 1], device=device),
        torch.ones(1, dtype=torch.int64, device=device), m.page_table,
        torch.zeros(1, rows, dtype=torch.int64, device=device), torch.ones(1, device=device))
    output = torch.empty_like(q)
    gate = torch.randn(rows, 8, device=device).bfloat16() if gated else None
    expected = torch.zeros_like(output)
    for row in range(rows - 1):
        positions = torch.arange(max(0, start + row - 511), start, device=device)
        keys = torch.cat((hk[1, positions % 512, 0].float(), k[:row + 1, 0].float()))
        values = torch.cat((hv[1, positions % 512, 0].float(), v[:row + 1, 0].float()))
        expected[row] = (q[row].float() @ keys.T / 192 ** .5).softmax(-1) @ values
    if gate is not None:
        expected *= gate.sigmoid()[..., None]
    unused = torch.ones((*hk.shape[:-1], 1), dtype=torch.bool, device=device)
    unused[1, torch.arange(max(0, start - 511), start, device=device) % 512] = False
    with RecordedPreparationSession(device=device, checkpoint="window-gqa", workload_schema="window-gqa",
                                    numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        operation = PreparedWindowAttention(q, k, v, hk, hv, m, output, output_gate=gate,
            session=session, workspace=arena, name="window")
        graph = _capture(operation, session)
        try:
            _replay(graph)
            baseline = output.clone()
            torch.testing.assert_close(baseline, expected, atol=.004, rtol=.02)
            for operand in ("k", "v", "both"):
                hk.copy_(clean_hk)
                hv.copy_(clean_hv)
                if operand in ("k", "both"):
                    _poison(hk, unused)
                if operand in ("v", "both"):
                    _poison(hv, unused)
                saved_hk, saved_hv = hk.view(torch.uint8).clone(), hv.view(torch.uint8).clone()
                for _ in range(3):
                    _replay(graph)
                    assert torch.isfinite(output).all(), operand
                    torch.testing.assert_close(output, baseline, atol=0, rtol=0)
                torch.testing.assert_close(hk.view(torch.uint8), saved_hk)
                torch.testing.assert_close(hv.view(torch.uint8), saved_hv)
            for target in (hk, hv):
                hk.copy_(clean_hk)
                hv.copy_(clean_hv)
                if kv_dtype == torch.float8_e4m3fn:
                    target[1, (start - 1) % 512].view(torch.uint8).fill_(0x7F)
                else:
                    target[1, (start - 1) % 512].fill_(float("nan"))
                _replay(graph)
                assert not torch.isfinite(output[0]).all()
        finally:
            graph.reset()


@pytest.mark.parametrize("math_mode,width", [("flash", 64), ("eager_bf16", 72)])
@pytest.mark.parametrize("window", [None, (16, 16)])
@torch.no_grad()
def test_varlen_attention_ignores_storage_past_live_rows(math_mode, width, window):
    if math_mode == "eager_bf16" and window is not None:
        pytest.skip("eager BF16 attention requires an unrestricted window")
    device = torch.device("cuda", 0)
    torch.manual_seed(214)
    q, k, v = [torch.randn(257, 8, width, device=device).bfloat16() for _ in range(3)]
    clean_k, clean_v = k.clone(), v.clone()
    output = torch.empty_like(q)
    cu = torch.tensor([0, 17, 65], dtype=torch.int32, device=device)
    with RecordedPreparationSession(device=device, checkpoint="varlen-attention", workload_schema="varlen-attention",
                                    numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        operation = PreparedVarlenAttention(q, k, v, cu, output, math_mode=math_mode, window=window,
            session=session, workspace=arena, name="varlen")
        graph = _capture(operation, session)
        try:
            for length in (65, 129, 257):
                cu[2] = length
                k.copy_(clean_k)
                v.copy_(clean_v)
                _replay(graph)
                baseline = output[:length].clone()
                assert torch.isfinite(baseline).all()
                for begin, end in ((0, 17), (17, length)):
                    for token in (begin, (begin + end) // 2, end - 1):
                        first = begin if window is None else max(begin, token - window[0])
                        last = end if window is None else min(end, token + window[1] + 1)
                        if math_mode == "eager_bf16":
                            scores = torch.einsum("hd,khd->hk", q[token], k[first:last]) * width ** -.5
                            probabilities = scores.softmax(-1, dtype=torch.float32).bfloat16()
                            expected = torch.einsum("hk,khd->hd", probabilities, v[first:last]).float()
                        else:
                            scores = torch.einsum("hd,khd->hk", q[token].float(), k[first:last].float()) / width ** .5
                            probabilities = scores.softmax(-1)
                            expected = torch.einsum("hk,khd->hd", probabilities, v[first:last].float())
                        torch.testing.assert_close(output[token].float(), expected, atol=.004, rtol=.02)
                for operand in ("k", "v", "both"):
                    k.copy_(clean_k)
                    v.copy_(clean_v)
                    if operand in ("k", "both"):
                        k[length:].fill_(float("nan"))
                    if operand in ("v", "both"):
                        v[length:].fill_(float("nan"))
                    saved_k, saved_v = k.view(torch.uint8).clone(), v.view(torch.uint8).clone()
                    _replay(graph)
                    torch.testing.assert_close(output[:length], baseline, atol=0, rtol=0)
                    torch.testing.assert_close(k.view(torch.uint8), saved_k)
                    torch.testing.assert_close(v.view(torch.uint8), saved_v)
            for target in (k, v):
                k.copy_(clean_k)
                v.copy_(clean_v)
                target[0].fill_(float("nan"))
                _replay(graph)
                assert not torch.isfinite(output[0]).all()
        finally:
            graph.reset()


@pytest.mark.parametrize("mode,rows", [("decode", 1), ("extend", 17)])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("window", [None, 32])
@torch.no_grad()
def test_dense_mla_ignores_unused_records(mode, rows, kv_dtype, window):
    from b12x.attention import dense_mla

    device = torch.device("cuda", 0)
    torch.manual_seed(528)
    capacity = max(2, rows)
    record_width = 1088 if kv_dtype == torch.float8_e4m3fn else 640
    q = (torch.randn(capacity, 8, 576, device=device) * .1).bfloat16()
    cache = (torch.randn(8, 128, record_width, device=device) * .1).to(kv_dtype)
    clean_cache = cache.clone()
    output = torch.empty(capacity, 8, 512, device=device, dtype=torch.bfloat16)
    lse = torch.empty(capacity, 8, device=device)
    pages = torch.tensor([[3, 1, 5], [6, 4, 7]], dtype=torch.int32, device=device)
    lengths = torch.tensor([257, 0], dtype=torch.int32, device=device)
    cu = torch.tensor([0, rows, rows], dtype=torch.int32, device=device)
    caps = dense_mla.Caps(device=device, mode=mode, kv_dtype=kv_dtype, q_dtype=q.dtype,
        num_q_heads=8, page_size=128, max_total_q=capacity, max_batch=2, max_cache_tokens=257,
        max_page_table_width=3, num_cache_pages=8, physical_record_width=record_width,
        window_size=window, use_cuda_graph=True)
    with RecordedPreparationSession(device=device, checkpoint="dense-mla", workload_schema="dense-mla",
                                    numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        operation = PreparedDenseMLA(q, cache, output, pages, lengths, cu, caps=caps, scale=576 ** -.5,
            lse=lse, session=session, workspace=arena, name="dense-mla")
        graph = _capture(operation, session)
        try:
            for length in (1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 257):
                count = min(rows, length)
                lengths[0] = length
                cu[1:] = count
                cache.copy_(clean_cache)
                _replay(graph)
                baseline, baseline_lse = output[:count].clone(), lse[:count].clone()
                expected, expected_lse = dense_mla.reference(q[:count], cache, pages[:1], lengths[:1], cu[:2],
                    kv_scale=operation.arguments["kv_scale"], q_scale=operation.arguments["q_scale"],
                    sm_scale=576 ** -.5, window_size=window)
                torch.testing.assert_close(baseline.float(), expected.float(), atol=5e-4, rtol=.02)
                torch.testing.assert_close(baseline_lse, expected_lse, atol=2e-5, rtol=2e-5)
                unused = torch.ones_like(cache, dtype=torch.bool)
                logical = torch.arange(length, device=device)
                unused[pages[0, logical // 128].long(), logical % 128, :576] = False
                _poison(cache, unused)
                saved_cache = cache.view(torch.uint8).clone()
                for _ in range(3):
                    _replay(graph)
                    assert torch.isfinite(output[:count]).all()
                    torch.testing.assert_close(output[:count], baseline, atol=0, rtol=0)
                    torch.testing.assert_close(lse[:count], baseline_lse, atol=0, rtol=0)
                torch.testing.assert_close(cache.view(torch.uint8), saved_cache)
            for first, last in ((0, 512), (512, 576)):
                cache.copy_(clean_cache)
                live = torch.zeros_like(cache, dtype=torch.bool)
                live[5, 0, first:last] = True
                _poison(cache, live)
                _replay(graph)
                assert not torch.isfinite(output[rows - 1]).all()
                assert torch.isnan(lse[rows - 1]).all()
        finally:
            graph.reset()


@pytest.mark.parametrize("mode,rows", [("verify", 8), ("prefill", 33)])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@torch.no_grad()
def test_block_selected_attention_ignores_unused_cache(mode, rows, kv_dtype):
    device = torch.device("cuda", 0)
    torch.manual_seed(412)
    q = (torch.randn(rows, 16, 128, device=device) * .1).bfloat16()
    k = (torch.randn(8, 128, 1, 128, device=device) * .1).to(kv_dtype)
    v = (torch.randn(8, 128, 1, 128, device=device) * .1).to(kv_dtype)
    clean_k, clean_v = k.clone(), v.clone()
    output = torch.empty_like(q)
    pages = torch.tensor([[3, 1, 5]], dtype=torch.int32, device=device)
    lengths = torch.tensor([257], dtype=torch.int32, device=device)
    cu = torch.tensor([0, rows], dtype=torch.int32, device=device)
    selected = torch.full((1, rows, 16), -1, dtype=torch.int32, device=device)
    selected[:, :, :2] = torch.tensor([0, 2], dtype=torch.int32, device=device)
    with RecordedPreparationSession(device=device, checkpoint="block-selected-gqa", workload_schema="block-selected-gqa",
                                    numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        operation = PreparedPagedAttention(q, k, v, pages, lengths, cu, output, selected_blocks=selected,
            mode=mode, width=rows if mode == "verify" else 1, session=session, workspace=arena, name="msa")
        assert not operation.decode and operation.tma is None
        graph = _capture(operation, session)
        try:
            for length in (1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 257):
                count = min(rows, length)
                lengths[0] = length
                cu[1] = count
                k.copy_(clean_k)
                v.copy_(clean_v)
                _replay(graph)
                baseline = output[:count].clone()
                for row in range(count):
                    end = length - count + row + 1
                    logical = torch.arange(end, device=device)
                    logical = logical[(logical // 128 == 0) | (logical // 128 == 2)]
                    physical = pages[0, logical // 128].long()
                    keys, values = (tensor[physical, logical % 128, 0].float() for tensor in (k, v))
                    expected = (q[row].float() @ keys.T / 128 ** .5).softmax(-1) @ values
                    torch.testing.assert_close(output[row].float(), expected, atol=.002, rtol=.02)
                unused = torch.ones((*k.shape[:-1], 1), dtype=torch.bool, device=device)
                logical = torch.arange(length, device=device)
                logical = logical[(logical // 128 == 0) | (logical // 128 == 2)]
                unused[pages[0, logical // 128].long(), logical % 128] = False
                for operand in ("k", "v", "both"):
                    k.copy_(clean_k)
                    v.copy_(clean_v)
                    if operand in ("k", "both"):
                        _poison(k, unused)
                    if operand in ("v", "both"):
                        _poison(v, unused)
                    saved_k, saved_v = k.view(torch.uint8).clone(), v.view(torch.uint8).clone()
                    _replay(graph)
                    assert torch.isfinite(output[:count]).all(), operand
                    torch.testing.assert_close(output[:count], baseline, atol=0, rtol=0)
                    torch.testing.assert_close(k.view(torch.uint8), saved_k)
                    torch.testing.assert_close(v.view(torch.uint8), saved_v)
            for target in (k, v):
                k.copy_(clean_k)
                v.copy_(clean_v)
                live = torch.zeros_like(unused)
                live[5, 0] = True
                _poison(target, live)
                _replay(graph)
                assert not torch.isfinite(output[rows - 1]).all()
        finally:
            graph.reset()
