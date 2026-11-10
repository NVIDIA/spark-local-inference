# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared mixed-record attention through compiler mutation boundaries and replay."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference, pack_deepseek_v41_cache_reference,
)

from ignition.components.compressed_mla import PreparedCompressedCacheWriter, PreparedCompressedMLA

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("mode", ["decode", "extend"])
@pytest.mark.parametrize("window_size", [128, 136])
@pytest.mark.parametrize("rows", [4, 24])
@torch.no_grad()
def test_mixed_cache_writers_and_captured_attention(ratio, mode, window_size, rows, functional):
    device = torch.device("cuda:0")
    torch.manual_seed(4101)
    window_values = torch.randn((window_size, 512), device=device, dtype=torch.bfloat16) * .2
    compressed_values = torch.randn((128 // ratio, 512), device=device, dtype=torch.bfloat16) * .2
    query = torch.randn((rows, 16, 512), device=device, dtype=torch.bfloat16) * .2
    # The window is a strided view into a complete continuation row.
    backing = torch.zeros((3, window_size * 528 + 1024), device=device, dtype=torch.uint8)
    window = backing[:, :window_size * 528]
    compressed = torch.zeros((2, 128 // ratio * 288), device=device, dtype=torch.uint8)
    window_slots = torch.arange(window_size, device=device) + window_size
    compressed_slots = torch.arange(128 // ratio, device=device) + 128 // ratio
    window_indices = window_slots.int().repeat(rows, 1)
    compressed_indices = torch.arange(128 // ratio, dtype=torch.int32, device=device).repeat(rows, 1)
    lengths = torch.tensor([128, 83, 1, 0], dtype=torch.int32, device=device).repeat(rows // 4)
    indexed_lengths = lengths // ratio
    table = torch.ones((rows, 1), device=device, dtype=torch.int32)
    sink = torch.randn(16, device=device)
    output = torch.empty_like(query)
    metadata = (window_slots, compressed_slots, window_indices, compressed_indices,
                lengths, indexed_lengths, table)
    saved_metadata = [value.clone() for value in metadata]
    for value in metadata:
        value.fill_(2 ** 30)
    backing.fill_(231)
    compressed.fill_(231)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        write_window = PreparedCompressedCacheWriter(window_values, window, window_slots,
            page_size=window_size, kind="swa", session=session, name="window")
        write_compressed = PreparedCompressedCacheWriter(compressed_values, compressed, compressed_slots,
            page_size=128 // ratio, kind="indexed", session=session, name="compressed")
        attention = PreparedCompressedMLA(query, window, window_indices, lengths, sink, output,
            indexed=compressed, indexed_indices=compressed_indices, indexed_lengths=indexed_lengths,
            page_table=table, indexed_page_size=128 // ratio, window_page_size=window_size,
            mode=mode, decode_row_capacity=rows if mode == "decode" else None,
            session=session, name="attention", workspace=None)
        assert backing.eq(231).all() and compressed.eq(231).all()
        for value, saved in zip(metadata, saved_metadata, strict=True):
            assert value.eq(2 ** 30).all()
            value.copy_(saved)
        if functional:
            write_window.functional(session=session, cache=None)
            write_compressed.functional(session=session)
            attention.functional(session=session, window=None)

        def forward():
            if functional:
                write_window(window_values, window_slots, cache=window)
                write_compressed(compressed_values, compressed_slots)
                output.copy_(attention(query, window_indices, lengths, indexed_indices=compressed_indices,
                    indexed_lengths=indexed_lengths, page_table=table, window=window))
            else:
                write_window(window_values)
                write_compressed(compressed_values)
                attention()
        compiled = torch.compile(forward, fullgraph=True)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for _ in range(3):
            output.fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(window[1:2], pack_deepseek_v41_cache_reference(
            window_values, page_size=window_size, cache_kind="swa"))
        assert torch.equal(compressed[1:2], pack_deepseek_v41_cache_reference(
            compressed_values, page_size=128 // ratio, cache_kind="indexed"))
        expected = compressed_sparse_mla_reference(query, window, window_indices, lengths,
            sm_scale=512 ** -.5, attn_sink=sink, extra_k_cache=compressed,
            extra_indices=compressed_indices + 128 // ratio, extra_topk_lengths=indexed_lengths,
            swa_page_size=window_size, extra_page_size=128 // ratio, cache_format="deepseek_v41")
        torch.testing.assert_close(output.float(), expected.float(), atol=.012, rtol=.08)
        graph.reset()
