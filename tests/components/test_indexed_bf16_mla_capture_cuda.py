# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexed BF16 MLA retains arena bindings through compiled graph replay."""

import os

import pytest
import torch
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference, pack_deepseek_v41_cache_reference,
)

from ignition.components.compressed_mla import PreparedCompressedMLA
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("page_size", [pytest.param(128, id="c1"), pytest.param(64, id="c2")])
@torch.no_grad()
def test_indexed_bf16_mla_arena_survives_compiled_capture(page_size):
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device).manual_seed(41021)
    rows, heads, width, window_page = 8, 16, 512, 136
    logical_pages = width // page_size
    num_pages = logical_pages + 1
    window_values = (torch.randn(3 * window_page, 512, device=device, generator=gen) * .2).bfloat16()
    indexed_values = (torch.randn(num_pages * page_size, 512, device=device, generator=gen) * .2).bfloat16()
    window_backing = torch.full((3, window_page * 528 + 1024), 173, device=device, dtype=torch.uint8)
    indexed_backing = torch.full((num_pages, page_size * 288 + 256), 179, device=device, dtype=torch.uint8)
    window = window_backing[:, :window_page * 528]
    indexed = indexed_backing[:, :page_size * 288]
    window.copy_(pack_deepseek_v41_cache_reference(window_values, page_size=window_page, cache_kind="swa"))
    indexed.copy_(pack_deepseek_v41_cache_reference(indexed_values, page_size=page_size, cache_kind="indexed"))
    query = (torch.randn(rows, heads, 512, device=device, generator=gen) * .2).bfloat16()
    window_indices = (torch.arange(128, dtype=torch.int32, device=device) + window_page).expand(rows, -1).contiguous()
    indexed_indices = torch.arange(width, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
    window_lengths = torch.tensor([128, 83, 1, 0, 65, 127, 32, 0], dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([512, 479, 65, 0, 129, 511, 32, 0], dtype=torch.int32, device=device)
    table = torch.arange(1, num_pages, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
    original_table = table.clone()
    original_window_lengths = window_lengths.clone()
    original_indexed_lengths = indexed_lengths.clone()
    sink = torch.randn(heads, device=device, generator=gen)
    output_backing = torch.full((rows + 2, heads, 512), torch.nan, device=device, dtype=torch.bfloat16)
    output = output_backing[:rows]
    arena = ScratchArena(device)
    graph = torch.cuda.CUDAGraph()

    def expected():
        physical = torch.full_like(indexed_indices, -1)
        counts = torch.empty_like(indexed_lengths)
        for row in range(rows):
            length = max(0, min(width, int(indexed_lengths[row].item())))
            logical = indexed_indices[row, :length].long()
            page = logical // page_size
            pid = table[row].long().gather(0, page.clamp(0, logical_pages - 1))
            valid = (logical >= 0) & (page < logical_pages) & (pid >= 0) & (pid < num_pages)
            slots = (pid * page_size + logical % page_size)[valid].int()
            physical[row, :slots.numel()].copy_(slots)
            counts[row] = slots.numel()
        return compressed_sparse_mla_reference(query, window, window_indices, window_lengths,
            sm_scale=512 ** -.5, attn_sink=sink, extra_k_cache=indexed,
            extra_indices=physical, extra_topk_lengths=counts, swa_page_size=window_page,
            extra_page_size=page_size, cache_format="deepseek_v41")

    def check():
        reference = expected()
        torch.testing.assert_close(output, reference, atol=.035, rtol=.035)
        active = (window_lengths > 0) | (indexed_lengths > 0)
        if active.any():
            cosine = torch.nn.functional.cosine_similarity(output[active].float().flatten(),
                reference[active].float().flatten(), dim=0).item()
            assert cosine >= .995, cosine
            assert output[active].count_nonzero()
        assert not output[~active].count_nonzero()
        assert output_backing[rows:].isnan().all()

    with RecordedPreparationSession(device=device, checkpoint="indexed-compressed-mla",
            workload_schema="packed8-heads16-bf16", numerical_policy="checkpoint", compile_workers=2) as session:
        session.register_scratch(arena)
        attention = PreparedCompressedMLA(query, window, window_indices, window_lengths, sink, output,
            indexed=indexed, indexed_indices=indexed_indices, indexed_lengths=indexed_lengths,
            page_table=table, window_page_size=window_page, indexed_page_size=page_size,
            mode="decode", decode_row_capacity=rows, session=session, name="attention",
            workspace=arena, compute_mode="bf16")
        session.tune(show_progress=False)
        state = attention.plan.prepared.state
        assert state.mapper is None
        assert not state.launch.native_dsv41_fp8

        def forward():
            return attention()

        with torch._inductor.config.patch(compile_threads=2):
            compiled = torch.compile(forward, fullgraph=True)
            compiled()
        torch.cuda.synchronize()
        check()
        storage = (query, window_backing, indexed_backing, window_indices, indexed_indices,
                   window_lengths, indexed_lengths, table, sink, output_backing, arena.tensor, *attention.scratch)
        pointers = tuple(value.data_ptr() for value in storage)
        for value in attention.scratch:
            assert arena.tensor.data_ptr() <= value.data_ptr() < arena.tensor.data_ptr() + arena.nbytes
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        try:
            for scenario in range(3):
                if scenario == 0:
                    table.copy_(original_table.flip(1))
                    table[0, 0] = -1
                    table[1, -1] = num_pages
                    window_lengths.copy_(torch.tensor([31, 0, 1, 0, 65, 64, 0, 0], device=device))
                    indexed_lengths.copy_(torch.tensor([511, 128, 0, 0, 65, 257, 32, 0], device=device))
                elif scenario == 1:
                    table.fill_(-1)
                    window_lengths.zero_()
                    indexed_lengths.zero_()
                else:
                    table.copy_(original_table)
                    window_lengths.copy_(original_window_lengths)
                    indexed_lengths.copy_(original_indexed_lengths)
                window.copy_(pack_deepseek_v41_cache_reference(
                    (window_values.float() * (1 + .125 * (scenario + 1))).bfloat16(),
                    page_size=window_page, cache_kind="swa"))
                indexed.copy_(pack_deepseek_v41_cache_reference(
                    (indexed_values.float() + .03125 * (scenario + 1)).bfloat16(),
                    page_size=page_size, cache_kind="indexed"))
                saved_window, saved_indexed = window_backing.clone(), indexed_backing.clone()
                output_backing.fill_(torch.nan)
                before = torch.cuda.memory_stats(device)
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                assert tuple(value.data_ptr() for value in storage) == pointers
                assert torch.equal(window_backing, saved_window)
                assert torch.equal(indexed_backing, saved_indexed)
                check()
        finally:
            graph.reset()
