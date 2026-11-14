# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense MLA tuning and replay use every physical ragged query row."""

import os

import pytest
import torch
from b12x.attention import dense_mla

from ignition.components.dense_mla import PreparedDenseMLA
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("tokens,batch", [(5, 2), (9, 4)])
def test_ragged_tuning_partition_covers_physical_queries(monkeypatch, tokens, batch, functional):
    import ignition.components.dense_mla as module

    device = torch.device("cuda", 0)
    heads, page = 8, 128
    query = torch.randn((tokens, heads, 576), dtype=torch.bfloat16, device=device) * .1
    cache = torch.full((batch, page, 576), .125, dtype=torch.float8_e4m3fn, device=device)
    output = torch.empty((tokens, heads, 512), dtype=torch.bfloat16, device=device)
    table = torch.arange(batch, dtype=torch.int32, device=device).view(batch, 1)
    lengths = torch.full((batch,), 64, dtype=torch.int32, device=device)
    starts = torch.arange(batch + 1, dtype=torch.int32, device=device) * tokens // batch
    original, observed = module.benchmark_resource, []

    def checked_resource(session, key, factory):
        def allocate():
            arguments, query_seed, cache_seed = factory()
            actual = arguments["cu_seqlens_q"].cpu()
            expected = torch.arange(batch + 1, dtype=torch.int32) * tokens // batch
            torch.testing.assert_close(actual, expected)
            assert actual[-1] == arguments["q"].shape[0] == tokens
            observed.append(True)
            return arguments, query_seed, cache_seed
        return original(session, key, allocate)

    monkeypatch.setattr(module, "benchmark_resource", checked_resource)
    caps = dense_mla.Caps(device=device, mode="verify", kv_dtype=cache.dtype,
        q_dtype=query.dtype, num_q_heads=heads, page_size=page, max_total_q=tokens,
        max_batch=batch, max_cache_tokens=page, max_page_table_width=1,
        num_cache_pages=batch, use_cuda_graph=True)
    with RecordedPreparationSession(device=device, checkpoint="dense-mla-ragged",
            workload_schema="dense-mla-ragged", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedDenseMLA(query, cache, output, table, lengths, starts,
            caps=caps, scale=576 ** -.5, session=session, name="attention", workspace=arena)
        if functional:
            attention.functional(session=session)
        session.tune()
        if functional:
            arena.release_preparation()
            operation = attention
            attention = compile_forward(lambda: output.copy_(operation(query, table, lengths, starts)))
        assert observed
        cache.fill_(.125)
        attention(query, output)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            attention(query, output)
        session.freeze()
        for reverse in (False, True):
            counts = starts.diff().flip(0) if reverse else starts.diff()
            starts[1:].copy_(counts.cumsum(0))
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(output, torch.full_like(output, .125), atol=.002, rtol=.02)
        graph.reset()


def test_functional_broadcast_page_table_compilation_and_replay(monkeypatch):
    from ignition import runtime_patches
    from ignition.components.paged_batch import Metadata

    runtime_patches.apply()
    device = torch.device("cuda", 0)
    tokens, heads, columns, page = 7, 96, 8192, 128
    query = torch.zeros((tokens, heads, 576), dtype=torch.bfloat16, device=device)
    cache = torch.full((1, page, 576), .125, dtype=torch.bfloat16, device=device)
    output = torch.empty((tokens, heads, 512), dtype=torch.bfloat16, device=device)
    table = torch.zeros((tokens, columns), dtype=torch.int32, device=device)
    pages = torch.zeros((1, columns), dtype=torch.int32, device=device)
    lengths = torch.full((tokens,), 64, dtype=torch.int32, device=device)
    starts = torch.arange(tokens + 1, dtype=torch.int32, device=device)
    request_starts = torch.tensor([64], dtype=torch.int64, device=device)
    request_lengths = torch.tensor([tokens], dtype=torch.int32, device=device)
    rows = torch.zeros(1, dtype=torch.int64, device=device)
    token_ids = torch.zeros((1, tokens), dtype=torch.int64, device=device)
    live_rows = torch.ones(1, dtype=torch.int32, device=device)
    caps = dense_mla.Caps(device=device, mode="decode", kv_dtype=cache.dtype,
        q_dtype=query.dtype, num_q_heads=heads, page_size=page, max_total_q=tokens,
        max_batch=tokens, max_cache_tokens=columns * page, max_page_table_width=columns,
        num_cache_pages=1, use_cuda_graph=True, partial_dtype=torch.float32)
    with RecordedPreparationSession(device=device, checkpoint="dense-mla-broadcast-pages",
            workload_schema="decode7-context8192-pages", numerical_policy="bf16",
            autotune_effort="off", compile_workers=2) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedDenseMLA(query, cache, output, table, lengths, starts,
            caps=caps, scale=192 ** -.5, session=session, name="attention", workspace=arena)
        attention.functional(session=session)
        session.tune(show_progress=False)
        arena.release_preparation()

        def forward():
            metadata = Metadata.pack(request_starts, request_lengths, rows, pages, token_ids,
                live_rows, token_capacity=tokens, sample_count=tokens, max_query_tokens=tokens)
            result = attention(query, page_table=metadata.token_page_table, lengths=lengths, starts=starts)
            output.copy_(result)

        metadata = Metadata.pack(request_starts, request_lengths, rows, pages, token_ids,
            live_rows, token_capacity=tokens, sample_count=tokens, max_query_tokens=tokens)
        assert metadata.token_page_table.stride() == (0, 1)
        original_bind, observed = dense_mla.bind, []

        def checked_bind(plan, **kwargs):
            stride = kwargs["page_table"].stride()
            observed.append(stride)
            assert stride == (columns, 1)
            return original_bind(plan, **kwargs)

        monkeypatch.setattr(dense_mla, "bind", checked_bind)
        forward()
        compiled = compile_forward(forward)
        observed.clear()
        compiled()
        assert observed == [(columns, 1)]
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        assert observed == [(columns, 1), (columns, 1)]
        session.freeze()
        output.fill_(torch.nan)
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        assert torch.isfinite(output).all()
        graph.reset()
