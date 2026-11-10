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
        attention()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            attention()
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
