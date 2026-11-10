# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 prefill attention over context that decode context parallel ranks share, one window at a time."""

import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires CUDA")


@pytest.mark.parametrize("start,tokens", [(0, 100), (200, 128), (1000, 128), (1900, 100)])
def test_windowed_context_matches_attention_over_every_position(start, tokens):
    import torch
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.attention import PreparedContextPrefill
    from ignition.recipes.kimi3.config import PAGE_TOKENS
    from ignition.recipes.kimi3.dcp import RECORD_BYTES, ContextParallel, ContextShards

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    fp8 = torch.float8_e4m3fn
    world, heads, width, window, capacity = 2, 6, 8, 256, 128
    torch.manual_seed(start + tokens)
    # Rank 0 of two: position t lives on rank t % 2 at local index t // 2, both ranks addressing pages 0-7.
    caches = [(torch.randn((width, PAGE_TOKENS, RECORD_BYTES), device=device) * .5).to(fp8) for _ in range(world)]
    peers = torch.tensor([cache.data_ptr() for cache in caches], dtype=torch.int64, device=device)
    slab = torch.zeros((window + capacity, RECORD_BYTES), dtype=fp8, device=device)
    context = ContextParallel(ContextShards.on(0, world, device), heads=None, coordinator=None, peers=None,
                              slab=slab, window=window)
    metadata = Metadata.allocate(capacity, 1, width, device)
    metadata.page_table.copy_(torch.arange(width, dtype=torch.int32, device=device)[None])
    absorbed = (torch.randn((capacity, heads, RECORD_BYTES), device=device) * .5).bfloat16()
    latent = (torch.randn((capacity, RECORD_BYTES), device=device) * .5).bfloat16()
    attended = torch.zeros((capacity, heads, 512), dtype=torch.bfloat16, device=device)

    common = dict(device=device, kv_dtype=fp8, page_size=PAGE_TOKENS, use_cuda_graph=True,
                  partial_dtype=torch.float32)
    with RecordedPreparationSession(device=device, checkpoint="kimi3-context-prefill-test",
                                    workload_schema="kimi3-mla", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        op = PreparedContextPrefill(absorbed, attended, metadata, context, peers, session=session, name="mla",
                                    workspace=arena, scale=192 ** -.5, common=common)
        assert op.windows == width * PAGE_TOKENS * world // window
        session.tune(show_progress=False)
        metadata.starts.fill_(start)
        metadata.lengths.fill_(tokens)
        metadata.query_start_loc[1] = tokens
        metadata.positions.fill_(-1)
        metadata.positions[:tokens].copy_(torch.arange(start, start + tokens, device=device))
        arena.release_preparation()
        observed = torch.empty_like(attended)

        def forward():
            observed.copy_(op(absorbed, latent, metadata))

        compiled = compile_forward(forward)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        try:
            observed.fill_(float("nan"))
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_stats(device)
            graph.replay()
            torch.cuda.synchronize(device)
            after = torch.cuda.memory_stats(device)
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            output = observed.clone()
        finally:
            graph.reset()

    # Every position's record: earlier ones from their owners' caches, the chunk's from its E4M3 latents.
    positions = torch.arange(start, device=device)
    local = positions // world
    records = torch.empty((start + tokens, RECORD_BYTES), dtype=torch.float32, device=device)
    for owner in range(world):
        mask = positions % world == owner
        records[:start][mask] = caches[owner][local[mask] // PAGE_TOKENS, local[mask] % PAGE_TOKENS].float()
    records[start:] = latent[:tokens].float().clamp(-448., 448.).to(fp8).float()
    query = absorbed[:tokens].float().clamp(-448., 448.).to(fp8).float()
    scores = torch.einsum("thk,sk->hts", query, records) * 192 ** -.5
    visible = torch.arange(start + tokens, device=device)[None] <= torch.arange(start, start + tokens,
                                                                                device=device)[:, None]
    scores.masked_fill_(~visible[None], float("-inf"))
    expected = torch.einsum("hts,sk->thk", scores.softmax(-1), records[:, :512])
    actual = output[:tokens].float()
    error = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    assert torch.isfinite(actual).all() and error.item() < .02
