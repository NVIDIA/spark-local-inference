# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP16 sparse candidate consensus and distributed attention against a PyTorch oracle."""

from datetime import timedelta
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires sixteen explicitly assigned CUDA devices")

WORLD = 16
TOPK = 2048
RECORD_BYTES = 576
MAX_VISIBLE = 65536


def _scores(tokens, concentrated):
    if concentrated:
        return (tokens.remainder(WORLD) == 7).float() * 2
    return torch.where(tokens < 1536, 2., torch.where(tokens < 4096, 1., 0.))


def _records(tokens):
    channels = torch.arange(RECORD_BYTES, device=tokens.device)
    return (((tokens[:, None] * 17 + channels[None] * 13).remainder(257) - 128).float() / 512)


def _rank(rank, rendezvous):
    import torch.distributed as dist
    from b12x.attention import dense_mla
    from ignition.components.collectives import PreparedAllGather, PreparedBroadcast, PreparedLseReduceScatter
    from ignition.components.dense_mla import PreparedDenseMLA
    from ignition.components.row_topk import PreparedRowTopK
    from ignition.preparation.collectives import CollectivePreparation, create_head_exchange
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi31.dcp import ContextShards, pack_candidates, selection_slots

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=5))
    candidate_runtime = head_runtime = graph = None
    try:
        with PackageStorage(rank=rank, world=WORLD, device=rank) as storage:
            rows, local_heads, total_heads = 2, 6, 96
            logical_pages, physical_pages = MAX_VISIBLE // (128 * WORLD), MAX_VISIBLE // (128 * WORLD) + 3
            cache = torch.zeros((physical_pages, 128, RECORD_BYTES), device=device, dtype=torch.float8_e4m3fn)
            page_map = ((torch.arange(logical_pages, device=device) * 2 + 3) % physical_pages).to(torch.int32)
            tables = page_map[None].repeat(rows, 1)
            positions = torch.full((rows,), -1, device=device, dtype=torch.int64)
            local_ids = torch.full((rows, TOPK), -1, device=device, dtype=torch.int32)
            local_scores = torch.full((rows, TOPK), -torch.inf, device=device)
            packed = torch.zeros((rows, TOPK, 2), device=device)
            exchanged = torch.zeros((rows, WORLD * TOPK, 2), device=device)
            scores = torch.empty((rows, WORLD * TOPK), device=device)
            ids = torch.empty((rows, WORLD * TOPK), device=device, dtype=torch.int32)
            lengths = torch.zeros(rows, device=device, dtype=torch.int32)
            selected_scores = torch.empty((rows, TOPK), device=device)
            selected = torch.empty((rows, TOPK), device=device, dtype=torch.int32)
            owners, slots = torch.empty_like(selected), torch.empty_like(selected)
            counts = torch.zeros(rows, device=device, dtype=torch.int32)
            head_ids = torch.arange(rank * local_heads, (rank + 1) * local_heads, device=device)
            channels = torch.arange(RECORD_BYTES, device=device)
            query = (((head_ids[:, None] * 11 + channels[None] * 7).remainder(127) - 63).float() / 512)
            query = query.bfloat16().to(torch.float8_e4m3fn)[None].repeat(rows, 1, 1)
            gathered_query = torch.zeros((rows, total_heads, RECORD_BYTES), device=device, dtype=query.dtype)
            partial = torch.zeros((rows, total_heads, 512), device=device, dtype=torch.bfloat16)
            lse = torch.full((rows, total_heads), -torch.inf, device=device)
            output = torch.zeros((rows, local_heads, 512), device=device, dtype=torch.bfloat16)
            starts = torch.arange(rows + 1, device=device, dtype=torch.int32)
            shards = ContextShards.on(rank, WORLD, device)
            coordinator = CollectivePreparation(dist.group.WORLD)
            candidate_runtime = create_head_exchange(dist.group.WORLD, device, max_rows=rows,
                heads=WORLD, head_dim=TOPK * 4, ipc=storage.peer_storage)
            head_runtime = create_head_exchange(dist.group.WORLD, device, max_rows=rows,
                heads=total_heads, head_dim=512, query_head_dim=RECORD_BYTES, ipc=storage.peer_storage)
            with RecordedPreparationSession(device=device, checkpoint="kimi31-tp16-selected-attention",
                    workload_schema="tp16-global-top2048-and-lse", numerical_policy="default",
                    autotune_effort="off", coordinator=coordinator) as session:
                arena = ScratchArena(device)
                session.register_scratch(arena)
                exchange = PreparedAllGather(packed.view(torch.bfloat16).view(rows, -1),
                    exchanged.view(torch.bfloat16).view(rows, -1), runtime=candidate_runtime,
                    session=session, coordinator=coordinator, name="candidates")
                fold = PreparedRowTopK(scores, lengths, ids, selected_scores, selected,
                    session=session, name="global_top2048")
                consensus = PreparedBroadcast((selected,), runtime=candidate_runtime,
                    session=session, coordinator=coordinator, name="selected_consensus")
                gather_heads = PreparedAllGather(query.view(rows, -1), gathered_query.view(rows, -1),
                    runtime=head_runtime, session=session, coordinator=coordinator, name="query_heads")
                caps = dense_mla.Caps(device=device, mode="decode", q_dtype=query.dtype, kv_dtype=cache.dtype,
                    num_q_heads=total_heads, page_size=1, max_total_q=rows, max_batch=rows,
                    max_cache_tokens=TOPK, max_page_table_width=TOPK,
                    num_cache_pages=physical_pages * 128, partial_dtype=torch.float32, use_cuda_graph=True)
                attention = PreparedDenseMLA(gathered_query, cache.view(-1, 1, RECORD_BYTES), partial,
                    slots, counts, starts, caps=caps, scale=192 ** -.5, session=session,
                    name="local_selected_attention", workspace=arena, lse=lse)
                combine = PreparedLseReduceScatter(partial, lse, output, runtime=head_runtime,
                    session=session, coordinator=coordinator, name="combine_selected_attention")
                session.tune(show_progress=False)
                local_positions = torch.arange(MAX_VISIBLE // WORLD, device=device, dtype=torch.int64)
                physical_slots = page_map[local_positions // 128].long() * 128 + local_positions % 128
                cache.view(-1, RECORD_BYTES)[physical_slots] = _records(local_positions * WORLD + rank).to(cache.dtype)
                saved_cache = cache.view(torch.uint8).clone()

                def forward():
                    pack_candidates(local_ids, local_scores, packed, shards)
                    exchange(packed.view(torch.bfloat16).view(rows, -1))
                    scores.copy_(exchanged[:, :, 0])
                    ids.copy_(exchanged.view(torch.int32)[:, :, 1])
                    lengths.copy_(torch.where(positions >= 0, WORLD * TOPK, 0).to(torch.int32))
                    fold()
                    consensus()
                    selection_slots(selected, positions, tables, owners, slots, counts, shards,
                                    physical_pages, local=True)
                    gather_heads(query.view(rows, -1))
                    attention(gathered_query, partial)
                    combine(partial, lse)

                forward()
                torch.cuda.synchronize(device)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph, stream=stream):
                    forward()
                session.freeze()
                for visible, concentrated in ((17, False), (9943, False), (MAX_VISIBLE, False), (MAX_VISIBLE, True)):
                    global_ids = torch.arange(rank, visible, WORLD)
                    candidate_scores = _scores(global_ids, concentrated)
                    candidate_count = min(TOPK, len(global_ids))
                    values, indices = candidate_scores.topk(candidate_count)
                    local_ids.fill_(-1)
                    local_scores.fill_(-torch.inf)
                    local_ids[1, :candidate_count].copy_(indices.to(device, dtype=torch.int32))
                    local_scores[1, :candidate_count].copy_(values.to(device))
                    positions[1] = visible - 1
                    if visible == MAX_VISIBLE:
                        assert len(global_ids) > TOPK
                    for _ in range(3):
                        arena.tensor.fill_(255)
                        output.fill_(17)
                        partial.fill_(torch.nan)
                        lse.fill_(torch.nan)
                        selected.fill_(-2)
                        torch.cuda.synchronize(device)
                        dist.barrier()
                        before = torch.cuda.memory_stats(device)
                        graph.replay()
                        torch.cuda.synchronize(device)
                        after = torch.cuda.memory_stats(device)
                        for field in ("allocation.all.allocated", "allocation.all.freed",
                                      "segment.all.allocated", "segment.all.freed"):
                            assert before[field] == after[field], field
                        observed = selected.cpu().tolist()
                        all_selected = [None] * WORLD
                        dist.all_gather_object(all_selected, observed)
                        expected_count = min(visible, TOPK)
                        sets = []
                        for peer, selections in enumerate(all_selected):
                            assert all(token == -1 for token in selections[0]), peer
                            valid = [token for token in selections[1] if token >= 0]
                            assert len(valid) == len(set(valid)) == expected_count, (peer, visible)
                            assert all(token < visible for token in valid), (peer, visible)
                            sets.append(set(valid))
                        assert len(set.union(*sets)) == expected_count, (visible, concentrated)
                        assert all(selection == sets[0] for selection in sets), (visible, concentrated)
                        chosen = torch.tensor(sorted(sets[0]), dtype=torch.int64)
                        all_scores = _scores(torch.arange(visible), concentrated)
                        cutoff = all_scores.topk(expected_count).values[-1]
                        assert (all_scores[chosen] >= cutoff).all()
                        required = set(torch.where(all_scores > cutoff)[0].tolist())
                        assert required <= sets[0]
                        assert counts[0].item() == 0
                        assert counts[1].item() == sum(token % WORLD == rank for token in sets[0])
                        if concentrated and rank != 7:
                            assert counts[1].item() == 0
                        records = _records(chosen.to(device)).to(cache.dtype).float()
                        logits = query[1].float() @ records.T * 192 ** -.5
                        expected = logits.softmax(-1) @ records[:, :512]
                        torch.testing.assert_close(output[1].float(), expected, rtol=.03, atol=.002)
                        assert not output[0].count_nonzero()
                        assert torch.isfinite(output).all()
                        torch.testing.assert_close(cache.view(torch.uint8), saved_cache, rtol=0, atol=0)
                        dist.barrier()
                graph.reset()
                graph = None
            head_runtime.close()
            candidate_runtime.close()
            head_runtime = candidate_runtime = None
    finally:
        if graph is not None:
            graph.reset()
        if head_runtime is not None:
            head_runtime.close()
        if candidate_runtime is not None:
            candidate_runtime.close()
        dist.destroy_process_group()


def test_global_top2048_consensus_and_lse_attention_on_tp16(tmp_path):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == WORLD, "assign exactly sixteen CUDA devices"
    mp.spawn(_rank, args=((tmp_path / "selected-attention").as_uri(),), nprocs=WORLD, join=True)
