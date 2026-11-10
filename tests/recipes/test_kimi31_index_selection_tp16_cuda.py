# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Production TP16 FP8 selection, shared prefill slices, and selected MLA replay."""

from datetime import timedelta
import os
from types import SimpleNamespace

import pytest
import torch

from tests.recipes.test_kimi31_dsa_tp16_cuda import MAX_VISIBLE, RECORD_BYTES, TOPK, WORLD, _records, _scores

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires sixteen explicitly assigned CUDA devices")


def _check_selection(selected, positions, concentrated):
    import torch.distributed as dist

    observed = selected.cpu().tolist()
    peers = [None] * WORLD
    dist.all_gather_object(peers, observed)
    chosen_rows = []
    for row, position in enumerate(positions.cpu().tolist()):
        visible = position + 1
        count = min(visible, TOPK)
        common = None
        for rank, values in enumerate(peers):
            valid = [token for token in values[row] if token >= 0]
            chosen = set(valid)
            assert len(valid) == len(chosen) == count, (rank, row, position)
            assert all(0 <= token < visible for token in chosen), (rank, row, position)
            assert all(token == -1 for token in values[row] if token < 0), (rank, row)
            if common is None:
                common = chosen
            assert chosen == common, (rank, row, position, concentrated)
        chosen_rows.append(sorted(common))
        if count:
            scores = _scores(torch.arange(visible), concentrated)
            cutoff = scores.topk(count).values[-1]
            assert (scores[chosen_rows[-1]] >= cutoff).all(), (row, position)
            assert set(torch.where(scores > cutoff)[0].tolist()) <= common, (row, position)
    return chosen_rows


def _rank(rank, rendezvous):
    import torch.distributed as dist
    from b12x.attention import dense_mla

    from ignition.components.collectives import PreparedAllGather, PreparedLseReduceScatter
    from ignition.components.dense_mla import PreparedDenseMLA
    from ignition.preparation.collectives import CollectivePreparation, create_head_exchange
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi31.dcp import ContextShards, selection_slots
    from ignition.recipes.kimi31.indexer import PreparedIndexSelection

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=10))
    candidate_runtime = head_runtime = None
    graphs = []
    try:
        with PackageStorage(rank=rank, world=WORLD, device=rank) as storage:
            logical_pages = MAX_VISIBLE // (128 * WORLD)
            physical_pages = logical_pages + 3
            page_map = ((torch.arange(logical_pages, device=device) * 2 + 3) % physical_pages).int()
            cache = torch.zeros((physical_pages, 128, RECORD_BYTES), device=device, dtype=torch.float8_e4m3fn)
            index_cache = torch.zeros((physical_pages * 2, 64 * 132), device=device, dtype=torch.uint8)
            local_positions = torch.arange(MAX_VISIBLE // WORLD, device=device, dtype=torch.int64)
            tokens = local_positions * WORLD + rank
            slots = page_map[local_positions // 128].long() * 128 + local_positions % 128
            cache.view(-1, RECORD_BYTES)[slots] = _records(tokens).to(cache.dtype)
            saved_cache = cache.view(torch.uint8).clone()
            index_keys = torch.zeros((len(tokens), 128), device=device, dtype=torch.bfloat16)
            # The fixed maximum makes every key scale exactly one.
            index_keys[:, 127] = 448
            index_queries = torch.zeros((257, 32, 128), device=device, dtype=torch.bfloat16)
            index_queries[:, :, 0] = 1
            weights = torch.zeros((257, 32), device=device, dtype=torch.bfloat16)
            weights[:, 0] = 1
            prefill_metadata = SimpleNamespace(
                positions=torch.full((257,), -1, device=device, dtype=torch.int64),
                token_page_table=page_map[None].repeat(257, 1))
            decode_metadata = SimpleNamespace(
                positions=torch.full((4,), -1, device=device, dtype=torch.int64),
                token_page_table=page_map[None].repeat(4, 1))
            shards = ContextShards.on(rank, WORLD, device)
            coordinator = CollectivePreparation(dist.group.WORLD)
            candidate_runtime = create_head_exchange(dist.group.WORLD, device, max_rows=128,
                heads=WORLD, head_dim=TOPK * 4, ipc=storage.peer_storage)
            local_heads, total_heads = 6, 96
            head_runtime = create_head_exchange(dist.group.WORLD, device, max_rows=4,
                heads=total_heads, head_dim=512, query_head_dim=RECORD_BYTES, ipc=storage.peer_storage)
            context = SimpleNamespace(candidates=candidate_runtime, coordinator=coordinator)
            with RecordedPreparationSession(device=device, checkpoint="kimi31-production-index-tp16",
                    workload_schema="fp8-selection-and-selected-mla", numerical_policy="query-pow2-key-fp32",
                    autotune_effort="off", coordinator=coordinator) as session:
                arena = ScratchArena(device)
                session.register_scratch(arena)
                buffers = {}
                variants = []
                for mode, start, end, metadata in (
                        ("prefill", 0, 128, prefill_metadata),
                        ("prefill", 256, 257, prefill_metadata),
                        ("decode", 0, 4, decode_metadata),
                        ("decode", 0, 2, decode_metadata)):
                    rows = end - start
                    name = f"{mode}{rows}"
                    selector = PreparedIndexSelection(index_queries.view(257, -1), weights, index_cache,
                        metadata, shards, context, buffers, start=start, end=end, mode=mode,
                        session=session, name=name, workspace=arena, allocator=None, latent_cache=cache)
                    variant = dict(name=name, selector=selector, start=start, end=end, metadata=metadata)
                    if mode == "decode":
                        owners = torch.empty_like(selector.selected)
                        selected_slots = torch.empty_like(selector.selected)
                        counts = torch.empty(rows, device=device, dtype=torch.int32)
                        heads = torch.arange(rank * local_heads, (rank + 1) * local_heads, device=device)
                        channels = torch.arange(RECORD_BYTES, device=device)
                        query = (((heads[:, None] * 11 + channels[None] * 7).remainder(127) - 63).float() / 512)
                        query = query.bfloat16().to(cache.dtype)[None].repeat(rows, 1, 1)
                        gathered = torch.zeros((rows, total_heads, RECORD_BYTES), device=device, dtype=query.dtype)
                        partial = torch.zeros((rows, total_heads, 512), device=device, dtype=torch.bfloat16)
                        lse = torch.full((rows, total_heads), -torch.inf, device=device)
                        output = torch.zeros((rows, local_heads, 512), device=device, dtype=torch.bfloat16)
                        starts = torch.arange(rows + 1, device=device, dtype=torch.int32)
                        gather = PreparedAllGather(query.view(rows, -1), gathered.view(rows, -1),
                            runtime=head_runtime, session=session, coordinator=coordinator, name=name + ".heads")
                        caps = dense_mla.Caps(device=device, mode="decode", q_dtype=query.dtype, kv_dtype=cache.dtype,
                            num_q_heads=total_heads, page_size=1, max_total_q=rows, max_batch=rows,
                            max_cache_tokens=TOPK, max_page_table_width=TOPK,
                            num_cache_pages=physical_pages * 128, partial_dtype=torch.float32, use_cuda_graph=True)
                        attention = PreparedDenseMLA(gathered, cache.view(-1, 1, RECORD_BYTES), partial,
                            selected_slots, counts, starts, caps=caps, scale=192 ** -.5, session=session,
                            name=name + ".attention", workspace=arena, lse=lse)
                        combine = PreparedLseReduceScatter(partial, lse, output, runtime=head_runtime,
                            session=session, coordinator=coordinator, name=name + ".combine")
                        variant.update(owners=owners, slots=selected_slots, counts=counts, query=query,
                            partial=partial, lse=lse, output=output, gathered=gathered, gather=gather,
                            attention=attention, combine=combine)
                    variants.append(variant)
                large, tail = (variant["selector"] for variant in variants[:2])
                assert large.selected.data_ptr() == tail.selected.data_ptr()
                assert large.merge.packed.data_ptr() == tail.merge.packed.data_ptr()
                assert large.merge.gathered.data_ptr() == tail.merge.gathered.data_ptr()
                variants[0]["name"] = "prefill257"
                variants[0]["end"] = 257
                variants[0]["bounds"] = ((0, 128), (128, 256), (256, 257))
                variants[0]["selections"] = {128: large, 1: tail}
                for variant in variants:
                    start, end = variant["start"], variant["end"]
                    variant.setdefault("bounds", ((start, end),))
                    variant.setdefault("selections", {end - start: variant["selector"]})
                    variant["observed"] = {field: torch.empty((end - start, *value.shape[1:]),
                        dtype=value.dtype, device=device) for field, value in (
                            ("selected", variant["selector"].selected),
                            ("quantized", variant["selector"].indexer.quantized),
                            ("scales", variant["selector"].indexer.scales),
                            ("scaled_weights", variant["selector"].indexer.scaled_weights),
                            ("local", variant["selector"].local), ("scores", variant["selector"].scores))}
                session.tune(show_progress=False)

                def forward(variant):
                    metadata = variant["metadata"]
                    for start, end in variant["bounds"]:
                        selector = variant["selections"][end - start]
                        positions, tables = metadata.positions[start:end], metadata.token_page_table[start:end]
                        selected = selector(index_queries[start:end], weights[start:end], positions, tables)
                        offset = start - variant["start"]
                        for field in ("selected", "local", "scores"):
                            variant["observed"][field][offset:offset + end - start].copy_(getattr(selector, field))
                        for field in ("quantized", "scales", "scaled_weights"):
                            variant["observed"][field][offset:offset + end - start].copy_(
                                getattr(selector.indexer, field))
                        if selector.mode == "decode":
                            selection_slots(selected, positions, tables, variant["owners"],
                                variant["slots"], variant["counts"], shards, physical_pages, local=True)
                            variant["gather"](variant["query"].view(selected.shape[0], -1))
                            variant["attention"](variant["gathered"], variant["partial"])
                            variant["combine"](variant["partial"], variant["lse"])

                index_keys[:, 0] = _scores(tokens, False)
                large.indexer.write_keys(index_keys, slots)
                for variant in variants:
                    forward(variant)
                    torch.cuda.synchronize(device)
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    graph = torch.cuda.CUDAGraph()
                    with session.capture(), torch.cuda.graph(graph, stream=stream):
                        forward(variant)
                    graphs.append(graph)
                    variant["graph"] = graph
                session.freeze()
                for concentrated in (False, True):
                    index_keys[:, 0] = _scores(tokens, concentrated)
                    large.indexer.write_keys(index_keys, slots)
                    saved_index = index_cache.clone()
                    physical_keys = index_cache[:, :8192].view(torch.float8_e4m3fn).reshape(-1, 128)
                    physical_scales = index_cache[:, 8192:].view(torch.float32).flatten()
                    torch.testing.assert_close(physical_keys[slots].view(torch.uint8),
                                              index_keys.to(cache.dtype).view(torch.uint8), rtol=0, atol=0)
                    torch.testing.assert_close(physical_scales[slots], torch.ones_like(physical_scales[slots]),
                                              rtol=0, atol=0)
                    for long_context in (False, True):
                        prefill_metadata.positions.copy_(torch.arange(257, device=device)
                            + (MAX_VISIBLE - 257 if long_context else 0))
                        decode_metadata.positions.copy_(torch.tensor(
                            [-1, MAX_VISIBLE - 1, 16, 9942] if long_context else [-1, 0, 2048, 9942],
                            device=device))
                        tail_position = MAX_VISIBLE - 1 if long_context else 256
                        for index, tail_live in ((2, True), (0, True), (3, True), (1, True),
                                                 (0, False), (1, False)):
                            prefill_metadata.positions[-1] = tail_position if tail_live else -1
                            variant = variants[index]
                            selector = variant["selector"]
                            arena.tensor.fill_(255)
                            for operation in variant["selections"].values():
                                operation.selected.fill_(-2)
                            for value in variant["observed"].values():
                                value.fill_(-2)
                            if selector.mode == "decode":
                                variant["output"].fill_(17)
                                variant["partial"].fill_(torch.nan)
                                variant["lse"].fill_(torch.nan)
                            torch.cuda.synchronize(device)
                            dist.barrier()
                            before = torch.cuda.memory_stats(device)
                            variant["graph"].replay()
                            torch.cuda.synchronize(device)
                            after = torch.cuda.memory_stats(device)
                            for field in ("allocation.all.allocated", "allocation.all.freed",
                                          "segment.all.allocated", "segment.all.freed"):
                                assert before[field] == after[field], (variant["name"], field)
                            observed = variant["observed"]
                            chosen_rows = _check_selection(observed["selected"],
                                variant["metadata"].positions[variant["start"]:variant["end"]], concentrated)
                            expected_query = torch.zeros_like(observed["quantized"], dtype=torch.float32)
                            expected_query[:, :, 0] = 256
                            torch.testing.assert_close(observed["quantized"].float(), expected_query, rtol=0, atol=0)
                            torch.testing.assert_close(observed["scales"],
                                torch.full_like(observed["scales"], 1 / 256), rtol=0, atol=0)
                            expected_weights = torch.zeros_like(observed["scaled_weights"])
                            expected_weights[:, 0] = (1 / 256) * 128 ** -.5
                            torch.testing.assert_close(observed["scaled_weights"], expected_weights, rtol=0, atol=0)
                            local = observed["local"].long()
                            live = local >= 0
                            expected_scores = _scores(local * WORLD + rank, concentrated) * 128 ** -.5
                            torch.testing.assert_close(observed["scores"][live], expected_scores[live], rtol=0, atol=0)
                            if selector.mode == "decode":
                                for row, chosen in enumerate(chosen_rows):
                                    assert variant["counts"][row].item() == sum(token % WORLD == rank for token in chosen)
                                    if not chosen:
                                        assert not variant["output"][row].count_nonzero()
                                        continue
                                    records = _records(torch.tensor(chosen, device=device)).to(cache.dtype).float()
                                    logits = variant["query"][row].float() @ records.T * 192 ** -.5
                                    expected = logits.softmax(-1) @ records[:, :512]
                                    torch.testing.assert_close(variant["output"][row].float(), expected,
                                                              rtol=.03, atol=.002)
                                assert torch.isfinite(variant["output"]).all()
                            torch.testing.assert_close(cache.view(torch.uint8), saved_cache, rtol=0, atol=0)
                            torch.testing.assert_close(index_cache, saved_index, rtol=0, atol=0)
                            dist.barrier()
                for graph in graphs:
                    graph.reset()
                graphs.clear()
            head_runtime.close()
            candidate_runtime.close()
            head_runtime = candidate_runtime = None
    finally:
        for graph in graphs:
            graph.reset()
        if head_runtime is not None:
            head_runtime.close()
        if candidate_runtime is not None:
            candidate_runtime.close()
        dist.destroy_process_group()


def test_production_fp8_selection_and_shared_prefill_slices_on_tp16(tmp_path):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == WORLD, "assign exactly sixteen CUDA devices"
    mp.spawn(_rank, args=((tmp_path / "production-index-selection").as_uri(),), nprocs=WORLD, join=True)
