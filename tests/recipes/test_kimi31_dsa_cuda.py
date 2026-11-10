# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sparse Kimi record addressing, exact candidates, and selected-record replay."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@pytest.mark.parametrize("local", [False, True])
def test_selected_records_compact_causal_interleaved_owners(local):
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.kimi31.dcp import TOPK, ContextShards, index_pages, pack_candidates, selection_slots

    device = torch.device("cuda", 0)
    shards = ContextShards.on(7, 16, device)
    selected = torch.full((3, TOPK), -1, dtype=torch.int32, device=device)
    tokens = [0, 7, 15, 16, 63, 64, 127, 128, 2047, 2048, 2055, 4095, 4096, 6143]
    selected[:, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32, device=device)
    positions = torch.tensor([6143, 2054, -1], dtype=torch.int64, device=device)
    tables = torch.tensor([[3, 0, 2]] * 3, dtype=torch.int32, device=device)
    owners, slots = torch.empty_like(selected), torch.empty_like(selected)
    counts = torch.empty(3, dtype=torch.int32, device=device)
    subpages = torch.empty((3, 6), dtype=torch.int32, device=device)
    scores = torch.randn_like(selected, dtype=torch.float32)
    packed = torch.empty((3, TOPK, 2), dtype=torch.float32, device=device)

    def forward():
        index_pages(tables, subpages)
        pack_candidates(selected, scores, packed, shards)
        selection_slots(selected, positions, tables, owners, slots, counts, shards, 4, local=local)

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    graph.replay()
    torch.cuda.synchronize()
    for row, limit in enumerate(positions.tolist()):
        live = [token for token in tokens if token <= limit and (not local or token % 16 == 7)]
        assert counts[row].item() == len(live)
        assert owners[row, :len(live)].tolist() == [token % 16 for token in live]
        assert slots[row, :len(live)].tolist() == [tables[row, token // 2048].item() * 128
                                                  + token // 16 % 128 for token in live]
        assert (slots[row, len(live):] == -1).all()
    assert subpages[0].tolist() == [6, 7, 0, 1, 4, 5]
    expected_ids = torch.where(selected >= 0, selected * 16 + 7, -1)
    torch.testing.assert_close(packed.view(torch.bfloat16).view(torch.float32).view(torch.int32)[:, :, 1],
                              expected_ids, rtol=0, atol=0)
    graph.reset()


def _identity_fp8_reference(value, *, power_of_two):
    value = value.float()
    maximum = value.abs().amax(-1).clamp_min(1e-4)
    if power_of_two:
        scale = torch.exp2(torch.ceil(torch.log2(maximum / 448)))
        quantized = value / scale[..., None]
    else:
        scale = (maximum.double() / 448).float()
        reciprocal = (1 / scale.double()).float()
        quantized = value * reciprocal[..., None]
    return quantized.clamp(-448, 448).to(torch.float8_e4m3fn), scale


@torch.no_grad()
def test_fp8_top2048_scores_and_inactive_rows():
    from ignition.components.fp8_indexer import PreparedFP8Indexer
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    torch.manual_seed(7231)
    rows, heads, records = 4, 32, 4096
    query = torch.randn((rows, heads, 128), device=device, dtype=torch.bfloat16)
    query[0].zero_()
    query[1].fill_(1e-6)
    query[2, 0].zero_()
    weights = torch.randn((rows, heads), device=device, dtype=torch.float32)
    keys = torch.randn((records, 128), device=device, dtype=torch.bfloat16)
    physical_slots = (torch.arange(records, dtype=torch.int64, device=device) * 73 + 17) % records
    keys[0].zero_()
    keys[1].fill_(1e-6)
    slots = physical_slots.clone()
    cache = torch.zeros((records // 64, 64 * 132), device=device, dtype=torch.uint8)
    pages = torch.arange(records // 64 - 1, -1, -1, device=device, dtype=torch.int32)[None].expand(rows, -1)
    lengths = torch.tensor([0, 1, 2051, records], dtype=torch.int32, device=device)
    output = torch.empty((rows, 2048), dtype=torch.int32, device=device)
    scores = torch.empty((rows, 2048), dtype=torch.float32, device=device)
    with RecordedPreparationSession(device=device, checkpoint="kimi31-top2048",
            workload_schema="kimi31-index", numerical_policy="query-pow2-key-fp32",
            autotune_effort="off") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        indexer = PreparedFP8Indexer(query, weights, cache, pages, lengths, None, output,
            mode="decode", session=session, name="index", workspace=arena,
            buffers=GraphBufferPool(device).cursor(), topk=2048, scores=scores,
            head_weight_scale=1.0, hadamard=False, key_power_of_two=False)
        session.tune(show_progress=False)

        def forward():
            indexer.write_keys(keys, slots)
            indexer(query, weights)

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for field in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated"):
            assert before[field] == after[field]
        expected_q, query_scale = _identity_fp8_reference(query, power_of_two=True)
        expected_k, key_scale = _identity_fp8_reference(keys, power_of_two=False)
        physical_k = torch.empty_like(expected_k)
        physical_scale = torch.empty_like(key_scale)
        physical_k[physical_slots] = expected_k
        physical_scale[physical_slots] = key_scale
        actual_k = cache[:, :64 * 128].view(torch.float8_e4m3fn).reshape(records, 128)
        actual_scale = cache[:, 64 * 128:].view(torch.float32).flatten()
        torch.testing.assert_close(actual_k.view(torch.uint8), physical_k.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(actual_scale, physical_scale, rtol=0, atol=0)
        torch.testing.assert_close(indexer.quantized.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(indexer.scales, query_scale, rtol=0, atol=0)
        expected_weights = (weights * query_scale) * 128 ** -.5
        torch.testing.assert_close(indexer.scaled_weights, expected_weights, rtol=0, atol=0)
        decoded = physical_k.float() * physical_scale[:, None]
        history = decoded.view(-1, 64, 128).flip(0).flatten(0, 1)
        expected = torch.einsum("rhd,kd->rhk", expected_q.float(), history).relu()
        expected = (expected * expected_weights[:, :, None]).sum(1)
        assert (output[0] == -1).all()
        for row, count in enumerate(lengths.tolist()):
            valid = output[row] >= 0
            assert valid.sum().item() == min(count, 2048)
            if not count:
                continue
            ids = output[row, valid].long()
            assert ids.max().item() < count
            actual_scores = expected[row].gather(0, ids)
            torch.testing.assert_close(scores[row, valid], actual_scores, rtol=3e-5, atol=3e-4)
            cutoff = expected[row, :count].topk(min(count, 2048)).values[-1]
            assert actual_scores.min() >= cutoff - .001
        saved_keys, saved_scales = actual_k.clone(), actual_scale.clone()
        inactive = torch.arange(0, records, 17, device=device)
        slots[inactive] = -1
        keys.neg_()
        graph.replay()
        torch.cuda.synchronize()
        actual_k = cache[:, :64 * 128].view(torch.float8_e4m3fn).reshape(records, 128)
        actual_scale = cache[:, 64 * 128:].view(torch.float32).flatten()
        updated_keys, updated_scales = _identity_fp8_reference(keys, power_of_two=False)
        live = slots >= 0
        physical_k[physical_slots[live]] = updated_keys[live]
        physical_scale[physical_slots[live]] = updated_scales[live]
        torch.testing.assert_close(actual_k.view(torch.uint8), physical_k.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(actual_scale, physical_scale, rtol=0, atol=0)
        untouched = physical_slots[inactive]
        torch.testing.assert_close(actual_k[untouched].view(torch.uint8),
                                  saved_keys[untouched].view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(actual_scale[untouched].view(torch.uint8),
                                  saved_scales[untouched].view(torch.uint8), rtol=0, atol=0)
        graph.reset()


@torch.no_grad()
def test_candidate_fold_and_selected_peer_records_match_dense_reference():
    from b12x.attention import dense_mla
    from ignition.components.dense_mla import PreparedDenseMLA
    from ignition.components.peer_record_gather import PreparedPeerRecordGather
    from ignition.components.row_topk import PreparedRowTopK
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi31.dcp import TOPK, ContextShards, selection_slots

    device = torch.device("cuda", 0)
    torch.manual_seed(1307)
    rows, world, heads = 2, 16, 8
    caches = [(torch.randn((128, 576), device=device) * .1).to(torch.float8_e4m3fn) for _ in range(world)]
    peers = torch.tensor([cache.data_ptr() for cache in caches], dtype=torch.int64, device=device)
    scores = torch.randn((rows, world * TOPK), device=device)
    ids = torch.arange(world * TOPK, dtype=torch.int32, device=device)[None].expand(rows, -1).clone()
    ids[:, 256:] = -1
    scores[:, 256:] = -torch.inf
    lengths = torch.tensor([0, world * TOPK], dtype=torch.int32, device=device)
    selected = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
    selected_scores = torch.empty((rows, TOPK), dtype=torch.float32, device=device)
    owners, slots = torch.empty_like(selected), torch.empty_like(selected)
    counts = torch.empty(rows, dtype=torch.int32, device=device)
    positions = torch.tensor([-1, 255], dtype=torch.int64, device=device)
    tables = torch.zeros((rows, 1), dtype=torch.int32, device=device)
    records = torch.empty((rows, TOPK, 576), dtype=torch.float8_e4m3fn, device=device)
    query = (torch.randn((rows, heads, 576), device=device) * .1).bfloat16()
    output = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=device)
    pages = torch.arange(rows * TOPK, dtype=torch.int32, device=device).view(rows, TOPK)
    starts = torch.arange(rows + 1, dtype=torch.int32, device=device)
    shards = ContextShards.on(0, world, device)
    with RecordedPreparationSession(device=device, checkpoint="kimi31-selected-records",
            workload_schema="kimi31-selected-mla", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        fold = PreparedRowTopK(scores, lengths, ids, selected_scores, selected, session=session, name="fold")
        gather = PreparedPeerRecordGather(peers, owners, slots, counts, records,
            peer_records=128, session=session, name="gather")
        caps = dense_mla.Caps(device=device, mode="decode", q_dtype=query.dtype, kv_dtype=records.dtype,
            num_q_heads=heads, page_size=1, max_total_q=rows, max_batch=rows, max_cache_tokens=TOPK,
            max_page_table_width=TOPK, num_cache_pages=rows * TOPK, use_cuda_graph=True)
        attention = PreparedDenseMLA(query, records.view(-1, 1, 576), output, pages, counts, starts,
            caps=caps, scale=192 ** -.5, session=session, name="attention", workspace=arena)
        session.tune(show_progress=False)

        def forward():
            fold()
            selection_slots(selected, positions, tables, owners, slots, counts, shards, 1, local=False)
            gather()
            attention(query, output)

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        assert before["allocation.all.allocated"] == after["allocation.all.allocated"]
        assert counts.tolist() == [0, 256]
        assert set(selected[1, :256].tolist()) == set(range(256))
        expected_records = torch.stack([caches[token % world][token // world] for token in selected[1, :256].tolist()])
        torch.testing.assert_close(records[1, :256].float(), expected_records.float(), rtol=0, atol=0)
        q = query[1].float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
        logits = q @ expected_records.float().T * 192 ** -.5
        expected = logits.softmax(-1) @ expected_records[:, :512].float()
        torch.testing.assert_close(output[1].float(), expected, rtol=.03, atol=.002)
        assert torch.isfinite(output).all()
        assert not output[0].count_nonzero()
        graph.reset()


@pytest.mark.parametrize("compile_projection", [False, True])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@torch.no_grad()
def test_index_projections_layernorm_and_prefill_slice_boundary(compile_projection, eps):
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi31.dcp import ContextShards
    from ignition.recipes.kimi31.indexer import PreparedDSAIndexer

    device = torch.device("cuda", 0)
    torch.manual_seed(3181)
    rows = 257
    source = (torch.randn((rows, 7168), device=device) * .0005).bfloat16()
    query_latent = (torch.randn((rows, 1536), device=device) * .1).bfloat16()
    weights = {field: (torch.randn(shape, device=device) * .02).bfloat16() for field, shape in (
        ("indexer.wq_b", (4096, 1536)), ("indexer.wk", (128, 7168)),
        ("indexer.weights_proj", (32, 7168)), ("indexer.k_norm.weight", (128,)),
        ("indexer.k_norm.bias", (128,)))}
    weights["indexer.k_norm.weight"].add_(1)
    metadata = Metadata.allocate(rows, 1, 3, device)
    metadata.positions.copy_(torch.arange(rows, dtype=torch.int64, device=device))
    metadata.positions[-1] = -1
    metadata.page_table.copy_(torch.tensor([[2, 0, 1]], dtype=torch.int32, device=device))
    cache = torch.zeros((6, 64, 132), dtype=torch.uint8, device=device)
    shards = ContextShards.on(0, 1, device)
    with RecordedPreparationSession(device=device, checkpoint="kimi31-index-projections",
            workload_schema="kimi31-index", numerical_policy="query-pow2-key-fp32", autotune_effort="off") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        indexer = PreparedDSAIndexer(weights, source, query_latent, metadata, cache, shards,
            context=None, eps=eps, mode="prefill", session=session, name="index", workspace=arena, buffers={},
            latent_cache=torch.zeros((3, 128, 576), dtype=torch.float8_e4m3fn, device=device))
        assert indexer.bounds == ((0, 128), (128, 256), (256, 257))
        assert tuple(indexer.selections) == (128, 1)
        selected = torch.empty((rows, 2048), dtype=torch.int32, device=device)
        quantized = torch.empty((rows, 32, 128), dtype=torch.float8_e4m3fn, device=device)
        scales = torch.empty((rows, 32), dtype=torch.float32, device=device)
        scaled_weights = torch.empty_like(scales)
        session.tune(show_progress=False)
        cache.zero_()

        observed_query = torch.empty((rows, 4096), dtype=source.dtype, device=device)
        observed_key = torch.empty((rows, 128), dtype=source.dtype, device=device)
        observed_normalized_key = torch.empty_like(observed_key)
        observed_weights = torch.empty((rows, 32), dtype=source.dtype, device=device)

        def forward():
            query, key, normalized_key, head_weights = indexer.project_result(source, query_latent, metadata)
            observed_query.copy_(query)
            observed_key.copy_(key)
            observed_normalized_key.copy_(normalized_key)
            observed_weights.copy_(head_weights)
            for start, end in indexer.bounds:
                part = indexer.selections[end - start]
                packed, _, _, quantized_part, scales_part, scaled_part = part.prepare_local_result(
                    query[start:end], head_weights[start:end], metadata.positions[start:end],
                    metadata.token_page_table[start:end])
                selected[start:end].copy_(part.exchange(packed, metadata.positions[start:end]))
                quantized[start:end].copy_(quantized_part)
                scales[start:end].copy_(scales_part)
                scaled_weights[start:end].copy_(scaled_part)

        compiled = compile_forward(forward) if compile_projection else forward
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for iteration in range(3):
            source.mul_(-1)
            query_latent.mul_(-1)
            arena.tensor.fill_(255 if iteration % 2 else 0)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats(device)
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats(device)
            for field in ("allocation.all.allocated", "allocation.all.freed",
                          "segment.all.allocated", "segment.all.freed"):
                assert before[field] == after[field]
        expected_key = (source.float() @ weights["indexer.wk"].float().T).bfloat16()
        torch.testing.assert_close(observed_key.float(), expected_key.float(), rtol=.02, atol=.002)
        expected_weights = torch.mm(source, weights["indexer.weights_proj"].T) * 32 ** -.5
        assert observed_weights.dtype == torch.bfloat16
        torch.testing.assert_close(observed_weights.float(), expected_weights.float(), rtol=.008, atol=1e-5)
        expected_norm = torch.nn.functional.layer_norm(observed_key.float(), (128,),
            weights["indexer.k_norm.weight"].float(), weights["indexer.k_norm.bias"].float(), eps).bfloat16()
        torch.testing.assert_close(observed_normalized_key.float(), expected_norm.float(), rtol=0, atol=0)
        wrong_norm = torch.nn.functional.layer_norm(observed_key.float(), (128,),
            weights["indexer.k_norm.weight"].float(), weights["indexer.k_norm.bias"].float(),
            1e-5 if eps == 1e-6 else 1e-6).bfloat16()
        assert not torch.allclose(expected_norm.float(), wrong_norm.float(), rtol=.1, atol=.001)
        expected_k, expected_key_scale = _identity_fp8_reference(observed_normalized_key[:256], power_of_two=False)
        physical_keys = cache.view(-1, 64 * 132)
        actual_k = physical_keys[[4, 5, 0, 1], :8192].view(torch.float8_e4m3fn).reshape(256, 128)
        actual_key_scale = physical_keys[[4, 5, 0, 1], 8192:].view(torch.float32).flatten()
        torch.testing.assert_close(actual_k.view(torch.uint8), expected_k.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(actual_key_scale, expected_key_scale, rtol=0, atol=0)
        expected_q, expected_query_scale = _identity_fp8_reference(observed_query.view(rows, 32, 128), power_of_two=True)
        torch.testing.assert_close(quantized.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(scales, expected_query_scale, rtol=0, atol=0)
        expected_scaled = (observed_weights.float() * scales) * 128 ** -.5
        torch.testing.assert_close(scaled_weights, expected_scaled, rtol=0, atol=0)
        for row, chosen in enumerate(selected[:-1].cpu().tolist()):
            assert set(token for token in chosen if token >= 0) == set(range(row + 1))
        assert selected[-1].eq(-1).all()
        assert not cache[2:4].count_nonzero()
        graph.reset()


def _cache_barrier_peer_read_rank(rank, rendezvous, world, rows):
    from datetime import timedelta

    import torch.distributed as dist

    from ignition.components.collectives import PreparedAllGather
    from ignition.components.peer_record_gather import PreparedPeerRecordGather
    from ignition.preparation.collectives import CollectivePreparation, create_head_exchange
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi31.dcp import RECORD_BYTES, TOPK, write_records

    local_records = 128
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    graph = runtime = None
    try:
        with CoreStorage(rank=rank, world=world, device=rank) as storage:
            cache = storage.device_allocate("latent", (1, local_records, RECORD_BYTES), torch.float8_e4m3fn,
                group="KV_PAGES", maximum=1)
            region = next(region for region in storage.dynamic_regions if region.address == cache.data_ptr())
            addresses = storage.peer_storage.share_regions(dist.group.WORLD, [region])[0]
            peers = torch.tensor(addresses, dtype=torch.int64, device=device)
            latent = torch.zeros((local_records, RECORD_BYTES), dtype=torch.bfloat16, device=device)
            record_slots = torch.arange(local_records, dtype=torch.int64, device=device)
            candidates = torch.full((rows, TOPK * 4), rank, dtype=torch.bfloat16, device=device)
            exchanged = torch.empty((rows, TOPK * 4 * world), dtype=torch.bfloat16, device=device)
            owners = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
            slots = (torch.arange(TOPK, dtype=torch.int32, device=device) % local_records)[None].repeat(rows, 1)
            counts = torch.full((rows,), TOPK, dtype=torch.int32, device=device)
            output = torch.empty((rows, TOPK, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device)
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_head_exchange(dist.group.WORLD, device, max_rows=rows, heads=world,
                head_dim=TOPK * 4, ipc=storage.peer_storage)
            with RecordedPreparationSession(device=device, checkpoint="kimi31-cache-publish",
                    workload_schema="cache-write-candidate-barrier-peer-read", numerical_policy="exact-bytes",
                    coordinator=coordinator) as session:
                exchange = PreparedAllGather(candidates, exchanged, runtime=runtime, session=session,
                    coordinator=coordinator, name="candidates", dependencies=(cache,))
                gather = PreparedPeerRecordGather(peers, owners, slots, counts, output,
                    peer_records=local_records, session=session, name="records")
                session.tune(show_progress=False)

                def forward():
                    write_records(latent, record_slots, cache)
                    exchange(candidates)
                    owners.copy_(exchanged.view(rows, world, TOPK * 4)[:, :, :TOPK // world]
                                 .reshape(rows, TOPK).to(torch.int32))
                    gather()

                compiled = compile_forward(forward)
                compiled()
                torch.cuda.synchronize(device)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph, stream=stream):
                    compiled()
                session.freeze()
                for step in range(6):
                    latent.fill_((rank + 1) * (step + 1) * .125)
                    cache.fill_(17)
                    output.fill_(19)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize(device)
                    assert before == torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    torch.testing.assert_close(cache.flatten().float(), latent.to(cache.dtype).flatten().float(), rtol=0, atol=0)
                    for peer in range(world):
                        actual = output[:, peer * (TOPK // world):(peer + 1) * (TOPK // world)].float()
                        expected = torch.full_like(actual, (peer + 1) * (step + 1) * .125).to(output.dtype).float()
                        torch.testing.assert_close(actual, expected,
                                                  rtol=0, atol=0)
                    dist.barrier()
    finally:
        if graph is not None:
            graph.reset()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                    reason="requires explicitly assigned peer GPUs")
@pytest.mark.parametrize("world,rows", [(2, 2), (16, 128)])
def test_compiled_cache_write_precedes_candidate_barrier_and_peer_read(tmp_path, world, rows):
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} assigned GPUs")
    mp.spawn(_cache_barrier_peer_read_rank, args=((tmp_path / "cache-publish").as_uri(), world, rows),
             nprocs=world, join=True)


@torch.no_grad()
def test_selected_attention_constants_survive_disposable_scratch_reset():
    from b12x.attention import dense_mla
    from ignition.components.dense_mla import PreparedDenseMLA
    from ignition.components.peer_record_gather import PreparedPeerRecordGather
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.disposable import storage_ranges
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.decoder import DecoderStorage
    from ignition.recipes.kimi31.dcp import AttentionConstants, PREFILL_ROWS, RECORD_BYTES, TOPK
    from ignition.recipes.kimi31.indexer import SharedBuffers

    device = torch.device("cuda", 0)
    torch.manual_seed(3147)
    constants = AttentionConstants.allocate(device)
    scratch = {}
    buffers = SharedBuffers(scratch, device)
    query = buffers.buffer("query", (2 * PREFILL_ROWS + 1, 8, RECORD_BYTES), torch.bfloat16)
    cache = buffers.buffer("records", (PREFILL_ROWS, TOPK, RECORD_BYTES), torch.float8_e4m3fn)
    output = buffers.buffer("output", (2 * PREFILL_ROWS + 1, 8, 512), torch.bfloat16)
    counts = buffers.buffer("counts", (PREFILL_ROWS,), torch.int32)
    query_seed = (torch.randn(query.shape, device=device) * .1).bfloat16()
    record_seed = (torch.randn((PREFILL_ROWS, 17, RECORD_BYTES), device=device) * .1).to(cache.dtype)
    count_seed = torch.arange(PREFILL_ROWS, dtype=torch.int32, device=device) % 18
    count_seed[0] = 17
    peers = torch.full((16,), record_seed.data_ptr(), dtype=torch.int64, device=device)
    owners = buffers.buffer("owners", (PREFILL_ROWS, TOPK), torch.int32)
    slots = buffers.buffer("slots", (PREFILL_ROWS, TOPK), torch.int32)
    slot_seed = torch.zeros_like(slots)
    slot_seed[:, :17] = torch.arange(PREFILL_ROWS * 17, device=device).view(PREFILL_ROWS, 17)
    storage = DecoderStorage(scratch, hidden_columns=None, latent_begin=None)
    with RecordedPreparationSession(device=device, checkpoint="kimi31-attention-constants",
            workload_schema="selected-page-attention", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        operations = []
        for layer in range(2):
            parts = {}
            for begin, end in ((0, PREFILL_ROWS), (PREFILL_ROWS, 2 * PREFILL_ROWS),
                               (2 * PREFILL_ROWS, 2 * PREFILL_ROWS + 1)):
                rows = end - begin
                if rows in parts:
                    operations.append((begin, end, parts[rows]))
                    continue
                gather = PreparedPeerRecordGather(peers, owners[:rows], slots[:rows], counts[:rows], cache[:rows],
                    peer_records=PREFILL_ROWS * 17, session=session, name=f"layer{layer}.records{rows}")
                caps = dense_mla.Caps(device=device, mode="decode", q_dtype=query.dtype, kv_dtype=cache.dtype,
                    num_q_heads=8, page_size=1, max_total_q=rows, max_batch=rows, max_cache_tokens=TOPK,
                    max_page_table_width=TOPK, num_cache_pages=rows * TOPK, use_cuda_graph=True)
                attention = PreparedDenseMLA(query[begin:end], cache[:rows].view(-1, 1, RECORD_BYTES),
                    output[begin:end], constants.pages[:rows], counts[:rows], constants.starts[:rows + 1],
                    caps=caps, scale=192 ** -.5, session=session, name=f"layer{layer}.rows{rows}", workspace=arena,
                    kv_scale=constants.scale, q_scale=constants.scale)
                for name, original in (("page_table", constants.pages), ("cu_seqlens_q", constants.starts),
                                       ("kv_scale", constants.scale), ("q_scale", constants.scale)):
                    assert attention.arguments[name].untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
                parts[rows] = (gather, attention)
                operations.append((begin, end, parts[rows]))
        assert operations[0][2] is operations[1][2]
        assert operations[3][2] is operations[4][2]
        assert operations[0][2] is not operations[3][2]
        session.tune(show_progress=False)
        scratch["workspace"] = arena.tensor
        disposable = storage_ranges(storage.scratch_tensors())
        for constant in (constants.pages, constants.starts, constants.scale):
            address = constant.untyped_storage().data_ptr()
            extent = constant.untyped_storage().nbytes()
            assert all(address + extent <= begin or begin + size <= address for begin, size in disposable)
        query.copy_(query_seed)
        slots.copy_(slot_seed)
        counts.copy_(count_seed)
        graphs = []
        def make_forward(layer_operations):
            def forward():
                for begin, end, (gather, attention) in layer_operations:
                    gather()
                    attention(query[begin:end], output[begin:end])
            return forward

        for layer in range(2):
            layer_operations = tuple(operations[layer * 3:(layer + 1) * 3])
            compiled = compile_forward(make_forward(layer_operations))
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs.append((layer_operations, compiled, graph))
        session.freeze()
        for layer_operations, compiled, graph in graphs:
            for tensor in storage.scratch_tensors():
                if isinstance(tensor, torch.Tensor):
                    tensor.zero_()
            query.copy_(query_seed)
            output.fill_(torch.nan)
            slots.copy_(slot_seed)
            counts.copy_(count_seed)
            for _ in range(3):
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                graph.replay()
                torch.cuda.synchronize(device)
                after = torch.cuda.memory_stats(device)
                for field in ("allocation.all.allocated", "allocation.all.freed",
                              "segment.all.allocated", "segment.all.freed"):
                    assert before[field] == after[field]
            for begin, end, _ in layer_operations:
                rows = end - begin
                quantized = query_seed[begin:end].float().to(cache.dtype).float()
                records = record_seed[:rows].float()
                logits = torch.einsum("rhd,rkd->rhk", quantized, records) * 192 ** -.5
                visible = torch.arange(17, device=device)[None] < count_seed[:rows, None]
                probabilities = logits.masked_fill(~visible[:, None], -torch.inf).softmax(-1)
                probabilities = torch.where(count_seed[:rows, None, None] > 0, probabilities, 0)
                expected = torch.einsum("rhk,rkd->rhd", probabilities, records[:, :, :512])
                torch.testing.assert_close(output[begin:end].float(), expected, rtol=.03, atol=.002)
            assert constants.pages[-1, -1].item() == PREFILL_ROWS * TOPK - 1
            assert constants.starts[-1].item() == PREFILL_ROWS
            assert constants.scale.item() == 1
            graph.reset()


@pytest.mark.parametrize("stage", ["index", "fold", "gather", "attention"])
@torch.no_grad()
def test_inactive_local_dsa_stage_replay_with_overwritten_workspace(stage):
    """Inactive rows preserve cache bytes and clear selections, records, and attention outputs."""
    from b12x.attention import dense_mla
    from ignition.components.dense_mla import PreparedDenseMLA
    from ignition.components.fp8_indexer import PreparedFP8Indexer
    from ignition.components.peer_record_gather import PreparedPeerRecordGather
    from ignition.components.row_topk import PreparedRowTopK
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi31.dcp import (
        AttentionConstants, ContextShards, PREFILL_ROWS, RECORD_BYTES, TOPK, selection_slots,
    )

    device = torch.device("cuda", 0)
    rows, world, index_columns, backed_pages = PREFILL_ROWS, 16, 1024, 20
    torch.manual_seed(3179)
    query = torch.randn((rows, 32, 128), dtype=torch.bfloat16, device=device)
    weights = torch.randn((rows, 32), dtype=torch.float32, device=device)
    keys = torch.randn((rows, 128), dtype=torch.bfloat16, device=device)
    key_slots = torch.full((rows,), -1, dtype=torch.int64, device=device)
    index_cache = torch.zeros((backed_pages, 64 * 132), dtype=torch.uint8, device=device)
    pages = torch.full((1, index_columns), -1, dtype=torch.int32, device=device).expand(rows, -1)
    lengths = torch.zeros(rows, dtype=torch.int32, device=device)
    local = torch.full((rows, TOPK), -1, dtype=torch.int32, device=device)
    local_scores = torch.full((rows, TOPK), -torch.inf, dtype=torch.float32, device=device)
    candidate_scores = torch.full((rows, world * TOPK), -torch.inf, dtype=torch.float32, device=device)
    candidate_ids = torch.full((rows, world * TOPK), -1, dtype=torch.int32, device=device)
    selected = torch.full_like(local, -1)
    selected_scores = torch.full_like(local_scores, -torch.inf)
    positions = torch.full((rows,), -1, dtype=torch.int64, device=device)
    tables = torch.full((rows, index_columns // 2), -1, dtype=torch.int32, device=device)
    owners, slots = torch.full_like(local, -1), torch.full_like(local, -1)
    counts = torch.zeros(rows, dtype=torch.int32, device=device)
    peer_cache = torch.zeros((128, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device)
    peers = torch.full((world,), peer_cache.data_ptr(), dtype=torch.int64, device=device)
    records = torch.zeros((rows, TOPK, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device)
    absorbed = (torch.randn((rows, 6, RECORD_BYTES), device=device) * .1).bfloat16()
    attended = torch.zeros((rows, 6, 512), dtype=torch.bfloat16, device=device)
    shards = ContextShards.on(2, world, device)
    constants = AttentionConstants.allocate(device)
    with RecordedPreparationSession(device=device, checkpoint="kimi31-inactive-local-stage",
            workload_schema="inactive-dsa-128", numerical_policy="default", autotune_effort="off") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        if stage == "index":
            indexer = PreparedFP8Indexer(query, weights, index_cache, pages, lengths, None, local,
                mode="prefill", session=session, name="index", workspace=arena,
                buffers=GraphBufferPool(device).cursor(), topk=TOPK, scores=local_scores,
                prefill_records_per_call=131072, hadamard=False, key_power_of_two=False)

            def forward():
                indexer.write_keys(keys, key_slots)
                indexer(query, weights)
        elif stage == "fold":
            forward = PreparedRowTopK(candidate_scores, lengths, candidate_ids, selected_scores, selected,
                session=session, name="fold")
        elif stage == "gather":
            gather = PreparedPeerRecordGather(peers, owners, slots, counts, records,
                peer_records=peer_cache.shape[0], session=session, name="gather")

            def forward():
                selection_slots(selected, positions, tables, owners, slots, counts, shards,
                                backed_pages // 2, local=False)
                gather()
        else:
            caps = dense_mla.Caps(device=device, mode="decode", q_dtype=absorbed.dtype, kv_dtype=records.dtype,
                num_q_heads=6, page_size=1, max_total_q=rows, max_batch=rows, max_cache_tokens=TOPK,
                max_page_table_width=TOPK, num_cache_pages=rows * TOPK, use_cuda_graph=True)
            attention = PreparedDenseMLA(absorbed, records.view(-1, 1, RECORD_BYTES), attended,
                constants.pages, counts, constants.starts, caps=caps, scale=192 ** -.5,
                session=session, name="attention", workspace=arena,
                kv_scale=constants.scale, q_scale=constants.scale)

            def forward():
                attention(absorbed, attended)
        session.tune(show_progress=False)
        arena.tensor.fill_(255)
        forward()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        for iteration in range(96):
            arena.tensor.fill_(255 if iteration % 2 else 0)
            local.fill_(-2)
            local_scores.fill_(torch.nan)
            selected_scores.fill_(torch.nan)
            if stage == "fold":
                selected.fill_(-2)
            if stage == "gather":
                records.fill_(17)
            attended.fill_(19)
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert before == torch.cuda.memory_stats(device)["allocation.all.allocated"]
            if stage == "index":
                assert local.eq(-1).all()
                assert local_scores.isneginf().all()
                assert not index_cache.count_nonzero()
            elif stage == "fold":
                assert selected.eq(-1).all()
                assert selected_scores.isneginf().all()
            elif stage == "gather":
                assert not counts.count_nonzero()
                assert not records.count_nonzero()
            else:
                assert not attended.count_nonzero()
        graph.reset()


@torch.no_grad()
def test_bf16_head_gate_rounding_and_normalized_fp8_weights_under_replay():
    from ignition.components.fp8_indexer import PreparedFP8Indexer
    from ignition.components.linear import UnquantizedProjection
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    rows, heads, hidden = 4, 32, 7168
    values = torch.arange(64, 193, dtype=torch.float32) / 128
    left, right = torch.meshgrid(values, values, indexing="ij")
    products = left * right
    rounded_projection = products.bfloat16()
    expected = rounded_projection.mul(32 ** -.5)
    single_rounding = (products * 32 ** -.5).bfloat16()
    adversarial = (expected != single_rounding).nonzero()[:heads]
    assert adversarial.shape[0] == heads
    a = left[adversarial[:, 0], adversarial[:, 1]].to(device)
    b = right[adversarial[:, 0], adversarial[:, 1]].to(device)
    source = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=device)
    source[:, :heads].copy_(a[None].expand(rows, -1))
    source[1].neg_()
    source[2].mul_(2)
    source[3].mul_(.5)
    weight = torch.zeros((heads, hidden), dtype=torch.bfloat16, device=device)
    weight[torch.arange(heads, device=device), torch.arange(heads, device=device)] = b.bfloat16()
    gates = torch.empty((rows, heads), dtype=torch.bfloat16, device=device)
    query = torch.randn((rows, heads, 128), dtype=torch.bfloat16, device=device)
    query.mul_(torch.tensor([.125, .5, 2, 8], device=device)[:, None, None])
    cache = torch.zeros((2, 64 * 132), dtype=torch.uint8, device=device)
    pages = torch.tensor([[0, 1]], dtype=torch.int32, device=device).expand(rows, -1)
    lengths = torch.tensor([0, 1, 64, 128], dtype=torch.int32, device=device)
    selected = torch.empty((rows, 512), dtype=torch.int32, device=device)
    with RecordedPreparationSession(device=device, checkpoint="kimi31-rounded-head-gates",
            workload_schema="bf16-head-gate", numerical_policy="two-bf16-roundings", autotune_effort="off") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        projection = UnquantizedProjection(source, weight, gates, session=session, name="gate")
        indexer = PreparedFP8Indexer(query, gates, cache, pages, lengths, None, selected,
            mode="decode", session=session, name="index", workspace=arena,
            buffers=GraphBufferPool(device).cursor(), head_weight_scale=1.0, hadamard=False, key_power_of_two=False)
        session.tune(show_progress=False)

        def forward():
            projection(source)
            gates.mul_(32 ** -.5)
            indexer(query, gates)

        forward()
        torch.cuda.synchronize(device)
        rounded = (source[:, :heads].float() * b[None]).bfloat16().mul(32 ** -.5)
        torch.testing.assert_close(gates, rounded, rtol=0, atol=0)
        torch.testing.assert_close(indexer.scaled_weights,
            (rounded.float() * indexer.scales) * 128 ** -.5, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        for replay in range(4):
            if replay:
                source.neg_()
                query.mul_(2)
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert before == torch.cuda.memory_stats(device)["allocation.all.allocated"]
            products = source[:, :heads].float() * b[None]
            rounded = products.bfloat16().mul(32 ** -.5)
            torch.testing.assert_close(gates, rounded, rtol=0, atol=0)
            assert not torch.equal(rounded, (products * 32 ** -.5).bfloat16())
            assert not torch.equal(rounded.float(), products.bfloat16().float() * 32 ** -.5)
            expected_weights = (rounded.float() * indexer.scales) * 128 ** -.5
            torch.testing.assert_close(indexer.scaled_weights, expected_weights, rtol=0, atol=0)
            assert selected[0].eq(-1).all()
        graph.reset()


@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@torch.no_grad()
def test_mla_query_and_latent_norm_use_configured_epsilon(eps):
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.attention import PreparedMLAProjection

    device = torch.device("cuda", 0)
    rows, hidden, heads, query_width = 3, 7168, 6, 1536
    latent_width = query_width + 576
    source = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=device)
    source[:, 0] = torch.tensor([2 ** -11, 2 ** -10, 2 ** -9], dtype=torch.bfloat16, device=device)
    coefficients = torch.linspace(-2, 2, latent_width + heads * 128, device=device).bfloat16()
    latent_weight = torch.zeros((coefficients.numel(), hidden), dtype=torch.bfloat16, device=device)
    latent_weight[:, 0].copy_(coefficients)
    weights = {
        "latent_gate": latent_weight,
        "query_norm": torch.linspace(.75, 1.25, query_width, device=device).bfloat16(),
        "kv_norm": torch.linspace(.5, 1.5, 512, device=device).bfloat16(),
        "q_b": torch.zeros((heads * 192, query_width), dtype=torch.bfloat16, device=device),
        "key": torch.zeros((heads, 128, 512), dtype=torch.bfloat16, device=device),
        "value": torch.zeros((heads, 512, 128), dtype=torch.bfloat16, device=device),
        "output": torch.zeros((hidden, heads * 128), dtype=torch.bfloat16, device=device),
    }
    projected = (source[:, :1].float() * coefficients[None].float()).bfloat16()

    def reference(value, weight, epsilon):
        normalized = (value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + epsilon)).bfloat16()
        return (normalized.float() * weight.float()).bfloat16()

    expected_query = reference(projected[:, :query_width], weights["query_norm"], eps)
    expected_latent = reference(projected[:, query_width:query_width + 512], weights["kv_norm"], eps)
    alternate_eps = 1e-5 if eps == 1e-6 else 1e-6
    assert not torch.allclose(expected_query.float(),
        reference(projected[:, :query_width], weights["query_norm"], alternate_eps).float(), rtol=.1, atol=.001)
    assert not torch.allclose(expected_latent.float(),
        reference(projected[:, query_width:query_width + 512], weights["kv_norm"], alternate_eps).float(),
        rtol=.1, atol=.001)
    with RecordedPreparationSession(device=device, checkpoint="kimi-mla-normalization",
            workload_schema="configured-epsilon", numerical_policy="bf16-rmsnorm", autotune_effort="off") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        projection = PreparedMLAProjection(weights, source, None, None, eps=eps, mode="prefill",
            session=session, name="projection", workspace=arena, buffers={})
        projection.functional(session=session)
        session.tune(show_progress=False)
        result = projection.project(source)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(result.query_norm, expected_query, rtol=0, atol=0)
        torch.testing.assert_close(result.latent[:, :512], expected_latent, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            result = projection.project(source)
        session.freeze()
        result.query_norm.fill_(19)
        result.latent.fill_(19)
        before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert before == torch.cuda.memory_stats(device)["allocation.all.allocated"]
        torch.testing.assert_close(result.query_norm, expected_query, rtol=0, atol=0)
        torch.testing.assert_close(result.latent[:, :512], expected_latent, rtol=0, atol=0)
        torch.testing.assert_close(result.latent[:, 512:], projected[:, query_width + 512:latent_width], rtol=0, atol=0)
        graph.reset()
