# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tuned sparse index selection preserves caller storage through native replay."""

import os

import pytest
import torch
from b12x.attention import dsa_indexer
from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference

from ignition.components.indexer import PreparedSparseIndexer
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


def _assert_selection(actual, actual_scores, expected, expected_scores):
    actual_ids, actual_order = actual.sort()
    expected_ids, expected_order = expected.sort()
    actual_scores = actual_scores.gather(1, actual_order)
    expected_scores = expected_scores.gather(1, expected_order)
    same = actual_ids == expected_ids
    torch.testing.assert_close(actual_scores[same], expected_scores[same], rtol=0, atol=0)
    for row in (~same).any(1).nonzero().flatten().tolist():
        a = dict(zip(actual_ids[row].tolist(), actual_scores[row].tolist()))
        b = dict(zip(expected_ids[row].tolist(), expected_scores[row].tolist()))
        assert len(a) == len(b)
        for index in a.keys() & b.keys():
            assert a[index] == b[index]
        cutoff = min(value for index, value in b.items() if index >= 0)
        assert all(a[index] == cutoff for index in a.keys() - b.keys())
        assert all(b[index] == cutoff for index in b.keys() - a.keys())


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("mode,rows", [("decode", 2), ("prefill", 4096)])
@torch.no_grad()
def test_tuned_sparse_indexer_native_replay(mode, rows, functional):
    device = torch.device("cuda:0")
    pages, heads, topk = 32, 32, 512
    query = torch.randn((rows, heads, 128), device=device).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, heads), device=device)
    cache = pack_index_k_cache_reference(torch.randn((pages * 64, 128), device=device))
    table = torch.randperm(pages, device=device, dtype=torch.int32).unsqueeze(0).expand(rows, -1)
    if mode == "decode":
        table = table.contiguous()
    lengths = torch.full((rows,), pages * 64, dtype=torch.int32, device=device)
    width = torch.tensor([pages * 64], dtype=torch.int32, device=device)
    output = torch.empty((rows, topk), dtype=torch.int32, device=device)
    scores = torch.empty((rows, topk), device=device)
    observed, observed_scores = torch.empty_like(output), torch.empty_like(scores)
    arena = ScratchArena(device)
    worker = None
    graph = torch.cuda.CUDAGraph()
    with RecordedPreparationSession(device=device, checkpoint="sparse-indexer-component",
            workload_schema="glm-indexer-v1", numerical_policy="default") as session:
        try:
            session.register_scratch(arena)
            caps = dsa_indexer.Caps(device=device, num_q_heads=heads, max_q_rows=rows,
                max_page_table_width=pages, topk=topk, mode=mode)
            indexer = PreparedSparseIndexer(query, weights, cache, table, lengths, width, output,
                caps=caps, session=session, name="indexer", workspace=arena, scores=scores)
            old_storage = arena.tensor
            arena.reserve(arena.nbytes + 1024)
            saved = tuple(tensor.clone() for tensor in (query, weights, cache, table, lengths, width))
            if functional:
                indexer.functional(session=session)
            session.tune()
            for actual, expected in zip((query, weights, cache, table, lengths, width), saved, strict=True):
                torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
            if functional:
                assert indexer.binding is indexer.output is indexer.scores is None
            else:
                assert indexer.binding.runtime.scratch.shared_scratch.data_ptr() == arena.tensor.data_ptr()
            assert arena.tensor.data_ptr() != old_storage.data_ptr()
            del old_storage, saved
            if functional:
                arena.release_preparation()

            def forward():
                if functional:
                    result, result_scores = indexer(query, weights, table, lengths, width)
                    observed.copy_(result)
                    observed_scores.copy_(result_scores)
                else:
                    observed.copy_(indexer())
                    observed_scores.copy_(scores)
                arena.tensor.fill_(255)

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            session.freeze()
            bundle = PreparedGraphBundle(device=0, graphs={mode: graph},
                buffers={"query": query, "cache": cache, "output": observed},
                resources=(indexer, compiled, arena, session))
            worker = bundle.start_worker()
            for count in (256, 1024, pages * 64):
                query.copy_(torch.randn(query.shape, device=device).to(query.dtype))
                weights.normal_()
                cache.copy_(pack_index_k_cache_reference(torch.randn((pages * 64, 128), device=device)))
                lengths.copy_((torch.arange(rows, device=device, dtype=torch.int32) // 4 + count).clamp_max(pages * 64))
                lengths[0] = 0
                forward()
                expected, expected_scores = observed.clone(), observed_scores.clone()
                observed.fill_(-42)
                observed_scores.fill_(42)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                ticket = worker.submit(0, repetitions=3)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                _assert_selection(observed, observed_scores, expected, expected_scores)
        finally:
            if worker is not None:
                worker.close()
            graph.reset()


@torch.no_grad()
def test_prefill_candidate_windows_preserve_full_context_native_selection():
    device = torch.device("cuda", 0)
    torch.manual_seed(6817)
    rows, records, heads = 4096, 262144, 64
    query = torch.randn((rows, heads, 128), device=device).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, heads), device=device) / heads
    cache = pack_index_k_cache_reference(torch.randn((records, 128), device=device).bfloat16())
    table = torch.arange(records // 64, device=device, dtype=torch.int32)[None].expand(rows, -1)
    lengths = torch.full((rows,), 2048, device=device, dtype=torch.int32)
    active = torch.tensor([2048], device=device, dtype=torch.int32)
    arena = ScratchArena(device)
    outputs, scores, graphs, owners, operations = {}, {}, {}, [], {}
    worker = None
    with RecordedPreparationSession(device=device, checkpoint="fp8-full-context",
            workload_schema="prefill-candidate-windows-v1", numerical_policy="fp8") as session:
        session.register_scratch(arena)
        try:
            for width in (32768, 131072):
                outputs[width] = torch.empty((rows, 512), device=device, dtype=torch.int32)
                scores[width] = torch.empty((rows, 512), device=device)
                caps = dsa_indexer.Caps(device=device, num_q_heads=heads, max_q_rows=rows,
                    max_page_table_width=records // 64, topk=512, mode="prefill", supertile_k=width)
                operations[width] = PreparedSparseIndexer(query, weights, cache, table, lengths,
                    active, outputs[width], caps=caps, session=session, name=f"window{width}",
                    workspace=arena, scores=scores[width])
            session.tune()
            for width, operation in operations.items():
                def forward(operation=operation):
                    operation()
                    arena.tensor.fill_(255)

                compiled = compile_forward(forward)
                compiled()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    compiled()
                graphs[str(width)] = graph
                owners.append(compiled)
            session.freeze()
            bundle = PreparedGraphBundle(device=0, graphs=graphs,
                buffers={"query": query, "cache": cache},
                resources=(operations, outputs, scores, arena, owners))
            worker = bundle.start_worker()
            for count in (1024, 32768, 131073, records, 2048):
                lengths.copy_((torch.arange(rows, device=device, dtype=torch.int32) // 4
                               + count - rows // 4).clamp(min=0, max=records))
                lengths[0] = 0
                lengths[1] = 127
                lengths[-1] = count
                active.fill_(count)
                # Rotate physical pages between launches while retaining logical index space.
                table[0].copy_(torch.roll(table[0], shifts=17))
                for width in operations:
                    outputs[width].fill_(-42)
                    scores[width].fill_(float("nan"))
                    arena.tensor.fill_(90)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    ticket = worker.submit(bundle.graph_indices[str(width)], repetitions=3)
                    completion = worker.poll(timeout_ms=120000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                    after = torch.cuda.memory_stats()
                    for key in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key]
                    assert (arena.tensor == 255).all()
                    selected = outputs[width]
                    assert ((selected == -1) | ((selected >= 0) & (selected < lengths[:, None]))).all()
                    assert (selected[0] == -1).all()
                    assert (selected[1] >= 0).sum() == 127
                _assert_selection(outputs[131072], scores[131072], outputs[32768], scores[32768])
        finally:
            if worker is not None:
                worker.close()
            for graph in graphs.values():
                graph.reset()
