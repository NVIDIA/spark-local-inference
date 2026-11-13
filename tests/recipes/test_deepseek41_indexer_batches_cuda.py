# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Batched sparse selection preserves scores, candidate bounds, and native storage."""

import os

import pytest
import torch

from ignition.components.mxfp4_indexer import PreparedMXFP4Indexer
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from tests.tensor_examples import TensorExamples
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.attention import INDEXER_QUERY_ROWS


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


def _quantized_values(value):
    groups = value.float().reshape(*value.shape[:-1], 4, 32)
    maximum = groups.abs().amax(-1).clamp_min(6 * 2.0**-126)
    bits = (maximum / 6).contiguous().view(torch.int32)
    scale_bytes = ((bits >> 23) + ((bits & 0x7FFFFF) != 0)).to(torch.uint8)
    scale = (scale_bytes.int() << 23).view(torch.float32)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=value.device)
    even_first = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=value.device)
    nearest = ((groups / scale[..., None]).abs()[..., None] - levels[even_first]).abs().argmin(-1)
    magnitude = levels[even_first[nearest]] * scale[..., None]
    return torch.where(torch.signbit(groups), -magnitude, magnitude).bfloat16().reshape_as(value)


def _scores(query, keys, weights):
    query, keys = _quantized_values(query), _quantized_values(keys)
    reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        dot = torch.einsum("rhd,kd->rhk", query, keys)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduction
    products = (dot.relu() * weights[..., None]).float()
    total = torch.zeros_like(products[:, 0])
    for head in range(products.shape[1]):
        total.add_(products[:, head])
    return total.bfloat16().float()


def _check_topk(indices, scores, *, allowed=None):
    valid = indices >= 0
    assert ((indices[:, 1:] > indices[:, :-1]) | ~valid[:, 1:]).all()
    # The mandatory newest candidate block has +inf priority; only -inf is padding.
    assert torch.equal(valid.sum(1), (~torch.isneginf(scores)).sum(1).clamp(max=indices.shape[1]))
    if allowed is None:
        chosen = indices.clamp_min(0).long()
    else:
        ordered = torch.where(allowed >= 0, allowed, torch.iinfo(allowed.dtype).max).contiguous()
        chosen = torch.searchsorted(ordered, indices.clamp_min(0)).clamp_max(allowed.shape[1] - 1)
        assert ((allowed.gather(1, chosen) == indices) | ~valid).all()
    actual = scores.gather(1, chosen).masked_fill(~valid, -torch.inf)
    expected = scores.topk(indices.shape[1], dim=1).values
    # Equal-score cutoff alternatives are valid; score values remain exact.
    torch.testing.assert_close(actual.sort().values, expected.sort().values, rtol=0, atol=0)


class BatchedSelection:
    def __init__(self, query, weights, cache, pages, lengths, active, output, *, page_size,
                 session, arena, name, candidates=None, counts=None, produce=False, buffers=None,
                 batch_rows=INDEXER_QUERY_ROWS, mode="decode", functional=False):
        self.operations = []
        self.functional = functional
        self.operands = query, weights, pages, lengths, active, output, candidates, counts, produce
        self.batch_rows = batch_rows
        self.declarations = []
        for first in range(0, query.shape[0], batch_rows):
            last = min(first + batch_rows, query.shape[0])
            options = {}
            if candidates is not None:
                options = (dict(candidates_out=candidates[first:last], counts_out=counts[first:last])
                           if produce else dict(candidates=candidates[first:last], counts=counts[first:last]))
            self.operations.append(PreparedMXFP4Indexer(query[first:last], weights[first:last],
                cache, pages[first:last], lengths[first:last], active, output[first:last],
                page_size=page_size, session=session, name=f"{name}.{first}", workspace=arena,
                buffers=buffers, mode=mode, **options))
            self.declarations.extend((self.operations[-1].packed, self.operations[-1].scales))
            if functional:
                self.operations[-1].functional(session=session)

    def __call__(self):
        query, weights, pages, lengths, active, output, candidates, counts, produce = self.operands
        for index, operation in enumerate(self.operations):
            if not self.functional:
                operation()
                continue
            first = index * self.batch_rows
            last = min(first + self.batch_rows, query.shape[0])
            options = {} if candidates is None or produce else dict(
                candidates=candidates[first:last], counts=counts[first:last])
            result = operation(query[first:last], weights[first:last], pages[first:last],
                               lengths[first:last], active, **options)
            if produce:
                selected, candidate_rows, candidate_counts = result
                candidates[first:last].copy_(candidate_rows)
                counts[first:last].copy_(candidate_counts)
            else:
                selected = result
            output[first:last].copy_(selected)


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("rows,independent_pages,batch_rows,index_mode", [
    (257, False, 128, "decode"), (257, False, 128, "prefill"),
    (257, False, 256, "prefill"), (64, True, 128, "decode"),
])
@torch.no_grad()
def test_compiled_query_batches_match_independent_scores_with_candidate_reuse(
        page_size, rows, independent_pages, batch_rows, index_mode, functional):
    device = torch.device("cuda", 0)
    torch.manual_seed(41128)
    records, heads = 32768, 32
    capacity = 1048576 // (128 // page_size)
    page_count = capacity // page_size
    queries = [torch.randn((rows, heads, 128), device=device, dtype=torch.bfloat16) * .2 for _ in range(2)]
    weights = [torch.randn((rows, heads), device=device, dtype=torch.bfloat16) / heads for _ in range(2)]
    keys = torch.randn((records, 128), device=device, dtype=torch.bfloat16) * .2
    cache = torch.zeros((page_count, page_size * 68), device=device, dtype=torch.uint8)
    permutation = torch.randperm(page_count, device=device, dtype=torch.int32)
    table = permutation[None].expand(rows, -1)
    logical_keys = torch.arange(records, device=device)[None].expand(rows, -1)
    if independent_pages:
        table = table.clone()
        visible_pages = records // page_size
        page_order = (torch.arange(visible_pages, device=device)[None]
                      + 7 * torch.arange(rows, device=device)[:, None]) % visible_pages
        table[:, :visible_pages] = permutation[page_order]
        # Requests share physical storage but have distinct logical key sequences.
        logical_keys = (page_order[:, :, None] * page_size
                        + torch.arange(page_size, device=device)).reshape(rows, records)
    slots = permutation[torch.arange(records, device=device) // page_size].long() * page_size
    slots += torch.arange(records, device=device) % page_size
    lengths = torch.full((rows,), records, device=device, dtype=torch.int32)
    active = torch.tensor([records], device=device, dtype=torch.int32)
    candidates = torch.zeros((rows, 16384), device=device, dtype=torch.int32)
    counts = torch.full((rows,), 16384, device=device, dtype=torch.int32)
    outputs = {mode: torch.empty((rows, 512), device=device, dtype=torch.int32)
               for mode in ("full", "producer", "consumer")}
    arena = ScratchArena(device)
    buffers = TensorExamples(device)
    operations, functions, graphs = {}, {}, {}
    worker = None
    with RecordedPreparationSession(device=device, checkpoint="ds41-index-batches",
            workload_schema=f"page{page_size}", numerical_policy="mxfp4") as session:
        session.register_scratch(arena)
        try:
            for mode in outputs:
                source = int(mode == "consumer")
                operations[mode] = BatchedSelection(queries[source], weights[source], cache, table,
                    lengths, active, outputs[mode], page_size=page_size, session=session, arena=arena,
                    name=mode, candidates=None if mode == "full" else candidates,
                    counts=counts, produce=mode == "producer", buffers=buffers,
                    batch_rows=batch_rows, mode=index_mode, functional=functional)
            session.tune()
            operations["full"].operations[0].write_keys(keys, slots)
            saved_cache = cache.clone()
            for mode, operation in operations.items():
                functions[mode] = compile_forward(operation)
                functions[mode]()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    functions[mode]()
                graphs[mode] = graph
            session.freeze()
            bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"cache": cache},
                resources=(operations, functions, session, arena, queries, weights, table, lengths,
                           active, candidates, counts, outputs))
            worker = bundle.start_worker()
            for context in (32768, 16441, 0, 512):
                lengths.copy_((torch.arange(rows, device=device) + context - rows + 1).clamp_min(0))
                lengths[0] = 0
                lengths[1] = min(127, context)
                lengths[-5:] = 0
                active.fill_(context)
                for query in queries:
                    query.neg_()
                references = [_scores(query, keys, weight).gather(1, logical_keys).masked_fill(
                    torch.arange(records, device=device)[None] >= lengths[:, None], -torch.inf)
                    for query, weight in zip(queries, weights, strict=True)]
                saved = [value.clone() for value in (*queries, *weights, table, lengths, active)]
                for mode in ("full", "producer", "consumer", "full"):
                    outputs[mode].fill_(-42)
                    arena.tensor.fill_(0x5A)
                    for operation in operations.values():
                        for declaration in operation.declarations:
                            declaration.fill_(0xA5)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    ticket = worker.submit(bundle.graph_indices[mode], repetitions=2)
                    completion = worker.poll(timeout_ms=120000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                    after = torch.cuda.memory_stats()
                    for name in ("allocation.all.allocated", "allocation.all.freed",
                                 "segment.all.allocated", "segment.all.freed"):
                        assert before[name] == after[name], name
                    if mode == "consumer":
                        reference = references[1].gather(1, candidates.clamp_min(0).long())
                        reference.masked_fill_(candidates < 0, -torch.inf)
                        _check_topk(outputs[mode], reference, allowed=candidates)
                    else:
                        _check_topk(outputs[mode], references[0])
                    if mode == "producer":
                        assert torch.equal(counts.long(), (candidates >= 0).sum(1))
                        block_ids = candidates[:, ::8] // 8
                        block_scores = references[0].reshape(rows, -1, 8).amax(2)
                        newest = ((lengths.long() - 1) // 8).clamp_min(0)
                        block_scores.scatter_(1, newest[:, None],
                            torch.where(lengths > 0, torch.inf, -torch.inf)[:, None])
                        _check_topk(block_ids, block_scores)
                        expanded = block_ids[:, :, None] * 8 + torch.arange(8, device=device)
                        expected = torch.where((block_ids[:, :, None] >= 0)
                            & (expanded < lengths[:, None, None]), expanded, -1).reshape_as(candidates)
                        torch.testing.assert_close(candidates.long(), expected, rtol=0, atol=0)
                assert torch.equal(cache, saved_cache)
                for value, original in zip((*queries, *weights, table, lengths, active), saved, strict=True):
                    assert torch.equal(value, original)
        finally:
            if worker is not None:
                worker.close()
            for graph in graphs.values():
                graph.reset()
