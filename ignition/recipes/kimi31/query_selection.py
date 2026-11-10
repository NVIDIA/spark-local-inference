# SPDX-License-Identifier: Apache-2.0

"""Shard prefill index queries while preserving stored key and scale bytes."""

import torch
import triton
import triton.language as tl

from ignition.components.collectives import PreparedAllGather
from ignition.components.fp8_indexer import PreparedFP8Indexer
from ignition.components.planar_index_history import PreparedPlanarIndexHistory, global_index_pages

from .dcp import PREFILL_ROWS, TOPK


@triton.jit
def _owned_queries(Query, Weights, Positions, Rank, LocalQuery, LocalWeights, Lengths,
                   query_stride, weight_stride, ROWS: tl.constexpr, WORLD: tl.constexpr):
    row = tl.program_id(0)
    original = row * WORLD + tl.load(Rank)
    position = tl.load(Positions + original, original < ROWS, other=-1)
    active = (original < ROWS) & (position >= 0)
    column = tl.arange(0, 4096)
    query = tl.load(Query + original * query_stride + column, active, other=0)
    tl.store(LocalQuery + row * 4096 + column, query)
    head = tl.arange(0, 32)
    weight = tl.load(Weights + original * weight_stride + head, active, other=0)
    tl.store(LocalWeights + row * 32 + head, weight)
    tl.store(Lengths + row, tl.where(active, position + 1, 0).to(tl.int32))


class PreparedQuerySelection:
    """Score interleaved query rows over replicated owner-encoded index history.

    Projection shapes and rounding remain those of the complete prefill graph.
    The selected-ID gather restores original query order and preserves each
    owner's selection order. Projection replicas and top-k ties still require
    numerical qualification; this decomposition does not promise bitwise
    equivalence to a distributed candidate fold.
    """

    def __init__(self, query, weights, cache, metadata, shards, context, buffers, *,
                 session, name, workspace, allocator):
        from .indexer import SharedBuffers

        rows = query.shape[0]
        if (rows < 4096 or query.shape != (rows, 4096) or query.dtype != torch.bfloat16
                or weights.shape != (rows, 32) or weights.dtype != torch.bfloat16
                or query.stride(1) != 1 or weights.stride(1) != 1
                or metadata.starts.numel() != 1 or context is None
                or context.index_staging is None or context.index_records is None
                or context.index_columns is None or context.selections is None):
            raise ValueError("query-sharded indexing requires large single-request BF16 prefill and shared index storage")
        self.query, self.weights, self.metadata, self.shards = query, weights, metadata, shards
        self.local_rows = triton.cdiv(rows, shards.world)
        storage = SharedBuffers(buffers, query.device, allocator)
        self.end = storage.buffer("end", (1,), torch.int64)
        self.local_query = storage.buffer("query", (self.local_rows, 32, 128), torch.bfloat16)
        self.local_weights = storage.buffer("weights", (self.local_rows, 32), torch.bfloat16)
        self.lengths = storage.buffer("lengths", (self.local_rows,), torch.int32)
        self.local_selected = storage.buffer("local_selected", (self.local_rows, TOPK), torch.int32)
        self.gathered = storage.buffer("gathered", (self.local_rows, TOPK * shards.world), torch.int32)
        self.selected = self.gathered.view(self.local_rows * shards.world, TOPK)[:rows]
        history = context.index_records
        self.columns = context.index_columns
        if self.columns.shape != (history.shape[0],) or self.columns.dtype != torch.int32:
            raise ValueError("query-sharded indexing requires one int32 column per global index page")
        self.pages = storage.buffer("pages", (1, history.shape[0]), torch.int32)
        self.cache_pages = cache.shape[0] // 2
        self.history = PreparedPlanarIndexHistory(cache, metadata.page_table[0], self.end,
            shards.index, context.index_staging.tensor.view(-1, 64 * 132), context.index_staging.addresses,
            history, context.index_signals, runtime=context.stage_runtime, session=session,
            coordinator=context.coordinator, name=name + ".history")
        self.indexers = {}
        for begin in range(0, self.local_rows, PREFILL_ROWS):
            end = min(begin + PREFILL_ROWS, self.local_rows)
            count = end - begin
            if count in self.indexers:
                continue
            lengths = storage.buffer("slice_lengths", (count,), torch.int32)
            selected = storage.buffer("slice_selected", (count, TOPK), torch.int32)
            indexer = PreparedFP8Indexer(self.local_query[begin:end], self.local_weights[begin:end],
                history, self.pages.expand(count, -1), lengths, None,
                selected, mode="prefill", session=session, name=f"{name}.rows{count}",
                workspace=workspace, buffers=SharedBuffers(buffers, query.device, allocator, prefix="fp8."),
                prefill_records_per_call=131072, topk=TOPK, head_weight_scale=1.0,
                hadamard=False, key_power_of_two=False)
            self.indexers[count] = indexer, lengths, selected
        self.send = self.local_selected.view(torch.bfloat16).view(self.local_rows, TOPK * 2)
        self.gather = PreparedAllGather(self.send,
            self.gathered.view(torch.bfloat16).view(self.local_rows, TOPK * shards.world * 2),
            runtime=context.selections, session=session, coordinator=context.coordinator,
            name=name + ".selected", dependencies=(history, self.history.completed))

    def __call__(self):
        self.end.copy_((self.metadata.starts + self.metadata.lengths) * (self.metadata.lengths > 0))
        self.history()
        global_index_pages(self.metadata.page_table[0], self.columns, self.pages,
                           world=self.shards.world, cache_pages=self.cache_pages)
        torch.library.wrap_triton(_owned_queries)[(self.local_rows,)](
            self.query, self.weights, self.metadata.positions, self.shards.index,
            self.local_query, self.local_weights, self.lengths,
            self.query.stride(0), self.weights.stride(0),
            ROWS=self.query.shape[0], WORLD=self.shards.world, num_warps=8)
        for begin in range(0, self.local_rows, PREFILL_ROWS):
            end = min(begin + PREFILL_ROWS, self.local_rows)
            indexer, lengths, selected = self.indexers[end - begin]
            lengths.copy_(self.lengths[begin:end])
            indexer(self.local_query[begin:end], self.local_weights[begin:end])
            self.local_selected[begin:end].copy_(selected)
        self.gather(self.send)
        return self.selected
