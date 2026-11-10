# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-token Kimi sparse indexing over rank-local FP8 key pages."""

import torch

from ignition.components.collectives import PreparedAllGather, PreparedBroadcast
from ignition.components.fp8_indexer import PreparedFP8Indexer
from ignition.components.linear import UnquantizedProjection
from ignition.components.row_topk import PreparedRowTopK
from ignition.preparation.scratch import buffer_rows

from .dcp import TOPK, PREFILL_ROWS, index_pages, pack_candidates


class SharedBuffers:
    """Named row-capacity storage reused by serial layers and query slices."""

    def __init__(self, values, device, allocator=None, prefix=""):
        self.values, self.device, self.allocator, self.prefix = values, device, allocator, prefix
        self.ordinal = 0

    def buffer(self, name, shape, dtype):
        name = self.prefix + name
        value = self.values.get(name)
        if value is None or value.shape[0] < shape[0]:
            value = (torch.zeros(shape, dtype=dtype, device=self.device) if self.allocator is None
                     else self.allocator.zeros(shape, dtype))
            self.values[name] = value
        if value.shape[1:] != shape[1:] or value.dtype != dtype:
            raise ValueError("Kimi index buffers require matching non-row dimensions and dtype")
        return buffer_rows(self.values, name, shape[0])

    def zeros(self, shape, dtype):
        name = str(self.ordinal)
        self.ordinal += 1
        return self.buffer(name, shape, dtype)


class PreparedIndexSelection:
    def __init__(self, query, weights, cache, metadata, shards, context, buffers, *, start, end,
                 mode, session, name, workspace, allocator, latent_cache):
        rows, width = end - start, metadata.token_page_table.shape[1]
        self.shards, self.mode = shards, mode
        storage = SharedBuffers(buffers, query.device, allocator)
        self.pages = storage.buffer("pages", (1 if mode == "prefill" else rows, width * 2), torch.int32)
        self.lengths = storage.buffer("local_lengths", (rows,), torch.int32)
        self.local = storage.buffer("local_ids", (rows, TOPK), torch.int32)
        self.scores = storage.buffer("local_scores", (rows, TOPK), torch.float32)
        self.packed = storage.buffer("candidates", (rows, TOPK, 2), torch.float32)
        self.gathered = storage.buffer("gathered", (rows, TOPK * shards.world, 2), torch.float32)
        self.global_scores = storage.buffer("global_scores", (rows, TOPK * shards.world), torch.float32)
        self.global_ids = storage.buffer("global_ids", (rows, TOPK * shards.world), torch.int32)
        self.fold_lengths = storage.buffer("fold_lengths", (rows,), torch.int32)
        self.selected_scores = storage.buffer("selected_scores", (rows, TOPK), torch.float32)
        self.selected = storage.buffer("selected", (rows, TOPK), torch.int32)
        pages = self.pages.expand(rows, -1) if mode == "prefill" else self.pages
        self.indexer = PreparedFP8Indexer(query[start:end].view(rows, 32, 128), weights[start:end],
            cache, pages, self.lengths, None, self.local, mode="prefill" if mode == "prefill" else "decode", session=session,
            name=name + ".local", workspace=workspace,
            buffers=SharedBuffers(buffers, query.device, allocator, prefix="fp8."), topk=TOPK, scores=self.scores,
            prefill_records_per_call=131072, head_weight_scale=1.0, hadamard=False, key_power_of_two=False)
        runtime = None if context is None else context.candidates
        if shards.world > 1 and runtime is None:
            raise ValueError("Kimi sparse selection requires a candidate exchange for every context rank")
        self.gather = PreparedAllGather(self.packed.view(torch.bfloat16).view(rows, TOPK * 4),
            self.gathered.view(torch.bfloat16).view(rows, TOPK * 4 * shards.world), runtime=runtime,
            session=session, coordinator=None if context is None else context.coordinator,
            name=name + ".candidates", dependencies=(latent_cache,))
        self.fold = PreparedRowTopK(self.global_scores, self.fold_lengths, self.global_ids,
            self.selected_scores, self.selected, session=session, name=name + ".fold")
        self.broadcast = PreparedBroadcast((self.selected,), runtime=runtime, session=session,
            coordinator=None if context is None else context.coordinator, name=name + ".selected",
            buffers=SharedBuffers(buffers, query.device, allocator, prefix="consensus."))

    def __call__(self, query, weights, positions, tables):
        index_pages(tables[:1] if self.mode == "prefill" else tables, self.pages)
        self.lengths.copy_(self.shards.local_tokens((positions + 1).clamp(min=0)).to(torch.int32))
        self.indexer(query.view(query.shape[0], 32, 128), weights)
        pack_candidates(self.local, self.scores, self.packed, self.shards)
        self.gather(self.packed.view(torch.bfloat16).view(self.packed.shape[0], -1))
        self.global_scores.copy_(self.gathered[:, :, 0])
        self.global_ids.copy_(self.gathered.view(torch.int32)[:, :, 1])
        self.fold_lengths.copy_(torch.where(positions >= 0, self.global_ids.shape[1], 0))
        self.fold()
        self.broadcast()
        return self.selected


class PreparedDSAIndexer:
    """BF16 projections and head normalization, FP32 LayerNorm, and global top-2048 selection."""

    def __init__(self, weights, source, query_latent, metadata, cache, shards, *, context, eps, mode,
                 session, name, workspace, buffers, latent_cache, allocator=None):
        rows = source.shape[0]
        expected = {"indexer.wq_b": (4096, 1536), "indexer.wk": (128, 7168),
            "indexer.weights_proj": (32, 7168), "indexer.k_norm.weight": (128,),
            "indexer.k_norm.bias": (128,)}
        if source.shape != (rows, 7168) or query_latent.shape != (rows, 1536):
            raise ValueError("Kimi sparse indexing requires hidden[rows,7168] and normalized query[rows,1536]")
        for field, shape in expected.items():
            if weights[field].shape != shape or weights[field].dtype != torch.bfloat16:
                raise ValueError(f"Kimi sparse index weight {field} requires BF16{shape}")
        if cache.dtype != torch.uint8 or cache.numel() % (64 * 132):
            raise ValueError("Kimi sparse index cache requires planar 64-record uint8 pages")
        storage = SharedBuffers(buffers, source.device, allocator)
        self.query = storage.buffer("query", (rows, 4096), torch.bfloat16)
        self.key = storage.buffer("key", (rows, 128), torch.bfloat16)
        self.normalized_key = storage.buffer("normalized_key", (rows, 128), torch.bfloat16)
        self.head_weights = storage.buffer("weights", (rows, 32), torch.bfloat16)
        self.weights, self.metadata, self.shards = weights, metadata, shards
        self.eps = eps
        self.slots = storage.buffer("slots", (rows,), torch.int64)
        self.projections = tuple(UnquantizedProjection(x, weights[field], output,
            session=session, name=name + "." + field) for field, x, output in (
                ("indexer.wq_b", query_latent, self.query), ("indexer.wk", source, self.key),
                ("indexer.weights_proj", source, self.head_weights)))
        self.query_latent = query_latent
        step = PREFILL_ROWS if mode == "prefill" else rows
        self.bounds = tuple((start, min(start + step, rows)) for start in range(0, rows, step))
        selection_buffers = buffers.setdefault("selection", {})
        cache_pages = cache.view(-1, 64 * 132)
        self.selections = {}
        for start, end in self.bounds:
            rows = end - start
            if rows not in self.selections:
                self.selections[rows] = PreparedIndexSelection(self.query, self.head_weights, cache_pages,
                    metadata, shards, context, selection_buffers, start=start, end=end, mode=mode,
                    session=session, name=f"{name}.slice{start}", workspace=workspace, allocator=allocator,
                    latent_cache=latent_cache)

    def project(self, source):
        self.projections[0](self.query_latent)
        self.projections[1](source)
        self.projections[2](source)
        self.head_weights.mul_(32 ** -.5)
        key = self.key.float()
        centered = key - key.mean(-1, keepdim=True)
        variance = centered.square().mean(-1, keepdim=True)
        self.normalized_key.copy_(centered * torch.rsqrt(variance + self.eps)
            * self.weights["indexer.k_norm.weight"].float() + self.weights["indexer.k_norm.bias"].float())
        self.slots.copy_(self.shards.token_slots(self.metadata.positions, self.metadata.token_page_table))
        self.selections[self.bounds[0][1]].indexer.write_keys(self.normalized_key, self.slots)
