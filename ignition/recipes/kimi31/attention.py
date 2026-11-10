# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gated sparse MLA over globally selected interleaved FP8 latent records."""

import torch
from b12x.attention import dense_mla

from ignition.components.collectives import PreparedAllGather, PreparedLseReduceScatter
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.peer_record_gather import PreparedPeerRecordGather
from ignition.recipes.kimi3.attention import FP8, PreparedMLAProjection
from ignition.recipes.kimi3.dcp import ContextShards

from .dcp import RECORD_BYTES, TOPK, selection_slots, write_records
from .indexer import PreparedDSAIndexer, SharedBuffers


class PreparedDSA(PreparedMLAProjection):
    """Kimi latent projections with per-token global top-2048 sparse attention.

    Decode and verification gather absorbed query heads and combine local selected-record
    partials by their LSE. Prefill keeps local heads and gathers only each
    128-query slice's selected records into a reusable one-record-page slab.
    """

    def __init__(self, weights, source, metadata, cache, *, index_cache, eps, mode, session, name, workspace,
                 buffers, latents=None, coordinator=None, context=None, peers=None, allocator=None):
        if mode not in ("decode", "verify", "prefill"):
            raise ValueError("Kimi K3.1 sparse attention requires decode, verify, or prefill")
        super().__init__(weights, source, metadata, cache, eps=eps, mode=mode, session=session, name=name,
            workspace=workspace, buffers=buffers, latents=latents, coordinator=coordinator,
            context=context, peers=peers, allocator=allocator)
        self.shards = ContextShards() if context is None else context.shards
        if self.shards.world != 16:
            raise ValueError("Kimi K3.1 sparse attention requires sixteen context ranks")
        if index_cache.shape[0] != 2 * cache.shape[0]:
            raise ValueError("Kimi sparse index cache requires two 64-record subpages per latent page")
        self.indexer = PreparedDSAIndexer(weights, source, self.query_norm, metadata, index_cache, self.shards,
            context=context, eps=eps, mode=mode, session=session, name=name + ".index", workspace=workspace,
            buffers=buffers.setdefault("index", {}), latent_cache=cache, allocator=allocator)
        self.slots = self.buffer("cache_slots", dtype=torch.int64)
        self.mode = mode
        self.parts = {}
        storage = SharedBuffers(buffers.setdefault("selected_attention", {}), source.device, allocator)
        heads = self.absorbed.shape[1]
        common = dict(device=source.device, kv_dtype=FP8, page_size=1, use_cuda_graph=True,
                      partial_dtype=torch.float32)
        constants = context.attention_constants
        cache_records = cache.view(-1, 1, RECORD_BYTES)
        for start, end in self.indexer.bounds:
            rows = end - start
            if rows in self.parts:
                continue
            selector = self.indexer.selections[rows]
            owners = storage.buffer("owners", (rows, TOPK), torch.int32)
            slots = storage.buffer("slots", (rows, TOPK), torch.int32)
            counts = storage.buffer("counts", (rows,), torch.int32)
            starts, pages = constants.starts[:rows + 1], constants.pages[:rows]
            part = dict(selector=selector, owners=owners, slots=slots, counts=counts)
            if mode == "prefill":
                if peers is None or peers.shape != (self.shards.world,):
                    raise ValueError("Kimi sparse prefill requires every rank's latent-cache address")
                selected = storage.buffer("records", (rows, TOPK, RECORD_BYTES), FP8)
                part["gather"] = PreparedPeerRecordGather(peers, owners, slots, counts, selected,
                    peer_records=cache.shape[0] * cache.shape[1], session=session, name=f"{name}.records{start}")
                caps = dense_mla.Caps(mode="decode", q_dtype=torch.bfloat16, num_q_heads=heads,
                    max_total_q=rows, max_batch=rows, max_cache_tokens=TOPK, max_page_table_width=TOPK,
                    num_cache_pages=rows * TOPK, **common)
                part["attention"] = PreparedDenseMLA(self.absorbed[start:end], selected.view(-1, 1, RECORD_BYTES),
                    self.attended[start:end], pages, counts, starts, caps=caps, scale=192 ** -.5,
                    session=session, name=f"{name}.attention{start}", workspace=workspace,
                    kv_scale=constants.scale, q_scale=constants.scale)
            else:
                total = heads * self.shards.world
                quantized = storage.buffer("query", (rows, heads, RECORD_BYTES), FP8)
                gathered = storage.buffer("gathered_query", (rows, total, RECORD_BYTES), FP8)
                partial = storage.buffer("partial", (rows, total, 512), torch.bfloat16)
                lse = storage.buffer("lse", (rows, total), torch.float32)
                part.update(quantized=quantized, gathered=gathered, partial=partial, lse=lse)
                part["gather"] = PreparedAllGather(quantized.view(rows, -1), gathered.view(rows, -1),
                    runtime=context.heads, session=session, coordinator=context.coordinator,
                    name=name + ".gather_heads")
                caps = dense_mla.Caps(mode="decode", q_dtype=FP8, num_q_heads=total,
                    max_total_q=rows, max_batch=rows, max_cache_tokens=TOPK, max_page_table_width=TOPK,
                    num_cache_pages=cache.shape[0] * cache.shape[1], **common)
                part["attention"] = PreparedDenseMLA(gathered, cache_records, partial,
                    slots, counts, starts, caps=caps, scale=192 ** -.5, session=session,
                    name=name + ".attention", workspace=workspace, lse=lse,
                    kv_scale=constants.scale, q_scale=constants.scale)
                part["combine"] = PreparedLseReduceScatter(partial, lse, self.attended,
                    runtime=context.heads, session=session, coordinator=context.coordinator, name=name + ".combine",
                    threads=512, block_limit=8)
            self.parts[rows] = part

    def __call__(self, source):
        self.project(source)
        self.slots.copy_(self.shards.token_slots(self.metadata.positions, self.metadata.token_page_table))
        write_records(self.latent, self.slots, self.cache.view(-1, RECORD_BYTES))
        self.indexer.project(source)
        for start, end in self.indexer.bounds:
            part = self.parts[end - start]
            positions = self.metadata.positions[start:end]
            tables = self.metadata.token_page_table[start:end]
            selected = part["selector"](self.indexer.query[start:end], self.indexer.head_weights[start:end],
                positions, tables)
            selection_slots(selected, positions, tables, part["owners"], part["slots"],
                part["counts"], self.shards, self.cache.shape[0], local=self.mode != "prefill")
            if self.mode == "prefill":
                part["gather"]()
                part["attention"](self.absorbed[start:end], self.attended[start:end])
            else:
                part["quantized"].copy_(self.absorbed.float().clamp(-448., 448.).to(FP8))
                part["gather"](part["quantized"].view(self.absorbed.shape[0], -1))
                part["attention"](part["gathered"], part["partial"])
                part["combine"](part["partial"], part["lse"])
        return self.finish()
