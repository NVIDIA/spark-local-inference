# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gated sparse MLA over globally selected interleaved FP8 latent records."""

from typing import NamedTuple

import torch
from b12x.attention import dense_mla

from ignition.components.collectives import ConcurrentAllGather, PreparedAllGather, PreparedLseReduceScatter
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.peer_record_gather import PreparedPeerRecordGather
from ignition.components.interleaved_history import PreparedInterleavedHistory
from ignition.recipes.kimi3.attention import FP8, MLAProjectionOutput, PreparedMLAProjection
from ignition.recipes.kimi3.dcp import ContextShards

from .dcp import RECORD_BYTES, TOPK, selection_slots, write_records
from .indexer import PreparedDSAIndexer, SharedBuffers
from ignition.preparation.tuning import register_finalizer


class _AttentionStage(NamedTuple):
    quantized: torch.Tensor | None
    gathered: torch.Tensor | None
    partial: torch.Tensor | None
    lse: torch.Tensor | None
    selected: torch.Tensor
    selected_scores: torch.Tensor | None
    owners: torch.Tensor
    slots: torch.Tensor
    counts: torch.Tensor
    index: tuple | None


class _AttentionResult(NamedTuple):
    output: torch.Tensor
    projected: MLAProjectionOutput
    attended: torch.Tensor
    last_stage: _AttentionStage


class PreparedDSA(PreparedMLAProjection):
    """Kimi latent projections with per-token global top-2048 sparse attention.

    Decode and verification gather absorbed query heads and combine local selected-record
    partials by their LSE. Prefill keeps local heads and gathers only each
    128-query slice's selected records into a reusable one-record-page slab.
    """

    def __init__(self, weights, source, metadata, cache, *, index_cache, eps, mode, session, name, workspace,
                 buffers, latents=None, coordinator=None, context=None, peers=None):
        if mode not in ("decode", "verify", "prefill"):
            raise ValueError("Kimi K3.1 sparse attention requires decode, verify, or prefill")
        super().__init__(weights, source, metadata, cache, eps=eps, mode=mode, session=session, name=name,
            workspace=workspace, buffers=buffers, latents=latents, coordinator=coordinator,
            context=context, peers=peers)
        self.shards = ContextShards() if context is None else context.shards
        if self.shards.world not in (8, 16):
            raise ValueError("Kimi K3.1 sparse attention requires DCP8 or DCP16 context ranks")
        if index_cache.shape[0] != 2 * cache.shape[0]:
            raise ValueError("Kimi sparse index cache requires two 64-record subpages per latent page")
        self.indexer = PreparedDSAIndexer(weights, source, self.query_norm, metadata, index_cache, self.shards,
            context=context, eps=eps, mode=mode, session=session, name=name + ".index", workspace=workspace,
            buffers=buffers.setdefault("index", {}), latent_cache=cache)
        self.slots = self.buffer("cache_slots", dtype=torch.int64)
        self.mode = mode
        self.history = None
        if mode == "prefill" and source.shape[0] >= 4096:
            if context.staging is None or context.staged_records is None or metadata.starts.numel() != 1:
                raise ValueError("large sparse prefill requires fixed staging for one request's history")
            history_storage = SharedBuffers(buffers.setdefault("history", {}), source.device)
            self.history_end = history_storage.buffer("end", (1,), torch.int64)
            self.history_records = context.staged_records
            self.history = PreparedInterleavedHistory(cache, metadata.page_table[0], self.history_end,
                self.shards.index, context.staging.tensor.view(-1, RECORD_BYTES), context.staging.addresses,
                self.history_records, context.stage_signals, runtime=context.stage_runtime,
                session=session, coordinator=coordinator, name=name + ".history")
        self.parts = {}
        storage = SharedBuffers(buffers.setdefault("selected_attention", {}), source.device)
        heads = self.absorbed.shape[1]
        common = dict(device=source.device, kv_dtype=FP8, page_size=1, use_cuda_graph=True,
                      partial_dtype=torch.float32)
        constants = context.attention_constants
        cache_records = cache.view(-1, 1, RECORD_BYTES)
        for start, end in self.indexer.bounds:
            rows = end - start
            if rows in self.parts:
                continue
            selector = None if self.indexer.sharded is not None else self.indexer.selections[rows]
            owners = storage.buffer("owners", (rows, TOPK), torch.int32)
            slots = storage.buffer("slots", (rows, TOPK), torch.int32)
            counts = storage.buffer("counts", (rows,), torch.int32)
            starts, pages = constants.starts[:rows + 1], constants.pages[:rows]
            part = dict(selector=selector, owners=owners, slots=slots, counts=counts, starts=starts, pages=pages)
            if mode == "prefill":
                if peers is None or peers.shape != (self.shards.world,):
                    raise ValueError("Kimi sparse prefill requires every rank's latent-cache address")
                selected = storage.buffer("records", (rows, TOPK, RECORD_BYTES), FP8)
                local_history = self.history is not None
                part["gather"] = PreparedPeerRecordGather(
                    torch.tensor([self.history_records.data_ptr()], dtype=torch.int64, device=source.device)
                    if local_history else peers,
                    owners, slots, counts, selected,
                    peer_records=self.history_records.shape[0] if local_history else cache.shape[0] * cache.shape[1],
                    session=session, name=f"{name}.records{start}",
                    dependencies=(self.history_records, self.history.completed) if local_history else ())
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
        self.query_overlap = None
        if mode == "decode" and self.absorbed.shape[0] == 1 and workspace is not None and workspace.overlap:
            part, = self.parts.values()
            # Both candidate transports consume the joined query before reusing the channel.
            part["gather"].dependencies.append(cache)
            part["selector"].merge.dependencies.append(part["gather"].output)
            self.query_overlap = ConcurrentAllGather(part["gather"], workspace=workspace, session=session,
                                                     parallel_plans=self.indexer.local_plans)

        self.functional(session=session)
        if self.history is not None:
            self.history.functional()
        for part in self.parts.values():
            part["gather"].functional(session=session)
            if mode == "prefill":
                part["attention"].functional(session=session, cache=None)
            else:
                part["attention"].functional(session=session)
                part["combine"].functional(session=session)
        if self.query_overlap is not None:
            self.query_overlap.functional(session=session)
        register_finalizer(session, self._release_dsa_examples)

    def _release_dsa_examples(self):
        self.slots = self.history_end = self.history_records = None
        for part in self.parts.values():
            for name in ("owners", "slots", "counts", "quantized", "gathered", "partial", "lse"):
                part.pop(name, None)
            if self.history is not None:
                part["gather"].peers = None

    def __call__(self, source, metadata=None, *, shared_index=None):
        return self._run(source, metadata, shared_index=shared_index).output

    def _run(self, source, metadata, *, shared_index=None):
        if metadata is None:
            raise ValueError("sparse MLA execution requires invocation metadata")
        projected = self.project(source, metadata)
        slots = (self.shards.token_slots(metadata.positions, metadata.token_page_table)
                 if shared_index is None else shared_index.slots)
        write_records(projected.latent, slots, self.cache.view(-1, RECORD_BYTES))
        pending = None
        if self.query_overlap is not None:
            quantized = projected.absorbed.float().clamp(-448., 448.).to(FP8)
            pending = self.query_overlap.start(quantized.view(source.shape[0], -1), dependencies=(self.cache,))
        query, weights = (self.indexer.project(source, projected.query_norm, metadata)
                          if shared_index is None else
                          self.indexer.project(source, projected.query_norm, metadata, slots=slots))
        selected_all = None if self.indexer.sharded is None else self.indexer.sharded(query, weights, metadata)
        history = None
        if self.history is not None:
            end = (metadata.starts + metadata.lengths) * (metadata.lengths > 0)
            history = self.history(metadata.page_table[0], end)
        attended = torch.empty((source.shape[0], self.heads, 512), dtype=source.dtype, device=source.device)
        for start, end in self.indexer.bounds:
            rows = end - start
            part = self.parts[rows]
            positions, tables = metadata.positions[start:end], metadata.token_page_table[start:end]
            gathered = partial = lse = index = selected_scores = None
            if self.mode == "prefill":
                quantized = None
            if selected_all is not None:
                selected = selected_all[start:end]
            else:
                selector = part["selector"]
                index = (selector.prepare_local_result(query[start:end], weights[start:end], positions, tables)
                         if shared_index is None else
                         selector.prepare_local_result(query[start:end], weights[start:end], positions, tables,
                                                       shared_index=shared_index))
                packed = index[0]
                if pending is not None:
                    gathered = self.query_overlap.finish(pending, packed)
                selected, selected_scores = selector.exchange_result(packed, positions,
                    dependencies=() if gathered is None else (gathered,))
            owners = torch.empty((rows, TOPK), dtype=torch.int32, device=source.device)
            slots = torch.empty_like(owners)
            counts = torch.empty((rows,), dtype=torch.int32, device=source.device)
            selection_slots(selected, positions, tables, owners, slots, counts, self.shards,
                self.cache.shape[0], local=self.mode != "prefill", global_records=history is not None)
            if self.mode == "prefill":
                records = part["gather"](owners, slots, counts, records=history,
                    dependencies=() if history is None else (self.history.completed,))
                output = part["attention"](projected.absorbed[start:end],
                    cache=records.view(-1, 1, RECORD_BYTES), page_table=part["pages"],
                    lengths=counts, starts=part["starts"])
            else:
                if gathered is None:
                    quantized = projected.absorbed[start:end].float().clamp(-448., 448.).to(FP8)
                    gathered = part["gather"](quantized.view(rows, -1), dependencies=(self.cache,))
                partial, lse = part["attention"](gathered.view(rows, self.heads * self.shards.world, RECORD_BYTES),
                    page_table=slots, lengths=counts, starts=part["starts"])
                output = part["combine"](partial, lse)
            attended[start:end].copy_(output)
        last_stage = _AttentionStage(quantized, gathered, partial, lse, selected,
            selected_scores, owners, slots, counts, index)
        return _AttentionResult(self.finish(attended, projected), projected, attended, last_stage)
