# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gated NoPE MLA over paged E4M3 latent records, sharded over decode context ranks.

As vLLM's Kimi K3 serving stores its latent cache (--kv-cache-dtype fp8), every record is E4M3 with a
unit scale. Without context ranks a rank attends with its heads over the whole cache. With them
(decode context parallel, vLLM's --decode-context-parallel-size 16), each rank stores the positions it
owns; decode and verification quantize their absorbed queries, gather every rank's heads, attend over
this rank's records with b12x dense MLA and keep this rank's heads by combining the partial outputs by
their LSE (b12x DcpAllToAll). Prefill copies the request's earlier context from every rank's cache into
a slab beside the chunk's own records and attends with this rank's heads.
"""

from typing import NamedTuple

import torch
from b12x.attention import dense_mla

from ignition.components.collectives import PreparedAllGather, PreparedChunkedAllGather, PreparedLseReduceScatter
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.linear import BatchedUnquantizedProjection
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tuning import register_finalizer

from .config import PAGE_TOKENS
from .dcp import RECORD_BYTES, ContextShards, pull_context, write_records
from .pointwise import rms_norm_columns, sigmoid_gate, unpack_and_rms_norm_qkv
from .projections import prepare_projection

FP8 = torch.float8_e4m3fn


class MLAProjectionOutput(NamedTuple):
    query_norm: torch.Tensor
    latent: torch.Tensor
    absorbed: torch.Tensor
    gate: torch.Tensor


class PreparedMLAProjection:
    """Kimi latent projections, optional rotary positions, and absorbed query arithmetic."""

    def __init__(self, weights, source, metadata, cache, *, eps, mode, session, name, workspace, buffers, latents=None,
                 coordinator=None, context=None, peers=None, gated=True, rotary=None):
        tokens, hidden = source.shape
        heads = weights["key"].shape[0]
        device = source.device

        def buffer(key, *shape, dtype=None, rows=None):
            role = 'kimi.hidden_output' if key == 'output' else 'mla.' + key
            return declaration_tensor(buffers, role, (tokens, *shape),
                source.dtype if dtype is None else dtype, device)

        latent_width = weights["query_norm"].shape[0] + RECORD_BYTES
        self.query_norm = buffer("query_norm", latent_width - RECORD_BYTES)
        self.query = buffer("query", heads * 192).view(tokens, heads, 192)
        self.latent = buffer("latent", RECORD_BYTES)
        self.absorbed = buffer("absorbed", heads, RECORD_BYTES)
        self.attended = buffer("attended", heads, 512)
        self.expanded = buffer("expanded", heads * 128)
        self.weights, self.metadata, self.mode, self.cache = weights, metadata, mode, cache
        self.gated, self.rotary = gated, rotary
        self.heads, self.latent_width = heads, latent_width
        self.latent_world = 1 if latents is None else latents.world_size
        self.latent_columns = None if latents is None else latents.query_head_dim
        self.eps = eps
        # Each rank projects only its query and KV latent rows beside its output gate, and the ranks gather the
        # latents, as vLLM's VLLM_KIMI_SHARD_QKV_A does: every rank keeps and reads a sixteenth of the latent
        # projection. A single rank (``latents`` None) projects every latent row itself.
        self.latent_gather = None
        local_width = (weights["latent_gate"].shape[0] if isinstance(weights["latent_gate"], torch.Tensor)
                       else weights["latent_gate"].out_features)
        if latents is None:
            self.local = buffer("local", local_width)
            self.input = self.local[:, :latent_width]
            self.gate = self.local[:, latent_width:]
        else:
            rows, width = latents.max_batch_size, latents.query_head_dim
            self.input = buffer("input", latent_width)
            self.local = buffer("local", local_width, rows=rows)
            self.gate = self.local[:, width:]
            # The exchange reads contiguous rows, which a single row's latent columns already are.
            self.staged = tokens > 1
            self.send = buffer("send", width, rows=rows) if self.staged else self.local[:, :width]
            self.gathered_latents = buffer("gathered_latents", latents.world_size * width, rows=rows)
            self.latent_gather = PreparedChunkedAllGather(self.send, self.gathered_latents, runtime=latents,
                session=session, coordinator=coordinator, name=name + ".gather_latents")
        self.projections = {key: prepare_projection(x, weights[key], y, session=session,
            name=name + "." + key, workspace=workspace)
            for key, x, y in (("latent_gate", source, self.local), ("q_b", self.query_norm, self.query.view(tokens, -1)),
                             ("output", self.expanded, buffer("output", hidden)))}
        self.buffer = buffer
        self.absorb = BatchedUnquantizedProjection(
            self.query[:, :, :128].transpose(0, 1), weights["key"],
            self.absorbed[:, :, :512].transpose(0, 1), major="n", session=session, name=name + ".absorb")
        self.expand = BatchedUnquantizedProjection(
            self.attended.transpose(0, 1), weights["value"],
            self.expanded.view(tokens, heads, 128).transpose(0, 1), major="n", session=session, name=name + ".expand")

    def functional(self, *, session):
        for projection in self.projections.values():
            projection.functional(session=session)
        self.absorb.functional(session=session)
        self.expand.functional(session=session)
        if self.latent_gather is not None:
            self.latent_gather.functional(session=session)
        register_finalizer(session, self._release_projection_examples)
        return self

    def _release_projection_examples(self):
        for name in ("query_norm", "query", "latent", "absorbed", "attended", "expanded",
                     "local", "input", "gate", "send", "gathered_latents", "metadata", "buffer"):
            if hasattr(self, name):
                setattr(self, name, None)

    def project(self, source, metadata=None):
        local = self.projections["latent_gate"](source)
        tokens = source.shape[0]
        query_norm = torch.empty((tokens, self.latent_width - RECORD_BYTES), dtype=source.dtype, device=source.device)
        latent = torch.empty((tokens, RECORD_BYTES), dtype=source.dtype, device=source.device)
        if self.latent_gather is None:
            inputs, gate = local[:, :self.latent_width], local[:, self.latent_width:]
            rms_norm_columns(inputs[:, :-RECORD_BYTES], self.weights["query_norm"], query_norm, self.eps)
            rms_norm_columns(inputs[:, -RECORD_BYTES:-64], self.weights["kv_norm"], latent[:, :512], self.eps)
            latent[:, 512:].copy_(inputs[:, -64:])
        else:
            width, world = self.latent_columns, self.latent_world
            gate = local[:, width:]
            gathered = self.latent_gather(local[:, :width].contiguous()).view(tokens, world, width)
            query = self.latent_width - RECORD_BYTES
            unpack_and_rms_norm_qkv(gathered, query // world, RECORD_BYTES // world,
                self.weights["query_norm"], self.weights["kv_norm"], query_norm, latent, self.eps)
        query = self.projections["q_b"](query_norm).view(tokens, self.heads, 192)
        if self.rotary is not None:
            self.rotary(query[:, :, 128:], latent[:, 512:], metadata.positions)
        absorbed = torch.empty((tokens, self.heads, RECORD_BYTES), dtype=source.dtype, device=source.device)
        self.absorb(query[:, :, :128].transpose(0, 1), absorbed[:, :, :512].transpose(0, 1))
        absorbed[:, :, 512:].copy_(query[:, :, 128:])
        return MLAProjectionOutput(query_norm, latent, absorbed, gate)

    def finish(self, attended, projected):
        tokens = attended.shape[0]
        expanded = torch.empty((tokens, self.heads * 128), dtype=attended.dtype, device=attended.device)
        self.expand(attended.transpose(0, 1), expanded.view(tokens, self.heads, 128).transpose(0, 1))
        if self.gated:
            sigmoid_gate(expanded, projected.gate)
        return self.projections["output"](expanded)


class PreparedMLA(PreparedMLAProjection):
    def __init__(self, weights, source, metadata, cache, *, eps, mode, session, name, workspace, buffers, latents=None,
                 coordinator=None, context=None, peers=None):
        super().__init__(weights, source, metadata, cache, eps=eps, mode=mode, session=session, name=name,
                         workspace=workspace, buffers=buffers, latents=latents, coordinator=coordinator,
                         context=context, peers=peers)
        tokens, heads = source.shape[0], weights["key"].shape[0]
        device, buffer = source.device, self.buffer
        self.shards = ContextShards() if context is None else context.shards
        self.peers = peers
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        world, width = self.shards.world, metadata.page_table.shape[1]
        common = dict(device=device, kv_dtype=FP8, page_size=PAGE_TOKENS, use_cuda_graph=True,
                      partial_dtype=torch.float32)
        scale = 192 ** -.5
        if world == 1:
            # Every rank holds every record: attend with this rank's heads; b12x quantizes the BF16 query.
            self.lengths = torch.zeros_like(metadata.lengths)
            caps = dense_mla.Caps(mode=mode if mode in ("decode", "verify") else "extend", q_dtype=torch.bfloat16,
                num_q_heads=heads, max_total_q=tokens, max_batch=metadata.lengths.numel(),
                max_cache_tokens=width * PAGE_TOKENS, max_page_table_width=width, num_cache_pages=cache.shape[0],
                **common)
            self.attention = PreparedDenseMLA(self.absorbed, cache, self.attended, metadata.page_table, self.lengths,
                metadata.query_start_loc, caps=caps, scale=scale, session=session, name=name + ".attention",
                workspace=workspace)
        elif mode == "prefill":
            self.prefill = PreparedContextPrefill(self.absorbed, self.attended, metadata, context, peers,
                session=session, name=name, workspace=workspace, buffer=buffer, scale=scale, common=common)
        else:
            # Each query row attends to the records this rank owns at or before its position, as its own sequence.
            total, rows = heads * world, context.heads.max_batch_size
            self.quantized = buffer("quantized", heads, RECORD_BYTES, dtype=FP8, rows=rows)
            self.gathered = buffer("gathered", total, RECORD_BYTES, dtype=FP8, rows=rows)
            self.partial = buffer("partial", total, 512, rows=rows)
            self.lse = buffer("lse", total, dtype=torch.float32, rows=rows)
            self.token_pages = buffer("token_pages", width, dtype=torch.int32, rows=rows)
            self.token_lengths = buffer("token_lengths", dtype=torch.int32, rows=rows)
            self.token_starts = persistent_copy(torch.arange(tokens + 1, dtype=torch.int32, device=device))
            self.gather = PreparedAllGather(self.quantized.view(tokens, -1), self.gathered.view(tokens, -1),
                runtime=context.heads, session=session, coordinator=context.coordinator, name=name + ".gather_heads")
            caps = dense_mla.Caps(mode="decode", q_dtype=FP8, num_q_heads=total, max_total_q=tokens,
                max_batch=tokens, max_cache_tokens=width * PAGE_TOKENS, max_page_table_width=width,
                num_cache_pages=cache.shape[0], **common)
            self.attention = PreparedDenseMLA(self.gathered, cache, self.partial, self.token_pages,
                self.token_lengths, self.token_starts, caps=caps, scale=scale, session=session,
                name=name + ".attention", workspace=workspace, lse=self.lse)
            self.combine = PreparedLseReduceScatter(self.partial, self.lse, self.attended, runtime=context.heads,
                session=session, coordinator=context.coordinator, name=name + ".combine")

        self.functional(session=session)
        if hasattr(self, "attention"):
            self.attention.functional(session=session)
        if hasattr(self, "gather"):
            self.gather.functional(session=session)
            self.combine.functional(session=session)
        register_finalizer(session, self._release_attention_examples)

    def _release_attention_examples(self):
        for name in ("slots", "lengths", "quantized", "gathered", "partial", "lse",
                     "token_pages", "token_lengths"):
            if hasattr(self, name):
                setattr(self, name, None)

    def __call__(self, source, metadata=None):
        if metadata is None:
            raise ValueError("MLA execution requires invocation metadata")
        m, shards = metadata, self.shards
        projected = self.project(source, m)
        slots = shards.token_slots(m.positions, m.token_page_table)
        write_records(projected.latent, slots, self.cache.view(-1, RECORD_BYTES))
        if shards.world == 1:
            lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0)
            attended = self.attention(projected.absorbed, page_table=m.page_table,
                lengths=lengths, starts=m.query_start_loc)
        elif self.mode == "prefill":
            attended = self.prefill(projected.latent, projected.absorbed, m)
        else:
            tokens = source.shape[0]
            quantized = projected.absorbed.float().clamp(-448., 448.).to(FP8)
            gathered = self.gather(quantized.view(tokens, -1)).view(tokens, self.heads * shards.world, RECORD_BYTES)
            lengths = shards.local_tokens(m.positions + 1).to(torch.int32)
            partial, lse = self.attention(gathered, page_table=m.token_page_table,
                lengths=lengths, starts=self.token_starts)
            attended = self.combine(partial, lse)
        return self.finish(attended, projected)


class PreparedContextPrefill:
    """One packed prefill request's attention with this rank's heads over its whole context, whose positions
    every rank of the decode context group shares.

    Earlier positions arrive from every rank's cache one window at a time, back from the chunk: the latest
    window and the chunk's own records attend causally, each earlier window without a mask, and the partial
    outputs combine by their log-sum-exp, as vLLM's chunked MLA prefill does. Earlier chunks' records were
    written before this forward's collectives ordered the ranks.
    """

    def __init__(self, absorbed, attended, metadata, context, peers, *, session, name, workspace, buffer, scale,
                 common):
        tokens, heads = absorbed.shape[:2]
        device, world, width = absorbed.device, context.shards.world, metadata.page_table.shape[1]
        slab, window = context.slab, context.window
        pages = slab.shape[0] // PAGE_TOKENS
        if (peers is None or peers.shape != (world,) or slab.shape[0] % PAGE_TOKENS or window % PAGE_TOKENS
                or slab.shape[0] < window + tokens):
            raise ValueError("context-parallel prefill requires every rank's cache address and a context slab")
        self.metadata, self.peers, self.shards, self.attended = metadata, peers, context.shards, attended
        self.absorbed = absorbed
        self.slab, self.window = slab, window
        self.words = slab.view(torch.uint8).view(torch.int32).view(slab.shape[0], RECORD_BYTES // 4)[:window]
        self.windows = -(-width * PAGE_TOKENS * world // window)
        self.offsets = persistent_copy(torch.arange(self.windows, dtype=torch.int64, device=device) * window)
        self.bounds = torch.zeros((self.windows, 2), dtype=torch.int64, device=device)
        self.records = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.lengths = torch.zeros_like(metadata.lengths)
        self.lse = buffer("prefill_lse", heads, dtype=torch.float32)
        cache = slab.view(pages, PAGE_TOKENS, RECORD_BYTES)
        caps = dense_mla.Caps(mode="extend", q_dtype=torch.bfloat16, num_q_heads=heads, max_total_q=tokens,
            max_batch=1, max_cache_tokens=pages * PAGE_TOKENS, max_page_table_width=pages, num_cache_pages=pages,
            **common)
        self.table = persistent_copy(torch.arange(pages, dtype=torch.int32, device=device)[None])
        self.attention = PreparedDenseMLA(absorbed, cache, attended, self.table, self.lengths,
            metadata.query_start_loc, caps=caps, scale=scale, session=session, name=name + ".attention",
            workspace=workspace, lse=self.lse)
        self.context_attention = None
        if self.windows > 1:
            columns = window // PAGE_TOKENS
            self.merged = buffer("merged", heads, 512, dtype=torch.float32)
            self.window_lengths = torch.zeros_like(metadata.lengths)
            self.window_lse = buffer("window_lse", heads, dtype=torch.float32)
            self.window_attended = buffer("window_attended", heads, 512)
            self.window_table = persistent_copy(torch.arange(columns, dtype=torch.int32, device=device)[None])
            caps = dense_mla.Caps(mode="extend", causal=False, q_dtype=torch.bfloat16, num_q_heads=heads,
                max_total_q=tokens, max_batch=1, max_cache_tokens=window, max_page_table_width=columns,
                num_cache_pages=pages, **common)
            self.context_attention = PreparedDenseMLA(absorbed, cache, self.window_attended, self.window_table,
                self.window_lengths, metadata.query_start_loc, caps=caps, scale=scale, session=session,
                name=name + ".context_attention", workspace=workspace, lse=self.window_lse)

        self.slab_layout = tuple(slab.shape), slab.dtype
        self.attention.functional(session=session, cache=None)
        if self.context_attention is not None:
            self.context_attention.functional(session=session, cache=None)
        register_finalizer(session, self._release_examples)

    def _release_examples(self):
        for name in ("metadata", "attended", "absorbed", "slab", "words", "bounds", "records", "lengths", "lse",
                     "merged", "window_lengths", "window_lse", "window_attended"):
            if hasattr(self, name):
                setattr(self, name, None)

    def __call__(self, latent, absorbed, metadata):
        m, table = metadata, metadata.page_table[0]
        shape, dtype = self.slab_layout
        slab = torch.empty(shape, dtype=dtype, device=latent.device)
        cache = slab.view(-1, PAGE_TOKENS, RECORD_BYTES)
        words = slab.view(torch.uint8).view(torch.int32).view(shape[0], RECORD_BYTES // 4)[:self.window]
        ends = m.starts[:1].to(torch.int64) - self.offsets
        bounds = torch.stack(((ends - self.window).clamp(min=0), ends.clamp(min=0)), -1)
        pull_context(self.peers, table, bounds[0], words, self.shards)
        latest = bounds[0, 0]
        records = torch.where(m.positions >= 0, m.positions - latest, -1)
        write_records(latent, records, slab)
        lengths = torch.where(m.lengths > 0, m.starts - latest + m.lengths, 0)
        attended, lse = self.attention(absorbed, cache=cache, page_table=self.table,
            lengths=lengths, starts=m.query_start_loc)
        if self.context_attention is None:
            return attended
        merged = attended.float()
        for index in range(1, self.windows):
            window_bounds = bounds[index]
            pull_context(self.peers, table, window_bounds, words, self.shards)
            present = window_bounds[1] > window_bounds[0]
            window_lengths = (window_bounds[1] - window_bounds[0]).to(m.lengths.dtype).expand_as(m.lengths).contiguous()
            window_attended, window_lse = self.context_attention(absorbed, cache=cache,
                page_table=self.window_table, lengths=window_lengths, starts=m.query_start_loc)
            peak = torch.maximum(lse, torch.where(present, window_lse, -torch.inf)).clamp(min=-1e30)
            kept = torch.exp(lse - peak)
            added = torch.where(present, torch.exp(window_lse - peak), 0.)
            total = (kept + added).clamp(min=1e-30)
            merged = (merged * kept[..., None] + torch.where(present, window_attended.float(),
                0.) * added[..., None]) / total[..., None]
            lse = peak + torch.log(total)
        return merged.to(absorbed.dtype)
