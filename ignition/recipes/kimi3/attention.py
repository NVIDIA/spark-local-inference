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

import torch
from b12x.attention import dense_mla

from ignition.components.collectives import PreparedAllGather, PreparedChunkedAllGather, PreparedLseReduceScatter
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.preparation.examples import declaration_tensor

from .config import PAGE_TOKENS
from .dcp import RECORD_BYTES, ContextShards, pull_context, write_records
from .pointwise import rms_norm_columns, sigmoid_gate
from .projections import prepare_projection

FP8 = torch.float8_e4m3fn


class PreparedMLA:
    def __init__(self, weights, source, metadata, cache, *, mode, session, name, workspace, latents=None,
                 coordinator=None, context=None, peers=None, examples=None):
        tokens, hidden = source.shape
        heads, device = weights["key"].shape[0], source.device
        def buffer(role, *shape, dtype=None):
            return declaration_tensor(examples, "mla." + role, shape,
                                      source.dtype if dtype is None else dtype, device)
        latent_width = weights["query_norm"].shape[0] + RECORD_BYTES
        local_width = (weights["latent_gate"].shape[0] if isinstance(weights["latent_gate"], torch.Tensor)
                       else weights["latent_gate"].out_features)
        local, query_norm = buffer("local", tokens, local_width), buffer("query_norm", tokens, latent_width - RECORD_BYTES)
        query = buffer("query", tokens, heads, 192)
        absorbed, attended = buffer("absorbed", tokens, heads, RECORD_BYTES), buffer("attended", tokens, heads, 512)
        expanded = buffer("expanded", tokens, heads * 128)
        self.weights, self.mode, self.cache = weights, mode, cache
        self.heads, self.latent_width = heads, latent_width
        self.latent_gather = None
        self.gather_width = None if latents is None else latents.query_head_dim
        if latents is not None:
            send = buffer("latent_send", tokens, self.gather_width)
            gathered = buffer("latent_gathered", tokens, latents.world_size * self.gather_width)
            self.latent_gather = PreparedChunkedAllGather(send, gathered, runtime=latents,
                session=session, coordinator=coordinator, name=name + ".gather_latents").functional(session=session)
        self.projections = {key: prepare_projection(x, weights[key], y, session=session,
            name=name + "." + key, workspace=workspace).functional(session=session)
            for key, x, y in (("latent_gate", source, local), ("q_b", query_norm, query.view(tokens, -1)),
                             ("output", expanded, buffer("output", tokens, hidden)))}
        self.shards = ContextShards() if context is None else context.shards
        world, width = self.shards.world, metadata.page_table.shape[1]
        common = dict(device=device, kv_dtype=FP8, page_size=PAGE_TOKENS, use_cuda_graph=True,
                      partial_dtype=torch.float32)
        scale = 192 ** -.5
        if world == 1:
            caps = dense_mla.Caps(mode=mode if mode in ("decode", "verify") else "extend", q_dtype=torch.bfloat16,
                num_q_heads=heads, max_total_q=tokens, max_batch=metadata.lengths.numel(),
                max_cache_tokens=width * PAGE_TOKENS, max_page_table_width=width, num_cache_pages=cache.shape[0],
                **common)
            self.attention = PreparedDenseMLA(absorbed, cache, attended, metadata.page_table,
                torch.zeros_like(metadata.lengths), metadata.query_start_loc, caps=caps, scale=scale,
                session=session, name=name + ".attention", workspace=workspace).functional(session=session)
        elif mode == "prefill":
            self.prefill = PreparedContextPrefill(absorbed, attended, metadata, context, peers,
                session=session, name=name, workspace=workspace, scale=scale, common=common, examples=examples)
        else:
            total = heads * world
            quantized, gathered = buffer("quantized", tokens, heads, RECORD_BYTES, dtype=FP8), buffer("gathered_heads", tokens, total, RECORD_BYTES, dtype=FP8)
            partial, lse = buffer("partial", tokens, total, 512), buffer("lse", tokens, total, dtype=torch.float32)
            self.gather = PreparedAllGather(quantized.view(tokens, -1), gathered.view(tokens, -1),
                runtime=context.heads, session=session, coordinator=context.coordinator,
                name=name + ".gather_heads").functional(session=session)
            caps = dense_mla.Caps(mode="decode", q_dtype=FP8, num_q_heads=total, max_total_q=tokens,
                max_batch=tokens, max_cache_tokens=width * PAGE_TOKENS, max_page_table_width=width,
                num_cache_pages=cache.shape[0], **common)
            self.attention = PreparedDenseMLA(gathered, cache, partial, metadata.token_page_table,
                torch.zeros(tokens, dtype=torch.int32, device=device), torch.arange(tokens + 1, dtype=torch.int32, device=device),
                caps=caps, scale=scale, session=session, name=name + ".attention", workspace=workspace,
                lse=lse).functional(session=session)
            self.combine = PreparedLseReduceScatter(partial, lse, attended, runtime=context.heads,
                session=session, coordinator=context.coordinator, name=name + ".combine").functional(session=session)

    def __call__(self, source, metadata):
        p, m, shards = self.projections, metadata, self.shards
        tokens, heads = source.shape[0], self.heads
        local = p["latent_gate"](source)
        if self.latent_gather is None:
            inputs, gate = local[:, :self.latent_width], local[:, self.latent_width:]
        else:
            width = self.gather_width
            gathered = self.latent_gather(local[:, :width].contiguous()).view(tokens, -1, width)
            world = gathered.shape[1]
            query = self.latent_width - RECORD_BYTES
            q, kv = query // world, RECORD_BYTES // world
            inputs = torch.cat((gathered[:, :, :q].reshape(tokens, query),
                                gathered[:, :, q:q + kv].reshape(tokens, RECORD_BYTES)), -1)
            gate = local[:, width:]
        query_norm = torch.empty((tokens, self.latent_width - RECORD_BYTES), dtype=source.dtype, device=source.device)
        rms_norm_columns(inputs[:, :-RECORD_BYTES], self.weights["query_norm"], query_norm, 1e-6)
        latent = torch.empty((tokens, RECORD_BYTES), dtype=source.dtype, device=source.device)
        rms_norm_columns(inputs[:, -RECORD_BYTES:-64], self.weights["kv_norm"], latent[:, :512], 1e-6)
        latent[:, 512:].copy_(inputs[:, -64:])
        query = p["q_b"](query_norm).view(tokens, heads, 192)
        absorbed = torch.empty((tokens, heads, RECORD_BYTES), dtype=source.dtype, device=source.device)
        absorbed[:, :, :512].copy_(torch.bmm(query[:, :, :128].transpose(0, 1), self.weights["key"]).transpose(0, 1))
        absorbed[:, :, 512:].copy_(query[:, :, 128:])
        slots = shards.token_slots(m.positions, m.token_page_table)
        write_records(latent, slots, self.cache.view(-1, RECORD_BYTES))
        if shards.world == 1:
            lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0).to(torch.int32)
            attended = self.attention(absorbed, m.page_table, lengths, m.query_start_loc)
        elif self.mode == "prefill":
            attended = self.prefill(absorbed, latent, m)
        else:
            quantized = absorbed.float().clamp(-448., 448.).to(FP8)
            gathered = self.gather(quantized.view(tokens, -1)).view(tokens, -1, RECORD_BYTES)
            lengths = shards.local_tokens(m.positions + 1).to(torch.int32)
            starts = torch.arange(tokens + 1, dtype=torch.int32, device=source.device)
            partial, lse = self.attention(gathered, m.token_page_table, lengths, starts)
            attended = self.combine(partial, lse)
        expanded = torch.bmm(attended.transpose(0, 1), self.weights["value"]).transpose(0, 1).flatten(1).contiguous()
        sigmoid_gate(expanded, gate)
        return p["output"](expanded)


class PreparedContextPrefill:
    """One packed prefill request's attention with this rank's heads over its whole context, whose positions
    every rank of the decode context group shares.

    Earlier positions arrive from every rank's cache one window at a time, back from the chunk: the latest
    window and the chunk's own records attend causally, each earlier window without a mask, and the partial
    outputs combine by their log-sum-exp, as vLLM's chunked MLA prefill does. Earlier chunks' records were
    written before this forward's collectives ordered the ranks.
    """

    def __init__(self, absorbed, attended, metadata, context, peers, *, session, name, workspace, scale, common, examples=None):
        tokens, heads = absorbed.shape[:2]
        device, world, width = absorbed.device, context.shards.world, metadata.page_table.shape[1]
        slab, window = context.slab, context.window
        pages = slab.shape[0] // PAGE_TOKENS
        if (peers is None or peers.shape != (world,) or slab.shape[0] % PAGE_TOKENS or window % PAGE_TOKENS
                or slab.shape[0] < window + tokens):
            raise ValueError("context-parallel prefill requires every rank's cache address and a context slab")
        self.peers, self.shards = peers, context.shards
        self.slab_shape, self.slab_dtype, self.window = tuple(slab.shape), slab.dtype, window
        self.windows = -(-width * PAGE_TOKENS * world // window)
        cache = slab.view(pages, PAGE_TOKENS, RECORD_BYTES)
        caps = dense_mla.Caps(mode="extend", q_dtype=torch.bfloat16, num_q_heads=heads, max_total_q=tokens,
            max_batch=1, max_cache_tokens=pages * PAGE_TOKENS, max_page_table_width=pages, num_cache_pages=pages,
            **common)
        table = torch.arange(pages, dtype=torch.int32, device=device)[None]
        lengths = torch.zeros_like(metadata.lengths)
        lse = declaration_tensor(examples, "mla.prefill.lse", (tokens, heads), torch.float32, device)
        self.attention = PreparedDenseMLA(absorbed, cache, attended, table, lengths,
            metadata.query_start_loc, caps=caps, scale=scale, session=session, name=name + ".attention",
            workspace=workspace, lse=lse).functional(session=session, cache=None)
        self.context_attention = None
        if self.windows > 1:
            columns = window // PAGE_TOKENS
            window_table = torch.arange(columns, dtype=torch.int32, device=device)[None]
            caps = dense_mla.Caps(mode="extend", causal=False, q_dtype=torch.bfloat16, num_q_heads=heads,
                max_total_q=tokens, max_batch=1, max_cache_tokens=window, max_page_table_width=columns,
                num_cache_pages=pages, **common)
            self.context_attention = PreparedDenseMLA(absorbed, cache, attended, window_table,
                lengths, metadata.query_start_loc, caps=caps, scale=scale, session=session,
                name=name + ".context_attention", workspace=workspace, lse=lse).functional(session=session, cache=None)

    def __call__(self, absorbed, latent, metadata):
        m, table = metadata, metadata.page_table[0]
        slab = torch.empty(self.slab_shape, dtype=self.slab_dtype, device=latent.device)
        words = slab.view(torch.uint8).view(torch.int32).view(slab.shape[0], RECORD_BYTES // 4)[:self.window]
        offsets = torch.arange(self.windows, dtype=torch.int64, device=latent.device) * self.window
        ends = m.starts[:1].to(torch.int64) - offsets
        bounds = torch.stack(((ends - self.window).clamp(min=0), ends.clamp(min=0)), -1)
        pull_context(self.peers, table, bounds[0], words, self.shards)
        latest = bounds[0, 0]
        records = torch.where(m.positions >= 0, m.positions - latest, -1)
        write_records(latent, records, slab)
        lengths = torch.where(m.lengths > 0, m.starts - latest + m.lengths, 0).to(torch.int32)
        pages = slab.shape[0] // PAGE_TOKENS
        cache = slab.view(pages, PAGE_TOKENS, RECORD_BYTES)
        page_table = torch.arange(pages, dtype=torch.int32, device=latent.device)[None]
        attended, lse = self.attention(absorbed, page_table, lengths, m.query_start_loc, cache=cache)
        if self.context_attention is None:
            return attended
        merged = attended.float()
        window_table = torch.arange(self.window // PAGE_TOKENS, dtype=torch.int32, device=latent.device)[None]
        for index in range(1, self.windows):
            current = bounds[index]
            pull_context(self.peers, table, current, words, self.shards)
            present = current[1] > current[0]
            window_lengths = (current[1] - current[0]).to(torch.int32).expand_as(m.lengths).clone(
                memory_format=torch.contiguous_format)
            window_attended, window_lse = self.context_attention(absorbed, window_table, window_lengths,
                                                                m.query_start_loc, cache=cache)
            peak = torch.maximum(lse, torch.where(present, window_lse, -torch.inf)).clamp(min=-1e30)
            kept = torch.exp(lse - peak)
            added = torch.where(present, torch.exp(window_lse - peak), 0.)
            total = (kept + added).clamp(min=1e-30)
            merged = (merged * kept[..., None] + torch.where(present, window_attended.float(),
                                                            0.) * added[..., None]) / total[..., None]
            lse = peak + torch.log(total)
        return merged.to(attended.dtype)
