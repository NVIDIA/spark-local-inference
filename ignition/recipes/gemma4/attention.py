# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 sliding-window, full-context, and KV-shared paged attention."""

import math
from typing import NamedTuple
import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.parallel import TensorParallelShard
from ignition.components.tensor_fp8 import TensorFP8Projection, TensorFP8Weight
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy


def prepare_projection(source, weight, output, *, session, name, workspace=None):
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name)
    if isinstance(weight, TensorFP8Weight):
        return TensorFP8Projection(source, weight, output, session=session, name=name)
    return QuantizedProjection(
        source, weight, output, activation_mode="auto", session=session, workspace=workspace, name=name
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def rms_norm_no_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return (x * cos[:, None, :]) + (rotated * sin[:, None, :])


class CacheBindings(NamedTuple):
    slots: torch.Tensor
    lengths: torch.Tensor


def cache_bindings(metadata):
    from .config import PAGE_TOKENS

    m = metadata
    columns = (m.positions.clamp_min(0) // PAGE_TOKENS).clamp(max=m.token_page_table.shape[1] - 1)
    rows = torch.arange(m.positions.numel(), device=m.positions.device)
    pages = m.token_page_table[rows, columns]
    slots = torch.where((m.positions >= 0) & (pages >= 0),
        pages.long() * PAGE_TOKENS + m.positions.remainder(PAGE_TOKENS), -1)
    lengths = torch.where(m.lengths > 0, (m.starts + m.lengths).int(), 0)
    return CacheBindings(slots, lengths)


class PreparedAttention:
    def __init__(
        self,
        layer: int,
        weights: dict,
        config,
        source: torch.Tensor,
        metadata,
        pool,
        *,
        shard: TensorParallelShard = TensorParallelShard(0, 1),
        runtime=None,
        coordinator=None,
        mode: str,
        session,
        name: str,
        workspace,
        examples,
        cache_metadata,
        unit_scale: torch.Tensor,
        k_descale: torch.Tensor,
    ):
        self.layer = layer
        self.config = config
        self.eps = config.rms_norm_eps
        self.is_sliding = config.is_sliding(layer)
        self.shared_kv = config.is_kv_shared(layer)
        self.heads = config.local_heads(shard)
        self.kv_heads = config.local_kv_heads(layer, shard)
        self.dim = config.layer_head_dim(layer)
        tokens = source.shape[0]

        batch = metadata.lengths.numel()
        width = 1 if mode == "prefill" else tokens // batch
        gqa_group = self.heads // self.kv_heads
        q_cap = (
            tokens
            if self.is_sliding or mode == "prefill"
            else batch * max(width, (32 // gqa_group) + 1, 2)
        )

        def zeros(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "attention." + role, shape, dtype, source.device)

        self.q_norm = weights["q_norm"]
        self.q_raw = zeros("q_raw", (tokens, self.heads * self.dim))
        self.q_full = zeros("q_full", (q_cap, self.heads, self.dim))
        self.q = self.q_full[:tokens]
        self.attended = zeros("attended", (tokens, self.heads * self.dim))
        self.partial_output = zeros("partial_output", (tokens, config.hidden))
        self.output = zeros("output", (tokens, config.hidden))
        self.q_proj = prepare_projection(
            source, weights["q_proj"], self.q_raw, session=session, name=name + ".q_proj", workspace=workspace
        )
        self.o_proj = prepare_projection(
            self.attended,
            weights["o_proj"],
            self.partial_output,
            session=session,
            name=name + ".o_proj",
            workspace=workspace,
        )
        self.reduce = PreparedAllReduce(
            self.partial_output,
            self.output,
            runtime=runtime,
            session=session,
            coordinator=coordinator,
            name=name + ".reduce",
        )

        self.k_proj = None
        self.v_proj = None
        if not self.shared_kv:
            self.k_norm = weights["k_norm"]
            self.k_raw = zeros("k_raw", (tokens, self.kv_heads * self.dim))
            self.key = zeros("key", (tokens, self.kv_heads, self.dim))
            self.k_proj = prepare_projection(
                source, weights["k_proj"], self.k_raw, session=session, name=name + ".k_proj", workspace=workspace
            )
            if "v_proj" in weights:
                self.v_raw = zeros("v_raw", (tokens, self.kv_heads * self.dim))
                self.v_proj = prepare_projection(
                    source, weights["v_proj"], self.v_raw, session=session, name=name + ".v_proj", workspace=workspace
                )
            else:
                self.v_raw = self.k_raw

        if self.is_sliding:
            key_cache, value_cache = pool.attention(layer, config, shard=shard)
            self.writer = None
            if not self.shared_kv:
                self.value = zeros("value", (tokens, self.kv_heads, self.dim))
                self.writer = PreparedPagedKVWriter(
                    self.key,
                    self.value,
                    key_cache,
                    value_cache,
                    cache_metadata.slots,
                    session=session,
                    name=name + ".cache_write",
                    k_scale=unit_scale,
                    v_scale=unit_scale,
                    caller_owned_placeholders=True,
                )
            self.attention = PreparedPagedAttention(
                self.q,
                key_cache,
                value_cache,
                metadata.page_table,
                cache_metadata.lengths,
                metadata.query_start_loc,
                self.attended.view(tokens, self.heads, self.dim),
                mode=mode,
                window=config.sliding_window - 1,
                width=width,
                session=session,
                name=name + ".attention",
                workspace=workspace,
                caller_owned_placeholders=True,
                tuning_page_count=int(pool.memory_capacity[1].item()),
                k_descale=k_descale,
                v_descale=unit_scale,
            )
        else:
            key_cache, value_lo_cache, value_hi_cache = pool.attention(layer, config, shard=shard)
            half_dim = self.dim // 2
            self.writer_v = None
            self.writer_k = None
            if not self.shared_kv:
                self.value_lo = zeros("value_lo", (tokens, self.kv_heads, half_dim))
                self.value_hi = zeros("value_hi", (tokens, self.kv_heads, half_dim))
                self.writer_v = PreparedPagedKVWriter(
                    self.value_lo,
                    self.value_hi,
                    value_lo_cache,
                    value_hi_cache,
                    cache_metadata.slots,
                    session=session,
                    name=name + ".cache_write_v",
                    k_scale=unit_scale,
                    v_scale=unit_scale,
                    caller_owned_placeholders=True,
                )
                self.writer_k = PreparedPagedKVWriter(
                    self.key,
                    self.value_hi,
                    key_cache,
                    value_hi_cache,
                    cache_metadata.slots,
                    session=session,
                    name=name + ".cache_write_k",
                    k_scale=unit_scale,
                    v_scale=unit_scale,
                    caller_owned_placeholders=True,
                )
            self.attended_lo_full = zeros("attended_lo_full", (q_cap, self.heads, half_dim))
            self.attended_hi_full = zeros("attended_hi_full", (q_cap, self.heads, half_dim))
            self.attended_lo = self.attended_lo_full[:tokens]
            self.attended_hi = self.attended_hi_full[:tokens]
            self.attention_lo = PreparedPagedAttention(
                self.q_full,
                key_cache,
                value_lo_cache,
                metadata.page_table,
                cache_metadata.lengths,
                metadata.query_start_loc,
                self.attended_lo_full,
                mode="prefill",
                window=-1,
                width=1,
                session=session,
                name=name + ".attention_lo",
                workspace=workspace,
                caller_owned_placeholders=True,
                tuning_page_count=int(pool.memory_capacity[1].item()),
                k_descale=k_descale,
                v_descale=unit_scale,
            )
            self.attention_hi = PreparedPagedAttention(
                self.q_full,
                key_cache,
                value_hi_cache,
                metadata.page_table,
                cache_metadata.lengths,
                metadata.query_start_loc,
                self.attended_hi_full,
                mode="prefill",
                window=-1,
                width=1,
                session=session,
                name=name + ".attention_hi",
                workspace=workspace,
                caller_owned_placeholders=True,
                tuning_page_count=int(pool.memory_capacity[1].item()),
                k_descale=k_descale,
                v_descale=unit_scale,
            )

        self.q_cap = q_cap
        for projection in (self.q_proj, self.o_proj, self.k_proj, self.v_proj):
            if projection is not None:
                projection.functional(session=session)
        self.reduce.functional(session=session)
        for field in ("writer", "writer_v", "writer_k"):
            operation = getattr(self, field, None)
            if operation is not None:
                setattr(self, field, operation.functional())
        for field in ("attention", "attention_lo", "attention_hi"):
            operation = getattr(self, field, None)
            if operation is not None:
                setattr(self, field, operation.functional(session=session, page_table=None,
                    cache_seqlens=None, cu_seqlens_q=None))
        for field in ("q_raw", "q_full", "q", "attended", "partial_output", "output", "k_raw", "key",
                      "v_raw", "value", "value_lo", "value_hi", "attended_lo_full", "attended_hi_full",
                      "attended_lo", "attended_hi"):
            if hasattr(self, field):
                setattr(self, field, None)

    def __call__(self, source, cos, sin, *, metadata, cache_metadata):
        tokens = source.shape[0]
        raw = self.q_proj(source)
        q = rms_norm(raw.view(tokens, self.heads, self.dim), self.q_norm, self.eps)
        query = apply_rotary_pos_emb(q, cos, sin)
        if not self.shared_kv:
            k_raw = self.k_proj(source)
            v_raw = k_raw if self.v_proj is None else self.v_proj(source)
            k = rms_norm(k_raw.view(tokens, self.kv_heads, self.dim), self.k_norm, self.eps)
            key = apply_rotary_pos_emb(k, cos, sin)
            value = rms_norm_no_scale(v_raw.view(tokens, self.kv_heads, self.dim), self.eps)
            if self.is_sliding:
                self.writer(key, value, cache_metadata.slots)
            else:
                half_dim = self.dim // 2
                lo, hi = value[:, :, :half_dim].contiguous(), value[:, :, half_dim:].contiguous()
                self.writer_v(lo, hi, cache_metadata.slots)
                self.writer_k(key, hi, cache_metadata.slots)
        arguments = dict(page_table=metadata.page_table, cache_seqlens=cache_metadata.lengths,
                         cu_seqlens_q=metadata.query_start_loc)
        if self.is_sliding:
            attended = self.attention(query, **arguments)
        else:
            full_query = torch.zeros((self.q_cap, self.heads, self.dim), dtype=query.dtype, device=query.device)
            full_query[:tokens].copy_(query)
            lo = self.attention_lo(full_query, **arguments)[:tokens]
            hi = self.attention_hi(full_query, **arguments)[:tokens]
            attended = torch.cat((lo, hi), dim=-1)
        return self.reduce(self.o_proj(attended.reshape(tokens, -1)))


def attention_descales(config, device):
    unit = torch.ones(1, dtype=torch.float32, device=device)
    sliding_descale = torch.tensor([math.sqrt(config.text["head_dim"])], dtype=torch.float32, device=device)
    full_descale = torch.tensor([math.sqrt(config.text["global_head_dim"])], dtype=torch.float32, device=device)
    return tuple(persistent_copy(value) for value in (unit, sliding_descale, full_descale))
