# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized gated attention with paged FP8 and accepted circular histories."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.linear import UnquantizedProjection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.preparation.persistent import persistent_copy
from ignition.components.window_attention import stage_windows, commit_window

from .pool import KV_DTYPE, PAGE_TOKENS


@dataclass(frozen=True)
class AttentionScratch:
    projected: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    gated: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    window_key: torch.Tensor
    window_value: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, device, *, requests=1):
        geometries = [config.attention(i) for i in range(config.layers)]
        if len({(a.heads, a.kv_heads, a.key_dim, a.value_dim) for a in geometries}) != 1:
            raise ValueError("Muse Spark attention scratch requires matching global and sliding head geometry")
        geometry = geometries[0]
        heads, kv_heads = geometry.local(shard)
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        window = max(a.window for a in geometries)
        pages = max((tokens + max(window, 0) + PAGE_TOKENS - 1) // PAGE_TOKENS,
                    requests * ((max(window, 0) + 8 + PAGE_TOKENS - 1) // PAGE_TOKENS))
        return cls(zeros(tokens, 2 * heads * geometry.key_dim + kv_heads * (geometry.key_dim + geometry.value_dim)),
            zeros(tokens, heads, geometry.key_dim), zeros(tokens, kv_heads, geometry.key_dim),
            zeros(tokens, kv_heads, geometry.value_dim), zeros(tokens, heads, geometry.value_dim),
            zeros(tokens, heads, geometry.value_dim),
            zeros(tokens, config.text["hidden_size"]), zeros(tokens, config.text["hidden_size"]),
            zeros(pages, PAGE_TOKENS, kv_heads, geometry.key_dim, dtype=KV_DTYPE),
            zeros(pages, PAGE_TOKENS, kv_heads, geometry.value_dim, dtype=KV_DTYPE))

    def rows(self, tokens):
        return type(self)(*(getattr(self, name) if name.startswith("window_") else getattr(self, name)[:tokens]
                            for name in self.__dataclass_fields__))


def head_norm(values, eps):
    value = values.float()
    return (value * torch.pow(value.square().mean(-1, keepdim=True) + eps, -0.5)).to(values.dtype)


class PreparedAttention(torch.nn.Module):
    def __init__(self, layer, weights, config, shard, source, scratch, metadata, pool, *, mode,
                 runtime, coordinator, session, name, workspace):
        super().__init__()
        self.geometry = config.attention(layer)
        self.query_shape, self.key_shape, self.value_shape = (tuple(value.shape) for value in (scratch.query, scratch.key, scratch.value))
        if mode not in ("prefill", "decode", "verify"):
            raise ValueError(f"unsupported Muse Spark attention mode: {mode}")
        self.mode, self.pool, self.layer = mode, pool, layer
        self.projection = UnquantizedProjection(source, weights.qkv, scratch.projected,
            session=session, name=name + ".qkv_gate").functional(session=session)
        self.output_projection = UnquantizedProjection(scratch.gated.flatten(1), weights.output, scratch.partial,
            session=session, name=name + ".output").functional(session=session)
        self.reduction = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime,
                                           coordinator=coordinator, session=session, name=name + ".reduce").functional(session=session)
        geometry = self.geometry
        self.inverse = (1.0 / (geometry.theta ** (torch.arange(0, geometry.rotary_dim, 2,
            dtype=torch.float32, device=source.device) / geometry.rotary_dim))
            if geometry.rotary_dim else torch.empty(0, dtype=torch.float32, device=source.device))
        self.eps = config.text["rms_norm_eps"]
        self.query_factor = config.text["qk_scale_factor"]
        self.lengths = torch.zeros_like(metadata.lengths)
        self.slots = torch.full_like(metadata.positions, -1)
        self.key_cache, self.value_cache = pool.cache(layer, "key"), pool.cache(layer, "value")
        self.commit_writer = None
        window = -1
        if geometry.window > 0:
            if mode in ("prefill", "verify"):
                key, value = scratch.window_key, scratch.window_value
                rows = metadata.lengths.numel()
                if mode == "verify":
                    pages = (geometry.window + metadata.max_query_tokens + PAGE_TOKENS - 1) // PAGE_TOKENS
                    key, value = key[:rows * pages], value[:rows * pages]
                    retained_key = torch.zeros_like(scratch.key)
                    retained_value = torch.zeros_like(scratch.value)
                    self.commit_writer = PreparedPagedKVWriter(retained_key, retained_value,
                        self.key_cache, self.value_cache, self.slots, session=session, name=name + ".commit", caller_owned_placeholders=True).functional()
                self.staged_key, self.staged_value = key, value
                self.pages = torch.arange(key.shape[0], dtype=torch.int32, device=source.device).view(rows, -1)
                window = geometry.window - 1
            else:
                key, value = pool.window_pages(layer, "key"), pool.window_pages(layer, "value")
                self.window_row_stride = pool.window_page_stride(layer)
                window_columns = torch.arange(geometry.window // PAGE_TOKENS,
                    dtype=torch.int32, device=source.device)
                self.pages = torch.zeros((metadata.lengths.numel(), window_columns.numel()),
                    dtype=torch.int32, device=source.device)
        else:
            key, value, self.pages = self.key_cache, self.value_cache, metadata.page_table
        self.writer = None
        if geometry.window < 0 or mode == "decode":
            self.writer = PreparedPagedKVWriter(scratch.key, scratch.value, key, value, self.slots,
                                                  session=session, name=name + ".cache_write", caller_owned_placeholders=True).functional()
        if geometry.window > 0:
            tuning_page_count = (key.shape[0] if mode in ("prefill", "verify") else
                                 (int(pool.memory_capacity[0].item()) - 1) * self.window_row_stride + geometry.window // PAGE_TOKENS)
        else:
            tuning_page_count = int(pool.memory_capacity[1].item())
        self.attention = PreparedPagedAttention(scratch.query, key, value, self.pages, self.lengths,
            metadata.query_start_loc, scratch.attended, mode=mode, window=window,
            tuning_page_count=tuning_page_count,
            width=1 if mode == "prefill" else metadata.max_query_tokens,
            session=session, workspace=workspace, name=name + ".attention", caller_owned_placeholders=True).functional(session=session,
                k_cache=None, v_cache=None, page_table=None, cache_seqlens=None, cu_seqlens_q=None)

        self.inverse = persistent_copy(self.inverse)
        self.staged_key_shape, self.staged_value_shape = tuple(key.shape), tuple(value.shape)
        self.attention_key, self.attention_value = key, value
        if geometry.window > 0 and mode in ("prefill", "verify"):
            self.attention_key = self.attention_value = None
        self.pages = self.lengths = self.slots = None
        self.staged_key = self.staged_value = None

    def _rotary(self, values, cos, sin):
        rotary, tail = values.split((self.geometry.rotary_dim, values.shape[-1] - self.geometry.rotary_dim), -1)
        left, right = rotary.chunk(2, -1)
        rotated = torch.cat((-right, left), -1)
        return torch.cat((rotary * cos[:, None] + rotated * sin[:, None], tail), -1)

    def project(self, hidden_states, metadata):
        m, a = metadata, self.geometry
        projected = self.projection(hidden_states.reshape(-1, hidden_states.shape[-1]))
        heads, kv_heads = self.query_shape[1], self.key_shape[1]
        q, k, v, gate = projected.split((heads * a.key_dim, kv_heads * a.key_dim,
                                        kv_heads * a.value_dim, heads * a.value_dim), -1)
        q = head_norm(q.view(self.query_shape), self.eps) * self.query_factor
        k = head_norm(k.view(self.key_shape), self.eps)
        if a.rotary_dim:
            frequencies = m.positions.clamp_min(0).float()[:, None] * self.inverse[None]
            angles = torch.cat((frequencies, frequencies), -1)
            cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
            q, k = self._rotary(q, cos, sin), self._rotary(k, cos, sin)
        v = v.view(self.value_shape).contiguous()
        return q, k, v, gate

    def forward(self, hidden_states, *, metadata):
        return self.attend(*self.project(hidden_states, metadata), metadata, hidden_states.shape)

    def attend(self, q, k, v, gate, metadata, output_shape):
        m, a = metadata, self.geometry
        records = None
        slots = None
        if a.window < 0:
            columns = (m.positions.clamp_min(0) // PAGE_TOKENS).clamp(max=m.token_page_table.shape[1] - 1)
            rows = torch.arange(m.positions.numel(), device=m.positions.device)
            pages = m.token_page_table[rows, columns]
            slots = torch.where(m.positions >= 0, pages.long() * PAGE_TOKENS + m.positions.remainder(PAGE_TOKENS), -1)
            pages = m.page_table
            lengths = (m.starts + m.lengths).int()
        elif self.mode == "decode":
            columns = torch.arange(a.window // PAGE_TOKENS, dtype=torch.int32, device=m.positions.device)
            pages = (m.state_rows[:, None] * self.window_row_stride + columns[None]).to(torch.int32)
            slots = torch.where(m.positions >= 0, m.state_rows[m.requests] * self.window_row_stride * PAGE_TOKENS
                                + m.positions.remainder(a.window), -1)
            lengths = (m.starts + m.lengths).clamp(max=a.window).int()
        else:
            staged_key = torch.empty(self.staged_key_shape, dtype=self.key_cache.dtype, device=k.device)
            staged_value = torch.empty(self.staged_value_shape, dtype=self.value_cache.dtype, device=v.device)
            stage_windows(self.key_cache, k, staged_key, m)
            stage_windows(self.value_cache, v, staged_value, m)
            if self.mode == "verify":
                records = k, v
            pages = torch.arange(staged_key.shape[0], dtype=torch.int32, device=k.device).view(m.lengths.numel(), -1)
            lengths = m.starts.clamp(max=a.window).int() + m.lengths
        if self.writer is not None:
            self.writer(k, v, slots)
        key_cache = self.attention_key if a.window < 0 or self.mode == "decode" else staged_key
        value_cache = self.attention_value if a.window < 0 or self.mode == "decode" else staged_value
        attended = self.attention(q, k_cache=key_cache, v_cache=value_cache, page_table=pages,
            cache_seqlens=lengths, cu_seqlens_q=m.query_start_loc)
        if a.window > 0 and self.mode == "prefill":
            commit_window(self.key_cache, k, m.state_rows[0], m.starts[0], m.lengths[0])
            commit_window(self.value_cache, v, m.state_rows[0], m.starts[0], m.lengths[0])
        attended = head_norm(attended, self.eps) * torch.sigmoid(gate.view(self.query_shape[0], self.query_shape[1], a.value_dim))
        return self.reduction(self.output_projection(attended.flatten(1))).view(output_shape), records

    def commit(self, counts, records, metadata):
        if self.commit_writer is not None:
            m, window = metadata, self.geometry.window
            key, value = records
            offset = m.positions - m.starts[m.requests]
            live = (m.positions >= 0) & (offset < counts[m.requests])
            live &= offset >= 0
            slots = torch.where(live, m.state_rows[m.requests] * window + m.positions.remainder(window), -1)
            self.commit_writer(key, value, slots)
