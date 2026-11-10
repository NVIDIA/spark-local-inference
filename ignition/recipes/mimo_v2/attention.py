# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiMo partial RoPE, paged global attention and circular sliding histories."""

from dataclasses import dataclass

import torch

from ignition.components.block_fp8 import BlockFP8Projection, BlockFP8Weight
from ignition.components.collectives import PreparedAllReduce
from ignition.components.linear import UnquantizedProjection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.preparation.persistent import persistent_copy
from ignition.components.window_attention import stage_windows, commit_window

from .pool import PAGE_TOKENS


@dataclass(frozen=True)
class AttentionScratch:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    window_key: torch.Tensor
    window_value: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, device, *, requests=1):
        geometries = [config.attention(i) for i in range(config.layers)]
        if len({(a.heads, a.kv_heads, a.key_dim, a.value_dim) for a in geometries}) != 1:
            raise ValueError("MiMo attention scratch requires matching global and sliding head geometry")
        geometry = geometries[0]
        heads, kv_heads = geometry.local(shard)
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        window = max(a.window for a in geometries)
        pages = max((tokens + max(window, 0) + PAGE_TOKENS - 1) // PAGE_TOKENS,
                    requests * ((max(window, 0) + 8 + PAGE_TOKENS - 1) // PAGE_TOKENS))
        return cls(zeros(tokens, heads, geometry.key_dim), zeros(tokens, kv_heads, geometry.key_dim),
            zeros(tokens, kv_heads, geometry.value_dim), zeros(tokens, heads, geometry.value_dim),
            zeros(tokens, config.text["hidden_size"]), zeros(tokens, config.text["hidden_size"]),
            zeros(pages, PAGE_TOKENS, kv_heads, geometry.key_dim, dtype=torch.float8_e4m3fn),
            zeros(pages, PAGE_TOKENS, kv_heads, geometry.value_dim, dtype=torch.float8_e4m3fn))

    def rows(self, tokens):
        return type(self)(*(getattr(self, name) if name.startswith("window_") else getattr(self, name)[:tokens]
                            for name in self.__dataclass_fields__))


def stage_window(history, values, output, row, start, length):
    """Linearize a circular prefix and append a single request's prefill chunk."""
    window = history.shape[1]
    indices = torch.arange(output.shape[0] * output.shape[1], device=values.device)
    prefix = start.clamp(max=window)
    previous = history[row.expand_as(indices), (start - prefix + indices).remainder(window)].to(values.dtype)
    current = values[(indices - prefix).clamp(0, values.shape[0] - 1)]
    valid = indices < prefix + length
    staged = torch.where(valid[:, None, None],
        torch.where((indices < prefix)[:, None, None], previous, current), 0)
    if output.dtype == torch.float8_e4m3fn:
        staged = staged.clamp(-448, 448)
    output.view(-1, *values.shape[1:]).copy_(staged)


def partial_rotary(values, positions, inverse):
    dimension = inverse.numel() * 2
    frequencies = positions.clamp_min(0).float()[:, None] * inverse[None]
    angles = torch.cat((frequencies, frequencies), -1)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    rotary, tail = values.split((dimension, values.shape[-1] - dimension), -1)
    left, right = rotary.chunk(2, -1)
    return torch.cat((rotary * cos[:, None] + torch.cat((-right, left), -1) * sin[:, None], tail), -1)


class PreparedAttention(torch.nn.Module):
    def __init__(self, layer, weights, config, shard, source, scratch, metadata, pool, *, mode,
                 runtime, coordinator, session, name, workspace):
        super().__init__()
        self.geometry = config.attention(layer)
        self.query_shape, self.key_shape, self.value_shape = (tuple(value.shape) for value in (scratch.query, scratch.key, scratch.value))
        if mode not in ("prefill", "decode", "verify"):
            raise ValueError(f"unsupported MiMo attention mode: {mode}")
        self.mode, self.pool, self.layer = mode, pool, layer
        self.chunks = config.qkv_chunks(layer, shard)
        self.projection = BlockFP8Projection(source, weights.qkv, session=session, name=name + ".qkv").functional(session=session)
        attended = scratch.attended.flatten(1)
        if isinstance(weights.output, BlockFP8Weight):
            self.output_projection = BlockFP8Projection(attended, weights.output, session=session, name=name + ".output").functional(session=session)
        else:
            self.output_projection = UnquantizedProjection(attended, weights.output, scratch.partial,
                                                           session=session, name=name + ".output").functional(session=session)
        self.reduction = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime,
                                           coordinator=coordinator, session=session, name=name + ".reduce").functional(session=session)
        geometry = self.geometry
        self.inverse = 1.0 / (geometry.theta ** (torch.arange(0, geometry.rotary_dim, 2,
                                   dtype=torch.float32, device=source.device) / geometry.rotary_dim))
        self.value_scale = config.text.get("attention_value_scale")
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
                    width = metadata.max_query_tokens
                    pages = (geometry.window + width + PAGE_TOKENS - 1) // PAGE_TOKENS
                    key, value = key[:rows * pages], value[:rows * pages]
                    retained_key, retained_value = torch.zeros_like(scratch.key), torch.zeros_like(scratch.value)
                    self.commit_writer = PreparedPagedKVWriter(retained_key, retained_value,
                        self.key_cache, self.value_cache, self.slots, session=session, name=name + ".commit", caller_owned_placeholders=True).functional()
                self.staged_key, self.staged_value = key, value
                self.pages = torch.arange(key.shape[0], dtype=torch.int32, device=source.device).view(rows, -1)
                window = geometry.window - 1
            else:
                key, value = self.key_cache, self.value_cache
                self.pages = torch.zeros((metadata.lengths.numel(), 1), dtype=torch.int32, device=source.device)
        else:
            key, value, self.pages = self.key_cache, self.value_cache, metadata.page_table
        self.writer = None
        if geometry.window < 0 or mode == "decode":
            self.writer = PreparedPagedKVWriter(scratch.key, scratch.value, key, value, self.slots,
                                                  session=session, name=name + ".cache_write", caller_owned_placeholders=True).functional()
        if geometry.window > 0:
            tuning_page_count = (key.shape[0] if mode in ("prefill", "verify") else
                                 int(pool.memory_capacity[0].item()))
        else:
            tuning_page_count = int(pool.memory_capacity[1].item())
        self.attention = PreparedPagedAttention(scratch.query, key, value, self.pages, self.lengths,
            metadata.query_start_loc, scratch.attended, sink=weights.sink, mode=mode, window=window,
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

    def forward(self, hidden_states, *, metadata):
        m, a = metadata, self.geometry
        projected = self.projection(hidden_states.reshape(-1, hidden_states.shape[-1]))
        query, key, value, offset = [], [], [], 0
        for chunk in self.chunks:
            q_end, k_end = offset + chunk.query_rows, offset + chunk.query_rows + chunk.key_rows
            query.append(projected[:, offset:q_end])
            key.append(projected[:, q_end:k_end])
            value.append(projected[:, k_end:k_end + chunk.value_rows])
            offset += chunk.padded_rows
        q = partial_rotary(torch.cat(query, -1).view(self.query_shape), m.positions, self.inverse)
        k = partial_rotary(torch.cat(key, -1).view(self.key_shape), m.positions, self.inverse)
        v = torch.cat(value, -1).view(self.value_shape)
        v = (v if self.value_scale is None else v * self.value_scale).contiguous()
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
            pages = m.state_rows[:, None].int()
            slots = torch.where(m.positions >= 0, m.state_rows[m.requests] * a.window + m.positions.remainder(a.window), -1)
            lengths = (m.starts + m.lengths).clamp(max=a.window).int()
        else:
            staged_key = torch.empty(self.staged_key_shape, dtype=self.key_cache.dtype, device=k.device)
            staged_value = torch.empty(self.staged_value_shape, dtype=self.value_cache.dtype, device=v.device)
            if self.mode == "prefill":
                stage_window(self.key_cache, k, staged_key, m.state_rows[0], m.starts[0], m.lengths[0])
                stage_window(self.value_cache, v, staged_value, m.state_rows[0], m.starts[0], m.lengths[0])
            else:
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
        return self.reduction(self.output_projection(attended.flatten(1))).view_as(hidden_states), records

    def commit(self, counts, records, metadata):
        if self.commit_writer is not None:
            m, window = metadata, self.geometry.window
            key, value = records
            offset = m.positions - m.starts[m.requests]
            live = (m.positions >= 0) & (offset < counts[m.requests])
            slots = torch.where(live, m.state_rows[m.requests] * window + m.positions.remainder(window), -1)
            self.commit_writer(key, value, slots)
