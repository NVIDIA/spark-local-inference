# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged learned-relative-bias attention with complete FP32 convolution state."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedChunkedAllReduce
from ignition.components.linear import UnquantizedProjection
from ignition.components.paged_attention import PreparedPagedKVWriter
from ignition.components.relative_paged_attention import PreparedRelativePagedAttention
from ignition.components.residual_short_conv import PreparedResidualShortConvolution
from ignition.components.window_attention import stage_windows, commit_window
from .model import norm
from .pool import KV_DTYPE, PAGE_TOKENS


@dataclass(frozen=True)
class AttentionScratch:
    projected: torch.Tensor
    query: torch.Tensor
    key_source: torch.Tensor
    value_source: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    relative_source: torch.Tensor
    relative: torch.Tensor
    attended: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    window_key: torch.Tensor
    window_value: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, device, *, requests, buffers):
        geometries = [config.attention(i) for i in range(config.layers)]
        if len({(a.heads, a.kv_heads, a.key_dim) for a in geometries}) != 1:
            raise ValueError("Inkling scratch requires matching global and sliding head geometry")
        a = geometries[0]
        heads, kv = a.local(shard)
        drel = config.text["d_rel"]
        window, extent = max(a.window for a in geometries), max(a.relative_extent for a in geometries)
        pages = requests * -(-(window + (tokens + requests - 1) // requests) // PAGE_TOKENS)
        def zeros(*shape):
            return buffers.zeros(shape, torch.bfloat16)
        return cls(zeros(tokens, (heads + 2 * kv) * a.key_dim + heads * drel),
            zeros(tokens, heads, a.key_dim), zeros(tokens, kv * a.key_dim), zeros(tokens, kv * a.key_dim),
            zeros(tokens, kv, a.key_dim), zeros(tokens, kv, a.key_dim),
            zeros(tokens, heads, drel), zeros(tokens, heads, extent), zeros(tokens, heads, a.key_dim),
            zeros(tokens, config.hidden), zeros(tokens, config.hidden),
            zeros(pages, PAGE_TOKENS, kv, a.key_dim), zeros(pages, PAGE_TOKENS, kv, a.key_dim))

    def rows(self, tokens):
        return type(self)(*(getattr(self, name) if name.startswith("window_") else getattr(self, name)[:tokens]
                            for name in self.__dataclass_fields__))


class PreparedAttention(torch.nn.Module):
    def __init__(self, layer, weights, config, shard, source, scratch, metadata, pool, *, mode,
                 runtime, coordinator, session, name, workspace, buffers):
        super().__init__()
        self.source = source
        if mode not in ("prefill", "decode"):
            raise ValueError("Inkling supports ordinary prefill and decode")
        self.geometry, self.scratch, self.metadata, self.mode = config.attention(layer), scratch, metadata, mode
        c, a, s = config.text, self.geometry, scratch
        self.projection = UnquantizedProjection(source, weights.qkv_relative, s.projected,
            session=session, name=name + ".qkv_relative")
        self.query_norm, self.key_norm = norm(weights.query_norm, c["rms_norm_eps"]), norm(weights.key_norm, c["rms_norm_eps"])
        self.key_conv = PreparedResidualShortConvolution(s.key_source, weights.key_conv,
            pool.convolution(layer, "key"), metadata, s.key.flatten(1), session=session, workspace=workspace, name=name + ".key_conv")
        self.value_conv = PreparedResidualShortConvolution(s.value_source, weights.value_conv,
            pool.convolution(layer, "value"), metadata, s.value.flatten(1), session=session, workspace=workspace, name=name + ".value_conv")
        self.relative = s.relative.flatten()[:s.query.shape[0] * s.query.shape[1] * a.relative_extent].view(
            s.query.shape[0], s.query.shape[1], a.relative_extent)
        self.relative_projection = UnquantizedProjection(s.relative_source.flatten(0, 1), weights.relative,
            self.relative.flatten(0, 1), session=session, name=name + ".relative")
        self.log_floor, self.log_alpha = c["log_scaling_n_floor"], c["log_scaling_alpha"]
        self.output_projection = UnquantizedProjection(s.attended.flatten(1), weights.output, s.partial,
            session=session, name=name + ".output")
        self.reduction = PreparedChunkedAllReduce(s.partial, s.output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".reduce")
        self.lengths = buffers.zeros(metadata.lengths.shape, torch.int32)
        self.slots = buffers.zeros(metadata.positions.shape, torch.int64).fill_(-1)
        self.key_cache, self.value_cache = pool.cache(layer, "key"), pool.cache(layer, "value")
        if a.window > 0:
            count = metadata.lengths.numel() * -(-(a.window + metadata.max_query_tokens) // PAGE_TOKENS)
            self.staged_key, self.staged_value = s.window_key[:count], s.window_value[:count]
            self.pages = torch.arange(count, dtype=torch.int32, device=source.device).view(metadata.lengths.numel(), -1)
            key, value = self.staged_key, self.staged_value
            self.window_stride = pool.window_page_stride(layer)
            self.writer = PreparedPagedKVWriter(s.key, s.value, pool.window_pages(layer, "key"),
                pool.window_pages(layer, "value"), self.slots, session=session, name=name + ".ring_write") if mode == "decode" else None
        else:
            key, value, self.pages = self.key_cache, self.value_cache, metadata.page_table
            self.writer = PreparedPagedKVWriter(s.key, s.value, key, value, self.slots,
                session=session, name=name + ".cache_write")
        self.attention = PreparedRelativePagedAttention(s.query, key, value, self.pages, self.lengths,
            metadata.query_start_loc, self.relative, s.attended, window=a.window - 1 if a.window > 0 else -1,
            softmax_scale=1. / a.key_dim, session=session, workspace=workspace, name=name + ".attention")

    def forward(self, hidden_states, **kwargs):
        hidden = self.source.copy_(hidden_states.reshape_as(self.source))
        s, m, a = self.scratch, self.metadata, self.geometry
        heads, kv, dim = s.query.shape[1], s.key.shape[1], a.key_dim
        q, k, v, r = self.projection(hidden).split((heads * dim, kv * dim, kv * dim,
                                                  s.relative_source.shape[1] * s.relative_source.shape[2]), -1)
        s.query.copy_(self.query_norm(q.view_as(s.query)))
        s.key_source.copy_(k)
        s.value_source.copy_(v)
        self.key_conv()
        self.value_conv()
        s.key.copy_(self.key_norm(s.key))
        s.relative_source.copy_(r.view_as(s.relative_source))
        self.relative_projection(s.relative_source.flatten(0, 1))
        if a.window < 0 and self.log_floor is not None:
            tau = 1. + self.log_alpha * torch.log(((m.positions.clamp_min(0) + 1).float() / self.log_floor).clamp_min(1.))
            s.query.copy_(s.query.float() * tau[:, None, None])
            self.relative.copy_(self.relative.float() * tau[:, None, None])
        if a.window < 0:
            columns = (m.positions.clamp_min(0) // PAGE_TOKENS).clamp(max=m.token_page_table.shape[1] - 1)
            pages = m.token_page_table[torch.arange(m.positions.numel(), device=m.positions.device), columns]
            self.slots.copy_(torch.where(m.positions >= 0, pages.long() * PAGE_TOKENS + m.positions.remainder(PAGE_TOKENS), -1))
            self.lengths.copy_((m.starts + m.lengths).int())
            self.writer()
        else:
            stage_windows(self.key_cache, s.key, self.staged_key, m)
            stage_windows(self.value_cache, s.value, self.staged_value, m)
            self.lengths.copy_(m.starts.clamp(max=a.window).int() + m.lengths)
        self.attention()
        if a.window > 0:
            if self.mode == "prefill":
                commit_window(self.key_cache, s.key, m.state_rows[0], m.starts[0], m.lengths[0])
                commit_window(self.value_cache, s.value, m.state_rows[0], m.starts[0], m.lengths[0])
            else:
                self.slots.copy_(torch.where(m.positions >= 0,
                    m.state_rows[m.requests] * self.window_stride * PAGE_TOKENS + m.positions.remainder(a.window), -1))
                self.writer()
        return self.reduction(self.output_projection(s.attended.flatten(1))).view_as(hidden_states), None

    def scratch_tensors(self):
        return self.lengths, self.slots
