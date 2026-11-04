# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical normalized, rotary, gated attention over BF16 paged histories."""

from dataclasses import dataclass
import torch

from ignition.components.linear import prepare_projection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from .model import norm
from .pool import PAGE_TOKENS


@dataclass(frozen=True)
class AttentionScratch:
    projected: torch.Tensor
    staging: torch.Tensor
    quantized: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    gated: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, config, tokens, device, *, buffers=None):
        c = config.text
        heads, kv, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        def zeros(*shape, dtype=torch.bfloat16):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        return cls(zeros(tokens, 2 * (heads + kv) * dim), zeros(tokens, heads * dim),
            zeros(tokens, max(config.hidden, heads * dim), dtype=torch.uint8),
            zeros(tokens, heads, dim), zeros(tokens, kv, dim), zeros(tokens, kv, dim),
            zeros(tokens, heads, dim), zeros(tokens, heads, dim), zeros(tokens, config.hidden))

    def rows(self, tokens):
        return type(self)(*(getattr(self, name)[:tokens] for name in self.__dataclass_fields__))


class PreparedAttention(torch.nn.Module):
    def __init__(self, layer, weights, config, source, scratch, metadata, pool, *, mode,
                 session, name, workspace, buffers=None):
        super().__init__()
        if mode not in ("prefill", "decode"):
            raise ValueError(f"unsupported Muse Glimmer attention mode: {mode}")
        self.scratch, self.metadata = scratch, metadata
        c = config.text
        self.projection = prepare_projection(source, weights.qkv, scratch.projected,
            session=session, name=name + ".qkv_gate", workspace=workspace, staging=scratch.staging,
            quantized_staging=scratch.quantized)
        self.output_projection = prepare_projection(scratch.gated.flatten(1), weights.output, scratch.output,
            session=session, name=name + ".output", workspace=workspace, quantized_staging=scratch.quantized)
        self.qk_norm = norm("rms", eps=c["rms_norm_eps"])
        self.query_factor, self.theta = c["qk_scale_factor"], c["layer_rope_theta"][layer]
        dim = c["head_dim"]
        self.inverse = (1. / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32,
            device=source.device) / dim)) if self.theta else torch.empty(0, device=source.device))
        self.lengths = (torch.zeros_like(metadata.lengths) if buffers is None
                        else buffers.zeros(metadata.lengths.shape, metadata.lengths.dtype))
        self.slots = (torch.full_like(metadata.positions, -1) if buffers is None
                      else buffers.zeros(metadata.positions.shape, metadata.positions.dtype).fill_(-1))
        key, value = pool.cache(layer, "key"), pool.cache(layer, "value")
        self.writer = PreparedPagedKVWriter(scratch.key, scratch.value, key, value, self.slots,
            session=session, name=name + ".cache_write")
        self.attention = PreparedPagedAttention(scratch.query, key, value, metadata.page_table, self.lengths,
            metadata.query_start_loc, scratch.attended, mode=mode, window=config.window(layer),
            tuning_page_count=int(pool.memory_capacity[1].item()), width=1,
            session=session, workspace=workspace, name=name + ".attention")

    @staticmethod
    def _rotary(value, cos, sin):
        left, right = value.chunk(2, -1)
        return value * cos[:, None] + torch.cat((-right, left), -1) * sin[:, None]

    def forward(self, hidden_states, **kwargs):
        s, m = self.scratch, self.metadata
        projected = self.projection(hidden_states.reshape(-1, hidden_states.shape[-1]))
        qdim, kvdim = s.query.shape[1] * s.query.shape[2], s.key.shape[1] * s.key.shape[2]
        q, k, v, gate = projected.split((qdim, kvdim, kvdim, qdim), -1)
        q = self.qk_norm(q.view_as(s.query)) * self.query_factor
        k = self.qk_norm(k.view_as(s.key))
        if self.theta:
            frequencies = m.positions.clamp_min(0).float()[:, None] * self.inverse[None]
            angles = torch.cat((frequencies, frequencies), -1)
            cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
            q, k = self._rotary(q, cos, sin), self._rotary(k, cos, sin)
        s.query.copy_(q)
        s.key.copy_(k)
        s.value.copy_(v.view_as(s.value))
        columns = (m.positions.clamp_min(0) // PAGE_TOKENS).clamp(max=m.token_page_table.shape[1] - 1)
        rows = torch.arange(m.positions.numel(), device=m.positions.device)
        pages = m.token_page_table[rows, columns]
        self.slots.copy_(torch.where(m.positions >= 0,
            pages.long() * PAGE_TOKENS + m.positions.remainder(PAGE_TOKENS), -1))
        self.lengths.copy_((m.starts + m.lengths).int())
        self.writer()
        self.attention()
        s.gated.copy_(s.attended * torch.sigmoid(gate.view_as(s.gated)))
        return self.output_projection(s.gated.flatten(1)).view_as(hidden_states), None

    def scratch_tensors(self):
        quantized = tuple(p.quantized for p in getattr(self.projection, "projections", (self.projection,)) if hasattr(p, "quantized"))
        return self.slots, self.lengths, quantized, getattr(self.output_projection, "quantized", ())
