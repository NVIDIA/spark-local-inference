# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated GQA over sliding continuations and selected eight-token cache regions."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.gated_compression import commit_gated
from ignition.components.linear import UnquantizedProjection
from ignition.components.qk_norm_rope import PreparedQKNormRoPE
from ignition.components.region_selection import PreparedRegionSelection
from ignition.components.selected_attention import PreparedSelectedAttention
from ignition.components.window_attention import PreparedWindowAttention

from .compression import compress
from .model import rotary_inverse


@dataclass(frozen=True)
class AttentionScratch:
    projected: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    projected_output: torch.Tensor
    index_query: torch.Tensor
    index_key: torch.Tensor
    index_gate: torch.Tensor
    index_weight: torch.Tensor
    selected: torch.Tensor
    selected_tokens: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, context, device, *, requests=1, buffers=None):
        a = config.attention(0)
        heads, groups = a.local(shard)
        ih = config.sparse["sparse_indexer_num_heads"] // config.sparse["num_provider_groups"]
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)
        index_width = (groups * ih + 2) * 256 + groups * ih
        return cls(zeros(tokens, (heads + 2 * groups) * a.dimension + heads + index_width),
            zeros(tokens, heads, a.dimension), zeros(tokens, groups, a.dimension), zeros(tokens, groups, a.dimension),
            zeros(tokens, heads, a.dimension), zeros(tokens, config.hidden, dtype=torch.float32),
            zeros(tokens, config.hidden, dtype=torch.float32), zeros(tokens, config.hidden),
            zeros(tokens, groups, ih, 256, dtype=torch.float8_e4m3fn), zeros(tokens, 256), zeros(tokens, 256),
            zeros(tokens, groups, ih, dtype=torch.float32),
            zeros(tokens, groups, config.sparse["topk"], dtype=torch.int32),
            zeros(tokens, groups, (config.sparse["topk"] + 1) * 8, dtype=torch.int32))

    def rows(self, tokens):
        return type(self)(*(getattr(self, field)[:tokens]
                           for field in self.__dataclass_fields__))


class PreparedAttention(torch.nn.Module):
    def __init__(self, layer, weights, config, shard, source, scratch, metadata, pool, *, mode,
                 runtime, coordinator, session, name, workspace, first_position=0, proposal=None, reduce_output=True, buffers=None):
        super().__init__()
        self.scratch, self.metadata, self.mode = scratch, metadata, mode
        self.geometry = a = config.attention(layer)
        self.first_position, self.proposal = first_position, proposal
        self.retained = None
        if mode == "verify":
            shapes = (scratch.key, scratch.value) if a.window > 0 else (scratch.index_key, scratch.index_gate)
            self.retained = tuple(torch.zeros_like(x) if buffers is None else buffers.zeros(x.shape, x.dtype)
                                  for x in shapes)
        self.projected = scratch.projected.flatten()[:source.shape[0] * weights.qkvg.shape[0]].view(
            source.shape[0], weights.qkvg.shape[0])
        self.projection = UnquantizedProjection(source, weights.qkvg, self.projected, session=session, name=name + ".qkvg")
        heads, groups = scratch.query.shape[1], scratch.key.shape[1]
        gate_offset = (heads + 2 * groups) * a.dimension
        self.gate = self.projected[:, gate_offset:gate_offset + heads]
        self.output_projection = UnquantizedProjection(scratch.attended.flatten(1), weights.output,
            scratch.projected_output, session=session, name=name + ".output")
        self.reduction = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".reduce") if reduce_output else None
        scaling = config.text.get("rope_scaling") if a.window < 0 else None
        self.inverse = rotary_inverse(a.rotary, a.theta, scaling, source.device)
        self.key, self.value = pool.cache(layer, "key"), pool.cache(layer, "value")
        self.prepare_qkv = PreparedQKNormRoPE(self.projected, weights.q_norm, weights.k_norm,
            self.inverse, metadata, scratch.query, scratch.key, value=scratch.value,
            key_cache=self.key if a.window < 0 else None, value_cache=self.value if a.window < 0 else None,
            epsilon=config.text["rms_norm_eps"], session=session, name=name + ".qk_norm_rope")
        self.indexer = a.window < 0
        if a.window < 0:
            self.index_inverse = rotary_inverse(config.sparse["sparse_indexer_rope_dim"], a.theta, scaling, source.device)
            main_width = (scratch.query.shape[1] + 2 * scratch.key.shape[1]) * a.dimension + scratch.query.shape[1]
            self.prepare_index = PreparedQKNormRoPE(self.projected[:, main_width:], weights.indexer["q_norm"],
                weights.indexer["k_norm"], self.index_inverse, metadata,
                scratch.index_query.flatten(1, 2), scratch.index_key[:, None], bias=weights.indexer["k_bias"],
                epsilon=config.text["rms_norm_eps"], session=session, name=name + ".index_norm_rope")
            self.summaries = pool.cache(layer, "index")
            self.pending_key, self.pending_gate = pool.cache(layer, "pending_key"), pool.cache(layer, "pending_gate")
            self.selection = PreparedRegionSelection(scratch.index_query, scratch.index_weight, self.summaries,
                metadata, scratch.selected, ratio=8, session=session, workspace=workspace, name=name + ".selection", buffers=buffers)
            self.attention = PreparedSelectedAttention(scratch.query, self.key, self.value, metadata.page_table,
                metadata.requests, metadata.positions, scratch.selected_tokens, scratch.attended, output_gate=self.gate,
                session=session, workspace=workspace, name=name + ".attention")
        else:
            self.attention = PreparedWindowAttention(scratch.query, scratch.key, scratch.value, self.key, self.value,
                metadata, scratch.attended, first_position=first_position,
                proposal=proposal, output_gate=self.gate, session=session, workspace=workspace, name=name + ".attention")

    def forward(self, hidden, *, cache_only=False):
        s, m, a = self.scratch, self.metadata, self.geometry
        projected = self.projection(hidden)
        q_width, k_width = s.query.shape[1] * a.dimension, s.key.shape[1] * a.dimension
        main_width = q_width + 2 * k_width + s.query.shape[1]
        self.prepare_qkv()
        if cache_only:
            if self.proposal is None:
                commit_gated(s.key.flatten(1), s.value.flatten(1), self.key, self.value, m, m.lengths)
            else:
                key, value, step = self.proposal
                key[:, step].copy_(s.key)
                value[:, step].copy_(s.value)
            return None
        if self.indexer:
            index = projected[:, main_width:]
            index_heads = s.index_weight.shape[1] * s.index_weight.shape[2]
            _, _, iz, iw = index.split((index_heads * 256, 256, 256, index_heads), -1)
            self.prepare_index()
            s.index_gate.copy_(iz)
            s.index_weight.copy_(iw.view_as(s.index_weight).float() * (s.index_weight.shape[2] ** -.5))
            compress(s.index_key, s.index_gate, self.pending_key, self.pending_gate, m, self.summaries)
            self.selection()
            regions = torch.cat((s.selected, (m.positions // 8)[:, None, None].expand(-1, s.selected.shape[1], 1)), -1)
            selected = regions[..., None] * 8 + torch.arange(8, device=regions.device)
            s.selected_tokens.copy_(torch.where(regions[..., None] >= 0, selected, -1).flatten(2))
        self.attention()
        if self.retained is not None:
            values = (s.key, s.value) if a.window > 0 else (s.index_key, s.index_gate)
            for destination, value in zip(self.retained, values, strict=True):
                destination.copy_(value)
        elif self.proposal is None:
            self.commit(m.lengths)
        projected = self.output_projection(s.attended.flatten(1))
        if self.reduction is None:
            return projected
        s.partial.copy_(projected)
        return self.reduction(s.partial).bfloat16()

    def commit(self, counts):
        s = self.scratch
        if self.geometry.window > 0:
            key, value = (s.key, s.value) if self.retained is None else self.retained
            commit_gated(key.flatten(1), value.flatten(1), self.key, self.value, self.metadata, counts)
        else:
            key, gate = (s.index_key, s.index_gate) if self.retained is None else self.retained
            commit_gated(key, gate, self.pending_key, self.pending_gate, self.metadata, counts)

    def scratch_tensors(self):
        temporary = ((self.selection.requests, self.selection.positions, self.selection.lengths)
                     if self.indexer else self.attention.scratch_tensors())
        return self.retained, temporary
