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

from ignition.preparation.persistent import persistent_copy

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
    def allocate(cls, config, shard, tokens, context, device, *, requests=1):
        a = config.attention(0)
        heads, groups = a.local(shard)
        ih = config.sparse["sparse_indexer_num_heads"] // config.sparse["num_provider_groups"]
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
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
                 runtime, coordinator, session, name, workspace, first_position=0, proposal=None, reduce_output=True):
        super().__init__()
        self.scratch, self.metadata, self.mode = scratch, metadata, mode
        self.geometry = a = config.attention(layer)
        self.first_position, self.proposal = first_position, proposal
        self.verify = mode == "verify"
        self.projected = scratch.projected.flatten()[:source.shape[0] * weights.qkvg.shape[0]].view(
            source.shape[0], weights.qkvg.shape[0])
        self.projection = UnquantizedProjection(source, weights.qkvg, self.projected, session=session, name=name + ".qkvg").functional(session=session)
        heads, groups = scratch.query.shape[1], scratch.key.shape[1]
        gate_offset = (heads + 2 * groups) * a.dimension
        self.gate = self.projected[:, gate_offset:gate_offset + heads]
        self.gate_offset, self.main_width = gate_offset, gate_offset + heads
        self.index_shape = tuple(scratch.index_query.shape)
        self.output_projection = UnquantizedProjection(scratch.attended.flatten(1), weights.output,
            scratch.projected_output, session=session, name=name + ".output").functional(session=session)
        self.reduction = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".reduce").functional(session=session) if reduce_output else None
        scaling = config.text.get("rope_scaling") if a.window < 0 else None
        self.inverse = persistent_copy(rotary_inverse(a.rotary, a.theta, scaling, source.device))
        self.key, self.value = pool.cache(layer, "key"), pool.cache(layer, "value")
        self.prepare_qkv = PreparedQKNormRoPE(self.projected, weights.q_norm, weights.k_norm,
            self.inverse, metadata, scratch.query, scratch.key, value=scratch.value,
            key_cache=self.key if a.window < 0 else None, value_cache=self.value if a.window < 0 else None,
            epsilon=config.text["rms_norm_eps"], session=session, name=name + ".qk_norm_rope").functional(session=session)
        self.indexer = a.window < 0
        if a.window < 0:
            self.index_inverse = persistent_copy(rotary_inverse(config.sparse["sparse_indexer_rope_dim"], a.theta, scaling, source.device))
            main_width = (scratch.query.shape[1] + 2 * scratch.key.shape[1]) * a.dimension + scratch.query.shape[1]
            self.prepare_index = PreparedQKNormRoPE(self.projected[:, main_width:], weights.indexer["q_norm"],
                weights.indexer["k_norm"], self.index_inverse, metadata,
                scratch.index_query.flatten(1, 2), scratch.index_key[:, None], bias=weights.indexer["k_bias"],
                epsilon=config.text["rms_norm_eps"], session=session, name=name + ".index_norm_rope").functional(session=session)
            self.summaries = pool.cache(layer, "index")
            self.pending_key, self.pending_gate = pool.cache(layer, "pending_key"), pool.cache(layer, "pending_gate")
            self.selection = PreparedRegionSelection(scratch.index_query, scratch.index_weight, self.summaries,
                metadata, scratch.selected, ratio=8, session=session, workspace=workspace, name=name + ".selection").functional(session=session)
            self.attention = PreparedSelectedAttention(scratch.query, self.key, self.value, metadata.page_table,
                metadata.requests, metadata.positions, scratch.selected_tokens, scratch.attended, output_gate=self.gate,
                session=session, workspace=workspace, name=name + ".attention").functional(session=session)
        else:
            self.attention = PreparedWindowAttention(scratch.query, scratch.key, scratch.value, self.key, self.value,
                metadata, scratch.attended, first_position=first_position,
                proposal=proposal, output_gate=self.gate, session=session, workspace=workspace, name=name + ".attention").functional(session=session)

    def forward(self, hidden, *, metadata, cache_only=False, proposal=None):
        projected = self.projection(hidden)
        query, key, value = self.prepare_qkv(projected, metadata)
        gate = projected[:, self.gate_offset:self.main_width]
        if cache_only:
            if proposal is None:
                commit_gated(key.flatten(1), value.flatten(1), self.key, self.value, metadata, metadata.lengths)
            else:
                proposal_key, proposal_value, step = proposal
                proposal_key[:, step].copy_(key)
                proposal_value[:, step].copy_(value)
            return None, None
        if self.indexer:
            index = projected[:, self.main_width:]
            index_heads = self.index_shape[1] * self.index_shape[2]
            _, _, index_gate, index_weights = index.split((index_heads * 256, 256, 256, index_heads), -1)
            index_query, index_key, _ = self.prepare_index(index, metadata)
            index_query = index_query.view(self.index_shape)
            index_key, index_gate = index_key[:, 0], index_gate.contiguous()
            index_weights = index_weights.view(self.index_shape[:3]).float() * (self.index_shape[2] ** -.5)
            compress(index_key, index_gate, self.pending_key, self.pending_gate, metadata, self.summaries)
            selected = self.selection(index_query, index_weights, metadata)
            regions = torch.cat((selected, (metadata.positions // 8)[:, None, None].expand(-1, selected.shape[1], 1)), -1)
            positions = regions[..., None] * 8 + torch.arange(8, device=regions.device)
            selected_tokens = torch.where(regions[..., None] >= 0, positions, -1).flatten(2).to(torch.int32)
            attended = self.attention(query, metadata.page_table, metadata.requests, metadata.positions,
                                      selected_tokens, output_gate=gate)
            records = index_key, index_gate
        else:
            attended = self.attention(query, key, value, metadata, output_gate=gate, proposal=proposal)
            records = key, value
        if not self.verify and proposal is None:
            self.commit(metadata.lengths, records, metadata)
        output = self.output_projection(attended.flatten(1))
        if self.reduction is not None:
            output = self.reduction(output.float()).bfloat16()
        return output, records if self.verify else None

    def commit(self, counts, records, metadata):
        if records is None:
            return
        key, value = records
        if self.geometry.window > 0:
            commit_gated(key.flatten(1), value.flatten(1), self.key, self.value, metadata, counts)
        else:
            commit_gated(key, value, self.pending_key, self.pending_gate, metadata, counts)

    def release_examples(self):
        self.scratch = self.metadata = self.projected = self.gate = self.proposal = None

    def scratch_tensors(self):
        return ()
