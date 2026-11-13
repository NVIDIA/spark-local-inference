# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured DFlash context projections and noncausal parallel proposals."""

from dataclasses import dataclass
from functools import partial

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

from ignition.components.collectives import PreparedAllGather, PreparedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.speculation import SpeculativeRngState
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import partial_rotary, stage_windows
from .decoder import PreparedDecoder, canonical_norm


def inverse_frequencies(geometry, device):
    return 1.0 / (geometry.theta ** (torch.arange(0, geometry.rotary_dim, 2,
        dtype=torch.float32, device=device) / geometry.rotary_dim))


class PreparedContext:
    """Project accepted target features once into all draft layers' context K/V."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, runtime, coordinator):
        self.metadata, self.pool = metadata, pool
        c, a = weights.config.draft, weights.config.draft.attention
        self.window = a.window
        self.value_scale = c.text["dflash_config"].get("attention_value_scale", 1.)
        tokens, device = auxiliary.shape[0], auxiliary.device
        heads, kv_heads = a.local(weights.shard)
        hidden = c.text["hidden_size"]
        self.local = torch.zeros((tokens, hidden // weights.shard.world_size), dtype=auxiliary.dtype, device=device)
        self.projected = torch.zeros((tokens, hidden), dtype=auxiliary.dtype, device=device)
        self.project = UnquantizedProjection(auxiliary, weights.draft.projection, self.local,
                                             session=session, name=name + ".project")
        self.gather = PreparedAllGather(self.local, self.projected, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".gather")
        self.norm = canonical_norm(Qwen3RMSNorm, weights.draft.hidden_norm, c.text["rms_norm_eps"])
        self.kv = torch.zeros((tokens, weights.draft.context_kv.shape[0]), dtype=auxiliary.dtype, device=device)
        self.project_kv = UnquantizedProjection(self.projected, weights.draft.context_kv, self.kv,
                                               session=session, name=name + ".kv")
        self.key = torch.zeros((tokens, kv_heads, a.key_dim), dtype=auxiliary.dtype, device=device)
        self.value = torch.zeros((tokens, kv_heads, a.value_dim), dtype=auxiliary.dtype, device=device)
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.inverse = inverse_frequencies(a, device)
        self.stages = []
        for index, layer in enumerate(weights.draft.layers):
            norm = canonical_norm(Qwen3RMSNorm, layer.key_norm, c.text["rms_norm_eps"])
            writer = PreparedPagedKVWriter(self.key, self.value, pool.reserved_region(f"draft{index}.key"),
                pool.reserved_region(f"draft{index}.value"), self.slots, session=session, name=f"{name}.{index}.write")
            self.stages.append((norm, writer))

    def __call__(self, auxiliary, counts):
        m = self.metadata
        offset = m.positions - m.starts[m.requests]
        count = counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        self.slots.copy_(torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1))
        self.projected.copy_(self.norm(self.gather(self.project(auxiliary))))
        self.project_kv(self.projected)
        width = self.key.shape[1] * self.key.shape[2]
        chunks = self.kv.view(self.kv.shape[0], len(self.stages), -1)
        for index, (norm, writer) in enumerate(self.stages):
            key = norm(chunks[:, index, :width].view_as(self.key))
            self.key.copy_(partial_rotary(key, m.positions, self.inverse))
            self.value.copy_(chunks[:, index, width:].view_as(self.value) * self.value_scale)
            writer()

    def scratch_tensors(self):
        return (self.local, self.projected, self.kv, self.key, self.value, self.slots)


@dataclass(frozen=True)
class DraftScratch:
    embedding: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    window_key: torch.Tensor
    window_value: torch.Tensor
    mlp: MLPScratch
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, device):
        a, c = config.attention, config.text
        heads, kv_heads = a.local(shard)
        rows = tokens // config.block_size
        pages = rows * ((a.window + config.block_size + 127) // 128)
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        hidden = c["hidden_size"]
        return cls(zeros(tokens, hidden), zeros(tokens, heads * a.key_dim + kv_heads * (a.key_dim + a.value_dim)),
            zeros(tokens, heads, a.key_dim), zeros(tokens, kv_heads, a.key_dim), zeros(tokens, kv_heads, a.value_dim),
            zeros(tokens, heads, a.value_dim), zeros(tokens, hidden), zeros(tokens, hidden),
            zeros(pages, 128, kv_heads, a.key_dim, dtype=torch.float8_e4m3fn),
            zeros(pages, 128, kv_heads, a.value_dim, dtype=torch.float8_e4m3fn),
            MLPScratch.allocate(tokens, hidden, c["intermediate_size"] // shard.world_size, device),
            zeros(tokens, hidden), zeros(tokens, c["vocab_size"] // shard.world_size), zeros(tokens, c["vocab_size"]))

    def rows(self, tokens, samples):
        if samples != tokens:
            raise ValueError("DFlash projects every block position")
        return type(self)(*(getattr(self, field) if field.startswith("window_")
                           else self.mlp.rows(tokens) if field == "mlp"
                           else getattr(self, field)[:tokens] for field in self.__dataclass_fields__))


class DraftAttention(torch.nn.Module):
    def __init__(self, index, weights, config, scratch, metadata, pool, *, runtime, coordinator, session, name, workspace):
        super().__init__()
        self.scratch, self.metadata = scratch, metadata
        a = config.attention
        self.value_scale = config.text["dflash_config"].get("attention_value_scale", 1.)
        self.qkv = UnquantizedProjection(scratch.embedding, weights.qkv, scratch.qkv, session=session, name=name + ".qkv")
        self.q_norm = canonical_norm(Qwen3RMSNorm, weights.query_norm, config.text["rms_norm_eps"])
        self.k_norm = canonical_norm(Qwen3RMSNorm, weights.key_norm, config.text["rms_norm_eps"])
        self.inverse = inverse_frequencies(a, scratch.embedding.device)
        self.key_cache, self.value_cache = (pool.reserved_region(f"draft{index}.{kind}") for kind in ("key", "value"))
        rows = metadata.lengths.numel()
        count = (a.window + config.block_size + 127) // 128
        self.key, self.value = scratch.window_key[:rows * count], scratch.window_value[:rows * count]
        self.pages = torch.arange(rows * count, device=scratch.embedding.device, dtype=torch.int32).view(rows, count)
        self.lengths = torch.zeros_like(metadata.lengths)
        self.window = a.window
        self.attention = PreparedPagedAttention(scratch.query, self.key, self.value, self.pages, self.lengths,
            metadata.query_start_loc, scratch.attended, sink=weights.sink, mode="verify", width=config.block_size,
            window=a.window - 1, causal=False, session=session, workspace=workspace, name=name + ".attention")
        self.output = UnquantizedProjection(scratch.attended.flatten(1), weights.output, scratch.partial,
                                            session=session, name=name + ".output")
        self.reduce = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".reduce")

    def forward(self, hidden_states, **kwargs):
        s, m = self.scratch, self.metadata
        q, k, v = self.qkv(hidden_states).split((s.query[0].numel(), s.key[0].numel(), s.value[0].numel()), -1)
        s.query.copy_(partial_rotary(self.q_norm(q.view_as(s.query)), m.positions, self.inverse))
        s.key.copy_(partial_rotary(self.k_norm(k.view_as(s.key)), m.positions, self.inverse))
        s.value.copy_(v.view_as(s.value) * self.value_scale)
        stage_windows(self.key_cache, s.key, self.key, m)
        stage_windows(self.value_cache, s.value, self.value, m)
        self.lengths.copy_(m.starts.clamp(max=self.window).int() + m.lengths)
        self.attention()
        return self.reduce(self.output(s.attended.flatten(1))), None


class DraftMLP(torch.nn.Module):
    def __init__(self, weights, scratch, *, runtime, coordinator, session, name, workspace):
        super().__init__()
        s = scratch.mlp
        self.mlp = PreparedMLP(scratch.embedding, weights, s.gate_up, s.intermediate, s.output,
                              session=session, workspace=workspace, name=name)
        self.reduce = PreparedAllReduce(s.output, scratch.output, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".reduce")

    def forward(self, hidden_states):
        return self.reduce(self.mlp(hidden_states))


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, runtime, vocabulary_runtime, coordinator,
                 session, name, workspace):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        c = weights.config.draft
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.layers = []
        for index, layer_weights in enumerate(weights.draft.layers):
            layer = Qwen3DecoderLayer.__new__(Qwen3DecoderLayer)
            torch.nn.Module.__init__(layer)
            layer.self_attn = DraftAttention(index, layer_weights, c, scratch, metadata, pool,
                runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.layer{index}.attention",
                workspace=workspace)
            layer.mlp = DraftMLP(layer_weights.mlp, scratch, runtime=runtime, coordinator=coordinator,
                                session=session, name=f"{name}.layer{index}.mlp", workspace=workspace)
            layer.input_layernorm = canonical_norm(Qwen3RMSNorm, layer_weights.input_norm, c.text["rms_norm_eps"])
            layer.post_attention_layernorm = canonical_norm(Qwen3RMSNorm, layer_weights.post_attention_norm,
                                                            c.text["rms_norm_eps"])
            self.layers.append(layer.eval())
        self.norm = canonical_norm(Qwen3RMSNorm, weights.draft.norm, c.text["rms_norm_eps"])
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")

    def __call__(self):
        hidden = torch.where((self.metadata.token_ids == self.weights.config.draft.noise_token)[:, None],
                             self.weights.draft.mask_embedding, self.embedding())
        for layer in self.layers:
            hidden = layer(hidden)
        self.scratch.hidden.copy_(self.norm(hidden[self.metadata.sample_indices]))
        self.head(self.scratch.hidden)


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object = PreparedDraft
    markov: object = None


class SpeculativePreparation:
    def __init__(self, weights, canonical, pool, requests, inputs, scratch, *, completion, frame, sequence,
                 session, workspace, runtime, vocabulary_runtime, context_runtime, coordinator):
        from .pool import PAGE_TOKENS

        self.page_tokens = PAGE_TOKENS
        self.weights, self.tables, self.pool = weights, canonical, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.markov_runtime = None
        self.forwards = DraftForwards(PreparedDecoder,
            partial(PreparedContext, runtime=context_runtime, coordinator=coordinator))
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        self.draft_scratch = DraftScratch.allocate(weights.config.draft, weights.shard,
            requests.capacity * self.block_size, pool.backing.device)
