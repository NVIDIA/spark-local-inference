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
from ignition.preparation.persistent import persistent_copy
from ignition.components.dflash2_preparation import DFlash2Preparation

from .attention import partial_rotary, stage_windows
from .decoder import PreparedDecoder, canonical_norm


def inverse_frequencies(geometry, device):
    return persistent_copy(1.0 / (geometry.theta ** (torch.arange(0, geometry.rotary_dim, 2,
        dtype=torch.float32, device=device) / geometry.rotary_dim)))


class PreparedContext:
    """Project accepted target features once into all draft layers' context K/V."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, runtime, coordinator):
        self.weights = weights
        c, a = weights.config.draft, weights.config.draft.attention
        self.window = a.window
        self.value_scale = c.text["dflash_config"].get("attention_value_scale", 1.)
        tokens, device = auxiliary.shape[0], auxiliary.device
        heads, kv_heads = a.local(weights.shard)
        hidden = c.text["hidden_size"]
        local = torch.zeros((tokens, hidden // weights.shard.world_size), dtype=auxiliary.dtype, device=device)
        projected = torch.zeros((tokens, hidden), dtype=auxiliary.dtype, device=device)
        self.project = UnquantizedProjection(auxiliary, weights.draft.projection, local,
                                             session=session, name=name + ".project").functional(session=session)
        self.gather = PreparedAllGather(local, projected, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".gather").functional(session=session)
        self.norm = canonical_norm(Qwen3RMSNorm, weights.draft.hidden_norm, c.text["rms_norm_eps"])
        kv = torch.zeros((tokens, weights.draft.context_kv.shape[0]), dtype=auxiliary.dtype, device=device)
        self.project_kv = UnquantizedProjection(projected, weights.draft.context_kv, kv,
                                               session=session, name=name + ".kv").functional(session=session)
        key = torch.zeros((tokens, kv_heads, a.key_dim), dtype=auxiliary.dtype, device=device)
        value = torch.zeros((tokens, kv_heads, a.value_dim), dtype=auxiliary.dtype, device=device)
        slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.inverse = inverse_frequencies(a, device)
        self.stages = []
        for index, layer in enumerate(weights.draft.layers):
            norm = canonical_norm(Qwen3RMSNorm, layer.key_norm, c.text["rms_norm_eps"])
            writer = PreparedPagedKVWriter(key, value, pool.reserved_region(f"draft{index}.key"),
                pool.reserved_region(f"draft{index}.value"), slots, session=session, name=f"{name}.{index}.write", caller_owned_placeholders=True).functional()
            self.stages.append((norm, writer))

        self.key_shape, self.value_shape = tuple(key.shape), tuple(value.shape)

    def functional(self):
        return self

    def __call__(self, auxiliary, counts, metadata):
        m = metadata
        offset = m.positions - m.starts[m.requests]
        count = counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        slots = torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1)
        projected = self.norm(self.gather(self.project(auxiliary)))
        kv = self.project_kv(projected)
        width = self.key_shape[1] * self.key_shape[2]
        chunks = kv.view(kv.shape[0], len(self.stages), -1)
        for index, (norm, writer) in enumerate(self.stages):
            key = partial_rotary(norm(chunks[:, index, :width].view(self.key_shape)), m.positions, self.inverse)
            value = (chunks[:, index, width:].view(self.value_shape) * self.value_scale).contiguous()
            writer(key, value, slots)


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
        self.query_shape, self.key_shape, self.value_shape = (tuple(value.shape) for value in (scratch.query, scratch.key, scratch.value))
        a = config.attention
        self.value_scale = config.text["dflash_config"].get("attention_value_scale", 1.)
        self.qkv = UnquantizedProjection(scratch.embedding, weights.qkv, scratch.qkv, session=session, name=name + ".qkv").functional(session=session)
        self.q_norm = canonical_norm(Qwen3RMSNorm, weights.query_norm, config.text["rms_norm_eps"])
        self.k_norm = canonical_norm(Qwen3RMSNorm, weights.key_norm, config.text["rms_norm_eps"])
        self.inverse = inverse_frequencies(a, scratch.embedding.device)
        self.key_cache, self.value_cache = (pool.reserved_region(f"draft{index}.{kind}") for kind in ("key", "value"))
        rows = metadata.lengths.numel()
        count = (a.window + config.block_size + 127) // 128
        key, value = scratch.window_key[:rows * count], scratch.window_value[:rows * count]
        self.staged_key_shape, self.staged_value_shape = tuple(key.shape), tuple(value.shape)
        pages = torch.arange(rows * count, device=scratch.embedding.device, dtype=torch.int32).view(rows, count)
        lengths = torch.zeros_like(metadata.lengths)
        self.window = a.window
        self.attention = PreparedPagedAttention(scratch.query, key, value, pages, lengths,
            metadata.query_start_loc, scratch.attended, sink=weights.sink, mode="verify", width=config.block_size,
            window=a.window - 1, causal=False, session=session, workspace=workspace, name=name + ".attention", caller_owned_placeholders=True).functional(session=session,
                k_cache=None, v_cache=None, page_table=None, cache_seqlens=None, cu_seqlens_q=None)
        self.output = UnquantizedProjection(scratch.attended.flatten(1), weights.output, scratch.partial,
                                            session=session, name=name + ".output").functional(session=session)
        self.reduce = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".reduce").functional(session=session)

    def forward(self, hidden_states, *, metadata):
        m = metadata
        widths = tuple(shape[1] * shape[2] for shape in (self.query_shape, self.key_shape, self.value_shape))
        q, k, v = self.qkv(hidden_states).split(widths, -1)
        query = partial_rotary(self.q_norm(q.view(self.query_shape)), m.positions, self.inverse)
        key = partial_rotary(self.k_norm(k.view(self.key_shape)), m.positions, self.inverse)
        value = (v.view(self.value_shape) * self.value_scale).contiguous()
        staged_key = torch.empty(self.staged_key_shape, dtype=self.key_cache.dtype, device=key.device)
        staged_value = torch.empty(self.staged_value_shape, dtype=self.value_cache.dtype, device=value.device)
        stage_windows(self.key_cache, key, staged_key, m)
        stage_windows(self.value_cache, value, staged_value, m)
        pages = torch.arange(staged_key.shape[0], dtype=torch.int32, device=key.device).view(m.lengths.numel(), -1)
        lengths = m.starts.clamp(max=self.window).int() + m.lengths
        attended = self.attention(query, k_cache=staged_key, v_cache=staged_value,
            page_table=pages, cache_seqlens=lengths, cu_seqlens_q=m.query_start_loc)
        return self.reduce(self.output(attended.flatten(1))), None


class DraftMLP(torch.nn.Module):
    def __init__(self, weights, scratch, *, runtime, coordinator, session, name, workspace):
        super().__init__()
        s = scratch.mlp
        self.mlp = PreparedMLP(scratch.embedding, weights, s.gate_up, s.intermediate, s.output,
                              session=session, workspace=workspace, name=name).functional(session=session)
        self.reduce = PreparedAllReduce(s.output, scratch.output, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".reduce").functional(session=session)

    def forward(self, hidden_states):
        return self.reduce(self.mlp(hidden_states))


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, runtime, vocabulary_runtime, coordinator,
                 session, name, workspace):
        self.weights = weights
        self.logit_shape, self.local_logit_shape = tuple(scratch.logits.shape), tuple(scratch.local_logits.shape)
        self.logit_dtype, self.local_logit_dtype = scratch.logits.dtype, scratch.local_logits.dtype
        c = weights.config.draft
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
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
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", workspace=workspace).functional(session=session)

    def release_outputs(self):
        return self

    def __call__(self, metadata):
        hidden = torch.where((metadata.token_ids == self.weights.config.draft.noise_token)[:, None],
            self.weights.draft.mask_embedding,
            self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1]))
        for layer in self.layers:
            attended, _ = layer.self_attn(layer.input_layernorm(hidden), metadata=metadata)
            hidden = hidden + attended
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = self.norm(hidden[metadata.sample_indices])
        return self.head(hidden), None


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object = PreparedDraft
    markov: object = None


class SpeculativePreparation:
    round = DFlash2Preparation.round
    def release_examples(self):
        self.scratch = self.draft_scratch = None

    def __init__(self, weights, canonical, pool, requests, inputs, scratch, *, completion, frame, sequence,
                 session, workspace, runtime, vocabulary_runtime, context_runtime, coordinator):
        from .pool import PAGE_TOKENS

        self.owners = []
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
