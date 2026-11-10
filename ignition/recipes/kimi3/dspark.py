# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 DSpark: accepted-state context, sliding-window Qwen3 proposals and the Markov transition.

Accepted target positions project their auxiliary AttnRes states once into every draft layer's key
and value window. A proposal block holds the pending token followed by mask tokens; the five causal
sliding-window Qwen3 layers predict one draft per block position, and a low-rank transition from the
previously sampled token biases each position's logits while drafts are sampled left to right.
"""

from dataclasses import dataclass
from functools import partial

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

from ignition.components.collectives import PreparedAllGather, PreparedAllReduce, PreparedChunkedAllGather
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.speculation import SpeculativeRngState
from ignition.components.vocabulary import PreparedVocabularyProjection

from .pointwise import rms_norm_columns


def canonical_norm(weight, eps):
    with torch.device("meta"):
        module = Qwen3RMSNorm(weight.numel(), eps=eps)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module.eval()


def inverse_frequencies(config, device):
    d = config.draft
    return 1.0 / (d.theta ** (torch.arange(0, d.head_dim, 2, dtype=torch.float32, device=device) / d.head_dim))


def rotate(values, positions, inverse):
    """Qwen3's full-head rotary embedding: FP32 angles, BF16 cosine and sine, rotate-half pairing."""
    frequencies = positions.clamp_min(0).float()[:, None] * inverse[None]
    angles = torch.cat((frequencies, frequencies), -1)
    cos, sin = angles.cos().to(values.dtype), angles.sin().to(values.dtype)
    left, right = values.chunk(2, -1)
    return values * cos[:, None] + torch.cat((-right, left), -1) * sin[:, None]


def stage_windows(history, values, output, metadata):
    """Linearize each request's window: committed history from its ring, then the block's own rows."""
    m = metadata
    rows, window = m.starts.numel(), history.shape[1]
    capacity = output.shape[0] * output.shape[1] // rows
    offsets = torch.arange(capacity, device=values.device)[None, :]
    prefix = m.starts.clamp(max=window)[:, None]
    positions = m.starts[:, None] - prefix + offsets
    previous = history[m.state_rows[:, None], positions.remainder(window)]
    indices = (m.query_start_loc[:-1, None] + offsets - prefix).clamp(0, values.shape[0] - 1)
    current = values[indices]
    valid = (m.lengths[:, None] > 0) & (offsets < prefix + m.lengths[:, None])
    output.view(rows, capacity, *values.shape[1:]).copy_(torch.where(valid[:, :, None, None],
        torch.where((offsets < prefix)[:, :, None, None], previous, current), 0))


def projection(source, weight, output, *, session, name, workspace):
    """A BF16 matrix projects as stored; online MXFP8 weights keep BF16 activations, like their BF16 inputs."""
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name)
    return QuantizedProjection(source, weight, output, session=session, name=name, activation_mode="a16",
                               workspace=workspace)


def local_heads(config, shard):
    d = config.draft
    return d.heads // shard.world_size, d.kv_heads // shard.world_size


# b12x decodes 128-wide heads; the draft's 64-wide heads attend zero-padded, with queries scaled so the kernel's
# 1/sqrt(128) softmax scale applies the canonical 1/sqrt(64).
PADDED_HEAD = 128


class PreparedContext:
    """Project accepted target auxiliary states once into every draft layer's context keys and values."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, runtime, coordinator, buffers):
        self.metadata, self.pool = metadata, pool
        config, draft = weights.config, weights.draft
        d, eps = config.draft, config.draft.text["rms_norm_eps"]
        self.window, self.eps = d.window, eps
        tokens, device = auxiliary.shape[0], auxiliary.device
        _, kv_heads = local_heads(config, weights.shard)
        hidden = config.hidden

        # Prefill and round graphs run serially, so every capacity shares the largest one's storage.
        def buffer(key, *shape, dtype=auxiliary.dtype):
            if key not in buffers:
                buffers[key] = torch.zeros((tokens, *shape), dtype=dtype, device=device)
            if buffers[key].shape[0] < tokens:
                raise ValueError("draft context storage must be allocated for the largest capacity first")
            return buffers[key][:tokens]
        self.local = buffer("local", hidden // weights.shard.world_size)
        self.projected = buffer("projected", hidden)
        self.normalized = buffer("normalized", hidden)
        self.project = projection(auxiliary, draft.projection, self.local, session=session, name=name + ".project",
                                  workspace=workspace)
        self.gather = PreparedChunkedAllGather(self.local, self.projected, runtime=runtime, coordinator=coordinator,
                                        session=session, name=name + ".gather")
        self.hidden_norm = draft.hidden_norm
        self.kv = buffer("kv", draft.context_kv.shape[0])
        self.project_kv = UnquantizedProjection(self.normalized, draft.context_kv, self.kv,
                                                session=session, name=name + ".kv")
        self.key = buffer("key", kv_heads, d.head_dim)
        self.value = buffer("value", kv_heads, d.head_dim)
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.inverse = inverse_frequencies(config, device)
        self.stages = []
        for index, layer in enumerate(draft.layers):
            writer = PreparedPagedKVWriter(self.key, self.value, pool.reserved_region(f"draft{index}.key"),
                pool.reserved_region(f"draft{index}.value"), self.slots, session=session, name=f"{name}.{index}.write")
            self.stages.append((canonical_norm(layer.key_norm, eps), writer))

    def __call__(self, auxiliary, counts):
        m = self.metadata
        offset = m.positions - m.starts[m.requests]
        count = counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        self.slots.copy_(torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1))
        self.gather(self.project(auxiliary))
        rms_norm_columns(self.projected, self.hidden_norm, self.normalized, self.eps)
        self.project_kv(self.normalized)
        chunks = self.kv.view(self.kv.shape[0], len(self.stages), 2, -1)
        for index, (norm, writer) in enumerate(self.stages):
            self.key.copy_(rotate(norm(chunks[:, index, 0].view_as(self.key)), m.positions, self.inverse))
            self.value.copy_(chunks[:, index, 1].view_as(self.value))
            writer()

    def scratch_tensors(self):
        return (self.local, self.projected, self.normalized, self.kv, self.key, self.value, self.slots)


@dataclass(frozen=True)
class DraftScratch:
    local_embedding: torch.Tensor
    embedding: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    heads_output: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    window_key: torch.Tensor
    window_value: torch.Tensor
    mlp: MLPScratch
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, block, device):
        d = config.draft
        heads, kv_heads = local_heads(config, shard)
        rows = tokens // block
        pages = rows * ((d.window + block + 127) // 128)

        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        hidden = config.hidden
        return cls(zeros(tokens, hidden // shard.world_size), zeros(tokens, hidden),
            zeros(tokens, (heads + 2 * kv_heads) * d.head_dim), zeros(tokens, heads, PADDED_HEAD),
            zeros(tokens, kv_heads, d.head_dim), zeros(tokens, kv_heads, d.head_dim),
            zeros(tokens, heads, PADDED_HEAD), zeros(tokens, heads * d.head_dim), zeros(tokens, hidden),
            zeros(tokens, hidden),
            zeros(pages, 128, kv_heads, PADDED_HEAD), zeros(pages, 128, kv_heads, PADDED_HEAD),
            MLPScratch.allocate(tokens, hidden, d.text["intermediate_size"] // shard.world_size, device),
            zeros(tokens, hidden), zeros(tokens, config.vocabulary // shard.world_size),
            zeros(tokens, config.vocabulary))

    def rows(self, tokens, samples):
        if samples != tokens:
            raise ValueError("DSpark projects every block position")
        return type(self)(*(getattr(self, field) if field.startswith("window_")
                           else self.mlp.rows(tokens) if field == "mlp"
                           else getattr(self, field)[:tokens] for field in self.__dataclass_fields__))

    def scratch_tensors(self):
        return (self,)


class DraftAttention(torch.nn.Module):
    def __init__(self, index, weights, config, scratch, metadata, pool, block, *, runtime, coordinator, session,
                 name, workspace):
        super().__init__()
        self.scratch, self.metadata = scratch, metadata
        d, eps = config.draft, config.draft.text["rms_norm_eps"]
        self.qkv = UnquantizedProjection(scratch.embedding, weights.qkv, scratch.qkv, session=session,
                                         name=name + ".qkv")
        self.q_norm, self.k_norm = canonical_norm(weights.query_norm, eps), canonical_norm(weights.key_norm, eps)
        self.inverse = inverse_frequencies(config, scratch.embedding.device)
        self.key_cache, self.value_cache = (pool.reserved_region(f"draft{index}.{kind}") for kind in ("key", "value"))
        rows = metadata.lengths.numel()
        count = (d.window + block + 127) // 128
        self.key, self.value = scratch.window_key[:rows * count], scratch.window_value[:rows * count]
        self.pages = torch.arange(rows * count, device=scratch.embedding.device, dtype=torch.int32).view(rows, count)
        self.lengths = torch.zeros_like(metadata.lengths)
        self.window = d.window
        # Each block position attends causally to the committed window and the block's earlier positions.
        self.attention = PreparedPagedAttention(scratch.query, self.key, self.value, self.pages, self.lengths,
            metadata.query_start_loc, scratch.attended, mode="verify", width=block, window=d.window - 1,
            causal=True, session=session, workspace=workspace, name=name + ".attention")
        self.output = projection(scratch.heads_output, weights.output, scratch.partial, session=session,
                                 name=name + ".output", workspace=workspace)
        self.head_dim, self.query_scale = d.head_dim, (PADDED_HEAD / d.head_dim) ** .5
        self.reduce = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime, coordinator=coordinator,
                                        session=session, name=name + ".reduce")

    def forward(self, hidden_states, **kwargs):
        s, m = self.scratch, self.metadata
        width = self.head_dim
        q, k, v = self.qkv(hidden_states).split((s.query.shape[1] * width, s.key[0].numel(), s.value[0].numel()), -1)
        query = self.q_norm(q.view(q.shape[0], -1, width))
        s.query[..., :width].copy_(rotate(query, m.positions, self.inverse) * self.query_scale)
        s.key.copy_(rotate(self.k_norm(k.view_as(s.key)), m.positions, self.inverse))
        s.value.copy_(v.view_as(s.value))
        # Padding columns of the staged windows stay zero.
        stage_windows(self.key_cache, s.key, self.key[..., :width], m)
        stage_windows(self.value_cache, s.value, self.value[..., :width], m)
        self.lengths.copy_(m.starts.clamp(max=self.window).int() + m.lengths)
        self.attention()
        s.heads_output.view(-1, s.query.shape[1], width).copy_(s.attended[..., :width])
        return self.reduce(self.output(s.heads_output)), None


class DraftMLP(torch.nn.Module):
    def __init__(self, weights, scratch, *, runtime, coordinator, session, name, workspace):
        super().__init__()
        s = scratch.mlp
        self.mlp = PreparedMLP(scratch.embedding, weights, s.gate_up, s.intermediate, s.output,
                               session=session, workspace=workspace, name=name, activation_mode="a16")
        self.reduce = PreparedAllReduce(s.output, scratch.output, runtime=runtime, coordinator=coordinator,
                                        session=session, name=name + ".reduce")

    def forward(self, hidden_states):
        return self.reduce(self.mlp(hidden_states))


class PreparedDraft:
    """Project and gather draft logits for sampled proposals."""

    def __init__(self, weights, scratch, metadata, pool, *, block, runtime, vocabulary_runtime, embedding_runtime,
                 coordinator, session, name, workspace):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        config, draft = weights.config, weights.draft
        eps = config.draft.text["rms_norm_eps"]
        self.embedding = PreparedShardedEmbedding(draft.embedding, metadata.token_ids, scratch.local_embedding,
            scratch.embedding, runtime=embedding_runtime, coordinator=coordinator,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.layers = []
        for index, layer_weights in enumerate(draft.layers):
            layer = Qwen3DecoderLayer.__new__(Qwen3DecoderLayer)
            torch.nn.Module.__init__(layer)
            layer.self_attn = DraftAttention(index, layer_weights, config, scratch, metadata, pool, block,
                runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.layer{index}.attention",
                workspace=workspace)
            layer.mlp = DraftMLP(layer_weights.mlp, scratch, runtime=runtime, coordinator=coordinator,
                                 session=session, name=f"{name}.layer{index}.mlp", workspace=workspace)
            layer.input_layernorm = canonical_norm(layer_weights.input_norm, eps)
            layer.post_attention_layernorm = canonical_norm(layer_weights.post_attention_norm, eps)
            self.layers.append(layer.eval())
        self.norm = canonical_norm(draft.norm, eps)
        head = weights.head if draft.head is None else draft.head
        self.head = PreparedVocabularyProjection(scratch.hidden, head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", workspace=workspace)

    def __call__(self):
        hidden = self.embedding.gather(self.embedding.lookup())
        for layer in self.layers:
            hidden = layer(hidden)
        self.scratch.hidden.copy_(self.norm(hidden[self.metadata.sample_indices]))
        self.head(self.scratch.hidden)


class PreparedMarkov:
    """Gather the preceding token's FP32 transition bias across vocabulary shards."""

    def __init__(self, weights, tokens, *, runtime, coordinator, session, name):
        from ignition.components.embedding import PreparedEmbedding

        rows, device = tokens.numel(), tokens.device
        draft, vocabulary = weights.draft, weights.config.vocabulary
        self.embedding = torch.zeros((rows, weights.config.draft.markov_rank), dtype=torch.bfloat16, device=device)
        self.local = torch.zeros((rows, vocabulary // weights.shard.world_size), dtype=torch.float32, device=device)
        self.logits = torch.zeros((rows, vocabulary), dtype=torch.float32, device=device)
        self.embed = PreparedEmbedding(draft.markov_embedding, tokens, self.embedding, session=session,
                                       name=name + ".embed")
        self.projection = UnquantizedProjection(self.embedding, draft.markov_head, self.local, session=session,
                                                name=name + ".head")
        # Gathering BF16 views transports the FP32 shards without conversion.
        self.gather = PreparedAllGather(self.local.view(torch.bfloat16), self.logits.view(torch.bfloat16),
            runtime=runtime, coordinator=coordinator, session=session, name=name + ".gather")

    def __call__(self):
        self.projection(self.embed())
        self.gather(self.local.view(torch.bfloat16))
        return self.logits


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object
    markov: object


class SpeculativePreparation:
    """Shared draft storage and preparation inputs of every DSpark round, consumed by PreparedParallelDraftRound.

    ``collectives`` are the target decoder's; the rounds pass only the reduction, vocabulary gather and
    coordinator to their draft modules. Verification records every KDA layer's candidates (``KDARecords``),
    so each request keeps one committed recurrent row.
    """

    def __init__(self, weights, pool, requests, inputs, scratch, *, completion, frame, sequence, session, workspace,
                 collectives, decoder_buffers, draft_tokens, context_runtime, page_tokens, markov_runtime):
        from ignition.components.kda_speculation import KDARecords
        from .decoder import PreparedDecoder

        self.page_tokens = page_tokens
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        coordinator = collectives["coordinator"]
        self.collectives = dict(runtime=collectives["runtime"], vocabulary_runtime=collectives["vocabulary_runtime"],
                                coordinator=coordinator)
        self.markov_runtime = markov_runtime
        kda_layers = {i: w["attention"] for i, w in enumerate(weights.layers) if weights.config.layers[i] == "kda"}
        heads = weights.config.text["linear_attn_config"]["num_heads"] // weights.shard.world_size
        self.kda_records = KDARecords.allocate(kda_layers, requests.capacity + 1, heads, pool.backing.device)

        def decoder(weights, tables, scratch, metadata, pool, *, mode, **kwargs):
            return PreparedDecoder(weights, scratch, metadata, pool, mode=mode, buffers=decoder_buffers,
                kda_records=self.kda_records if mode == "verify" else None, **{**collectives, **kwargs})

        # The anchor predicts the first draft, so a block holds one position per draft.
        self.block_size, self.logit_offset = max(1, draft_tokens), 0
        self.context_buffers = {}
        self.forwards = DraftForwards(decoder,
            partial(PreparedContext, runtime=context_runtime, coordinator=coordinator, buffers=self.context_buffers),
            partial(PreparedDraft, block=self.block_size, embedding_runtime=collectives["embedding_runtime"]),
            PreparedMarkov)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.draft_scratch = DraftScratch.allocate(weights.config, weights.shard,
            requests.capacity * self.block_size, self.block_size, pool.backing.device)
