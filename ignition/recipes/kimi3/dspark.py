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
from ignition.preparation.persistent import persistent_copy, persistent_zeros

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

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, runtime, coordinator):
        config, draft = weights.config, weights.draft
        d, eps = config.draft, config.draft.text["rms_norm_eps"]
        self.window, self.eps = d.window, eps
        tokens, device = auxiliary.shape[0], auxiliary.device
        _, kv_heads = local_heads(config, weights.shard)
        hidden = config.hidden

        def buffer(*shape, dtype=auxiliary.dtype):
            return torch.zeros((tokens, *shape), dtype=dtype, device=device)
        self.local = buffer(hidden // weights.shard.world_size)
        self.projected = buffer(hidden)
        self.normalized = buffer(hidden)
        self.project = projection(auxiliary, draft.projection, self.local, session=session, name=name + ".project",
                                  workspace=workspace)
        self.gather = PreparedChunkedAllGather(self.local, self.projected, runtime=runtime, coordinator=coordinator,
                                        session=session, name=name + ".gather")
        self.hidden_norm = draft.hidden_norm
        self.kv = buffer(draft.context_kv.shape[0])
        self.project_kv = UnquantizedProjection(self.normalized, draft.context_kv, self.kv,
                                                session=session, name=name + ".kv")
        self.key = buffer(kv_heads, d.head_dim)
        self.value = buffer(kv_heads, d.head_dim)
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.inverse = persistent_copy(inverse_frequencies(config, device))
        self.stages = []
        for index, layer in enumerate(draft.layers):
            writer = PreparedPagedKVWriter(self.key, self.value, pool.reserved_region(f"draft{index}.key"),
                pool.reserved_region(f"draft{index}.value"), self.slots, session=session, name=f"{name}.{index}.write")
            self.stages.append((canonical_norm(layer.key_norm, eps), writer.functional()))
        for operation in (self.project, self.gather, self.project_kv):
            operation.functional(session=session)
        self.key_shape = tuple(self.key.shape)
        self.local = self.projected = self.normalized = self.kv = self.key = self.value = self.slots = None

    def functional(self):
        return self

    def __call__(self, auxiliary, counts, metadata):
        m = metadata
        offset = m.positions - m.starts[m.requests]
        count = counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        slots = torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1)
        projected = self.gather(self.project(auxiliary))
        normalized = torch.empty_like(projected)
        rms_norm_columns(projected, self.hidden_norm, normalized, self.eps)
        kv = self.project_kv(normalized)
        chunks = kv.view(kv.shape[0], len(self.stages), 2, -1)
        for index, (norm, writer) in enumerate(self.stages):
            key = rotate(norm(chunks[:, index, 0].view(self.key_shape)), m.positions, self.inverse)
            value = chunks[:, index, 1].view(self.key_shape).contiguous()
            writer(key, value, slots=slots)

    def scratch_tensors(self):
        return ()


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
        self.query_shape, self.key_shape = tuple(scratch.query.shape), tuple(scratch.key.shape)
        d, eps = config.draft, config.draft.text["rms_norm_eps"]
        self.qkv = UnquantizedProjection(scratch.embedding, weights.qkv, scratch.qkv, session=session,
                                         name=name + ".qkv").functional(session=session)
        self.q_norm, self.k_norm = canonical_norm(weights.query_norm, eps), canonical_norm(weights.key_norm, eps)
        self.inverse = persistent_copy(inverse_frequencies(config, scratch.embedding.device))
        self.key_cache, self.value_cache = (pool.reserved_region(f"draft{index}.{kind}") for kind in ("key", "value"))
        rows = metadata.lengths.numel()
        count = (d.window + block + 127) // 128
        self.key, self.value = scratch.window_key[:rows * count], scratch.window_value[:rows * count]
        self.pages = persistent_copy(torch.arange(rows * count, device=scratch.embedding.device, dtype=torch.int32).view(rows, count))
        self.lengths = torch.zeros_like(metadata.lengths)
        self.window = d.window
        # Each block position attends causally to the committed window and the block's earlier positions.
        self.attention = PreparedPagedAttention(scratch.query, self.key, self.value, self.pages, self.lengths,
            metadata.query_start_loc, scratch.attended, mode="verify", width=block, window=d.window - 1,
            causal=True, session=session, workspace=workspace, name=name + ".attention",
            caller_owned_placeholders=True).functional(session=session, k_cache=None, v_cache=None,
                                                       cache_seqlens=None, cu_seqlens_q=None)
        self.output = projection(scratch.heads_output, weights.output, scratch.partial, session=session,
                                 name=name + ".output", workspace=workspace).functional(session=session)
        self.head_dim, self.query_scale = d.head_dim, (PADDED_HEAD / d.head_dim) ** .5
        self.reduce = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime, coordinator=coordinator,
                                        session=session, name=name + ".reduce").functional(session=session)
        self.window_shape = tuple(self.key.shape)
        self.key = self.value = self.lengths = None

    def forward(self, hidden_states, *, metadata, **kwargs):
        m, width = metadata, self.head_dim
        q_width, kv_width = self.query_shape[1]*width, self.key_shape[1]*width
        q, k, v = self.qkv(hidden_states).split((q_width, kv_width, kv_width), -1)
        query = torch.zeros(self.query_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        query[..., :width].copy_(rotate(self.q_norm(q.view(-1, self.query_shape[1], width)), m.positions, self.inverse) * self.query_scale)
        key = rotate(self.k_norm(k.view(self.key_shape)), m.positions, self.inverse)
        value = v.view(self.key_shape)
        staged_key = torch.zeros(self.window_shape, dtype=key.dtype, device=key.device)
        staged_value = torch.zeros_like(staged_key)
        stage_windows(self.key_cache, key, staged_key[..., :width], m)
        stage_windows(self.value_cache, value, staged_value[..., :width], m)
        lengths = m.starts.clamp(max=self.window).int() + m.lengths
        attended = self.attention(query, k_cache=staged_key, v_cache=staged_value,
                                 cache_seqlens=lengths, cu_seqlens_q=m.query_start_loc)
        heads = attended[..., :width].contiguous().flatten(1)
        return self.reduce(self.output(heads)), None


class DraftMLP(torch.nn.Module):
    def __init__(self, weights, scratch, *, runtime, coordinator, session, name, workspace):
        super().__init__()
        s = scratch.mlp
        self.mlp = PreparedMLP(scratch.embedding, weights, s.gate_up, s.intermediate, s.output,
                               session=session, workspace=workspace, name=name, activation_mode="a16").functional(session=session)
        self.reduce = PreparedAllReduce(s.output, scratch.output, runtime=runtime, coordinator=coordinator,
                                        session=session, name=name + ".reduce").functional(session=session)

    def forward(self, hidden_states):
        return self.reduce(self.mlp(hidden_states))


class PreparedDraft:
    """Project and gather draft logits for sampled proposals."""

    def __init__(self, weights, scratch, metadata, pool, *, block, runtime, vocabulary_runtime, embedding_runtime,
                 coordinator, session, name, workspace):
        self.logit_shape, self.local_logit_shape = tuple(scratch.logits.shape), tuple(scratch.local_logits.shape)
        self.logit_dtype, self.local_logit_dtype = scratch.logits.dtype, scratch.local_logits.dtype
        config, draft = weights.config, weights.draft
        eps = config.draft.text["rms_norm_eps"]
        self.embedding = PreparedShardedEmbedding(draft.embedding, metadata.token_ids, scratch.local_embedding,
            scratch.embedding, runtime=embedding_runtime, coordinator=coordinator,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
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
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head").functional(session=session)

    def release_outputs(self):
        return self

    def forward(self, metadata):
        hidden = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
        for layer in self.layers:
            hidden = layer(hidden, metadata=metadata)
        hidden = self.norm(hidden[metadata.sample_indices])
        return hidden, self.head(hidden)

    def __call__(self, metadata):
        return self.forward(metadata)[1]


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
                                       name=name + ".embed").functional(session=session, dynamic_inputs=True)
        self.projection = UnquantizedProjection(self.embedding, draft.markov_head, self.local, session=session,
                                                name=name + ".head").functional(session=session)
        # Gathering BF16 views transports the FP32 shards without conversion.
        self.gather = PreparedAllGather(self.local.view(torch.bfloat16), self.logits.view(torch.bfloat16),
            runtime=runtime, coordinator=coordinator, session=session, name=name + ".gather").functional(session=session)
        self.embedding = self.local = self.logits = None

    def forward(self, tokens):
        local = self.projection(self.embed(token_ids=tokens))
        return local, self.gather(local.view(torch.bfloat16)).view(torch.float32)

    def __call__(self, tokens):
        return self.forward(tokens)[1]


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
        context_type, draft_type, scratch_type = PreparedContext, PreparedDraft, DraftScratch
        if weights.config.dsa is not None:
            from ignition.recipes.kimi31.dspark import DraftScratch as scratch_type
            from ignition.recipes.kimi31.dspark import PreparedContext as context_type
            from ignition.recipes.kimi31.dspark import PreparedDraft as draft_type

            context_type = partial(context_type, context=collectives["context"])
            draft_type = partial(draft_type, context=collectives["context"],
                                 latents=collectives["latent_runtimes"][-1])
        self.forwards = DraftForwards(decoder,
            partial(context_type, runtime=context_runtime, coordinator=coordinator),
            partial(draft_type, block=self.block_size, embedding_runtime=collectives["embedding_runtime"]),
            PreparedMarkov)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = (persistent_zeros(requests.generation.shape, requests.generation.dtype,
            requests.generation.device) for _ in range(2))
        self.draft_scratch = scratch_type.allocate(weights.config, weights.shard,
            requests.capacity * self.block_size, self.block_size, pool.backing.device)

    def release_examples(self):
        self.scratch = self.draft_scratch = self.kda_records = self.forwards = None
