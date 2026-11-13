# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GQA DFlash2 context injection and parallel block proposals."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, rotate_half

from ignition.components.candidate_selector import PreparedCandidateSelector
from ignition.components.collectives import PreparedAllReduce, PreparedChunkedAllGather
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.grouped_convolution import PreparedGroupedConvolution
from ignition.components.linear import prepare_projection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.examples import declaration_tensor
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.window_staging import PreparedWindowStaging

PAGE_TOKENS = 128


def _norm(weight, eps):
    with torch.device("meta"):
        module = Qwen3RMSNorm(weight.numel(), eps=eps)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module.eval()


def _inverse(geometry, device):
    return persistent_copy(1.0 / geometry.theta ** (
        torch.arange(0, geometry.rotary_dim, 2, dtype=torch.float32, device=device) / geometry.rotary_dim
    ))


def _rotary(values, positions, inverse):
    frequencies = positions.clamp_min(0).float()[:, None] * inverse[None]
    angles = torch.cat((frequencies, frequencies), -1)
    cos, sin = angles.cos().to(values.dtype), angles.sin().to(values.dtype)
    return values * cos[:, None] + rotate_half(values) * sin[:, None]


class PreparedContextCapture:
    """Capture target features that can survive in a draft context ring.

    Compact prefill accepts every live input row. Verification retains all rows
    until its accepted prefix is known.
    """

    def __init__(self, metadata, columns, window, *, compact_prefill=False):
        rows = metadata.positions.numel()
        compact = compact_prefill and metadata.starts.numel() == 1 and rows > window
        if compact:
            rows = window
        self.shape = (rows, columns)
        self.device = metadata.positions.device
        self.compact = compact

    def functional(self):
        return self

    def _indices(self, metadata=None):
        if self.compact:
            m = metadata
            if m is None:
                raise ValueError("functional context capture requires runtime metadata")
            offsets = torch.arange(self.shape[0], device=m.positions.device)
            count = m.lengths[0].long()
            kept = count.clamp(max=self.shape[0])
            indices = (m.query_start_loc[0] + count - kept + offsets).clamp(
                0, m.positions.numel() - 1)
            live = (offsets < kept) & (m.positions[indices] >= 0)
            return indices, live
        return None, None

    def begin(self, metadata=None):
        return torch.empty(self.shape, dtype=torch.bfloat16, device=self.device), *self._indices(metadata)

    def __call__(self, state, hidden, column):
        output, indices, live = state
        values = hidden if indices is None else torch.where(live[:, None], hidden[indices], 0)
        output[:, column:column + hidden.shape[-1]].copy_(values)
        return state




class PreparedContext:
    """Write accepted target features into independent draft KV rings.

    Projection and staged key/value tensors are local to the traced forward.
    Compact prefill consumes a suffix captured from all live input rows; counts
    must equal the metadata lengths used by PreparedContextCapture.
    """

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace,
                 runtime, coordinator, compact_prefill=False, examples=None):
        self.weights = weights
        c, a = weights.config.draft, weights.config.draft.attention
        self.window = a.window
        tokens, hidden = auxiliary.shape[0], c.text["hidden_size"]
        compact = compact_prefill and metadata.starts.numel() == 1 and metadata.positions.numel() > self.window
        if compact:
            if tokens != self.window:
                raise ValueError("compact prefill context requires one retained window of auxiliary rows")
        self.compact = compact
        _, kv_heads = a.local(weights.shard)
        self.inverse = _inverse(a, auxiliary.device)
        local = declaration_tensor(examples, "dflash.context.local",
            (tokens, hidden // weights.shard.world_size), torch.bfloat16, auxiliary.device)
        projected = declaration_tensor(examples, "dflash.context.projected", (tokens, hidden), torch.bfloat16, auxiliary.device)
        self.project = prepare_projection(auxiliary, weights.draft.projection, local,
            session=session, name=name + ".project", workspace=workspace).functional(session=session)
        self.gather = PreparedChunkedAllGather(local, projected, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".gather")
        self.gather.functional(session=session)
        self.norm = _norm(weights.draft.hidden_norm, c.text["rms_norm_eps"])
        kv_shape = (tokens, c.layers * kv_heads * (a.key_dim + a.value_dim))
        kv = declaration_tensor(examples, "dflash.context.kv", kv_shape, torch.bfloat16, auxiliary.device)
        self.project_kv = prepare_projection(projected, weights.draft.context_kv, kv,
            session=session, name=name + ".kv", workspace=workspace).functional(session=session)
        self.layers = []
        for index, layer in enumerate(weights.draft.layers):
            norm = _norm(layer.key_norm, c.text["rms_norm_eps"])
            key = declaration_tensor(examples, "dflash.context.key", (tokens, kv_heads, a.key_dim), torch.bfloat16, auxiliary.device)
            value = declaration_tensor(examples, "dflash.context.value", (tokens, kv_heads, a.value_dim), torch.bfloat16, auxiliary.device)
            slots = torch.full((tokens,), -1, dtype=torch.int64, device=auxiliary.device)
            writer = PreparedPagedKVWriter(key, value,
                pool.reserved_region(f"draft{index}.key"), pool.reserved_region(f"draft{index}.value"),
                slots, session=session, name=f"{name}.{index}.write",
                caller_owned_placeholders=True).functional()
            self.layers.append((norm, writer))

    def functional(self):
        return self

    def __call__(self, auxiliary, counts, metadata):
        m = metadata
        if not self.compact:
            positions = m.positions
            offset = positions - m.starts[m.requests]
            count = counts[m.requests]
            valid = (positions >= 0) & (offset >= 0) & (offset < count)
            valid &= offset >= (count - self.window).clamp_min(0)
            states = m.state_rows[m.requests]
        else:
            count = counts[0].long()
            kept = count.clamp(max=self.window)
            offsets = torch.arange(auxiliary.shape[0], device=auxiliary.device)
            indices = (m.query_start_loc[0] + count - kept + offsets).clamp(
                0, m.positions.numel() - 1)
            positions = m.positions[indices]
            valid = (offsets < kept) & (positions >= 0)
            states = m.state_rows[0]
        slots = torch.where(valid, states * self.window + positions.remainder(self.window), -1)
        projected = self.norm(self.gather(self.project(auxiliary)))
        kv = self.project_kv(projected)
        a = self.weights.config.draft.attention
        _, kv_heads = a.local(self.weights.shard)
        key_shape = (auxiliary.shape[0], kv_heads, a.key_dim)
        width = kv_heads * a.key_dim
        chunks = kv.view(auxiliary.shape[0], len(self.layers), width + kv_heads * a.value_dim)
        for index, (norm, writer) in enumerate(self.layers):
            key = _rotary(norm(chunks[:, index, :width].view(key_shape)), positions, self.inverse)
            value = chunks[:, index, width:].view(auxiliary.shape[0], kv_heads, a.value_dim).contiguous()
            writer(key, value, slots)

    def scratch_tensors(self):
        return ()


@dataclass(frozen=True)
class DraftScratch:
    tokens: int
    hidden_size: int
    local_hidden_size: int
    local_vocabulary_size: int
    vocabulary_size: int
    block_size: int
    window: int

    @classmethod
    def allocate(cls, config, shard, tokens, device=None, *, block_size=None):
        block_size = config.block_size if block_size is None else block_size
        if tokens <= 0 or tokens % block_size:
            raise ValueError("DFlash2 scratch requires complete proposal blocks")
        c, a = config.text, config.attention
        vocab_width = ((c["vocab_size"] + shard.world_size - 1) // shard.world_size + 63) // 64 * 64
        return cls(tokens, c["hidden_size"], c["hidden_size"] // shard.world_size,
            vocab_width, c["vocab_size"], block_size, a.window)

    def rows(self, tokens, samples):
        if not 0 < tokens <= self.tokens or samples != tokens or tokens % self.block_size:
            raise ValueError("DFlash2 scratch requires complete proposal blocks")
        return type(self)(tokens, self.hidden_size, self.local_hidden_size, self.local_vocabulary_size,
                          self.vocabulary_size, self.block_size, self.window)

    @property
    def logit_shape(self):
        return self.tokens, self.vocabulary_size

    def scratch_tensors(self):
        return ()


def _draft_templates(config, shard, scratch, device, *, examples=None):
    attention, text = config.attention, config.text
    tokens, hidden, block = scratch.tokens, scratch.hidden_size, scratch.block_size
    heads, kv_heads = attention.local(shard)
    rows = tokens // block
    pages = rows * ((attention.window + block + PAGE_TOKENS - 1) // PAGE_TOKENS)

    def empty(role, shape, dtype=torch.bfloat16):
        return declaration_tensor(examples, "dflash.draft." + role, shape, dtype, device)

    return SimpleNamespace(
        block_size=block,
        local_embedding=empty("local_embedding", (tokens, scratch.local_hidden_size)),
        embedding=empty("embedding", (tokens, hidden)),
        normalized=empty("normalized", (tokens, hidden)),
        qkv=empty("qkv", (tokens, heads * attention.key_dim + kv_heads * (attention.key_dim + attention.value_dim))),
        query=empty("query", (tokens, heads, attention.key_dim)),
        key=empty("key", (tokens, kv_heads, attention.key_dim)),
        value=empty("value", (tokens, kv_heads, attention.value_dim)),
        attended=empty("attended", (tokens, heads, attention.value_dim)),
        partial=empty("partial", (tokens, hidden)),
        output=empty("output", (tokens, hidden)),
        window_key=empty("window_key", (pages, PAGE_TOKENS, kv_heads, attention.key_dim), torch.float8_e4m3fn),
        window_value=empty("window_value", (pages, PAGE_TOKENS, kv_heads, attention.value_dim), torch.float8_e4m3fn),
        mlp=SimpleNamespace(
            gate_up=empty("mlp.gate_up", (tokens, 2 * text["intermediate_size"] // shard.world_size)),
            intermediate=empty("mlp.intermediate", (tokens, text["intermediate_size"] // shard.world_size)),
            output=empty("mlp.output", (tokens, hidden)),
        ),
        hidden=empty("hidden", (tokens, hidden)),
        local_logits=empty("local_logits", (tokens, scratch.local_vocabulary_size)),
        gathered_logits=empty("gathered_logits", (tokens, scratch.local_vocabulary_size * shard.world_size)),
        convolution={
            "coefficients": empty("convolution.coefficients", (tokens, 2 * 2 * hidden // 16)),
            "before": empty("convolution.before", (tokens, hidden)),
            "after": empty("convolution.after", (tokens, hidden)),
        },
    )


class DraftAttention:
    """Noncausal GQA over accepted context followed by a parallel proposal block."""

    def __init__(self, index, weights, source, config, shard, scratch, metadata, pool, *,
                 runtime, coordinator, session, name, workspace):
        a, self.window = config.attention, config.attention.window
        self.block_size = scratch.block_size
        self.qkv = prepare_projection(source, weights.qkv, scratch.qkv,
            session=session, name=name + ".qkv", workspace=workspace).functional(session=session)
        self.q_norm = _norm(weights.query_norm, config.text["rms_norm_eps"])
        self.k_norm = _norm(weights.key_norm, config.text["rms_norm_eps"])
        self.inverse = _inverse(a, source.device)
        self.history_key, self.history_value = (pool.reserved_region(f"draft{index}.{kind}")
                                               for kind in ("key", "value"))
        rows = metadata.lengths.numel()
        count = (a.window + self.block_size + PAGE_TOKENS - 1) // PAGE_TOKENS
        self.staging = PreparedWindowStaging(self.history_key, self.history_value, scratch.key, scratch.value,
            scratch.window_key[:rows * count], scratch.window_value[:rows * count], metadata,
            session=session, name=name + ".staging").functional(session=session)
        self.pages = persistent_copy(
            torch.arange(rows * count, device=source.device, dtype=torch.int32).view(rows, count))
        staged_key = scratch.window_key[:rows * count]
        staged_value = scratch.window_value[:rows * count]
        self.attention = PreparedPagedAttention(scratch.query, staged_key,
            staged_value, self.pages, metadata.lengths, metadata.query_start_loc,
            scratch.attended, mode="verify", width=self.block_size, window=a.window - 1, causal=False,
            session=session, workspace=workspace, name=name + ".attention",
            caller_owned_placeholders=True).functional(session=session,
                k_cache=None, v_cache=None, cache_seqlens=None, cu_seqlens_q=None)
        self.output = prepare_projection(scratch.attended.flatten(1), weights.output, scratch.partial,
            session=session, name=name + ".output", workspace=workspace).functional(session=session)
        self.reduce = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime, coordinator=coordinator,
            session=session, name=name + ".reduce").functional(session=session)
        self.hidden_size = config.text["hidden_size"]
        self.query_shape = tuple(scratch.query.shape)
        self.key_shape = tuple(scratch.key.shape)
        self.value_shape = tuple(scratch.value.shape)

    def functional(self):
        return self

    def project(self, source, metadata):
        m = metadata
        q_width = self.query_shape[1] * self.query_shape[2]
        kv_width = self.key_shape[1] * self.key_shape[2]
        projected = self.qkv(source)
        q, k, v = projected.split((q_width, kv_width, self.value_shape[1] * self.value_shape[2]), -1)
        query = _rotary(self.q_norm(q.view(self.query_shape)), m.positions, self.inverse)
        key = _rotary(self.k_norm(k.view(self.key_shape)), m.positions, self.inverse)
        value = v.view(self.value_shape).contiguous()
        return projected, query, key, value

    def attend(self, query, key, value, metadata):
        m = metadata
        staged_key, staged_value = self.staging(key, value, metadata=m)
        lengths = torch.where(m.lengths > 0, m.starts.clamp(max=self.window) + m.lengths, 0).int()
        return self.attention(query, k_cache=staged_key, v_cache=staged_value,
                              cache_seqlens=lengths, cu_seqlens_q=m.query_start_loc)

    def __call__(self, source, metadata):
        _, query, key, value = self.project(source, metadata)
        attended = self.attend(query, key, value, metadata)
        return self.reduce(self.output(attended.flatten(1)))


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, runtime, vocabulary_runtime, embedding_runtime,
                 coordinator, session, name, workspace, embedding, head,
                 output_multiplier, final_logit_softcapping, examples=None,
                 convolution_runtime=None, templates=None, attention_factory=DraftAttention):
        self.logit_shape = scratch.logit_shape
        self.logit_dtype, self.local_logit_dtype = torch.float32, torch.bfloat16
        self.local_logit_shape = (scratch.tokens, scratch.local_vocabulary_size)
        c = weights.config.draft
        s = (_draft_templates(weights.config.draft, weights.shard, scratch, embedding.device, examples=examples)
             if templates is None else templates)
        self.embedding = PreparedShardedEmbedding(embedding, metadata.token_ids, s.local_embedding, s.embedding,
            runtime=embedding_runtime, coordinator=coordinator, session=session, name=name + ".embedding",
            num_rows=metadata.live_counts[:1]).functional(session=session, dynamic_inputs=True)
        self.layers = []
        for index, layer in enumerate(weights.draft.layers):
            prefix = f"{name}.layer{index}"
            attention_source, mlp_source = s.convolution["before"], s.convolution["before"]
            conv = {kind: PreparedGroupedConvolution(s.normalized, getattr(layer, kind), block_size=scratch.block_size,
                group_size=16, session=session, name=prefix + "." + kind, buffers=s.convolution,
                token_capacity=s.convolution["before"].shape[0], runtime=convolution_runtime,
                coordinator=coordinator) for kind in ("attention_conv", "mlp_conv")}
            for operation in conv.values():
                operation.functional(session=session)
            attention = attention_factory(index, layer, attention_source, c, weights.shard, s, metadata, pool,
                runtime=runtime, coordinator=coordinator, session=session, name=prefix + ".attention",
                workspace=workspace)
            mlp = PreparedMLP(mlp_source, layer.mlp, s.mlp.gate_up, s.mlp.intermediate, s.mlp.output,
                session=session, name=prefix + ".mlp", workspace=workspace).functional(session=session)
            reduction = PreparedAllReduce(s.mlp.output, s.output, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".mlp.reduce").functional(session=session)
            self.layers.append((_norm(layer.input_norm, c.text["rms_norm_eps"]), conv["attention_conv"], attention,
                _norm(layer.post_attention_norm, c.text["rms_norm_eps"]), conv["mlp_conv"], mlp, reduction))
        self.norm = _norm(weights.draft.norm, c.text["rms_norm_eps"])
        self.head = PreparedVocabularyProjection(s.hidden, head, s.local_logits, s.gathered_logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            activation_mode="a16", workspace=workspace).functional(session=session)
        self.selector = PreparedCandidateSelector(s.hidden.view(-1, scratch.block_size, s.hidden.shape[-1]),
            weights.draft.selector, top_k=c.text["dflash_config"]["selector_top_k"], session=session,
            name=name + ".selector").functional(session=session)
        self.block_size = scratch.block_size
        self.hidden_size = scratch.hidden_size
        self.vocabulary_size = scratch.vocabulary_size
        self.multiplier = output_multiplier
        self.softcap = final_logit_softcapping

    def release_outputs(self):
        return self

    def forward(self, metadata):
        m = metadata
        embedding = self.embedding(token_ids=m.token_ids, num_rows=m.live_counts[:1])
        for norm, conv, attention, post_norm, mlp_conv, mlp, reduce in self.layers:
            normalized = norm(embedding)
            attention_input, attention_coefficients = conv.prepare(normalized)
            attention_output = attention(attention_input, metadata=m)
            embedding = embedding + conv.finish(attention_output, attention_coefficients)
            normalized = post_norm(embedding)
            mlp_input, mlp_coefficients = mlp_conv.prepare(normalized)
            mlp_output = reduce(mlp(mlp_input))
            embedding = embedding + mlp_conv.finish(mlp_output, mlp_coefficients)
        hidden = self.norm(embedding[m.sample_indices])
        logits = self.head(hidden)
        unary = logits[:, :self.vocabulary_size].float() * self.multiplier
        logits = unary if self.softcap is None else (unary / self.softcap).tanh() * self.softcap
        selector_features = self.selector.prepare_features(hidden.view(-1, self.block_size, self.hidden_size))
        return hidden, logits, selector_features

    def __call__(self, metadata):
        _, logits, selector_features = self.forward(metadata)
        return logits, selector_features

    def scratch_tensors(self):
        return ()



@dataclass(frozen=True)
class PooledDraftScratch:
    local_embedding: torch.Tensor
    embedding: torch.Tensor
    normalized: torch.Tensor
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
    gathered_logits: torch.Tensor
    logits: torch.Tensor
    convolution: dict
    retained: list = field(default_factory=list)
    selectors: dict = field(default_factory=dict)
    block_size: int = 8

    @classmethod
    def allocate(cls, config, shard, tokens, device, *, buffers, block_size=None):
        block_size = config.block_size if block_size is None else block_size
        if tokens <= 0 or tokens % block_size:
            raise ValueError("DFlash2 scratch requires complete proposal blocks")
        s = buffers.fork()
        a, c = config.attention, config.text
        heads, kv_heads = a.local(shard)
        rows = tokens // block_size
        pages = rows * ((a.window + block_size + PAGE_TOKENS - 1) // PAGE_TOKENS)
        hidden = c["hidden_size"]
        vocab_width = ((c["vocab_size"] + shard.world_size - 1) // shard.world_size + 63) // 64 * 64
        def zeros(*shape, dtype=torch.bfloat16):
            return s.zeros(shape, dtype)
        convolution = dict(coefficients=zeros(tokens, 2 * 2 * hidden // 16),
                           before=zeros(tokens, hidden), after=zeros(tokens, hidden))
        return cls(zeros(tokens, hidden // shard.world_size), zeros(tokens, hidden), zeros(tokens, hidden),
            zeros(tokens, heads * a.key_dim + kv_heads * (a.key_dim + a.value_dim)),
            zeros(tokens, heads, a.key_dim), zeros(tokens, kv_heads, a.key_dim),
            zeros(tokens, kv_heads, a.value_dim), zeros(tokens, heads, a.value_dim),
            zeros(tokens, hidden), zeros(tokens, hidden),
            zeros(pages, PAGE_TOKENS, kv_heads, a.key_dim, dtype=torch.float8_e4m3fn),
            zeros(pages, PAGE_TOKENS, kv_heads, a.value_dim, dtype=torch.float8_e4m3fn),
            MLPScratch.allocate(tokens, hidden, c["intermediate_size"] // shard.world_size, device, buffers=s),
            zeros(tokens, hidden), zeros(tokens, vocab_width), zeros(tokens, vocab_width * shard.world_size),
            zeros(tokens, c["vocab_size"], dtype=torch.float32), convolution, block_size=block_size)

    def rows(self, tokens, samples):
        if (not 0 < tokens <= self.embedding.shape[0] or samples != tokens
                or tokens % self.block_size):
            raise ValueError("DFlash2 scratch requires complete proposal blocks")
        return type(self)(*(getattr(self, name) if name in
            ("window_key", "window_value", "convolution", "retained", "selectors", "block_size") else self.mlp.rows(tokens) if name == "mlp"
            else getattr(self, name)[:tokens] for name in self.__dataclass_fields__))

    def scratch_tensors(self):
        return tuple(getattr(self, name) for name in self.__dataclass_fields__ if name not in ("selectors", "block_size"))


class PooledDraftAttention:
    """Noncausal GQA over accepted context followed by a private eight-token block."""

    def __init__(self, index, weights, source, config, shard, scratch, metadata, pool, *,
                 runtime, coordinator, session, name, workspace, buffers):
        self.scratch, self.metadata = scratch, metadata
        a, self.window = config.attention, config.attention.window
        self.qkv = prepare_projection(source, weights.qkv, scratch.qkv, session=session, name=name + ".qkv", workspace=workspace)
        self.q_norm = _norm(weights.query_norm, config.text["rms_norm_eps"])
        self.k_norm = _norm(weights.key_norm, config.text["rms_norm_eps"])
        self.inverse = _inverse(a, source.device)
        self.history_key, self.history_value = (pool.reserved_region(f"draft{index}.{kind}")
                                               for kind in ("key", "value"))
        rows = metadata.lengths.numel()
        block_size = scratch.block_size
        count = (a.window + block_size + PAGE_TOKENS - 1) // PAGE_TOKENS
        self.staged_key, self.staged_value = scratch.window_key[:rows * count], scratch.window_value[:rows * count]
        self.staging = PreparedWindowStaging(self.history_key, self.history_value, scratch.key, scratch.value,
            self.staged_key, self.staged_value, metadata, session=session, name=name + ".staging")
        self.pages = torch.arange(rows * count, device=source.device, dtype=torch.int32).view(rows, count)
        self.lengths = buffers.zeros(metadata.lengths.shape, metadata.lengths.dtype)
        self.attention = PreparedPagedAttention(scratch.query, self.staged_key, self.staged_value,
            self.pages, self.lengths, metadata.query_start_loc, scratch.attended,
            mode="verify", width=scratch.block_size, window=a.window - 1, causal=False,
            session=session, workspace=workspace, name=name + ".attention")
        self.output = prepare_projection(scratch.attended.flatten(1), weights.output, scratch.partial,
            session=session, name=name + ".output", workspace=workspace)
        self.reduce = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime, coordinator=coordinator,
            session=session, name=name + ".reduce")

    def __call__(self, source):
        s, m = self.scratch, self.metadata
        q, k, v = self.qkv(source).split((s.query[0].numel(), s.key[0].numel(), s.value[0].numel()), -1)
        s.query.copy_(_rotary(self.q_norm(q.view_as(s.query)), m.positions, self.inverse))
        s.key.copy_(_rotary(self.k_norm(k.view_as(s.key)), m.positions, self.inverse))
        s.value.copy_(v.view_as(s.value))
        self.staging()
        self.lengths.copy_(torch.where(m.lengths > 0, m.starts.clamp(max=self.window) + m.lengths, 0).int())
        self.attention()
        return self.reduce(self.output(s.attended.flatten(1)))


class PooledPreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, runtime, vocabulary_runtime, embedding_runtime,
                 coordinator, session, name, workspace, buffers, embedding, head,
                 output_multiplier, final_logit_softcapping, attention_factory=PooledDraftAttention,
                 convolution_runtime=None):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        c, s = weights.config.draft, scratch
        stages = buffers.fork()
        self.embedding = PreparedShardedEmbedding(embedding, metadata.token_ids, s.local_embedding, s.embedding,
            runtime=embedding_runtime, coordinator=coordinator, session=session, name=name + ".embedding",
            num_rows=metadata.live_counts[:1])
        self.layers = []
        for index, layer in enumerate(weights.draft.layers):
            prefix = f"{name}.layer{index}"
            conv = {kind: PreparedGroupedConvolution(s.normalized, getattr(layer, kind), block_size=s.block_size,
                group_size=16, session=session, name=prefix + "." + kind, buffers=s.convolution,
                token_capacity=s.convolution["before"].shape[0], runtime=convolution_runtime,
                coordinator=coordinator) for kind in ("attention_conv", "mlp_conv")}
            attention = attention_factory(index, layer, conv["attention_conv"].before, c, weights.shard, s, metadata, pool,
                runtime=runtime, coordinator=coordinator, session=session, name=prefix + ".attention",
                workspace=workspace, buffers=stages)
            mlp = PreparedMLP(conv["mlp_conv"].before, layer.mlp, s.mlp.gate_up, s.mlp.intermediate, s.mlp.output,
                session=session, name=prefix + ".mlp", workspace=workspace)
            reduction = PreparedAllReduce(s.mlp.output, s.output, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".mlp.reduce")
            self.layers.append((_norm(layer.input_norm, c.text["rms_norm_eps"]), conv["attention_conv"], attention,
                _norm(layer.post_attention_norm, c.text["rms_norm_eps"]), conv["mlp_conv"], mlp, reduction))
        self.norm = _norm(weights.draft.norm, c.text["rms_norm_eps"])
        self.head = PreparedVocabularyProjection(s.hidden, head, s.local_logits, s.gathered_logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            activation_mode="a16", workspace=workspace)
        selector_key = (s.hidden.shape[0], id(weights.draft.selector))
        if selector_key not in s.selectors:
            selector = PreparedCandidateSelector(s.hidden.view(-1, s.block_size, s.hidden.shape[-1]),
                weights.draft.selector, top_k=c.text["dflash_config"]["selector_top_k"], session=session,
                name=name + ".selector")
            s.selectors[selector_key] = selector
            s.retained.append(selector.projected)
        self.selector = s.selectors[selector_key]
        self.multiplier = output_multiplier
        self.softcap = final_logit_softcapping

    def __call__(self):
        s, c = self.scratch, self.weights.config.draft
        self.embedding()
        for norm, conv, attention, post_norm, mlp_conv, mlp, reduce in self.layers:
            s.normalized.copy_(norm(s.embedding))
            s.embedding.add_(conv.finish(attention(conv.prepare(s.normalized))))
            s.normalized.copy_(post_norm(s.embedding))
            s.embedding.add_(mlp_conv.finish(reduce(mlp(mlp_conv.prepare(s.normalized)))))
        s.hidden.copy_(self.norm(s.embedding[self.metadata.sample_indices]))
        self.head(s.hidden)
        # The unary transform consumes the head result in FP32.
        unary = s.gathered_logits[:, :c.text["vocab_size"]].float() * self.multiplier
        s.logits.copy_(unary if self.softcap is None else (unary / self.softcap).tanh() * self.softcap)
        self.selector.prepare(s.hidden.view(-1, s.block_size, s.hidden.shape[-1]))

    def scratch_tensors(self):
        return self.scratch.scratch_tensors(), self.selector.projected



class PooledPreparedContext:
    """Write accepted target features into independent draft KV rings.

    The supplied cursor is dedicated to context scratch. Each graph variant forks
    its starting position; target auxiliary features must occupy disjoint storage.
    Compact prefill consumes a suffix captured from all live input rows; counts
    must equal the metadata lengths used by PreparedContextCapture.
    """

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace,
                 runtime, coordinator, buffers, compact_prefill=False):
        self.metadata = metadata
        c, a = weights.config.draft, weights.config.draft.attention
        self.window = a.window
        tokens, hidden = auxiliary.shape[0], c.text["hidden_size"]
        scratch = buffers.fork()
        self.compact_offsets = self.compact_positions = None
        if compact_prefill and metadata.starts.numel() == 1 and metadata.positions.numel() > self.window:
            if tokens != self.window:
                raise ValueError("compact prefill context requires one retained window of auxiliary rows")
            self.compact_offsets = torch.arange(tokens, device=auxiliary.device)
            self.compact_positions = scratch.full((tokens,), -1, metadata.positions.dtype)
        self.local = scratch.zeros((tokens, hidden // weights.shard.world_size), torch.bfloat16)
        self.projected = scratch.zeros((tokens, hidden), torch.bfloat16)
        _, kv_heads = a.local(weights.shard)
        self.kv = scratch.zeros((tokens, c.layers * kv_heads * (a.key_dim + a.value_dim)), torch.bfloat16)
        self.key = scratch.zeros((tokens, kv_heads, a.key_dim), torch.bfloat16)
        self.value = scratch.zeros((tokens, kv_heads, a.value_dim), torch.bfloat16)
        self.slots = scratch.full((tokens,), -1, torch.int64)
        self.inverse = _inverse(a, auxiliary.device)
        self.project = prepare_projection(auxiliary, weights.draft.projection, self.local,
            session=session, name=name + ".project", workspace=workspace)
        self.gather = PreparedChunkedAllGather(self.local, self.projected, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".gather")
        self.norm = _norm(weights.draft.hidden_norm, c.text["rms_norm_eps"])
        self.project_kv = prepare_projection(self.projected, weights.draft.context_kv, self.kv,
            session=session, name=name + ".kv", workspace=workspace)
        self.layers = []
        for index, layer in enumerate(weights.draft.layers):
            norm = _norm(layer.key_norm, c.text["rms_norm_eps"])
            writer = PreparedPagedKVWriter(self.key, self.value,
                pool.reserved_region(f"draft{index}.key"), pool.reserved_region(f"draft{index}.value"),
                self.slots, session=session, name=f"{name}.{index}.write")
            self.layers.append((norm, writer))

    def __call__(self, auxiliary, counts):
        m = self.metadata
        if self.compact_positions is None:
            positions = m.positions
            offset = positions - m.starts[m.requests]
            count = counts[m.requests]
            valid = (positions >= 0) & (offset >= 0) & (offset < count)
            valid &= offset >= (count - self.window).clamp_min(0)
            states = m.state_rows[m.requests]
        else:
            count = counts[0].long()
            kept = count.clamp(max=self.window)
            indices = (m.query_start_loc[0] + count - kept + self.compact_offsets).clamp(
                0, m.positions.numel() - 1)
            positions = m.positions[indices]
            valid = (self.compact_offsets < kept) & (positions >= 0)
            self.compact_positions.copy_(torch.where(valid, positions, -1))
            positions = self.compact_positions
            states = m.state_rows[0]
        self.slots.copy_(torch.where(valid,
            states * self.window + positions.remainder(self.window), -1))
        self.projected.copy_(self.norm(self.gather(self.project(auxiliary))))
        self.project_kv(self.projected)
        width = self.key.shape[1] * self.key.shape[2]
        chunks = self.kv.view(self.kv.shape[0], len(self.layers), 2 * width)
        for index, (norm, writer) in enumerate(self.layers):
            key = norm(chunks[:, index, :width].view_as(self.key))
            self.key.copy_(_rotary(key, positions, self.inverse))
            self.value.copy_(chunks[:, index, width:].view_as(self.value))
            writer()

    def scratch_tensors(self):
        tensors = self.local, self.projected, self.kv, self.key, self.value, self.slots
        return tensors if self.compact_positions is None else (*tensors, self.compact_positions)
