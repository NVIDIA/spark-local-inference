# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""LiLiCorr context injection and bidirectional draft blocks."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ignition.components.embedding import PreparedEmbedding
from ignition.components.grouped_convolution import PreparedGroupedConvolution
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.speculation import SpeculativeRngState
from ignition.components.state import save_hidden
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.window_attention import stage_windows
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.examples import declaration_tensor

from .config import PAGE_TOKENS
from .decoder import PreparedDecoder, Scratch
from .lilicorr import PreparedLatticeSelector


def _norm(source, weight, epsilon):
    values = source.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon) * weight.float()).to(source.dtype)


class Rotary:
    def __init__(self, config, device):
        c = config.text
        self.inverse = persistent_copy(1 / c["rope_theta"] ** (torch.arange(0, c["head_dim"], 2,
            dtype=torch.float32, device=device) / c["head_dim"]))

    def __call__(self, values, positions):
        angles = positions.clamp_min(0).float()[:, None] * self.inverse[None]
        cosine, sine = angles.cos().to(values.dtype).float()[:, None], angles.sin().to(values.dtype).float()[:, None]
        left, right = values.float().chunk(2, -1)
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).to(values.dtype)


class PreparedContext:
    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, examples=None):
        self.session = session
        self.anchor = pool.reserved_region("draft_anchor")
        c, draft = weights.config.draft, weights.draft
        tokens, hidden = auxiliary.shape[0], weights.config.hidden
        self.window, self.epsilon = c.text["sliding_window"], c.text["rms_norm_eps"]
        projected = declaration_tensor(examples, "nemotron.context.projected", (tokens, hidden), torch.bfloat16, auxiliary.device)
        normalized = declaration_tensor(examples, "nemotron.context.normalized", (tokens, hidden), torch.bfloat16, auxiliary.device)
        self.project = QuantizedProjection(auxiliary, draft.projection, projected,
            session=session, name=name + ".project", workspace=workspace, activation_mode="auto",
            examples=examples, example_role="nemotron.context.project")
        self.norm = PreparedRMSNorm(projected, draft.hidden_norm, normalized,
            eps=self.epsilon, session=session, name=name + ".norm")
        kv = declaration_tensor(examples, "nemotron.context.kv", (tokens, draft.context_kv.shape[0]), torch.bfloat16, auxiliary.device)
        self.project_kv = UnquantizedProjection(normalized, draft.context_kv, kv,
            session=session, name=name + ".kv")
        shape = (tokens, c.text["num_key_value_heads"], c.text["head_dim"])
        self.key_shape = shape
        key, value = (declaration_tensor(examples, "nemotron.context." + kind, shape, torch.bfloat16, auxiliary.device)
                      for kind in ("key", "value"))
        self.rotary = Rotary(c, auxiliary.device)
        self.caches = []
        self.stages = []
        for index, layer in enumerate(draft.layers):
            key_cache, value_cache = pool.reserved_region(f"draft{index}.key"), pool.reserved_region(f"draft{index}.value")
            writer = PreparedPagedKVWriter(key, value, key_cache, value_cache,
                torch.empty((tokens,), dtype=torch.int64, device=auxiliary.device),
                session=session, name=f"{name}.{index}.write", caller_owned_placeholders=True)
            self.caches.append((key_cache, value_cache))
            self.stages.append((layer["k_norm"], writer))

    def functional(self):
        self.project.functional(session=self.session)
        self.norm = self.norm.functional()
        self.project_kv.functional(session=self.session)
        self.stages = tuple((weight, writer.functional(k_cache=None, v_cache=None))
                            for weight, writer in self.stages)
        return self

    def __call__(self, auxiliary, counts, metadata):
        m = metadata
        offset, count = m.positions - m.starts[m.requests], counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        slots = torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1)
        normalized = self.norm(self.project(auxiliary))
        kv = self.project_kv(normalized)
        _, groups, dim = self.key_shape
        chunks = kv.view(kv.shape[0], len(self.stages), 2, groups, dim)
        for index, (norm, writer) in enumerate(self.stages):
            key = self.rotary(_norm(chunks[:, index, 0], norm, self.epsilon), m.positions)
            value = chunks[:, index, 1].contiguous()
            key_cache, value_cache = self.caches[index]
            writer(key, value, slots, k_cache=key_cache, v_cache=value_cache)
        last = (m.query_start_loc[:-1] + counts - 1).long().clamp(0, normalized.shape[0] - 1)
        save_hidden(normalized[last], self.anchor, m.state_rows, counts > 0)
        return normalized


class DraftAttention:
    def __init__(self, weights, source, metadata, pool, config, index, *, session, name, workspace, examples=None):
        self.weights = weights
        c, tokens = config.text, source.shape[0]
        self.window, self.epsilon = c["sliding_window"], c["rms_norm_eps"]
        qh, kh, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        projected = declaration_tensor(examples, "nemotron.draft.attention.projected",
            (tokens, (qh + 2 * kh) * dim), torch.bfloat16, source.device)
        self.project = UnquantizedProjection(source, weights["qkv"], projected, session=session, name=name + ".qkv")
        self.output = UnquantizedProjection(
            declaration_tensor(examples, "nemotron.draft.attention.attended_input", (tokens, qh * dim), torch.bfloat16, source.device),
            weights["output"], declaration_tensor(examples, "nemotron.draft.attention.output", source.shape, torch.bfloat16, source.device),
            session=session, name=name + ".output")
        self.rotary = Rotary(config, source.device)
        self.history = tuple(pool.reserved_region(f"draft{index}.{field}") for field in ("key", "value"))
        rows, pages = metadata.lengths.numel(), (self.window + config.block_size + PAGE_TOKENS - 1) // PAGE_TOKENS
        staged = tuple(declaration_tensor(examples, "nemotron.draft.attention.staged_" + kind,
            (rows * pages, PAGE_TOKENS, kh, dim), history.dtype, source.device)
            for kind, history in zip(("key", "value"), self.history, strict=True))
        query = declaration_tensor(examples, "nemotron.draft.attention.query", (tokens, qh, dim), torch.bfloat16, source.device)
        attended = declaration_tensor(examples, "nemotron.draft.attention.attended", (tokens, qh, dim), torch.bfloat16, source.device)
        self.attention = PreparedPagedAttention(query, *staged,
            torch.arange(rows * pages, device=source.device, dtype=torch.int32).view(rows, pages),
            torch.empty((rows,), dtype=torch.int32, device=source.device),
            metadata.query_start_loc, attended, sink=weights["sink"], mode="verify", width=config.block_size,
            window=self.window - 1, causal=False, session=session, workspace=workspace, name=name + ".attention",
            caller_owned_placeholders=True)
        self.window_pages, self.rows, self.qh, self.kh, self.dim = pages, rows, qh, kh, dim

    def functional(self, *, session):
        self.project.functional(session=session)
        self.output.functional(session=session)
        self.attention = self.attention.functional(session=session, k_cache=None, v_cache=None,
            cache_seqlens=None, page_table=None, cu_seqlens_q=None)
        return self

    def __call__(self, source, metadata):
        m = metadata
        projected = self.project(source)
        query_width, key_width = self.qh * self.dim, self.kh * self.dim
        q, k, v = projected.split((query_width, key_width, key_width), -1)
        query = self.rotary(_norm(q.view(-1, self.qh, self.dim), self.weights["q_norm"], self.epsilon), m.positions)
        key = self.rotary(_norm(k.view(-1, self.kh, self.dim), self.weights["k_norm"], self.epsilon), m.positions)
        value = v.view(-1, self.kh, self.dim)
        staged_key = torch.empty((self.rows * self.window_pages, PAGE_TOKENS, self.kh, self.dim),
                                 dtype=self.history[0].dtype, device=source.device)
        staged_value = torch.empty((self.rows * self.window_pages, PAGE_TOKENS, self.kh, self.dim),
                                   dtype=self.history[1].dtype, device=source.device)
        stage_windows(self.history[0], key, staged_key, m)
        stage_windows(self.history[1], value, staged_value, m)
        pages = torch.arange(self.rows * self.window_pages, device=source.device, dtype=torch.int32).view(
            self.rows, self.window_pages)
        lengths = m.starts.clamp(max=self.window).int() + m.lengths
        attended = self.attention(query, k_cache=staged_key, v_cache=staged_value,
            page_table=pages, cache_seqlens=lengths, cu_seqlens_q=m.query_start_loc)
        return self.output(attended.flatten(1))


class DraftMLP:
    def __init__(self, weights, source, *, session, name, workspace, examples=None):
        gate, up, down = weights
        shape = (source.shape[0], gate.out_features)
        self.gate, self.up = (QuantizedProjection(source, weight,
            declaration_tensor(examples, "nemotron.draft.mlp" + suffix, shape, torch.bfloat16, source.device),
            session=session, name=name + suffix, workspace=workspace, activation_mode="auto",
            examples=examples, example_role="nemotron.draft.mlp" + suffix)
            for weight, suffix in ((gate, ".gate"), (up, ".up")))
        self.down = QuantizedProjection(declaration_tensor(examples, "nemotron.draft.mlp.intermediate", shape, torch.bfloat16, source.device), down,
            declaration_tensor(examples, "nemotron.draft.mlp.output", source.shape, torch.bfloat16, source.device),
            session=session, name=name + ".down", workspace=workspace, activation_mode="auto",
            examples=examples, example_role="nemotron.draft.mlp.down")
        self.concurrent_up = (ConcurrentProjectionChain((self.up,), parallel_plans=(self.gate.plan,),
            session=session, workspace=workspace) if workspace.overlap else None)

    def functional(self, *, session):
        if self.concurrent_up is not None:
            self.concurrent_up.functional(session=session)
        else:
            self.up.functional(session=session)
        self.gate.functional(session=session)
        self.down.functional(session=session)
        return self

    def __call__(self, source):
        pending = self.concurrent_up.start(source) if self.concurrent_up is not None else None
        gate = self.gate(source)
        up = self.up(source) if pending is None else self.concurrent_up.finish(pending, gate)
        return self.down((F.silu(gate.float()) * up.float()).bfloat16())


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, vocabulary_runtime, coordinator,
                 session, name, workspace, runtime=None, examples=None):
        self.weights = weights
        c, s = weights.config.draft, scratch
        self.session = session
        self.logit_shape, self.logit_dtype = tuple(s.logits.shape), s.logits.dtype
        self.local_logit_shape, self.local_logit_dtype = tuple(s.local_logits.shape), s.local_logits.dtype
        self.hidden_shape = tuple(s.hidden.shape)
        self._examples = scratch, metadata
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, s.activations,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.layers, previous = [], None
        for index, w in enumerate(weights.draft.layers):
            prefix = f"{name}.layer{index}"
            norm = PreparedRMSNorm(s.activations, w["input_norm"], s.normalized,
                eps=c.text["rms_norm_eps"], residual=previous, session=session, name=prefix + ".norm")
            convs = []
            for kind in ("attention_conv", "mlp_conv"):
                coefficients = 2 * c.text["dflash_config"]["conv_kernel_size"] * weights.config.hidden // c.text["dflash_config"]["conv_group_size"]
                convs.append(PreparedGroupedConvolution(s.normalized, w[kind], block_size=c.block_size,
                    group_size=c.text["dflash_config"]["conv_group_size"], session=session, name=prefix + "." + kind,
                    buffers={key: declaration_tensor(examples, "nemotron.draft." + kind + "." + key,
                        (s.activations.shape[0], width), torch.bfloat16, s.activations.device)
                        for key, width in (("coefficients", coefficients), ("before", weights.config.hidden), ("after", weights.config.hidden))}))
            attention = DraftAttention(w, s.normalized, metadata, pool, c, index,
                session=session, name=prefix + ".attention", workspace=workspace, examples=examples)
            post = PreparedRMSNorm(s.activations, w["post_norm"], s.normalized,
                eps=c.text["rms_norm_eps"], residual=s.normalized, session=session, name=prefix + ".post_norm")
            mlp = DraftMLP(w["mlp"], s.normalized, session=session, name=prefix + ".mlp", workspace=workspace, examples=examples)
            self.layers.append((norm, convs[0], attention, post, convs[1], mlp))
            previous = s.normalized
        self.norm = PreparedRMSNorm(s.activations, weights.draft.norm, s.normalized,
            eps=c.text["rms_norm_eps"], residual=previous, session=session, name=name + ".norm")
        self.head = PreparedVocabularyProjection(s.hidden, weights.head, s.local_logits, s.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", activation_mode="auto", workspace=workspace)
        self.saved_anchor = pool.reserved_region("draft_anchor")
        self.selector = PreparedLatticeSelector(weights.draft.lattice, c, weights.embedding,
            s.hidden.view(-1, c.block_size, weights.config.hidden), s.logits,
            declaration_tensor(examples, "nemotron.draft.selector.anchor",
                (metadata.lengths.numel(), weights.config.hidden), torch.bfloat16, s.hidden.device),
            session=session, name=name + ".selector", workspace=workspace, examples=examples)

    def release_outputs(self):
        scratch, _ = self._examples
        self.embedding.functional(session=self.session, dynamic_inputs=True)
        self.layers = tuple((norm.functional(), conv.functional(session=self.session),
            attention.functional(session=self.session), post.functional(), mlp_conv.functional(session=self.session),
            mlp.functional(session=self.session))
            for norm, conv, attention, post, mlp_conv, mlp in self.layers)
        self.norm = self.norm.functional()
        self.head.functional(session=self.session)
        self.selector.functional(session=self.session)
        self._examples = None
        return self

    def forward(self, metadata):
        m = metadata
        residual, output = self.embedding(token_ids=m.token_ids, num_rows=m.live_counts[:1]), None
        for norm, conv, attention, post, mlp_conv, mlp in self.layers:
            normalized = norm(residual, output)
            before, coefficients = conv.prepare(normalized)
            output = conv.finish(attention(before, m), coefficients)
            normalized = post(residual, output)
            before, coefficients = mlp_conv.prepare(normalized)
            output = mlp_conv.finish(mlp(before), coefficients)
        hidden = self.norm(residual, output)[m.sample_indices]
        logits = self.head(hidden)
        c = self.weights.config.draft
        anchor = self.saved_anchor[m.state_rows]
        selector_features = self.selector.prepare(hidden.view(-1, c.block_size, self.weights.config.hidden),
            logits.view(-1, c.block_size, self.weights.config.vocabulary), anchor,
            (m.starts > 0) & (m.lengths > 0))
        return hidden, logits, selector_features

    def __call__(self, metadata):
        _, logits, selector_features = self.forward(metadata)
        return logits, selector_features


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object
    markov: object = None


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, *, completion, frame, sequence,
                 session, workspace, coordinator, examples=None):
        self.examples = examples
        self.page_tokens = PAGE_TOKENS
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, None
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=None, vocabulary_runtime=None, coordinator=coordinator)
        self.markov_runtime = None
        self.forwards = DraftForwards(self.decoder, self.context, self.draft)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        self.draft_scratch = None

    def begin(self, batch, width, target_token_capacity):
        self.scratch = Scratch.allocate(target_token_capacity, batch * width, self.weights.config, self.pool.backing.device, examples=self.examples)
        self.draft_scratch = Scratch.allocate(batch * self.block_size,
            batch * self.block_size, self.weights.config, self.pool.backing.device, examples=self.examples, role="nemotron.draft")

    def release_examples(self):
        self.scratch = self.draft_scratch = None
        self.forwards = self.examples = None

    def decoder(self, weights, tables, scratch, metadata, pool, *, runtime, **kwargs):
        return PreparedDecoder(weights, scratch, metadata, pool, examples=self.examples, **kwargs)

    def context(self, *args, **kwargs):
        return PreparedContext(*args, examples=self.examples, **kwargs)

    def draft(self, *args, **kwargs):
        return PreparedDraft(*args, examples=self.examples, **kwargs)
