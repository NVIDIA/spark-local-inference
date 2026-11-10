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

from .config import PAGE_TOKENS
from .decoder import PreparedDecoder, Scratch
from .lilicorr import PreparedLatticeSelector


def _norm(source, weight, epsilon):
    values = source.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon) * weight.float()).to(source.dtype)


class Rotary:
    def __init__(self, config, device):
        c = config.text
        self.inverse = 1 / c["rope_theta"] ** (torch.arange(0, c["head_dim"], 2,
            dtype=torch.float32, device=device) / c["head_dim"])

    def __call__(self, values, positions):
        angles = positions.clamp_min(0).float()[:, None] * self.inverse[None]
        cosine, sine = angles.cos().to(values.dtype).float()[:, None], angles.sin().to(values.dtype).float()[:, None]
        left, right = values.float().chunk(2, -1)
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).to(values.dtype)


class PreparedContext:
    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, buffers):
        self.metadata, self.pool = metadata, pool
        c, draft = weights.config.draft, weights.draft
        tokens, hidden = auxiliary.shape[0], weights.config.hidden
        self.window, self.epsilon = c.text["sliding_window"], c.text["rms_norm_eps"]
        self.projected = buffers.zeros((tokens, hidden), torch.bfloat16)
        self.normalized = buffers.zeros((tokens, hidden), torch.bfloat16)
        self.project = QuantizedProjection(auxiliary, draft.projection, self.projected,
            session=session, name=name + ".project", workspace=workspace, activation_mode="auto")
        self.norm = PreparedRMSNorm(self.projected, draft.hidden_norm, self.normalized,
            eps=self.epsilon, session=session, name=name + ".norm")
        self.kv = buffers.zeros((tokens, draft.context_kv.shape[0]), torch.bfloat16)
        self.project_kv = UnquantizedProjection(self.normalized, draft.context_kv, self.kv,
            session=session, name=name + ".kv")
        shape = (tokens, c.text["num_key_value_heads"], c.text["head_dim"])
        self.key, self.value = (buffers.zeros(shape, torch.bfloat16) for _ in range(2))
        self.slots = buffers.zeros((tokens,), torch.int64)
        self.rotary = Rotary(c, auxiliary.device)
        self.anchor = pool.reserved_region("draft_anchor")
        self.stages = []
        for index, layer in enumerate(draft.layers):
            writer = PreparedPagedKVWriter(self.key, self.value, pool.reserved_region(f"draft{index}.key"),
                pool.reserved_region(f"draft{index}.value"), self.slots, session=session, name=f"{name}.{index}.write")
            self.stages.append((layer["k_norm"], writer))

    def __call__(self, auxiliary, counts):
        m = self.metadata
        offset, count = m.positions - m.starts[m.requests], counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        self.slots.copy_(torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1))
        self.norm(self.project(auxiliary))
        self.project_kv(self.normalized)
        chunks = self.kv.view(self.kv.shape[0], len(self.stages), 2, *self.key.shape[1:])
        for index, (norm, writer) in enumerate(self.stages):
            self.key.copy_(self.rotary(_norm(chunks[:, index, 0], norm, self.epsilon), m.positions))
            self.value.copy_(chunks[:, index, 1])
            writer()
        last = (m.query_start_loc[:-1] + counts - 1).long().clamp(0, self.normalized.shape[0] - 1)
        save_hidden(self.normalized[last], self.anchor, m.state_rows, counts > 0)


class DraftAttention:
    def __init__(self, weights, source, metadata, pool, config, index, *, session, name, workspace, buffers):
        self.weights, self.metadata = weights, metadata
        c, tokens = config.text, source.shape[0]
        self.window, self.epsilon = c["sliding_window"], c["rms_norm_eps"]
        qh, kh, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.projected = buffers.zeros((tokens, (qh + 2 * kh) * dim), torch.bfloat16)
        self.query = buffers.zeros((tokens, qh, dim), torch.bfloat16)
        self.key, self.value = (buffers.zeros((tokens, kh, dim), torch.bfloat16) for _ in range(2))
        self.attended = buffers.zeros(self.query.shape, torch.bfloat16)
        self.project = UnquantizedProjection(source, weights["qkv"], self.projected, session=session, name=name + ".qkv")
        self.output = UnquantizedProjection(self.attended.flatten(1), weights["output"], buffers.zeros(source.shape, torch.bfloat16),
            session=session, name=name + ".output")
        self.rotary = Rotary(config, source.device)
        self.history = tuple(pool.reserved_region(f"draft{index}.{field}") for field in ("key", "value"))
        rows, pages = metadata.lengths.numel(), (self.window + config.block_size + PAGE_TOKENS - 1) // PAGE_TOKENS
        self.staged = tuple(buffers.zeros((rows * pages, PAGE_TOKENS, kh, dim), history.dtype)
                            for history in self.history)
        self.pages = torch.arange(rows * pages, device=source.device, dtype=torch.int32).view(rows, pages)
        self.lengths = buffers.zeros((rows,), torch.int32)
        self.attention = PreparedPagedAttention(self.query, *self.staged, self.pages, self.lengths,
            metadata.query_start_loc, self.attended, sink=weights["sink"], mode="verify", width=config.block_size,
            window=self.window - 1, causal=False, session=session, workspace=workspace, name=name + ".attention")

    def __call__(self, source):
        m = self.metadata
        q, k, v = self.project(source).split((self.query[0].numel(), self.key[0].numel(), self.value[0].numel()), -1)
        self.query.copy_(self.rotary(_norm(q.view_as(self.query), self.weights["q_norm"], self.epsilon), m.positions))
        self.key.copy_(self.rotary(_norm(k.view_as(self.key), self.weights["k_norm"], self.epsilon), m.positions))
        self.value.copy_(v.view_as(self.value))
        stage_windows(self.history[0], self.key, self.staged[0], m)
        stage_windows(self.history[1], self.value, self.staged[1], m)
        self.lengths.copy_(m.starts.clamp(max=self.window).int() + m.lengths)
        self.attention()
        return self.output(self.attended.flatten(1))


class DraftMLP:
    def __init__(self, weights, source, *, session, name, workspace, buffers):
        gate, up, down = weights
        shape = (source.shape[0], gate.out_features)
        self.intermediate = buffers.zeros(shape, torch.bfloat16)
        self.gate, self.up = (QuantizedProjection(source, weight, buffers.zeros(shape, torch.bfloat16),
            session=session, name=name + suffix, workspace=workspace, activation_mode="auto")
            for weight, suffix in ((gate, ".gate"), (up, ".up")))
        self.down = QuantizedProjection(self.intermediate, down, buffers.zeros(source.shape, torch.bfloat16),
            session=session, name=name + ".down", workspace=workspace, activation_mode="auto")
        self.concurrent_up = (ConcurrentProjectionChain((self.up,), parallel_plans=(self.gate.plan,),
            session=session, workspace=workspace) if workspace.overlap else None)

    def __call__(self, source):
        if self.concurrent_up is not None:
            self.concurrent_up.start(source)
        gate = self.gate(source)
        up = (self.up(source) if self.concurrent_up is None else self.concurrent_up.finish(source, gate))
        self.intermediate.copy_((F.silu(gate.float()) * up.float()).bfloat16())
        return self.down(self.intermediate)


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, vocabulary_runtime, coordinator,
                 session, name, workspace, buffers, layer_buffers, runtime=None):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        c, s = weights.config.draft, scratch
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, s.activations,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.layers, previous = [], None
        for index, w in enumerate(weights.draft.layers):
            prefix, storage = f"{name}.layer{index}", layer_buffers.cursor()
            norm = PreparedRMSNorm(s.activations, w["input_norm"], s.normalized,
                eps=c.text["rms_norm_eps"], residual=previous, session=session, name=prefix + ".norm")
            convs = []
            for kind in ("attention_conv", "mlp_conv"):
                coefficients = 2 * c.text["dflash_config"]["conv_kernel_size"] * weights.config.hidden // c.text["dflash_config"]["conv_group_size"]
                convs.append(PreparedGroupedConvolution(s.normalized, w[kind], block_size=c.block_size,
                    group_size=c.text["dflash_config"]["conv_group_size"], session=session, name=prefix + "." + kind,
                    buffers={key: storage.zeros((s.activations.shape[0], width), torch.bfloat16)
                        for key, width in (("coefficients", coefficients), ("before", weights.config.hidden), ("after", weights.config.hidden))}))
            attention = DraftAttention(w, s.normalized, metadata, pool, c, index,
                session=session, name=prefix + ".attention", workspace=workspace, buffers=storage)
            post = PreparedRMSNorm(s.activations, w["post_norm"], s.normalized,
                eps=c.text["rms_norm_eps"], residual=s.normalized, session=session, name=prefix + ".post_norm")
            mlp = DraftMLP(w["mlp"], s.normalized, session=session, name=prefix + ".mlp", workspace=workspace, buffers=storage)
            self.layers.append((norm, convs[0], attention, post, convs[1], mlp))
            previous = s.normalized
        self.norm = PreparedRMSNorm(s.activations, weights.draft.norm, s.normalized,
            eps=c.text["rms_norm_eps"], residual=previous, session=session, name=name + ".norm")
        self.head = PreparedVocabularyProjection(s.hidden, weights.head, s.local_logits, s.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", activation_mode="auto", workspace=workspace)
        self.anchor = buffers.zeros((metadata.lengths.numel(), weights.config.hidden), torch.bfloat16)
        self.saved_anchor = pool.reserved_region("draft_anchor")
        self.selector = PreparedLatticeSelector(weights.draft.lattice, c, weights.embedding,
            s.hidden.view(-1, c.block_size, weights.config.hidden), s.logits, self.anchor,
            session=session, name=name + ".selector", workspace=workspace, buffers=buffers, layer_buffers=layer_buffers)

    def __call__(self):
        residual, output = self.embedding(), None
        for norm, conv, attention, post, mlp_conv, mlp in self.layers:
            output = conv.finish(attention(conv.prepare(norm(residual, output))))
            output = mlp_conv.finish(mlp(mlp_conv.prepare(post(residual, output))))
        self.scratch.hidden.copy_(self.norm(residual, output)[self.metadata.sample_indices])
        self.head(self.scratch.hidden)
        c = self.weights.config.draft
        self.anchor.copy_(self.saved_anchor[self.metadata.state_rows])
        self.selector.prepare(self.scratch.hidden.view(-1, c.block_size, self.weights.config.hidden),
            self.scratch.logits.view(-1, c.block_size, self.weights.config.vocabulary), self.anchor,
            (self.metadata.starts > 0) & (self.metadata.lengths > 0))


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object
    markov: object = None


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, *, completion, frame, sequence,
                 session, workspace, coordinator, buffers, layer_buffers):
        self.page_tokens = PAGE_TOKENS
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, None
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=None, vocabulary_runtime=None, coordinator=coordinator)
        self.markov_runtime = None
        self.buffers, self.layer_buffers = buffers, layer_buffers
        self.cursor = None
        self.forwards = DraftForwards(self.decoder, self.context, self.draft)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        self.draft_scratch = None

    def begin(self, batch, width, target_token_capacity):
        self.cursor = self.buffers.cursor()
        self.scratch = Scratch.allocate(target_token_capacity, batch * width, self.weights.config, self.cursor)
        self.draft_scratch = Scratch.allocate(batch * self.block_size,
            batch * self.block_size, self.weights.config, self.cursor)

    def decoder(self, weights, tables, scratch, metadata, pool, *, runtime, **kwargs):
        return PreparedDecoder(weights, scratch, metadata, pool, buffers=self.cursor, layer_buffers=self.layer_buffers, **kwargs)

    def context(self, *args, **kwargs):
        return PreparedContext(*args, buffers=self.cursor, **kwargs)

    def draft(self, *args, **kwargs):
        return PreparedDraft(*args, buffers=self.cursor, layer_buffers=self.layer_buffers, **kwargs)
