# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared dense decoder preserving canonical normalization, SwiGLU and logit softcapping."""

from dataclasses import dataclass
import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import prepare_projection
from ignition.components.paged_resume import PreparedResume as _PreparedResume
from ignition.components.vocabulary import PreparedVocabularyProjection
from .attention import AttentionScratch, PreparedAttention
from .model import decoder_layer, norm


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    normalized: torch.Tensor
    attention: AttentionScratch
    gate_up: torch.Tensor
    staging: torch.Tensor
    quantized: torch.Tensor
    activated: torch.Tensor
    output: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, tokens, samples, device, *, buffers):
        def zeros(*shape):
            return buffers.zeros(shape, torch.bfloat16)
        inter = config.text["intermediate_size"]
        embedding, normalized = zeros(tokens, config.hidden), zeros(tokens, config.hidden)
        root = buffers
        attn, mlp = buffers.fork(), buffers.fork()
        attention = AttentionScratch.allocate(config, tokens, device, buffers=attn)
        buffers = mlp
        gate_up, staging, activated = zeros(tokens, 2 * inter), zeros(tokens, inter), zeros(tokens, inter)
        quantized = buffers.zeros((tokens, max(config.hidden, inter)), torch.uint8)
        output = zeros(tokens, config.hidden)
        root.join(attn, mlp)
        buffers = root
        return cls(embedding, normalized, attention, gate_up, staging, quantized, activated, output,
            zeros(samples, config.hidden), zeros(samples, config.vocabulary), zeros(samples, config.vocabulary))

    def rows(self, tokens, samples):
        return type(self)(*(getattr(self, name).rows(tokens) if name == "attention" else
            getattr(self, name)[:samples if name in ("hidden", "local_logits", "logits") else tokens]
            for name in self.__dataclass_fields__))


class PreparedFeedForward(torch.nn.Module):
    def __init__(self, weights, scratch, *, session, name, workspace):
        super().__init__()
        self.scratch = scratch
        self.gate_up = prepare_projection(scratch.normalized, weights.gate_up, scratch.gate_up,
            session=session, name=name + ".gate_up", workspace=workspace, staging=scratch.staging,
            quantized_staging=scratch.quantized)
        self.down = prepare_projection(scratch.activated, weights.down, scratch.output,
            session=session, name=name + ".down", workspace=workspace, quantized_staging=scratch.quantized)

    def forward(self, hidden_states):
        s = self.scratch
        s.normalized.copy_(hidden_states.reshape_as(s.normalized))
        gate, up = self.gate_up(s.normalized).chunk(2, -1)
        s.activated.copy_(torch.nn.functional.silu(gate) * up)
        return self.down(s.activated).view_as(hidden_states)

    def scratch_tensors(self):
        return tuple(p.quantized for p in getattr(self.gate_up, "projections", (self.gate_up,)) if hasattr(p, "quantized")), getattr(self.down, "quantized", ())


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, session, name, workspace, buffers=None, scoring=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        c = weights.config.text
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.embedding,
            session=session, name=name + ".embedding", num_rows=metadata.live_counts[:1])
        self.embedding_norm = norm("rms", eps=c["rms_norm_eps"])
        self.layers = []
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            attention = PreparedAttention(index, w, weights.config, scratch.normalized,
                scratch.attention, metadata, pool, mode=mode, session=session, name=prefix + ".attention",
                workspace=workspace, buffers=buffers)
            feedforward = PreparedFeedForward(w, scratch, session=session, name=prefix + ".mlp", workspace=workspace)
            self.layers.append(decoder_layer(weights.config, w, attention, feedforward))
        self.norm = norm("rms", weights.norm, eps=c["rms_norm_eps"])
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=weights.config.vocabulary, session=session, coordinator=None, name=name + ".prompt", copy_source=True)
        if self.scoring is not None:
            self.scoring.parts = [(source, SoftcappedProjection(projection, c["output_multiplier"],
                c["final_logit_softcapping"]), reduction) for source, projection, reduction in self.scoring.parts]
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=None, coordinator=None, session=session, name=name + ".head", buffers=buffers, workspace=workspace)
        self.output_multiplier, self.softcap = c["output_multiplier"], c["final_logit_softcapping"]

    def __call__(self):
        hidden = self.embedding_norm(self.embedding())
        for layer in self.layers:
            hidden = layer(hidden)
        if self.scoring is not None:
            self.scoring.source.copy_(self.norm(hidden))
            self.scoring()
            self.scratch.hidden.copy_(self.scoring.source[self.metadata.sample_indices])
        else:
            self.scratch.hidden.copy_(self.norm(hidden[self.metadata.sample_indices]))
        logits = self.head(self.scratch.hidden)
        logits.copy_(torch.tanh(logits * self.output_multiplier / self.softcap) * self.softcap)

    def scratch_tensors(self):
        return (self.head.gathered, *((layer.self_attn.scratch_tensors(), layer.mlp.scratch_tensors())
                                     for layer in self.layers))


class SoftcappedProjection:
    def __init__(self, projection, multiplier, softcap):
        self.projection, self.multiplier, self.softcap = projection, multiplier, softcap

    def __call__(self, source):
        value = self.projection(source)
        value.copy_(torch.tanh(value * self.multiplier / self.softcap) * self.softcap)
        return value


class PreparedResume(_PreparedResume):
    def __init__(self, weights, *args, **kwargs):
        super().__init__(weights, *args, **kwargs)
        c = weights.config.text
        self.projection = SoftcappedProjection(self.projection, c["output_multiplier"], c["final_logit_softcapping"])
