# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared Muse Glimmer text layers with dense NVFP4 SwiGLU and a softcapped vocabulary head."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.recipes.muse_spark.attention import AttentionScratch, PreparedAttention
from ignition.recipes.muse_spark.decoder import PreparedResume, SoftcappedProjection
from .model import decoder_layer, norm


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    local_embedding: torch.Tensor
    normalized: torch.Tensor
    attention: AttentionScratch
    mlp: MLPScratch
    output: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, samples, device, *, buffers=None):
        from ignition.preparation.scratch import GraphBufferPool

        if buffers is None:
            buffers = GraphBufferPool(device, chunk_bytes=2 << 20).cursor()
        def zeros(shape, dtype=torch.bfloat16):
            return buffers.zeros(shape, dtype)
        embedding = zeros((tokens, config.hidden))
        local_embedding = zeros((tokens, config.hidden // shard.world_size))
        normalized = zeros((tokens, config.hidden))
        # Verify staging must hold the drafter's whole block; without a drafter nothing is verified.
        width = 1 if config.draft is None else config.draft.block_size
        attention = AttentionScratch.allocate(config, shard, tokens, device, requests=samples, buffers=buffers,
                                              max_verify_width=width)
        mlp = MLPScratch.allocate(tokens, config.hidden, config.intermediate // shard.world_size, device, buffers=buffers)
        return cls(embedding, local_embedding, normalized, attention, mlp, zeros((tokens, config.hidden)),
            zeros((samples, config.hidden)), zeros((samples, config.vocabulary_partition(shard)[2])),
            zeros((samples, config.vocabulary)))

    def rows(self, tokens, samples):
        return type(self)(*(getattr(self, field).rows(tokens) if field in ("attention", "mlp")
            else getattr(self, field)[:samples if field in ("hidden", "local_logits", "logits") else tokens]
            for field in self.__dataclass_fields__))


class PreparedFeedForward(torch.nn.Module):
    def __init__(self, weights, scratch, *, runtime, coordinator, session, name, workspace):
        super().__init__()
        self.scratch = scratch
        self.mlp = PreparedMLP(scratch.normalized, weights.mlp, scratch.mlp.gate_up, scratch.mlp.intermediate,
                               scratch.mlp.output, session=session, name=name, workspace=workspace)
        self.reduce = PreparedAllReduce(scratch.mlp.output, scratch.output, runtime=runtime,
                                        coordinator=coordinator, session=session, name=name + ".reduce")

    def forward(self, hidden_states):
        s = self.scratch
        s.normalized.copy_(hidden_states.reshape_as(s.normalized))
        return self.reduce(self.mlp(s.normalized)).view_as(hidden_states)


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, embedding_runtime, vocabulary_runtime,
                 coordinator, session, name, workspace, buffers=None, scoring=None, vision_state=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        # Prefill with images: ``<|patch|>`` rows take published features after the embedding norm, where
        # the canonical ``MuseGlimmerModel.forward`` scatters them into the normed ``inputs_embeds``.
        self.vision_state = vision_state
        c = weights.config.text
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, scratch.local_embedding,
            scratch.embedding, runtime=embedding_runtime, coordinator=coordinator, session=session,
            name=name + ".embedding", num_rows=metadata.live_counts[:1])
        self.embedding_norm = norm("rms", dimension=c["hidden_size"], eps=c["rms_norm_eps"], with_scale=False)
        self.layers = []
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            attention = PreparedAttention(index, w, weights.config, weights.shard, scratch.normalized,
                scratch.attention, metadata, pool, mode=mode, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".attention", workspace=workspace, buffers=buffers, output_norm=False)
            feedforward = PreparedFeedForward(w, scratch, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".mlp", workspace=workspace)
            self.layers.append(decoder_layer(weights.config, w, attention, feedforward))
        self.norm = norm("rms", weights.norm, dimension=c["hidden_size"], eps=c["rms_norm_eps"])
        self.target_layers = () if weights.draft is None else weights.config.draft.target_layers
        shape = (scratch.embedding.shape[0], len(self.target_layers) * c["hidden_size"])
        self.auxiliary = (torch.zeros(shape, dtype=torch.bfloat16, device=scratch.embedding.device)
                          if buffers is None else buffers.zeros(shape, torch.bfloat16))
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=weights.config.vocabulary, session=session, coordinator=coordinator,
            name=name + ".prompt", copy_source=True)
        if self.scoring is not None:
            self.scoring.parts = [(source, SoftcappedProjection(projection, c["output_multiplier"],
                c["final_logit_softcapping"]), reduction) for source, projection, reduction in self.scoring.parts]
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", buffers=buffers)
        self.output_multiplier, self.softcap = c["output_multiplier"], c["final_logit_softcapping"]

    def __call__(self):
        hidden = self.embedding_norm(self.embedding())
        if self.vision_state is not None:
            self.vision_state.inject(self.metadata, hidden)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index in self.target_layers:
                column = self.target_layers.index(index) * hidden.shape[-1]
                self.auxiliary[:, column:column + hidden.shape[-1]].copy_(hidden)
        if self.scoring is not None:
            self.scoring.source.copy_(self.norm(hidden))
            self.scoring()
            self.scratch.hidden.copy_(self.scoring.source[self.metadata.sample_indices])
        else:
            self.scratch.hidden.copy_(self.norm(hidden[self.metadata.sample_indices]))
        logits = self.head(self.scratch.hidden)
        logits.copy_(torch.tanh(logits * self.output_multiplier / self.softcap) * self.softcap)

    def commit(self, counts):
        for layer in self.layers:
            layer.self_attn.commit(counts)

    def scratch_tensors(self):
        return (self.head.gathered, self.auxiliary, *(layer.self_attn.scratch_tensors() for layer in self.layers))

