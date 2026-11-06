# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared Inkling text layers, shared-expert sinks, and vocabulary projection."""

from dataclasses import dataclass

import torch

from ignition.components.bf16_experts import BF16ExpertScratch, PreparedBF16Experts
from ignition.components.collectives import PreparedChunkedAllReduce
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import PreparedRoutedExperts
from ignition.components.paged_attention import PagedAttentionMetadata
from ignition.components.residual_short_conv import PreparedResidualShortConvolution
from ignition.components.vocabulary import PreparedVocabularyProjection
from .attention import AttentionScratch, PreparedAttention
from .model import decoder_layer, mix_routes, norm, route
from .pool import PAGE_TOKENS
from .weights import Experts


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    local_embedding: torch.Tensor
    normalized: torch.Tensor
    attention: AttentionScratch
    convolved: torch.Tensor
    router_logits: torch.Tensor
    route_ids: torch.Tensor
    route_weights: torch.Tensor
    gammas: torch.Tensor
    routes: torch.Tensor
    reduced_routes: torch.Tensor
    bf16: BF16ExpertScratch
    dense: MLPScratch
    shared: MLPScratch
    partial: torch.Tensor
    reduced: torch.Tensor
    output: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, samples, device, *, buffers):
        c = config.text
        def zeros(shape, dtype=torch.bfloat16):
            return buffers.zeros(shape, dtype)
        embedding = zeros((tokens, config.hidden))
        local_embedding = zeros((tokens, config.hidden // shard.world_size))
        normalized = zeros((tokens, config.hidden))
        attention = AttentionScratch.allocate(config, shard, tokens, requests=samples, buffers=buffers)
        convolved = zeros((tokens, config.hidden))
        logits = zeros((tokens, c["n_routed_experts"] + c["n_shared_experts"]), torch.float32)
        ids = zeros((tokens, c["num_experts_per_tok"]), torch.int32)
        weights = zeros(ids.shape, torch.float32)
        gammas = zeros((tokens, c["n_shared_experts"]), torch.float32)
        shape = (tokens, c["num_experts_per_tok"], config.hidden)
        routes, reduced_routes = zeros(shape), zeros(shape)
        bf16 = BF16ExpertScratch.allocate(tokens, c["num_experts_per_tok"], c["n_routed_experts"],
            c["moe_intermediate_size"] // shard.world_size, buffers=buffers)
        dense = MLPScratch.allocate(tokens, config.hidden, c["intermediate_size"] // shard.world_size, device, buffers=buffers)
        shared = MLPScratch.allocate(tokens, config.hidden, c["moe_intermediate_size"] // shard.world_size, device, buffers=buffers)
        return cls(embedding, local_embedding, normalized, attention, convolved, logits, ids, weights, gammas,
            routes, reduced_routes, bf16, dense, shared, zeros((tokens, config.hidden)),
            zeros((tokens, config.hidden)), zeros((tokens, config.hidden)), zeros((samples, config.hidden)),
            zeros((samples, config.vocabulary_partition(shard)[2])), zeros((samples, config.vocabulary)))

    def rows(self, tokens, samples):
        return type(self)(*(getattr(self, field).rows(tokens, self.route_ids.shape[1]) if field == "bf16"
            else getattr(self, field).rows(tokens) if field in ("attention", "dense", "shared")
            else getattr(self, field)[:samples if field in ("hidden", "local_logits", "logits") else tokens]
            for field in self.__dataclass_fields__))


class PreparedFeedForward(torch.nn.Module):
    def __init__(self, weights, config, scratch, metadata, *, runtime, coordinator, session, name, workspace):
        super().__init__()
        self.scratch, self.metadata, self.config, self.weights = scratch, metadata, config, weights
        s = scratch
        self.dense = None if weights.dense is None else PreparedMLP(s.normalized, weights.dense, s.dense.gate_up,
            s.dense.intermediate, s.dense.output, session=session, name=name + ".dense", workspace=workspace, round_silu=False)
        self.reduce = PreparedChunkedAllReduce(s.partial, s.reduced, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".reduce")
        if self.dense is None:
            self.router = UnquantizedProjection(s.normalized, weights.router, s.router_logits,
                session=session, name=name + ".router")
            if isinstance(weights.experts, Experts):
                self.experts = PreparedBF16Experts(s.normalized, weights.experts.gate_up, weights.experts.down,
                    s.route_ids, s.route_weights, s.routes, metadata.live_counts[:1], scratch=s.bf16, round_silu=False)
            else:
                self.experts = PreparedRoutedExperts(s.normalized, weights.experts, s.route_ids,
                    s.route_weights, s.routes, session=session, name=name + ".experts", workspace=workspace)
            self.reduce_routes = PreparedChunkedAllReduce(s.routes.flatten(1), s.reduced_routes.flatten(1),
                runtime=runtime, coordinator=coordinator, session=session, name=name + ".reduce_routes")
            self.shared = tuple((
                UnquantizedProjection(s.normalized, w.gate_up, s.shared.gate_up, session=session, name=f"{name}.shared{index}.gate_up"),
                UnquantizedProjection(s.shared.intermediate, w.down, s.shared.output, session=session, name=f"{name}.shared{index}.down"))
                for index, w in enumerate(weights.shared))

    def forward(self, hidden_states):
        s, c = self.scratch, self.config.text
        s.normalized.copy_(hidden_states.reshape_as(s.normalized))
        if self.dense is not None:
            s.partial.copy_(self.dense(s.normalized))
            s.output.copy_(self.reduce(s.partial) * self.weights.scale)
            return s.output.view_as(hidden_states)
        ids, weights, gammas = route(self.router(s.normalized), self.weights.correction, self.weights.scale,
            top_k=c["num_experts_per_tok"], shared_experts=c["n_shared_experts"], route_scale=c["route_scale"])
        valid = torch.arange(s.normalized.shape[0], device=s.normalized.device)[:, None] < self.metadata.live_counts[0]
        s.route_ids.copy_(torch.where(valid, ids, -1).int())
        s.route_weights.copy_(torch.where(valid, weights, 0.))
        s.gammas.copy_(torch.where(valid, gammas, 0.))
        self.experts()
        s.routes.copy_(torch.where(s.route_ids[..., None] >= 0, s.routes, 0.))
        self.reduce_routes(s.routes.flatten(1))
        routed = mix_routes(s.reduced_routes, weights, ids)
        shared = torch.zeros_like(s.output, dtype=torch.float32)
        for index, (gate_up, down) in enumerate(self.shared):
            gate_up(s.normalized)
            gate, up = s.shared.gate_up.float().chunk(2, dim=-1)
            s.shared.intermediate.copy_(torch.nn.functional.silu(gate) * up * s.gammas[:, index, None])
            s.partial.copy_(down(s.shared.intermediate))
            shared = shared + self.reduce(s.partial).float()
        s.output.copy_(routed + shared.to(s.output.dtype))
        return s.output.view_as(hidden_states)


class PreparedConvolution(torch.nn.Module):
    def __init__(self, source, weight, history, metadata, output, **kwargs):
        super().__init__()
        self.source, self.output = source, output
        self.operation = PreparedResidualShortConvolution(source, weight, history, metadata, output, **kwargs)

    def forward(self, hidden_states, **kwargs):
        self.source.copy_(hidden_states.reshape_as(self.source))
        self.operation()
        return self.output.view_as(hidden_states)


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, embedding_runtime,
                 vocabulary_runtime, coordinator, session, name, workspace, buffers, scoring=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        c, s = weights.config.text, scratch
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, s.local_embedding,
            s.embedding, runtime=embedding_runtime, coordinator=coordinator, session=session,
            name=name + ".embedding", num_rows=metadata.live_counts[:1])
        self.embedding_norm = norm(weights.embedding_norm, c["rms_norm_eps"])
        self.cache_metadata = PagedAttentionMetadata(metadata, page_tokens=PAGE_TOKENS, buffers=buffers)
        self.layers = []
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            attention = PreparedAttention(index, w, weights.config, s.normalized,
                s.attention, metadata, pool, mode=mode, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".attention", buffers=buffers, cache_metadata=self.cache_metadata)
            feedforward = PreparedFeedForward(w, weights.config, s, metadata, runtime=runtime,
                coordinator=coordinator, session=session, name=prefix + ".mlp", workspace=workspace)
            def convolution(kind, source, weight):
                return PreparedConvolution(source, weight, pool.convolution(index, kind), metadata,
                    s.convolved, session=session, name=prefix + "." + kind + "_conv")
            self.layers.append(decoder_layer(weights.config, w, attention, feedforward,
                convolution("attention", s.attention.output, w.attention_conv),
                convolution("mlp", s.output, w.mlp_conv)))
        self.norm = norm(weights.norm, c["rms_norm_eps"])
        self.multiplier = c["logits_mup_width_multiplier"]
        self.scoring = None if scoring is None else scoring.prepare(s.embedding, weights.head,
            vocabulary=weights.config.vocabulary, session=session, coordinator=coordinator,
            name=name + ".prompt", copy_source=True)
        self.head = PreparedVocabularyProjection(s.hidden, weights.head, s.local_logits, s.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", buffers=buffers)

    def __call__(self):
        self.cache_metadata()
        s = self.scratch
        hidden = self.embedding_norm(self.embedding())
        for layer in self.layers:
            hidden = layer(hidden)
        if self.scoring is not None:
            self.scoring.source.copy_(self.norm(hidden) / self.multiplier)
            self.scoring()
            s.hidden.copy_(self.scoring.source[self.metadata.sample_indices])
        else:
            s.hidden.copy_(self.norm(hidden[self.metadata.sample_indices]) / self.multiplier)
        self.head(s.hidden)

    def scratch_tensors(self):
        return (self.head.gathered, self.cache_metadata.slots, self.cache_metadata.lengths,
                *(layer.self_attn.scratch_tensors() for layer in self.layers))
