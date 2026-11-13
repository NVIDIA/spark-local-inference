# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared Muse Spark text layers with normalized raw routes and vocabulary shards."""

from dataclasses import dataclass

import torch

from ignition.components.bf16_experts import BF16ExpertScratch, PreparedBF16Experts
from ignition.components.collectives import PreparedChunkedAllGather, PreparedChunkedAllReduce
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.moe import PreparedRoutedExperts, RouterScratch
from ignition.components.route_finalizer import PreparedRouteFinalizer
from ignition.components.vocabulary import PreparedVocabularyProjection
from .attention import AttentionScratch, PreparedAttention
from .model import decoder_layer, norm
from .weights import BF16Experts


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    local_embedding: torch.Tensor
    normalized: torch.Tensor
    attention: AttentionScratch
    router_source: torch.Tensor
    router_logits: torch.Tensor
    router: RouterScratch
    pre_local: torch.Tensor
    pre_full: torch.Tensor
    expert_source: torch.Tensor
    routes: torch.Tensor
    reduced_routes: torch.Tensor
    experts: torch.Tensor
    bf16: BF16ExpertScratch
    post_local: torch.Tensor
    output: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, samples, device, *, buffers=None):
        from ignition.preparation.scratch import GraphBufferPool

        if buffers is None:
            buffers = GraphBufferPool(device, chunk_bytes=2 << 20).cursor()
        c = config.text
        def zeros(shape, dtype=torch.bfloat16):
            return buffers.zeros(shape, dtype)
        embedding = zeros((tokens, config.hidden))
        local_embedding = zeros((tokens, config.hidden // shard.world_size))
        normalized = zeros((tokens, config.hidden))
        root_buffers = buffers
        attention_cursor = buffers.fork()
        attention = AttentionScratch.allocate(config, shard, tokens, device, requests=samples, buffers=attention_cursor)
        feedforward_cursor = buffers.fork()
        buffers = feedforward_cursor
        router_source = zeros((tokens, config.hidden), torch.float32)
        router_logits = zeros((tokens, c["num_local_experts"]), torch.float32)
        router = RouterScratch.allocate(tokens, device, experts=c["num_local_experts"], top_k=c["num_experts_per_tok"], buffers=buffers)
        width = c["moe_hidden_size"]
        pre_local, pre_full, expert_source = zeros((tokens, width // shard.world_size)), zeros((tokens, width)), zeros((tokens, width))
        route_shape = (tokens, c["num_experts_per_tok"], width)
        routes, reduced_routes = zeros(route_shape), zeros(route_shape)
        experts = zeros((tokens, width))
        bf16 = BF16ExpertScratch.allocate(tokens, c["num_experts_per_tok"], c["num_local_experts"],
            c["moe_intermediate_size"] // shard.world_size, buffers=buffers)
        post_local, output = zeros((tokens, config.hidden // shard.world_size)), zeros((tokens, config.hidden))
        root_buffers.join(attention_cursor, feedforward_cursor)
        buffers = root_buffers
        return cls(embedding, local_embedding, normalized, attention, router_source, router_logits, router,
            pre_local, pre_full, expert_source, routes, reduced_routes, experts, bf16, post_local, output,
            zeros((samples, config.hidden)), zeros((samples, config.vocabulary_partition(shard)[2])),
            zeros((samples, config.vocabulary)))

    def rows(self, tokens, samples):
        return type(self)(*(getattr(self, field).rows(tokens, self.router.ids.shape[1]) if field == "bf16"
            else getattr(self, field).rows(tokens) if field in ("attention", "router")
            else getattr(self, field)[:samples if field in ("hidden", "local_logits", "logits") else tokens]
            for field in self.__dataclass_fields__))


class PreparedFeedForward(torch.nn.Module):
    def __init__(self, weights, config, scratch, metadata, *, runtime, preexpert_runtime, embedding_runtime,
                 coordinator, session, name, workspace, buffers):
        super().__init__()
        self.scratch, self.metadata = scratch, metadata
        c = config.text
        self.multiplier, self.scaling = c["output_multiplier"], c["routed_scaling_factor"]
        self.correction = weights.correction
        self.router_projection = UnquantizedProjection(scratch.router_source, weights.router, scratch.router_logits,
            session=session, name=name + ".router")
        self.pre_projection = UnquantizedProjection(scratch.normalized, weights.pre_expert, scratch.pre_local,
            session=session, name=name + ".pre_projection")
        self.pre_gather = PreparedChunkedAllGather(scratch.pre_local, scratch.pre_full,
            runtime=preexpert_runtime, coordinator=coordinator, session=session, name=name + ".pre_gather")
        self.pre_norm = norm("centered", weights.pre_expert_norm, eps=c["rms_norm_eps"])
        if isinstance(weights.experts, BF16Experts):
            self.routed = PreparedBF16Experts(scratch.expert_source, weights.experts.gate_up, weights.experts.down,
                scratch.router.ids, scratch.router.weights, scratch.routes, metadata.live_counts[:1], scratch=scratch.bf16)
        else:
            self.routed = PreparedRoutedExperts(scratch.expert_source, weights.experts,
                scratch.router.ids, scratch.router.weights, scratch.routes, session=session, name=name + ".experts",
                workspace=workspace)
        route_width = c["num_experts_per_tok"] * c["moe_hidden_size"]
        self.reduce = PreparedChunkedAllReduce(scratch.routes.view(-1, route_width), scratch.reduced_routes.view(-1, route_width),
            runtime=runtime, coordinator=coordinator, session=session, name=name + ".reduce_routes")
        self.finalize = PreparedRouteFinalizer(scratch.reduced_routes, weights.post_expert_norm,
            scratch.router.weights, scratch.router.ids, metadata.live_counts[:1], scratch.experts, eps=c["post_norm_eps"], num_experts=c["num_local_experts"])
        self.post_projection = UnquantizedProjection(scratch.experts, weights.post_expert, scratch.post_local,
            session=session, name=name + ".post_projection")
        self.post_gather = PreparedChunkedAllGather(scratch.post_local, scratch.output,
            runtime=embedding_runtime, coordinator=coordinator, session=session, name=name + ".post_gather")

    def forward(self, hidden_states):
        s = self.scratch
        s.normalized.copy_(hidden_states.reshape_as(s.normalized))
        s.router_source.copy_(s.normalized)
        logits = self.router_projection(s.router_source) * self.multiplier
        scores = torch.sigmoid(logits)
        ids = torch.topk(scores + self.correction, k=s.router.ids.shape[1], dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-15) * self.scaling
        valid = torch.arange(s.normalized.shape[0], device=s.normalized.device)[:, None] < self.metadata.live_counts[0]
        s.router.ids.copy_(torch.where(valid, ids, -1).to(torch.int32))
        s.router.weights.copy_(torch.where(valid, weights, 0.))
        s.expert_source.copy_(self.pre_norm(self.pre_gather(self.pre_projection(s.normalized))))
        self.routed()
        self.reduce(s.routes.view(s.routes.shape[0], -1))
        self.finalize()
        return self.post_gather(self.post_projection(s.experts)).view_as(hidden_states)


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, embedding_runtime, preexpert_runtime,
                 vocabulary_runtime, coordinator, session, name, workspace, buffers=None, scoring=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
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
                session=session, name=prefix + ".attention", workspace=workspace, buffers=buffers)
            feedforward = PreparedFeedForward(w, weights.config, scratch, metadata, runtime=runtime,
                preexpert_runtime=preexpert_runtime, embedding_runtime=embedding_runtime, coordinator=coordinator,
                session=session, name=prefix + ".mlp", workspace=workspace, buffers=buffers)
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


class SoftcappedProjection:
    def __init__(self, projection, multiplier, softcap):
        self.projection, self.multiplier, self.softcap = projection, multiplier, softcap

    def __call__(self, source):
        value = self.projection(source)
        value.copy_(torch.tanh(value * self.multiplier / self.softcap) * self.softcap)
        return value


from ignition.components.paged_resume import PreparedResume as _PreparedResume


class PreparedResume(_PreparedResume):
    def __init__(self, weights, *args, **kwargs):
        super().__init__(weights, *args, **kwargs)
        c = weights.config.text
        self.projection = SoftcappedProjection(self.projection, c["output_multiplier"], c["final_logit_softcapping"])
