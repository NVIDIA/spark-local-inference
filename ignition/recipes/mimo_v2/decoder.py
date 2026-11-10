# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical MiMo residual layers with prepared attention, experts and vocabulary shards."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import AttentionScratch, PreparedAttention


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    normalized: torch.Tensor
    attention: AttentionScratch
    dense: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    reduced: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, samples, device):
        c = config.text
        def zeros(rows, columns, dtype=torch.bfloat16):
            return torch.zeros((rows, columns), dtype=dtype, device=device)
        embedding, normalized = zeros(tokens, c["hidden_size"]), zeros(tokens, c["hidden_size"])
        reduced = zeros(tokens, c["hidden_size"])
        attention = AttentionScratch.allocate(config, shard, tokens, device, requests=samples)
        dense = MLPScratch.allocate(tokens, c["hidden_size"], c["intermediate_size"] // shard.world_size, device)
        router_logits = zeros(tokens, c["n_routed_experts"], torch.float32)
        router = RouterScratch.allocate(tokens, device, experts=c["n_routed_experts"], top_k=c["num_experts_per_tok"])
        return cls(embedding, normalized, attention, dense, router_logits, router, reduced,
            zeros(samples, c["hidden_size"]), zeros(samples, c["vocab_size"] // shard.world_size),
            zeros(samples, c["vocab_size"]))

    def rows(self, tokens, samples):
        return type(self)(self.embedding[:tokens], self.normalized[:tokens], self.attention.rows(tokens),
            self.dense.rows(tokens), self.router_logits[:tokens], self.router.rows(tokens), self.reduced[:tokens],
            self.hidden[:samples], self.local_logits[:samples], self.logits[:samples])


class PreparedFeedForward(torch.nn.Module):
    def __init__(self, weights, config, scratch, *, runtime, coordinator, session, name, workspace):
        super().__init__()
        self.dense = self.router_projection = self.router = self.experts = None
        source, output = scratch.normalized, scratch.dense.output
        if weights.mlp is not None:
            self.dense = PreparedMLP(source, weights.mlp, scratch.dense.gate_up, scratch.dense.intermediate,
                output, session=session, name=name, workspace=workspace).functional(session=session)
        else:
            self.router_projection = UnquantizedProjection(source, weights.router, scratch.router_logits,
                                                           session=session, name=name + ".gate").functional(session=session)
            c = config.text
            self.router = PreparedRouter(scratch.router_logits, scratch.router, session=session,
                name=name + ".route", score_func=c["scoring_func"], renormalize=c["norm_topk_prob"],
                routed_scaling_factor=c.get("routed_scaling_factor") or 1., correction_bias=weights.correction).functional(session=session)
            self.experts = PreparedRoutedExperts(source, weights.experts, scratch.router.ids, scratch.router.weights,
                output, session=session, name=name + ".experts", workspace=workspace,
                router_plan=self.router.plan).functional(session=session)
        self.reduction = PreparedAllReduce(output, scratch.reduced, runtime=runtime,
                                           coordinator=coordinator, session=session, name=name + ".reduce").functional(session=session)

    def forward(self, hidden_states):
        source = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.dense is not None:
            output = self.dense(source)
        else:
            logits = self.router_projection(source)
            routes = self.experts.route(logits)
            output = self.experts(source, routes.ids, routes.weights,
                router_logits=logits, selected_logits=routes.selected_logits)
        return self.reduction(output).view_as(hidden_states)


def canonical_norm(norm_type, weight, eps):
    with torch.device("meta"):
        module = norm_type(weight.numel(), eps=eps)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module.eval()


class PreparedDecoder:
    def __init__(self, weights, environment, scratch, metadata, pool, *, mode, runtime,
                 vocabulary_runtime, coordinator, session, name, workspace, scoring=None):
        layer_type, norm_type = environment.canonical
        self.weights, self.scratch, self.pool = weights, scratch, pool
        self.vision = environment.vision if mode == "prefill" else None
        c = weights.config.text
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.layers = []
        for index, layer_weights in enumerate(weights.layers):
            # Populate the canonical layer's execution modules without constructing unloaded expert tensors.
            layer = layer_type.__new__(layer_type)
            torch.nn.Module.__init__(layer)
            layer.self_attn = PreparedAttention(index, layer_weights, weights.config, weights.shard,
                scratch.normalized, scratch.attention, metadata, pool, mode=mode, runtime=runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}.attention", workspace=workspace)
            layer.mlp = PreparedFeedForward(layer_weights, weights.config, scratch, runtime=runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}.mlp", workspace=workspace)
            layer.input_layernorm = canonical_norm(norm_type, layer_weights.input_norm, c["layernorm_epsilon"])
            layer.post_attention_layernorm = canonical_norm(norm_type, layer_weights.post_attention_norm,
                                                            c["layernorm_epsilon"])
            self.layers.append(layer.eval())
        self.norm = canonical_norm(norm_type, weights.norm, c["layernorm_epsilon"])
        self.target_layers = () if weights.draft is None else weights.config.draft.target_layers
        self.auxiliary_shape = (scratch.embedding.shape[0], len(self.target_layers) * c["hidden_size"])
        self.auxiliary = torch.empty(self.auxiliary_shape, dtype=torch.bfloat16, device=scratch.embedding.device)
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator,
            name=name + ".prompt")
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", workspace=workspace).functional(session=session)

    def release_outputs(self):
        self.scratch = self.auxiliary = None

    def __call__(self, metadata):
        hidden = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
        if self.vision is not None:
            self.vision.inject(metadata, hidden)
        auxiliary = torch.empty(self.auxiliary_shape, dtype=hidden.dtype, device=hidden.device)
        records = []
        for index, layer in enumerate(self.layers):
            attended, record = layer.self_attn(layer.input_layernorm(hidden), metadata=metadata)
            hidden = hidden + attended
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            records.append(record)
            if index in self.target_layers:
                column = self.target_layers.index(index) * hidden.shape[-1]
                auxiliary[:, column:column + hidden.shape[-1]].copy_(hidden)
        if self.scoring is not None:
            normalized = self.norm(hidden)
            self.scoring(normalized)
            selected = normalized[metadata.sample_indices]
        else:
            selected = self.norm(hidden[metadata.sample_indices])
        return selected, self.head(selected), auxiliary, tuple(records)

    def commit(self, counts, records, metadata):
        for layer, record in zip(self.layers, records, strict=True):
            layer.self_attn.commit(counts, record, metadata)
