# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Step-5 layers with FP32 residuals, expert overlap, and vocabulary sharding."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce, PreparedAllReduceRMSNorm
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.tuning import register_finalizer

from .attention import AttentionScratch, PreparedAttention


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    local_embedding: torch.Tensor
    normalized: torch.Tensor
    residual: torch.Tensor
    attention: AttentionScratch
    dense: MLPScratch
    shared: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    experts: torch.Tensor
    partial: torch.Tensor
    reduced: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, samples, context, device):
        c = config.text
        def zeros(rows, columns, dtype=torch.bfloat16):
            return torch.zeros((rows, columns), dtype=dtype, device=device)
        width = config.vocabulary_partition(shard)[2]
        embedding, local_embedding = zeros(tokens, config.hidden), zeros(tokens, config.hidden // shard.world_size)
        normalized, residual = zeros(tokens, config.hidden), zeros(tokens, config.hidden, torch.float32)
        reduced = zeros(tokens, config.hidden, torch.float32)
        attention = AttentionScratch.allocate(config, shard, tokens, context, device, requests=samples)
        dense = MLPScratch.allocate(tokens, config.hidden, c["intermediate_size"] // shard.world_size, device)
        shared = MLPScratch.allocate(tokens, config.hidden, c["share_expert_dim"] // shard.world_size, device)
        router_logits = zeros(tokens, c["moe_num_experts"], torch.float32)
        router = RouterScratch.allocate(tokens, device, experts=c["moe_num_experts"], top_k=c["moe_top_k"])
        experts = zeros(tokens, config.hidden)
        partial = zeros(tokens, config.hidden, torch.float32)
        return cls(embedding, local_embedding, normalized, residual, attention, dense, shared,
            router_logits, router, experts, partial, reduced, zeros(samples, config.hidden),
            zeros(samples, width), zeros(samples, config.vocabulary))

    def rows(self, tokens, samples):
        return type(self)(*(getattr(self, field).rows(tokens) if field in ("attention", "dense", "shared", "router")
            else getattr(self, field)[:samples if field in ("hidden", "local_logits", "logits") else tokens]
            for field in self.__dataclass_fields__))


class PreparedFeedForward(torch.nn.Module):
    def __init__(self, layer, weights, config, scratch, *, runtime, coordinator, session, name, workspace, overlap,
                 reduce_output=True, activation_mode="auto"):
        super().__init__()
        self.scratch = scratch
        source = scratch.normalized
        routed = weights.experts is not None
        mlp_scratch = scratch.shared if routed else scratch.dense
        self.mlp = PreparedMLP(source, weights.mlp, mlp_scratch.gate_up, mlp_scratch.intermediate, mlp_scratch.output,
            session=session, name=name + ".shared" if routed else name + ".dense", workspace=workspace,
            swiglu_limit=config.text["swiglu_limits_shared" if routed else "swiglu_limits"][layer],
            round_silu=False, clamp_after_silu=True, activation_mode=activation_mode)
        self.experts = self.concurrent = None
        if routed:
            c = config.text
            self.router_projection = UnquantizedProjection(source, weights.router, scratch.router_logits,
                                                           session=session, name=name + ".router").functional(session=session)
            self.router = PreparedRouter(scratch.router_logits, scratch.router, session=session, name=name + ".route",
                score_func="sigmoid", renormalize=True, routed_scaling_factor=c["moe_router_scaling_factor"],
                correction_bias=weights.correction)
            self.experts = PreparedRoutedExperts(source, weights.experts, scratch.router.ids, scratch.router.weights,
                scratch.experts, session=session, name=name + ".experts", workspace=workspace,
                router_plan=self.router.plan)
            if overlap and source.shape[0] <= 256:
                self.concurrent = ConcurrentSharedExpert(self.mlp, parallel_plans=(self.router_projection.plan,
                    self.router.plan, self.experts.plan), session=session, workspace=workspace)
                def finalize_overlap():
                    binding = getattr(self.experts, "binding", None)
                    self.concurrent.defer_launch = (
                        source.shape[0] <= 2 and binding is not None and binding.implementation == "micro")
                register_finalizer(session, finalize_overlap, workspace=workspace)
        self.reduction = PreparedAllReduce(scratch.partial, scratch.reduced, runtime=runtime, coordinator=coordinator,
                                           session=session, name=name + ".reduce").functional(session=session) if reduce_output else None
        if self.concurrent is not None:
            self.concurrent.functional(session=session)
        if self.experts is not None:
            self.experts.functional(session=session)
            self.router.functional(session=session)
        self.mlp.functional(session=session)

    def forward(self, hidden, *, metadata=None):
        if self.experts is None:
            projected = self.mlp(hidden)
            if self.reduction is None:
                return projected
            return self.reduction(projected.float()).bfloat16()
        pending = None if self.concurrent is None else self.concurrent.start(hidden)
        logits = self.router_projection(hidden)
        routes = self.router(logits)
        routed = self.experts(hidden, routes.ids, routes.weights, router_logits=logits)
        shared = self.mlp(hidden) if self.concurrent is None else self.concurrent.finish(pending, routed)
        partial = shared.float() + routed.float()
        return partial if self.reduction is None else self.reduction(partial)


class PreparedDecoderLayer(torch.nn.Module):
    def __init__(self, weights, attention, feed_forward, scratch, epsilon, *, session, name, accumulate=False,
                 runtime=None, coordinator=None, input_normalized=False, next_norm=None):
        super().__init__()
        self.attention, self.feed_forward = attention, feed_forward
        self.input_norm = None if input_normalized else PreparedRMSNorm(scratch.residual, weights.input_norm, scratch.normalized,
            eps=epsilon, session=session, name=name + ".input_norm", zero_centered=True,
            residual=scratch.reduced if accumulate else None).functional()
        self.fused_attention = attention.reduction is None
        if self.fused_attention:
            self.post_attention_norm = PreparedAllReduceRMSNorm(scratch.attention.projected_output, scratch.residual,
                weights.post_attention_norm, scratch.normalized, eps=epsilon, runtime=runtime, coordinator=coordinator,
                session=session, name=name + ".attention_reduce_norm", zero_centered=True, reduction_dtype=torch.bfloat16).functional(session=session)
        else:
            self.post_attention_norm = PreparedRMSNorm(scratch.residual, weights.post_attention_norm, scratch.normalized,
                eps=epsilon, session=session, name=name + ".post_attention_norm", zero_centered=True,
                residual=scratch.attention.projected_output).functional()
        mlp_output = scratch.dense.output if weights.experts is None else scratch.partial
        self.output_norm = None if next_norm is None else PreparedAllReduceRMSNorm(mlp_output, scratch.residual,
            next_norm, scratch.normalized, eps=epsilon, runtime=runtime, coordinator=coordinator, session=session,
            name=name + ".mlp_reduce_norm", zero_centered=True,
            reduction_dtype=torch.bfloat16 if weights.experts is None else None).functional(session=session)

    def forward(self, residual, update=None, *, metadata, proposal=None):
        hidden = update if self.input_norm is None else self.input_norm(residual, update)
        attended, records = self.attention(hidden, metadata=metadata, proposal=proposal)
        normalized = (self.post_attention_norm(attended, residual) if self.fused_attention
                      else self.post_attention_norm(residual, attended))
        update = self.feed_forward(normalized, metadata=metadata)
        return (update if self.output_norm is None else self.output_norm(update, residual)), records


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, embedding_runtime, vocabulary_runtime,
                 coordinator, session, name, workspace, overlap=True, vision=None, scoring=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.mode = mode
        self.vision = vision if mode == "prefill" else None
        self.auxiliary = ((torch.zeros_like(scratch.embedding))
                          if weights.draft is not None else None)
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, scratch.local_embedding,
            scratch.embedding, runtime=embedding_runtime, coordinator=coordinator, session=session,
            name=name + ".embedding", num_rows=metadata.live_counts[:1]).functional(session=session, dynamic_inputs=True)
        self.layers = []
        fused = PreparedAllReduceRMSNorm.fits(runtime, scratch.partial)
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            common = {"runtime": runtime, "coordinator": coordinator, "session": session, "workspace": workspace}
            attention = PreparedAttention(index, w, weights.config, weights.shard, scratch.normalized,
                scratch.attention, metadata, pool, mode=mode, name=prefix + ".attention", reduce_output=not fused,  **common)
            next_norm = weights.layers[index + 1].input_norm if fused and index + 1 < len(weights.layers) else None
            feed_forward = PreparedFeedForward(index, w, weights.config, scratch, name=prefix + ".mlp",
                                               overlap=overlap, reduce_output=next_norm is None, **common)
            self.layers.append(PreparedDecoderLayer(w, attention, feed_forward, scratch,
                weights.config.text["rms_norm_eps"], session=session, name=prefix, accumulate=index > 0,
                runtime=runtime, coordinator=coordinator, input_normalized=fused and index > 0, next_norm=next_norm))
        self.sampled_hidden = (torch.empty_like(scratch.hidden))
        self.norm = PreparedRMSNorm(self.sampled_hidden, weights.norm, scratch.hidden,
            eps=weights.config.text["rms_norm_eps"], session=session, name=name + ".norm", zero_centered=True).functional()
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator,
            name=name + ".prompt")
        self.score_norm = None if self.scoring is None else PreparedRMSNorm(scratch.embedding, weights.norm,
            scratch.embedding, eps=weights.config.text["rms_norm_eps"], session=session,
            name=name + ".prompt_norm", zero_centered=True).functional()
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",  workspace=workspace).functional(session=session)

    def __call__(self, metadata):
        local = self.embedding.lookup(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
        if self.vision is not None:
            self.vision.inject(metadata, local)
        embedded = self.embedding.gather(local)
        residual = embedded.float()
        update, records = None, []
        for layer in self.layers:
            update, retained = layer(residual, update, metadata=metadata)
            records.append(retained)
            if layer.output_norm is None:
                update = update.float()
        residual.add_(update)
        auxiliary = None if self.weights.draft is None else (residual.bfloat16(), embedded)
        hidden = self.norm(residual[metadata.sample_indices].bfloat16())
        if self.scoring is not None:
            self.scoring(self.score_norm(residual.bfloat16()))
        return hidden, self.head(hidden), auxiliary, tuple(records) if self.mode == "verify" else None

    def commit(self, counts, records, metadata):
        if records is None:
            return
        for layer, retained in zip(self.layers, records, strict=True):
            layer.attention.commit(counts, retained, metadata)

    def release_examples(self):
        self.scratch = self.metadata = self.auxiliary = self.sampled_hidden = None
        for layer in self.layers:
            layer.feed_forward.scratch = None
            layer.attention.release_examples()

    def release_outputs(self):
        self.release_examples()

    def scratch_tensors(self):
        return ()
