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
    def allocate(cls, config, shard, tokens, samples, context, device, *, buffers=None):
        from ignition.preparation.scratch import GraphBufferPool

        c = config.text
        if buffers is None:
            buffers = GraphBufferPool(device, chunk_bytes=2 << 20).cursor()
        def zeros(rows, columns, dtype=torch.bfloat16, *, cursor=buffers):
            return cursor.zeros((rows, columns), dtype)
        width = config.vocabulary_partition(shard)[2]
        embedding, local_embedding = zeros(tokens, config.hidden), zeros(tokens, config.hidden // shard.world_size)
        normalized, residual = zeros(tokens, config.hidden), zeros(tokens, config.hidden, torch.float32)
        reduced = zeros(tokens, config.hidden, torch.float32)
        attn, dense_stage, routed_stage = (buffers.fork() for _ in range(3))
        attention = AttentionScratch.allocate(config, shard, tokens, context, device, requests=samples, buffers=attn)
        dense = MLPScratch.allocate(tokens, config.hidden, c["intermediate_size"] // shard.world_size, device, buffers=dense_stage)
        shared = MLPScratch.allocate(tokens, config.hidden, c["share_expert_dim"] // shard.world_size, device, buffers=routed_stage)
        router_logits = zeros(tokens, c["moe_num_experts"], torch.float32, cursor=routed_stage)
        router = RouterScratch.allocate(tokens, device, experts=c["moe_num_experts"], top_k=c["moe_top_k"], buffers=routed_stage)
        experts = zeros(tokens, config.hidden, cursor=routed_stage)
        partial = zeros(tokens, config.hidden, torch.float32, cursor=routed_stage)
        buffers.join(attn, dense_stage, routed_stage)
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
                                                           session=session, name=name + ".router")
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
                                           session=session, name=name + ".reduce") if reduce_output else None

    def forward(self, hidden):
        s = self.scratch
        s.normalized.copy_(hidden)
        if self.experts is None:
            projected = self.mlp(s.normalized)
            if self.reduction is None:
                return projected
            s.partial.copy_(projected)
        else:
            if self.concurrent is not None:
                self.concurrent.start(s.normalized)
            self.router(self.router_projection(s.normalized))
            routed = self.experts()
            shared = self.mlp(s.normalized) if self.concurrent is None else self.concurrent.finish(s.normalized, routed)
            s.partial.copy_(shared.float() + routed.float())
        if self.reduction is None:
            return s.partial
        reduced = self.reduction(s.partial)
        return reduced.bfloat16() if self.experts is None else reduced


class PreparedDecoderLayer(torch.nn.Module):
    def __init__(self, weights, attention, feed_forward, scratch, epsilon, *, session, name, accumulate=False,
                 runtime=None, coordinator=None, input_normalized=False, next_norm=None):
        super().__init__()
        self.attention, self.feed_forward = attention, feed_forward
        self.input_norm = None if input_normalized else PreparedRMSNorm(scratch.residual, weights.input_norm, scratch.normalized,
            eps=epsilon, session=session, name=name + ".input_norm", zero_centered=True,
            residual=scratch.reduced if accumulate else None)
        self.fused_attention = attention.reduction is None
        if self.fused_attention:
            self.post_attention_norm = PreparedAllReduceRMSNorm(scratch.attention.projected_output, scratch.residual,
                weights.post_attention_norm, scratch.normalized, eps=epsilon, runtime=runtime, coordinator=coordinator,
                session=session, name=name + ".attention_reduce_norm", zero_centered=True, reduction_dtype=torch.bfloat16)
        else:
            self.post_attention_norm = PreparedRMSNorm(scratch.residual, weights.post_attention_norm, scratch.normalized,
                eps=epsilon, session=session, name=name + ".post_attention_norm", zero_centered=True,
                residual=scratch.attention.projected_output)
        mlp_output = scratch.dense.output if weights.experts is None else scratch.partial
        self.output_norm = None if next_norm is None else PreparedAllReduceRMSNorm(mlp_output, scratch.residual,
            next_norm, scratch.normalized, eps=epsilon, runtime=runtime, coordinator=coordinator, session=session,
            name=name + ".mlp_reduce_norm", zero_centered=True,
            reduction_dtype=torch.bfloat16 if weights.experts is None else None)

    def forward(self, residual, update=None):
        hidden = update if self.input_norm is None else self.input_norm(residual, update)
        attended = self.attention(hidden)
        normalized = (self.post_attention_norm(attended, residual) if self.fused_attention
                      else self.post_attention_norm(residual, attended))
        update = self.feed_forward(normalized)
        return update if self.output_norm is None else self.output_norm(update, residual)


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, embedding_runtime, vocabulary_runtime,
                 coordinator, session, name, workspace, overlap=True, vision=None, buffers=None, scoring=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.vision = vision if mode == "prefill" else None
        self.auxiliary = ((torch.zeros_like(scratch.embedding) if buffers is None
                           else buffers.zeros(scratch.embedding.shape, scratch.embedding.dtype))
                          if weights.draft is not None else None)
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, scratch.local_embedding,
            scratch.embedding, runtime=embedding_runtime, coordinator=coordinator, session=session,
            name=name + ".embedding", num_rows=metadata.live_counts[:1])
        self.layers = []
        fused = PreparedAllReduceRMSNorm.fits(runtime, scratch.partial)
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            common = {"runtime": runtime, "coordinator": coordinator, "session": session, "workspace": workspace}
            attention = PreparedAttention(index, w, weights.config, weights.shard, scratch.normalized,
                scratch.attention, metadata, pool, mode=mode, name=prefix + ".attention", reduce_output=not fused, buffers=buffers, **common)
            next_norm = weights.layers[index + 1].input_norm if fused and index + 1 < len(weights.layers) else None
            feed_forward = PreparedFeedForward(index, w, weights.config, scratch, name=prefix + ".mlp",
                                               overlap=overlap, reduce_output=next_norm is None, **common)
            self.layers.append(PreparedDecoderLayer(w, attention, feed_forward, scratch,
                weights.config.text["rms_norm_eps"], session=session, name=prefix, accumulate=index > 0,
                runtime=runtime, coordinator=coordinator, input_normalized=fused and index > 0, next_norm=next_norm))
        self.sampled_hidden = (torch.empty_like(scratch.hidden) if buffers is None
                               else buffers.zeros(scratch.hidden.shape, scratch.hidden.dtype))
        self.norm = PreparedRMSNorm(self.sampled_hidden, weights.norm, scratch.hidden,
            eps=weights.config.text["rms_norm_eps"], session=session, name=name + ".norm", zero_centered=True)
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator,
            name=name + ".prompt", copy_source=True)
        self.score_norm = None if self.scoring is None else PreparedRMSNorm(self.scoring.source, weights.norm,
            self.scoring.source, eps=weights.config.text["rms_norm_eps"], session=session,
            name=name + ".prompt_norm", zero_centered=True)
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", buffers=buffers)

    def __call__(self):
        local = self.embedding.lookup()
        if self.vision is not None:
            self.vision.inject(self.metadata, local)
        residual = self.scratch.residual
        residual.copy_(self.embedding.gather(local))
        update = None
        for layer in self.layers:
            update = layer(residual, update)
            if layer.output_norm is None:
                # Dense blocks narrow their reduced output before residual addition.
                update = update.float()
        residual.add_(update)
        if self.auxiliary is not None:
            self.auxiliary.copy_(residual)
        # The checkpoint narrows the residual before its final normalization.
        self.sampled_hidden.copy_(residual[self.metadata.sample_indices])
        self.norm(self.sampled_hidden)
        if self.scoring is not None:
            self.scoring.source.copy_(residual)
            self.score_norm(self.scoring.source)
            self.scoring()
        self.head(self.scratch.hidden)

    def commit(self, counts):
        for layer in self.layers:
            layer.attention.commit(counts)

    def scratch_tensors(self):
        return (self.auxiliary, self.sampled_hidden, self.head.gathered,
                *(layer.attention.scratch_tensors() for layer in self.layers))
