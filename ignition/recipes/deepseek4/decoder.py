# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Vision-aware routed experts and non-lagged manifold hyperconnections."""

from dataclasses import dataclass, fields, is_dataclass

import torch
import torch.nn.functional as F

from ignition.components.collectives import PreparedAllReduce
from ignition.components.dspark import PreparedFeedbackCapture
from ignition.components.embedding import PreparedEmbedding
from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import UnquantizedProjection
from ignition.components.mhc import (
    MHCScratch,
    MHCWeights,
    PreparedMHC,
    PreparedMHCHead,
    PreparedMHCPost,
)
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.rotary import CompressedRotary as Rotary
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.scratch import GraphBufferPool

from .attention import AttentionIndices, AttentionScratch, PreparedAttention


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    residual: torch.Tensor
    attention_hc: MHCScratch
    ffn_hc: MHCScratch
    attention: AttentionScratch
    attention_output: torch.Tensor
    shared: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    routed: torch.Tensor
    mlp_partial: torch.Tensor
    mlp_output: torch.Tensor
    head_residual: torch.Tensor
    collapsed: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor
    # Values retained through a forward, including speculative state awaiting acceptance.
    buffers: GraphBufferPool
    # Prefill commits each layer's state before the following layer reuses these intermediates.
    layer_buffers: GraphBufferPool

    def rows(self, tokens, samples):
        def select(value, count):
            if isinstance(value, torch.Tensor):
                return value[:count]
            if isinstance(value, GraphBufferPool):
                return value.cursor()
            if is_dataclass(value):
                return type(value)(*(select(getattr(value, f.name), count) for f in fields(value)))
            raise TypeError("decoder scratch must contain tensors and tensor dataclasses")

        return type(self)(
            *(
                getattr(self, f.name) if f.name == "layer_buffers" else select(
                    getattr(self, f.name),
                    samples
                    if f.name in ("head_residual", "collapsed", "hidden", "local_logits", "logits")
                    else tokens,
                )
                for f in fields(self)
            )
        )

    @classmethod
    def allocate(cls, tokens, samples, world, device, *, blocks=8192, experts=256, top_k=6, buffers=None):
        if buffers is None:
            buffers = GraphBufferPool(device, chunk_bytes=2 << 20).cursor()
        def zeros(*shape, dtype=torch.bfloat16, cursor=buffers):
            return cursor.zeros(shape, dtype)

        embedding, residual = zeros(tokens, 4096), zeros(tokens, 4, 4096)
        attention_hc = MHCScratch.allocate(tokens, 4096, device, buffers=buffers)
        ffn_hc = MHCScratch.allocate(tokens, 4096, device, buffers=buffers)
        attention_output, mlp_output = zeros(tokens, 4096), zeros(tokens, 4096)
        attn, ffn = buffers.fork(), buffers.fork()
        attention = AttentionScratch.allocate(tokens, world, device, blocks=blocks, buffers=attn)
        shared = MLPScratch.allocate(tokens, 4096, 2048 // world, device, buffers=ffn)
        router_logits = zeros(tokens, experts, dtype=torch.float32, cursor=ffn)
        router = RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k, buffers=ffn)
        routed, partial = zeros(tokens, 4096, cursor=ffn), zeros(tokens, 4096, cursor=ffn)
        buffers.join(attn, ffn)
        return cls(embedding, residual, attention_hc, ffn_hc, attention, attention_output,
            shared, router_logits, router, routed, partial, mlp_output,
            zeros(samples, 4, 4096), zeros(samples, 4096), zeros(samples, 4096),
            zeros(samples, 129280 // world), zeros(samples, 129280),
            GraphBufferPool(device, chunk_bytes=min(64 << 20, max(1 << 20, tokens << 14))),
            GraphBufferPool(device, chunk_bytes=min(64 << 20, max(1 << 20, tokens << 14))))



class PreparedRouting:
    def __init__(self, weights, scratch, metadata, *, mode, session, name, buffers):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        self.prefill = mode == "prefill"
        self.image_mask = (buffers.zeros(metadata.token_ids.shape, torch.bool)
                           if mode == "prefill" and weights.token_experts is None else None)
        self.text = (
            None
            if weights.token_experts is not None
            else PreparedRouter(
                scratch.router_logits,
                scratch.router,
                session=session,
                name=name + ".text",
                score_func="sqrtsoftplus",
                renormalize=True,
                routed_scaling_factor=1.5,
                correction_bias=weights.correction,
                image_correction_bias=weights.vision_correction if self.image_mask is not None else None,
                image_mask=self.image_mask,
            )
        )
        self.vision, self.vision_scratch = None, None
        if mode == "prefill" and self.text is None:
            tokens = scratch.router_logits.shape[0]
            self.vision_scratch = RouterScratch(
                buffers.zeros((tokens, 6), torch.float32),
                buffers.zeros((tokens, 6), torch.int32),
                buffers.full((tokens, 6), 1 / 6, torch.float32),
            )
            self.vision = PreparedRouter(
                scratch.router_logits,
                self.vision_scratch,
                session=session,
                name=name + ".vision",
                score_func="sqrtsoftplus",
                renormalize=True,
                routed_scaling_factor=1.5,
                correction_bias=weights.vision_correction,
            )

    def __call__(self, logits):
        r = self.scratch.router
        tokens = self.metadata.token_ids
        if self.image_mask is not None:
            self.image_mask.copy_(tokens >= 129280)
        if self.text is not None:
            self.text(logits)
        else:
            r.ids.copy_(self.weights.token_experts[tokens.clamp(0, 129279)])
            selected = F.softplus(logits).sqrt().gather(1, r.ids.long())
            r.weights.copy_(selected / selected.sum(-1, keepdim=True) * 1.5)
        if self.vision is not None:
            self.vision(logits)
            image = tokens[:, None] >= 129280
            r.ids.copy_(torch.where(image, self.vision_scratch.ids, r.ids))
            r.weights.copy_(torch.where(image, self.vision_scratch.weights, r.weights))
        if self.prefill:
            # b12x excludes negative expert IDs when constructing dynamic expert work.
            r.ids.copy_(torch.where(self.metadata.positions[:, None] >= 0, r.ids, -1))


class PreparedLayer:
    def __init__(
        self,
        index,
        weights,
        scratch,
        metadata,
        pool,
        rotary,
        *,
        shard,
        mode,
        runtime,
        coordinator,
        session,
        name,
        workspace,
        carry,
        defer_post,
        visible=None,
        indices=None,
    ):
        s, text = scratch, pool.config.text
        self.scratch, self.carry = s, carry
        buffers = s.layer_buffers.cursor() if mode == "prefill" else s.buffers
        options = {
            "rms_eps": text["rms_norm_eps"],
            "norm_eps": text["rms_norm_eps"],
            "hc_eps": text["hc_eps"],
            "sinkhorn_iters": text["hc_sinkhorn_iters"],
        }

        def mhc_weights(kind, norm):
            r = weights.residual
            return MHCWeights(
                r[f"hc_{kind}_fn"], r[f"hc_{kind}_scale"], r[f"hc_{kind}_base"], r[norm + ".weight"]
            )

        incoming = (s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb) if carry else (s.residual,)
        self.attention_hc = PreparedMHC(
            incoming,
            mhc_weights("attn", "attn_norm"),
            s.attention_hc,
            session=session,
            name=name + ".attn_hc",
            workspace=workspace,
            max_rows_per_call=2048 if mode == "prefill" and carry else None,
            **options,
        )
        self.attention = PreparedAttention(
            index,
            weights.attention,
            s.attention_hc.normalized,
            s.attention,
            metadata,
            pool,
            rotary,
            mode=mode,
            world=shard.world_size,
            session=session,
            name=name + ".attention",
            workspace=workspace,
            buffers=buffers,
            visible=visible,
            indices=indices,
        )
        self.attention_reduce = PreparedAllReduce(
            s.attention.output,
            s.attention_output,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".attention_reduce",
        )
        self.ffn_hc = PreparedMHC(
            (s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb),
            mhc_weights("ffn", "ffn_norm"),
            s.ffn_hc,
            session=session,
            name=name + ".ffn_hc",
            workspace=workspace,
            max_rows_per_call=2048 if mode == "prefill" else None,
            **options,
        )
        self.router_projection = UnquantizedProjection(
            s.ffn_hc.normalized,
            weights.experts.router,
            s.router_logits,
            session=session,
            name=name + ".router",
        )
        self.router = PreparedRouting(
            weights.experts, s, metadata, mode=mode, session=session, name=name + ".route", buffers=buffers
        )
        self.experts = PreparedRoutedExperts(
            s.ffn_hc.normalized,
            weights.experts.experts,
            s.router.ids,
            s.router.weights,
            s.routed,
            session=session,
            name=name + ".experts",
            workspace=workspace,
        )
        self.shared = PreparedMLP(
            s.ffn_hc.normalized,
            weights.experts.shared,
            s.shared.gate_up,
            s.shared.intermediate,
            s.shared.output,
            session=session,
            name=name + ".shared",
            workspace=workspace,
            swiglu_limit=text["swiglu_limit"],
            round_silu=False,
            activation_mode="auto",
        )
        self.concurrent_shared = None
        rows = s.ffn_hc.normalized.shape[0]
        overlap = rows >= 2048 if mode == "prefill" else rows <= 256
        if overlap and self.router.text is not None and workspace.overlap:
            parallel_plans = (self.router_projection.plan, self.router.text.plan, self.experts.plan)
            if self.router.vision is not None:
                parallel_plans += (self.router.vision.plan,)
            self.concurrent_shared = ConcurrentSharedExpert(
                self.shared,
                parallel_plans=parallel_plans,
                session=session,
                workspace=workspace,
                defer_launch=mode == "prefill" or rows <= 8,
            )
        self.mlp_reduce = PreparedAllReduce(
            s.mlp_partial,
            s.mlp_output,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".mlp_reduce",
        )
        self.post = (
            None
            if defer_post
            else PreparedMHCPost(
                s.mlp_output,
                s.ffn_hc.residual,
                s.ffn_hc.post,
                s.ffn_hc.comb,
                s.residual,
                session=session,
                name=name + ".post",
            )
        )

    def __call__(self):
        s = self.scratch
        incoming = (
            (s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb) if self.carry else (s.residual,)
        )
        self.attention_hc(*incoming)
        self.attention_reduce(self.attention(s.attention_hc.normalized))
        self.ffn_hc(s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb)
        if self.concurrent_shared is not None:
            self.concurrent_shared.start(s.ffn_hc.normalized)
        self.router(self.router_projection(s.ffn_hc.normalized))
        self.experts()
        if self.concurrent_shared is None:
            self.shared(s.ffn_hc.normalized)
        else:
            self.concurrent_shared.finish(s.ffn_hc.normalized, s.routed)
        s.mlp_partial.copy_(s.routed + s.shared.output)
        self.mlp_reduce(s.mlp_partial)
        if self.post is not None:
            self.post(s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb)


class PreparedDecoder:
    def __init__(
        self,
        weights,
        vision,
        scratch,
        metadata,
        pool,
        *,
        mode,
        runtime,
        vocabulary_runtime,
        coordinator,
        session,
        name,
        workspace,
        scoring=None,
    ):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.mode, self.vision = mode, vision
        self.aux_layers = () if weights.draft is None else weights.config.draft.target_layers
        self.feedback = PreparedFeedbackCapture(metadata, len(self.aux_layers) * 4096,
            scratch.buffers, compact_prefill=mode == "prefill")
        self.auxiliary = self.feedback.output
        self.safe_tokens = scratch.buffers.zeros(metadata.token_ids.shape, metadata.token_ids.dtype)
        self.embedding = PreparedEmbedding(
            weights.embedding,
            self.safe_tokens,
            scratch.embedding,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        )
        self.rotary = Rotary(weights.config, scratch.embedding.device)
        self.layers, carry = [], False
        visible = None if vision is None or mode != "prefill" else vision.visible(metadata)
        self.indices = AttentionIndices(
            metadata, mode=mode, tokens=metadata.token_ids.numel(), visible=visible, buffers=scratch.buffers
        )
        for index, layer in enumerate(weights.layers):
            defer = index + 1 < len(weights.layers) and index not in self.aux_layers
            self.layers.append(
                PreparedLayer(
                    index,
                    layer,
                    scratch,
                    metadata,
                    pool,
                    self.rotary,
                    shard=weights.shard,
                    mode=mode,
                    runtime=runtime,
                    coordinator=coordinator,
                    session=session,
                    name=f"{name}.layer{index}",
                    workspace=workspace,
                    carry=carry,
                    defer_post=defer,
                    visible=visible,
                    indices=self.indices,
                )
            )
            carry = defer
        self.collapse = PreparedMHCHead(
            scratch.head_residual,
            weights.head_mix,
            scratch.collapsed,
            rms_eps=weights.config.text["rms_norm_eps"],
            hc_eps=weights.config.text["hc_eps"],
            session=session,
            name=name + ".collapse",
        )
        self.norm = PreparedRMSNorm(
            scratch.collapsed,
            weights.norm,
            scratch.hidden,
            eps=weights.config.text["rms_norm_eps"],
            session=session,
            name=name + ".norm",
        )
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=129280, session=session, coordinator=coordinator, name=name + ".prompt", copy_source=True)
        self.score_collapse = None if self.scoring is None else PreparedMHCHead(scratch.residual,
            weights.head_mix, self.scoring.source, rms_eps=weights.config.text["rms_norm_eps"],
            hc_eps=weights.config.text["hc_eps"], session=session, name=name + ".prompt_collapse")
        self.score_norm = None if self.scoring is None else PreparedRMSNorm(self.scoring.source,
            weights.norm, self.scoring.source, eps=weights.config.text["rms_norm_eps"], session=session,
            name=name + ".prompt_norm")
        self.head = PreparedVocabularyProjection(
            scratch.hidden,
            weights.head,
            scratch.local_logits,
            scratch.logits,
            runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".head",
            workspace=workspace,
        )

    def __call__(self):
        s, m = self.scratch, self.metadata
        self.safe_tokens.copy_(m.token_ids.clamp(0, 129279))
        self.embedding()
        if self.vision is not None and self.mode == "prefill":
            self.vision.inject(m, s.embedding)
        s.residual.copy_(s.embedding[:, None, :])
        self.indices()
        self.feedback.begin()
        for index, layer in enumerate(self.layers):
            layer()
            if index in self.aux_layers:
                column = self.aux_layers.index(index)
                self.feedback(s.residual, column * 4096)
        s.head_residual.copy_(s.residual[m.sample_indices])
        self.norm(self.collapse(s.head_residual))
        if self.scoring is not None:
            self.score_norm(self.score_collapse(s.residual))
            self.scoring()
        self.head(s.hidden)
        return s.hidden

    def commit(self, counts):
        for layer in self.layers:
            layer.attention.commit(counts)
