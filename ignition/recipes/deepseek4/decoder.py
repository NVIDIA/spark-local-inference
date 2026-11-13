# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Vision-aware routed experts and non-lagged manifold hyperconnections."""

from dataclasses import dataclass, fields

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

    def rows(self, tokens, samples):
        def select(value, count):
            if isinstance(value, torch.Tensor):
                return value[:count]
            return type(value)(*(select(getattr(value, f.name), count) for f in fields(value)))
        sampled = ("head_residual", "collapsed", "hidden", "local_logits", "logits")
        return type(self)(*(select(getattr(self, f.name), samples if f.name in sampled else tokens)
                            for f in fields(self)))

    @classmethod
    def allocate(cls, tokens, samples, world, device, *, blocks=8192, experts=256, top_k=6):
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(zeros(tokens, 4096), zeros(tokens, 4, 4096),
            MHCScratch.allocate(tokens, 4096, device), MHCScratch.allocate(tokens, 4096, device),
            AttentionScratch.allocate(tokens, world, device, blocks=blocks), zeros(tokens, 4096),
            MLPScratch.allocate(tokens, 4096, 2048 // world, device), zeros(tokens, experts, dtype=torch.float32),
            RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k),
            zeros(tokens, 4096), zeros(tokens, 4096), zeros(tokens, 4096), zeros(samples, 4, 4096),
            zeros(samples, 4096), zeros(samples, 4096), zeros(samples, 129280 // world), zeros(samples, 129280))



class PreparedRouting:
    def __init__(self, weights, scratch, metadata, *, mode, session, name):
        self.token_experts = weights.token_experts
        self.prefill = mode == "prefill"
        self.image_mask = self.prefill and weights.token_experts is None
        mask = torch.zeros(metadata.token_ids.shape, dtype=torch.bool, device=metadata.token_ids.device)
        self.text = (None if weights.token_experts is not None else PreparedRouter(
            scratch.router_logits, scratch.router, session=session, name=name + ".text",
            score_func="sqrtsoftplus", renormalize=True, routed_scaling_factor=1.5,
            correction_bias=weights.correction,
            image_correction_bias=weights.vision_correction if self.image_mask else None,
            image_mask=mask if self.image_mask else None).functional(session=session))
        self.vision = None
        if self.prefill and self.text is None:
            routes = RouterScratch.allocate(scratch.router_logits.shape[0], scratch.router_logits.device,
                                            experts=scratch.router_logits.shape[1], top_k=6)
            self.vision = PreparedRouter(scratch.router_logits, routes, session=session, name=name + ".vision",
                score_func="sqrtsoftplus", renormalize=True, routed_scaling_factor=1.5,
                correction_bias=weights.vision_correction).functional(session=session)

    def __call__(self, logits, metadata):
        tokens = metadata.token_ids
        if self.text is not None:
            routes = self.text(logits, image_mask=tokens >= 129280 if self.image_mask else None)
        else:
            ids = self.token_experts[tokens.clamp(0, 129279)]
            selected = F.softplus(logits).sqrt().gather(1, ids.long())
            weights = selected / selected.sum(-1, keepdim=True) * 1.5
            routes = RouterScratch(torch.zeros_like(weights), ids, weights)
        if self.vision is not None:
            vision = self.vision(logits)
            image = tokens[:, None] >= 129280
            routes = RouterScratch(routes.selected_logits, torch.where(image, vision.ids, routes.ids),
                                   torch.where(image, vision.weights, routes.weights))
        if self.prefill:
            routes = RouterScratch(routes.selected_logits,
                torch.where(metadata.positions[:, None] >= 0, routes.ids, -1), routes.weights)
        return routes


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
        examples=None,
    ):
        s, text = scratch, pool.config.text
        self.carry = carry
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
        ).functional()
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
            visible=visible,
            indices=indices,
            examples=examples,
        )
        self.attention_reduce = PreparedAllReduce(
            s.attention.output,
            s.attention_output,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".attention_reduce",
        ).functional(session=session)
        self.ffn_hc = PreparedMHC(
            (s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb),
            mhc_weights("ffn", "ffn_norm"),
            s.ffn_hc,
            session=session,
            name=name + ".ffn_hc",
            workspace=workspace,
            max_rows_per_call=2048 if mode == "prefill" else None,
            **options,
        ).functional()
        self.router_projection = UnquantizedProjection(
            s.ffn_hc.normalized,
            weights.experts.router,
            s.router_logits,
            session=session,
            name=name + ".router",
        ).functional(session=session)
        self.router = PreparedRouting(
            weights.experts, s, metadata, mode=mode, session=session, name=name + ".route"
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
        self.experts.functional(session=session)
        if self.concurrent_shared is None:
            self.shared.functional(session=session)
        else:
            self.concurrent_shared.functional(session=session)
        self.mlp_reduce = PreparedAllReduce(
            s.mlp_partial,
            s.mlp_output,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".mlp_reduce",
        ).functional(session=session)
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

        if self.post is not None:
            self.post.functional()

    def __call__(self, incoming, metadata, indices):
        residual, post, comb, source = self.attention_hc(*incoming)
        attended, records = self.attention(source, metadata, indices)
        attended = self.attention_reduce(attended)
        residual, post, comb, normalized = self.ffn_hc(attended, residual, post, comb)
        pending = None if self.concurrent_shared is None else self.concurrent_shared.start(normalized)
        routes = self.router(self.router_projection(normalized), metadata)
        routed = self.experts(normalized, routes.ids, routes.weights)
        shared = (self.shared(normalized) if pending is None
                  else self.concurrent_shared.finish(pending, routed))
        output = self.mlp_reduce(routed + shared)
        if self.post is None:
            return (output, residual, post, comb), records
        result = self.post(output, residual, post, comb, torch.empty_like(residual))
        return result, records


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
        examples=None,
    ):
        self.weights, self.scratch, self.pool = weights, scratch, pool
        self.mode, self.vision = mode, vision
        self.aux_layers = () if weights.draft is None else weights.config.draft.target_layers
        self.feedback = (PreparedFeedbackCapture(metadata, len(self.aux_layers) * 4096,
            compact_prefill=mode == "prefill") if self.aux_layers else None)
        feedback_rows = min(128, metadata.token_ids.numel()) if mode == "prefill" else metadata.token_ids.numel()
        self.auxiliary = (None if self.feedback is None else torch.empty(
            (feedback_rows, len(self.aux_layers) * 4096), dtype=torch.bfloat16, device=scratch.embedding.device))
        safe_tokens = torch.zeros_like(metadata.token_ids)
        self.embedding = PreparedEmbedding(
            weights.embedding,
            safe_tokens,
            scratch.embedding,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        ).functional(session=session, dynamic_inputs=True)
        self.rotary = Rotary(weights.config, scratch.embedding.device)
        self.layers, carry = [], False
        visible = None if vision is None or mode != "prefill" else vision.visible(metadata)
        self.indices = AttentionIndices(
            metadata, mode=mode, tokens=metadata.token_ids.numel(), visible=visible
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
                    examples=examples,
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
        ).functional()
        self.norm = PreparedRMSNorm(
            scratch.collapsed,
            weights.norm,
            scratch.hidden,
            eps=weights.config.text["rms_norm_eps"],
            session=session,
            name=name + ".norm",
        ).functional()
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=129280, session=session, coordinator=coordinator, name=name + ".prompt")
        self.score_collapse = None if self.scoring is None else PreparedMHCHead(scratch.residual,
            weights.head_mix, scratch.embedding, rms_eps=weights.config.text["rms_norm_eps"],
            hc_eps=weights.config.text["hc_eps"], session=session, name=name + ".prompt_collapse").functional()
        self.score_norm = None if self.scoring is None else PreparedRMSNorm(scratch.embedding,
            weights.norm, scratch.embedding, eps=weights.config.text["rms_norm_eps"], session=session,
            name=name + ".prompt_norm").functional()
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
        ).functional(session=session)
        self.indices.functional()

    def release_outputs(self):
        self.scratch = self.auxiliary = None

    def __call__(self, metadata):
        m = metadata
        embedding = self.embedding(token_ids=m.token_ids.clamp(0, 129279), num_rows=m.live_counts[:1])
        if self.vision is not None and self.mode == "prefill":
            self.vision.inject(m, embedding)
        residual = embedding[:, None, :].expand(-1, 4, -1).clone()
        visible = None if self.vision is None or self.mode != "prefill" else self.vision.visible(m)
        indices = self.indices(m, visible=visible)
        capture = None if self.feedback is None else self.feedback.begin(m)
        records = []
        incoming = (residual,)
        for index, layer in enumerate(self.layers):
            result, record = layer(incoming, m, indices)
            records.append(record)
            incoming = result if isinstance(result, tuple) else (result,)
            if index in self.aux_layers:
                self.feedback(capture, result, self.aux_layers.index(index) * 4096)
        residual = result
        hidden = self.norm(self.collapse(residual[m.sample_indices]))
        if self.scoring is not None:
            self.scoring(self.score_norm(self.score_collapse(residual)))
        return hidden, self.head(hidden), None if capture is None else capture[0], tuple(records)

    def commit(self, counts, records, metadata):
        for layer, record in zip(self.layers, records, strict=True):
            layer.attention.commit(counts, record, metadata)
