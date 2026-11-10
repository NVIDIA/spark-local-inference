# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared M3 layers with Gemma residual boundaries and rank-local projections."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce, PreparedAllReduceAdd, PreparedAllReduceRMSNorm, WeightedGroupSource
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.components.msa_attention import PreparedMSAIndexer, PreparedPackedAttention
from ignition.components.paged_attention import PreparedPagedAttention
from ignition.components.paged_schedule import PreparedPagedSchedules
from ignition.components.rms_norm import PreparedIndexedRMSNorm, PreparedRMSNorm
from ignition.components.state import save_hidden
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.persistent import persistent_copy


def projection(source, weight, output, *, session, name, workspace):
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name).functional(session=session)
    return QuantizedProjection(source, weight, output, session=session, name=name,
                               workspace=workspace, activation_mode="auto").functional(session=session)


@dataclass(frozen=True)
class AttentionScratch:
    projected: torch.Tensor
    query: torch.Tensor
    index_query: torch.Tensor
    selected: torch.Tensor
    attended: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, *, device):
        c = config.text
        heads, kv = c["num_attention_heads"] // shard.world_size, c["num_key_value_heads"] // shard.world_size
        dim = c["head_dim"]
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(zeros((tokens, (heads + 3 * kv + 1) * dim), torch.bfloat16),
            zeros((tokens, heads, dim), torch.bfloat16),
            zeros((tokens, kv, dim), torch.float8_e4m3fn),
            zeros((kv, tokens, c["sparse_attention_config"]["sparse_topk_blocks"]), torch.int32),
            zeros((tokens, heads * dim), torch.bfloat16),
            zeros((tokens, config.hidden), torch.bfloat16))

    def rows(self, tokens):
        return type(self)(self.projected[:tokens], self.query[:tokens], self.index_query[:tokens],
                          self.selected[:, :tokens], self.attended[:tokens], self.output[:tokens])


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    local_embedding: torch.Tensor
    normalized: torch.Tensor
    residual: torch.Tensor
    reduced: torch.Tensor
    partial: torch.Tensor
    attention: AttentionScratch
    dense: MLPScratch
    shared: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    experts: torch.Tensor
    expert_routes: torch.Tensor
    auxiliary: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, samples, device):
        c = config.text
        def zeros(rows, columns, dtype=torch.bfloat16):
            return torch.zeros((rows, columns), dtype=dtype, device=device)
        embedding = zeros(tokens, config.hidden)
        local_embedding = zeros(tokens, config.hidden // shard.world_size)
        normalized, residual, reduced, partial = (zeros(tokens, config.hidden) for _ in range(4))
        attention = AttentionScratch.allocate(config, shard, tokens, device=device)
        dense = MLPScratch.allocate(tokens, config.hidden, c["dense_intermediate_size"] // shard.world_size,
                                    device)
        shared = MLPScratch.allocate(tokens, config.hidden, c["shared_intermediate_size"] // shard.world_size,
                                     device)
        router_logits = zeros(tokens, c["num_local_experts"], torch.float32)
        router = RouterScratch.allocate(tokens, device, experts=c["num_local_experts"],
            top_k=c["num_experts_per_tok"])
        experts = zeros(tokens, config.hidden)
        expert_routes = torch.zeros((tokens, c["num_experts_per_tok"], config.hidden), dtype=torch.bfloat16, device=device)
        auxiliary = (torch.zeros((tokens, len(config.target_layers), config.hidden), dtype=torch.bfloat16, device=device))
        width = config.vocabulary_partition(shard)[2]
        return cls(embedding, local_embedding, normalized, residual, reduced, partial, attention, dense, shared,
                   router_logits, router, experts, expert_routes, auxiliary, zeros(samples, config.hidden),
                   zeros(samples, width), zeros(samples, config.vocabulary))

    def rows(self, tokens, samples):
        return type(self)(*(getattr(self, name).rows(tokens) if name in ("attention", "dense", "shared", "router")
            else getattr(self, name)[:samples if name in ("hidden", "local_logits", "logits") else tokens]
            for name in self.__dataclass_fields__))


class PreparedAttention:
    def __init__(self, layer, weights, config, shard, source, scratch, metadata, pool, *, mode,
                 inverse_frequency, session, name, workspace, cache_lengths, schedules=None):
        self.scratch, self.lengths = scratch, cache_lengths
        self.input = projection(source, weights.qkv_index, scratch.projected, session=session,
                                name=name + ".input", workspace=workspace)
        key, value, index = (pool.cache(layer, kind) for kind in ("key", "value", "index"))
        self.pack = PreparedPackedAttention(scratch.projected, weights.q_norm, weights.k_norm,
            weights.index_q_norm, weights.index_k_norm, inverse_frequency, metadata,
            scratch.query, scratch.index_query, key, value, index,
            epsilon=config.text["rms_norm_eps"], session=session, name=name + ".norm_rope_write").functional(session=session)
        self.indexer = PreparedMSAIndexer(scratch.index_query, index, metadata.page_table, self.lengths,
            metadata.query_start_loc, metadata.positions, scratch.selected,
            session=session, workspace=workspace, name=name + ".indexer",
            max_q_per_request=metadata.max_query_tokens).functional(session=session)
        self.attention = PreparedPagedAttention(scratch.query, key, value, metadata.page_table,
            self.lengths, metadata.query_start_loc, scratch.attended.view_as(scratch.query),
            selected_blocks=scratch.selected, mode=mode, tuning_page_count=int(pool.memory_capacity[1].item()),
            width=metadata.max_query_tokens if mode == "verify" else 1,
            session=session, workspace=workspace, name=name + ".attention", schedules=schedules).functional(session=session, cache_seqlens=None, page_table=None, cu_seqlens_q=None, selected_blocks=None, schedule_storage=None)
        self.output = projection(scratch.attended, weights.output, scratch.output, session=session,
                                 name=name + ".output", workspace=workspace)

    def __call__(self, source, *, metadata, lengths, schedule_storage):
        projected = self.input(source)
        query, index_query = self.pack(projected, metadata)
        selected = self.indexer(index_query, metadata.page_table, lengths, metadata.query_start_loc, metadata.positions)
        attended = self.attention(query, page_table=metadata.page_table, cache_seqlens=lengths,
            cu_seqlens_q=metadata.query_start_loc, selected_blocks=selected, schedule_storage=schedule_storage)
        return self.output(attended.flatten(1))


class PreparedFeedForward:
    def __init__(self, weights, config, source, scratch, metadata, *, session, name, workspace, overlap, route_output=None):
        self.scratch = scratch
        routed = weights.experts is not None
        storage = scratch.shared if routed else scratch.dense
        c = config.text
        self.mlp = PreparedMLP(source, weights.mlp, storage.gate_up, storage.intermediate, storage.output,
            session=session, name=name + (".shared" if routed else ".dense"), workspace=workspace,
            swiglu_limit=c["swiglu_limit"], swiglu_alpha=c["swiglu_alpha"], swiglu_beta=1.,
            round_silu=False, activation_mode="auto")
        self.experts = self.concurrent = None
        if routed:
            self.router_projection = UnquantizedProjection(source, weights.router, scratch.router_logits,
                session=session, name=name + ".router").functional(session=session)
            self.router = PreparedRouter(scratch.router_logits, scratch.router, session=session, name=name + ".route",
                score_func="sigmoid", renormalize=True, routed_scaling_factor=c["routed_scaling_factor"],
                correction_bias=weights.correction, num_valid_tokens=metadata.live_counts[:1])
            self.experts = PreparedRoutedExperts(source, weights.experts, scratch.router.ids, scratch.router.weights,
                scratch.experts, session=session, name=name + ".experts", workspace=workspace, router_plan=self.router.plan, router=self.router, route_output=route_output)
            if overlap:
                self.concurrent = ConcurrentSharedExpert(self.mlp,
                    parallel_plans=(self.router_projection.plan, self.router.plan, self.experts.plan),
                    session=session, workspace=workspace)
        self.partial_dtype = scratch.partial.dtype
        self.output_source = None if self.experts is None else self.experts.output_source
        if self.concurrent is not None:
            self.concurrent.functional(session=session)
        if self.experts is not None:
            self.experts.functional(session=session)
            self.router.functional(session=session)
        self.mlp.functional(session=session)

    def __call__(self, source, *, metadata, combine_output=True):
        if self.experts is None:
            return self.mlp(source)
        pending = None if self.concurrent is None else self.concurrent.start(source)
        logits = self.router_projection(source)
        routes = self.experts.route(logits, num_valid_tokens=metadata.live_counts[:1])
        routed = self.experts(source, routes.ids, routes.weights, router_logits=logits,
                              selected_logits=routes.selected_logits, num_valid_tokens=metadata.live_counts[:1])
        shared = self.mlp(source) if self.concurrent is None else self.concurrent.finish(pending, routed)
        if not combine_output:
            return routed, shared
        return (routed.float() + shared.float()).to(self.partial_dtype)


class PreparedDecoderLayer:
    """Canonical M3 residual arithmetic with reductions folded into Gemma boundaries."""

    def __init__(self, attention, feed_forward, weights, scratch, *, epsilon, runtime, coordinator,
                 session, name, first, fused, next_norm):
        self.attention, self.feed_forward, self.scratch = attention, feed_forward, scratch
        self.input_norm = None if fused and not first else PreparedRMSNorm(
            scratch.residual, weights.input_norm, scratch.normalized, eps=epsilon, session=session,
            name=name + ".input_norm", zero_centered=True, residual=None if first else scratch.reduced).functional()
        self.attention_reduce = None
        self.output_norm = None
        self.feed_forward_reduce = None
        self.final_reduce = None
        self.fused = fused
        if fused:
            self.post_norm = PreparedAllReduceRMSNorm(scratch.attention.output, scratch.residual,
                weights.post_attention_norm, scratch.normalized, eps=epsilon, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".attention_norm",
                zero_centered=True, reduction_dtype=torch.bfloat16).functional(session=session)
            if next_norm is not None:
                output = scratch.dense.output if weights.experts is None else feed_forward.output_source
                input_addend = None if weights.experts is None else scratch.shared.output
                self.output_norm = PreparedAllReduceRMSNorm(output, scratch.residual, next_norm,
                    scratch.normalized, eps=epsilon, runtime=runtime, coordinator=coordinator,
                    session=session, name=name + ".feed_forward_norm", zero_centered=True,
                    reduction_dtype=torch.bfloat16, input_addend=input_addend).functional(session=session)
            else:
                output = scratch.dense.output if weights.experts is None else feed_forward.output_source
                self.final_reduce = PreparedAllReduceAdd(output, scratch.residual, runtime=runtime,
                    coordinator=coordinator, session=session, name=name + ".feed_forward_residual",
                    reduction_dtype=torch.bfloat16,
                    input_addend=None if weights.experts is None else scratch.shared.output).functional(session=session)
        else:
            self.attention_reduce = PreparedAllReduce(scratch.attention.output, scratch.reduced, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".attention_reduce").functional(session=session)
            self.post_norm = PreparedRMSNorm(scratch.residual, weights.post_attention_norm, scratch.normalized,
                residual=scratch.reduced, eps=epsilon, zero_centered=True, session=session,
                name=name + ".attention_norm").functional()
        if self.output_norm is None and self.final_reduce is None:
            output = scratch.dense.output if weights.experts is None else scratch.partial
            self.feed_forward_reduce = PreparedAllReduce(output, scratch.reduced, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".feed_forward_reduce").functional(session=session)

    def normalize(self, residual, update, normalized):
        return normalized if self.input_norm is None else self.input_norm(residual, update)

    def __call__(self, normalized, residual, *, metadata, lengths, schedule_storage):
        attended = self.attention(normalized, metadata=metadata, lengths=lengths, schedule_storage=schedule_storage)
        if self.attention_reduce is not None:
            attended = self.attention_reduce(attended)
        normalized = (self.post_norm(attended, residual) if self.fused else self.post_norm(residual, attended))
        fused = self.output_norm is not None or self.final_reduce is not None
        output = self.feed_forward(normalized, metadata=metadata, combine_output=not fused)
        routed, shared = output if isinstance(output, tuple) else (output, None)
        groups = {}
        if isinstance(routed, WeightedGroupSource):
            groups = dict(group_weights=routed.weights, group_ids=routed.ids)
            routed = routed.values
        if self.output_norm is not None:
            return self.output_norm(routed, residual, input_addend=shared, **groups), None
        if self.final_reduce is not None:
            self.final_reduce(routed, residual, input_addend=shared, **groups)
            return None, None
        return None, self.feed_forward_reduce(output)


class _Norm(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, hidden):
        return self.operation(hidden)


class _Attention(torch.nn.Module):
    def __init__(self, operation, reduction):
        super().__init__()
        self.operation, self.reduction = operation, reduction

    def forward(self, hidden_states, **kwargs):
        return self.reduction(self.operation(hidden_states, **kwargs)), None


class _FeedForward(torch.nn.Module):
    def __init__(self, operation, reduction):
        super().__init__()
        self.operation, self.reduction = operation, reduction

    def forward(self, hidden):
        return self.reduction(self.operation(hidden))


class CanonicalPreparedLayer:
    """Canonical M3 residual arithmetic with explicit per-call attention metadata."""

    def __init__(self, index, attention, feed_forward, weights, config, scratch, *, runtime,
                 coordinator, session, name):
        from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLDecoderLayer

        with torch.device("meta"):
            self.module = MiniMaxM3VLDecoderLayer(config.canonical_config(), index)
        self.attention = attention
        self.module.input_layernorm = _Norm(PreparedRMSNorm(scratch.residual, weights.input_norm,
            scratch.normalized, eps=config.text["rms_norm_eps"], zero_centered=True,
            session=session, name=name + ".input_norm").functional())
        self.module.post_attention_layernorm = _Norm(PreparedRMSNorm(scratch.residual,
            weights.post_attention_norm, scratch.normalized, eps=config.text["rms_norm_eps"],
            zero_centered=True, session=session, name=name + ".post_attention_norm").functional())
        attention_reduction = PreparedAllReduce(scratch.attention.output, scratch.reduced, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".attention_reduce").functional(session=session)
        output = scratch.dense.output if weights.experts is None else scratch.partial
        ffn_reduction = PreparedAllReduce(output, scratch.reduced, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".feed_forward_reduce").functional(session=session)
        self.module.self_attn = _Attention(attention, attention_reduction)
        self.module.mlp = _FeedForward(feed_forward, ffn_reduction)
        self.module.eval().requires_grad_(False)

    def __call__(self, hidden, *, metadata, lengths, schedule_storage):
        normalized = self.module.input_layernorm(hidden)
        attended, _ = self.module.self_attn(normalized, metadata=metadata, lengths=lengths, schedule_storage=schedule_storage)
        hidden = hidden + attended
        normalized = self.module.post_attention_layernorm(hidden)
        projected = self.module.mlp.operation(normalized, metadata=metadata)
        return hidden + self.module.mlp.reduction(projected)


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, embedding_runtime,
                 vocabulary_runtime, coordinator, session, name, workspace, overlap=True, vision=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.mode = mode
        self.vision = vision if mode == "prefill" else None
        self.auxiliary = scratch.auxiliary
        self.cache_lengths = (torch.zeros_like(metadata.lengths))
        self.schedules = PreparedPagedSchedules(self.cache_lengths, metadata.query_start_loc,
            len(weights.layers))
        c = weights.config.text
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids,
            scratch.local_embedding, scratch.embedding, runtime=embedding_runtime, coordinator=coordinator,
            session=session, name=name + ".embedding", num_rows=metadata.live_counts[:1]).functional(session=session, dynamic_inputs=True)
        inverse = persistent_copy(weights.config.rotary_frequencies(scratch.embedding.device))
        fused = PreparedAllReduceRMSNorm.fits(runtime, scratch.partial)
        self.fused = fused
        self.layers = []
        for index, layer in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            attention = PreparedAttention(index, layer, weights.config, weights.shard, scratch.normalized,
                scratch.attention, metadata, pool, mode=mode, inverse_frequency=inverse,
                session=session, name=prefix + ".attention", workspace=workspace,
                cache_lengths=self.cache_lengths, schedules=self.schedules)
            following = weights.layers[index + 1].input_norm if fused and index + 1 < len(weights.layers) else None
            grouped = WeightedGroupSource(scratch.expert_routes, scratch.router.weights, scratch.router.ids,
                                          c["num_local_experts"])
            route_output = (scratch.expert_routes if fused
                            and PreparedAllReduceRMSNorm.fits(runtime, grouped) else None)
            feed_forward = PreparedFeedForward(layer, weights.config, scratch.normalized, scratch, metadata,
                session=session, name=prefix + ".feed_forward", workspace=workspace, overlap=overlap,
                route_output=route_output)
            if fused:
                prepared = PreparedDecoderLayer(attention, feed_forward, layer, scratch,
                    epsilon=c["rms_norm_eps"], runtime=runtime, coordinator=coordinator, session=session,
                    name=prefix, first=index == 0, fused=True, next_norm=following)
            else:
                prepared = CanonicalPreparedLayer(index, attention, feed_forward, layer, weights.config, scratch,
                    runtime=runtime, coordinator=coordinator, session=session, name=prefix)
            self.layers.append(prepared)
        self.schedules.finalize(session)
        self.schedules.functional(session=session)
        self.norm = PreparedIndexedRMSNorm(scratch.residual, metadata.sample_indices, weights.norm, scratch.hidden,
            eps=c["rms_norm_eps"], zero_centered=True, session=session, name=name + ".norm").functional()
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            activation_mode="auto",  workspace=workspace).functional(session=session)
        self.taps = ({layer + 1: index for index, layer in enumerate(weights.config.target_layers)}
                     if weights.draft is not None else {})

    def __call__(self, metadata):
        lengths = torch.where(metadata.lengths > 0, (metadata.starts + metadata.lengths).int(), 0)
        schedule_storage = self.schedules(lengths, metadata.query_start_loc)
        local = self.embedding.lookup(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
        if self.vision is not None:
            self.vision.inject(metadata, local)
        hidden = self.embedding.gather(local)
        auxiliary = (torch.empty((hidden.shape[0], len(self.taps), hidden.shape[1]), dtype=hidden.dtype, device=hidden.device)
                     if self.taps else None)
        if self.fused:
            residual, normalized, update = hidden, None, None
            for index, layer in enumerate(self.layers):
                normalized = layer.normalize(residual, update, normalized)
                if index in self.taps:
                    auxiliary[:, self.taps[index]].copy_(residual)
                normalized, update = layer(normalized, residual, metadata=metadata, lengths=lengths, schedule_storage=schedule_storage)
            if update is not None:
                residual.add_(update)
            hidden = residual
        else:
            for index, layer in enumerate(self.layers):
                hidden = layer(hidden, metadata=metadata, lengths=lengths, schedule_storage=schedule_storage)
                if index + 1 in self.taps:
                    auxiliary[:, self.taps[index + 1]].copy_(hidden)
        if len(self.layers) in self.taps:
            auxiliary[:, self.taps[len(self.layers)]].copy_(hidden)
        if self.taps and self.mode != "verify":
            self.commit(metadata.lengths, auxiliary, metadata)
        sampled = self.norm(hidden, metadata.sample_indices)
        return sampled, self.head(sampled), auxiliary, auxiliary if self.mode == "verify" and self.taps else None

    def commit(self, counts, records, metadata):
        if not self.taps or records is None:
            return
        indices = (metadata.query_start_loc[:-1] + counts - 1).long().clamp(0, records.shape[0] - 1)
        selected = records[indices].reshape(counts.numel(), -1)
        destination = self.pool.region("target_hidden").view(self.pool.backing.shape[0], -1)
        save_hidden(selected, destination, metadata.state_rows, counts > 0)

    def release_outputs(self):
        self.scratch = self.metadata = self.auxiliary = self.cache_lengths = None
        for layer in self.layers:
            layer.attention.scratch = layer.attention.lengths = None
            if isinstance(layer, PreparedDecoderLayer):
                layer.scratch = None
                layer.feed_forward.scratch = layer.feed_forward.output_source = None
            else:
                layer.module.mlp.operation.scratch = layer.module.mlp.operation.output_source = None

    def scratch_tensors(self):
        return ()
