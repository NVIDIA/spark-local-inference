# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 attention residuals and latent MoE composed into complete forwards."""

from dataclasses import dataclass

import torch
import triton

from ignition.components.collectives import PreparedChunkedAllGather, PreparedChunkedAllReduce
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.components.recurrent import DeltaMetadata, _commit_convolution
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.kda_speculation import KDARecords, PreparedKDAReplay
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import register_finalizer

from .attention import PreparedMLA
from .kda import PreparedKDA
from .pointwise import rms_norm_columns, situ
from .projections import prepare_projection
from .residual import ADD, REPLACE, ResidualStream
from .routing import MAX_ROWS as ROUTE_ROWS, PreparedRouteGather


# A concurrent shared-expert MLP forks onto the auxiliary stream after its input is written and joins before its
# output is added; the explicit mutated outputs order both against the routed branch through compilation.
@torch.library.custom_op("flare::kimi3_start_shared_expert", mutates_args=("outputs", "workspace"))
def _start_shared_expert(source: torch.Tensor, weights: list[torch.Tensor], scales: list[torch.Tensor | None],
                         outputs: list[torch.Tensor], workspace: torch.Tensor, plans: list[int], auxiliary_stream: int) -> None:
    from ignition.components.sm103_projection import bf16_out, packed_out_workspace

    def project(value, index, output):
        if scales[index] is None:
            bf16_out(value, weights[index], output, plans[index])
        else:
            packed_out_workspace(value, weights[index], scales[index], None, None, output,
                                 workspace, "none", None, plans[index], "any")

    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    auxiliary.wait_stream(main)
    gate_up, intermediate, projected = outputs
    with torch.cuda.stream(auxiliary):
        project(source, 0, gate_up)
        situ(gate_up, intermediate)
        project(intermediate, 1, projected)


@_start_shared_expert.register_fake
def _fake_start_shared_expert(source, weights, scales, outputs, workspace, plans, auxiliary_stream) -> None:
    pass


@torch.library.custom_op("flare::kimi3_join_shared_expert", mutates_args=("outputs", "workspace"))
def _join_shared_expert(source: torch.Tensor, routed: torch.Tensor, outputs: list[torch.Tensor],
                        workspace: torch.Tensor, auxiliary_stream: int) -> None:
    main = torch.cuda.current_stream(source.device)
    main.wait_stream(torch.cuda.ExternalStream(auxiliary_stream, device=source.device))


@_join_shared_expert.register_fake
def _fake_join_shared_expert(source, routed, outputs, workspace, auxiliary_stream) -> None:
    pass


@dataclass(frozen=True)
class DecoderStorage:
    hidden_columns: torch.Tensor
    latent_begin: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, device):
        return cls(persistent_copy(torch.arange(*shard.partition(config.hidden), device=device)),
            persistent_copy(torch.tensor(shard.partition(config.text["routed_expert_hidden_size"])[:1],
                                         dtype=torch.int32, device=device)))


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, world_size, device):
        def zeros(shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        return cls(zeros((tokens, config.hidden)), zeros((rows, config.hidden)),
                   zeros((rows, config.vocabulary // world_size)), zeros((rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.hidden[:rows], self.local_logits[:rows], self.logits[:rows])


class PreparedMLP:
    """SiTU projections with local intermediates and explicit asynchronous operands."""

    def __init__(self, weights, source, *, session, name, workspace=None, parallel_plans=(), examples=None):
        rows, device = source.shape[0], source.device
        width = (weights["gate_up"].shape[0] if isinstance(weights["gate_up"], torch.Tensor)
                 else weights["gate_up"].out_features)
        projected = declaration_tensor(examples, "mlp.gate_up", (rows, width), source.dtype, device)
        intermediate = declaration_tensor(examples, "mlp.intermediate", (rows, width // 2), source.dtype, device)
        output = (torch.zeros_like(source) if examples is None else
                  declaration_tensor(examples, "mlp.output", source.shape, source.dtype, device))
        self.width = width
        self.gate_up = prepare_projection(source, weights["gate_up"], projected,
                                          session=session, name=name + ".gate_up", workspace=workspace)
        self.down = prepare_projection(intermediate, weights["down"], output,
                                       session=session, name=name + ".down", workspace=workspace)
        self.concurrent = bool(parallel_plans) and workspace is not None and workspace.overlap
        if self.concurrent:
            self.stream = workspace.auxiliary_stream
            self.stream_handle = self.stream.cuda_stream
            self.weights, self.scales = [], []
            for projection in (self.gate_up, self.down):
                if isinstance(projection, UnquantizedProjection):
                    self.weights.append(projection.weight)
                    self.scales.append(None)
                else:
                    values, scales, _ = projection.parts
                    self.weights.append(values)
                    self.scales.append(scales.view(torch.uint8))
                for main in parallel_plans:
                    workspace.register_parallel(main, projection.plan)
            def finalize():
                self.scratch_bytes = max(p.plan.memory_requirements().scratch_nbytes for p in (self.gate_up, self.down))
            register_finalizer(session, finalize, workspace=workspace)
        self.gate_up.functional(session=session)
        self.down.functional(session=session)

    def __call__(self, source):
        intermediate = torch.empty((source.shape[0], self.width // 2), dtype=source.dtype, device=source.device)
        situ(self.gate_up(source), intermediate)
        return self.down(intermediate)

    def start(self, source):
        outputs = [torch.empty((source.shape[0], self.width), dtype=source.dtype, device=source.device),
                   torch.empty((source.shape[0], self.width // 2), dtype=source.dtype, device=source.device),
                   torch.empty_like(source)]
        workspace = torch.empty((self.scratch_bytes,), dtype=torch.uint8, device=source.device)
        _start_shared_expert(source, self.weights, self.scales, outputs, workspace,
                             [self.gate_up.plan.handle, self.down.plan.handle], self.stream_handle)
        return source, outputs, workspace

    def finish(self, pending, routed):
        source, outputs, workspace = pending
        _join_shared_expert(source, routed, outputs, workspace, self.stream_handle)
        return outputs[-1]


class PreparedLatentMoE:
    def __init__(self, weights, source, config, shard, *, runtime, router_runtime, coordinator, session, name,
                 workspace, storage, route_runtime=None, examples=None):
        tokens, hidden = source.shape
        c, device = config.text, source.device
        latent = c["routed_expert_hidden_size"]
        def buffer(role, width, dtype=torch.bfloat16):
            return declaration_tensor(examples, "moe." + role, (tokens, width), dtype, device)
        local_source, latent_source = buffer("local_source", hidden // shard.world_size), buffer("latent_source", latent)
        expert_output, up_input, output = buffer("expert_output", latent), buffer("up_input", latent // shard.world_size), buffer("output", hidden)
        self.hidden_columns, self.latent_begin = storage.hidden_columns, storage.latent_begin
        self.local_latent = latent // shard.world_size
        self.latent_norm, self.eps = weights["latent_norm"], c["rms_norm_eps"]
        self.up = prepare_projection(up_input, weights["latent_up"], output,
            session=session, name=name + ".latent_up", workspace=workspace).functional(session=session)
        self.expert_reduce = PreparedChunkedAllReduce(expert_output, latent_source, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".expert_reduce").functional(session=session)
        local_logits = buffer("local_logits", weights["router"].shape[0], torch.float32)
        self.router_projection = UnquantizedProjection(source, weights["router"], local_logits,
            session=session, name=name + ".router").functional(session=session)
        routes = RouterScratch.allocate(tokens, device, experts=c["num_experts"], top_k=c["num_experts_per_token"])
        self.route_gather = None
        if route_runtime is not None and tokens <= ROUTE_ROWS:
            local_down = buffer("local_down", latent // shard.world_size)
            self.down = prepare_projection(source, weights["latent_down_rows"], local_down,
                session=session, name=name + ".latent_down_rows", workspace=workspace).functional(session=session)
            self.route_gather = PreparedRouteGather(local_down, local_logits, weights["correction"],
                latent_source, routes.weights, routes.ids, runtime=route_runtime, session=session,
                coordinator=coordinator, name=name + ".route_gather")
        else:
            down = buffer("down", latent)
            self.down = prepare_projection(local_source, weights["latent_down"], down,
                session=session, name=name + ".latent_down", workspace=workspace).functional(session=session)
            self.down_reduce = PreparedChunkedAllReduce(down, latent_source, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".down_reduce").functional(session=session)
            logits = buffer("logits", c["num_experts"], torch.float32)
            self.local_experts = local_logits.shape[1]
            self.world = shard.world_size
            self.router_width = (self.local_experts + 31) // 32 * 32
            router_input = buffer("router_input", self.router_width, torch.float32)
            gathered = buffer("gathered", self.router_width * self.world, torch.float32)
            self.router_gather = PreparedChunkedAllGather(router_input.view(torch.bfloat16), gathered.view(torch.bfloat16),
                runtime=router_runtime, coordinator=coordinator, session=session, name=name + ".router_gather").functional(session=session)
            self.router = PreparedRouter(logits, routes, score_func="sigmoid", correction_bias=weights["correction"],
                renormalize=True, routed_scaling_factor=c["routed_scaling_factor"], session=session,
                name=name + ".route").functional(session=session)
        self.experts = PreparedRoutedExperts(latent_source, weights["experts"], routes.ids, routes.weights,
            expert_output, workspace=workspace, session=session, name=name + ".experts").functional(session=session)
        self.shared = PreparedMLP(weights["shared"], source, session=session, name=name + ".shared", workspace=workspace,
            parallel_plans=(self.router_projection.plan, self.down.plan, self.experts.plan, self.up.plan)
            if self.route_gather is not None else (), examples=examples)

    def __call__(self, source):
        pending = self.shared.start(source) if self.shared.concurrent else None
        logits = self.router_projection(source)
        if self.route_gather is not None:
            latent, routes = self.route_gather(self.down(source), logits)
        else:
            router_input = torch.zeros((source.shape[0], self.router_width), dtype=torch.float32, device=source.device)
            router_input[:, :logits.shape[1]].copy_(logits)
            gathered = self.router_gather(router_input.view(torch.bfloat16)).view(torch.float32)
            logits = gathered.view(source.shape[0], self.world, self.router_width)[:, :, :self.local_experts].reshape(source.shape[0], -1)
            routes = self.router(logits)
            latent = self.down_reduce(self.down(source.index_select(1, self.hidden_columns)))
        latent_sum = self.expert_reduce(self.experts(latent, routes.ids, routes.weights))
        up_input = torch.empty((source.shape[0], self.local_latent), dtype=source.dtype, device=source.device)
        rms_norm_columns(latent_sum, self.latent_norm, up_input, self.eps, begin=self.latent_begin)
        output = self.up(up_input)
        shared = self.shared.finish(pending, output) if pending is not None else self.shared(source)
        return output + shared


class PreparedDecoder:
    """Packed decoder with invocation-local residual blocks and verification records."""

    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, vocabulary_runtime, embedding_runtime,
                 router_runtime, vision_state, coordinator, session, name, workspace, storage, kda_records=None,
                 route_runtime=None, context=None, latent_runtimes=(), scoring=None, examples=None):
        if mode not in ("prefill", "decode", "verify") or mode == "verify" and kda_records is None:
            raise ValueError("Kimi K3 verification requires KDA record geometry")
        self.weights, self.scratch, self.pool, self.mode = weights, scratch, pool, mode
        self.vision_state = vision_state if mode == "prefill" else None
        config, source = weights.config, scratch.activations
        tokens, hidden = source.shape
        self.verification_columns = 8 if mode == "verify" else 1
        local_embedding = declaration_tensor(examples, "decoder.local_embedding",
            (tokens, weights.embedding.shape[1]), source.dtype, source.device)
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, local_embedding, source,
            runtime=embedding_runtime, coordinator=coordinator, num_rows=metadata.live_counts[:1], session=session,
            name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        delta = DeltaMetadata.allocate(metadata.starts.numel(), source.device, verification_columns=self.verification_columns)
        self.layers = []
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            kwargs = dict(mode=mode, session=session, name=prefix + ".attention", workspace=workspace, examples=examples)
            attention = (PreparedKDA(w["attention"], source, pool.recurrent(index), delta,
                eps=config.text["rms_norm_eps"], records=kda_records.layer(index) if mode == "verify" else None,
                **kwargs) if config.layers[index] == "kda" else
                PreparedMLA(w["attention"], source, metadata, pool.attention(index), context=context,
                    peers=None if context is None else context.peers[index], coordinator=coordinator,
                    latents=next((r for r in latent_runtimes if tokens <= r.max_batch_size),
                                 latent_runtimes[-1] if latent_runtimes else None), **kwargs))
            mlp = (PreparedMLP(w["mlp"], source, session=session, name=prefix + ".mlp", workspace=workspace, examples=examples)
                if "mlp" in w else PreparedLatentMoE(w, source, config, weights.shard, runtime=runtime,
                    router_runtime=router_runtime, route_runtime=route_runtime if mode != "prefill" else None,
                    coordinator=coordinator, session=session, name=prefix + ".moe", workspace=workspace, storage=storage, examples=examples))
            reduced = (torch.zeros_like(source) if examples is None else
                       declaration_tensor(examples, "decoder.reduced", source.shape, source.dtype, source.device))
            reduction = PreparedChunkedAllReduce(source, reduced, runtime=runtime,
                coordinator=coordinator, session=session, name=prefix + ".reduce").functional(session=session)
            self.layers.append((w, attention, mlp, reduction))
        self.scoring = None if scoring is None else scoring.prepare(source, weights.head,
            vocabulary=config.text["vocab_size"], session=session, coordinator=coordinator, name=name + ".prompt")
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            workspace=workspace).functional(session=session)
        self.auxiliary_layers = config.draft.auxiliary_layers if weights.draft is not None else ()
        self.auxiliary = (declaration_tensor(examples, "decoder.auxiliary",
            (tokens, len(self.auxiliary_layers) * hidden), source.dtype, source.device)
            if self.auxiliary_layers else None)
        self.replay = None
        if mode == "verify":
            indices = kda_records.layers
            states = [pool.recurrent(i).recurrent[:kda_records.corrections.shape[1]] for i in indices]
            self.record_layers = kda_records.layers
            self.record_shape, self.kg_shape = tuple(kda_records.corrections.shape), tuple(kda_records.keys_and_gates.shape)
            self.record_A_log, self.record_dt_bias = kda_records.A_log, kda_records.dt_bias
            self.replay = PreparedKDAReplay(pool.backing, states, kda_records, self.layers[indices[0]][1].plan,
                session=session).functional()
            regions = [pool.layout.region(f"layer{i}.convolution") for i in indices]
            self.convolution_offsets = persistent_copy(torch.tensor([r.offset // 2 for r in regions], dtype=torch.int64, device=source.device))
            self.convolution_shape = regions[0].shape

    def release_outputs(self):
        self.scratch = self.auxiliary = None

    def delta_metadata(self, m):
        sequences = m.starts.numel()
        state_indices = torch.zeros((sequences, self.verification_columns), dtype=torch.int64, device=m.starts.device)
        state_indices[:, 0].copy_(m.state_rows)
        return DeltaMetadata(m.query_start_loc, torch.where(m.lengths > 0, m.state_rows, -1), m.starts == 0,
            torch.ones((sequences,), dtype=torch.int32, device=m.starts.device), state_indices,
            torch.where(m.starts > 0, m.state_rows, 0), m.state_rows,
            torch.zeros((sequences,), dtype=torch.int64, device=m.starts.device),
            torch.zeros((sequences,), dtype=torch.int32, device=m.starts.device), m.live_counts[1:], m.live_counts[:1])

    def commit(self, counts, records, metadata):
        counts = counts.to(torch.int32)
        self.replay(counts, records, self.delta_metadata(metadata))
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)

    def __call__(self, metadata):
        m, c = metadata, self.weights.config.text
        d = self.delta_metadata(m)
        local = self.embedding.lookup(token_ids=m.token_ids, num_rows=m.live_counts[:1])
        if self.vision_state is not None:
            self.vision_state.inject(m, local)
        delta, mode = self.embedding.gather(local), REPLACE
        tokens, hidden_width = delta.shape
        size, eps = c["attn_res_block_size"], c["rms_norm_eps"]
        residual = ResidualStream(torch.empty_like(delta),
            torch.empty((tokens, -(-len(self.layers) // size), hidden_width), dtype=delta.dtype, device=delta.device),
            torch.empty_like(delta))
        auxiliary = (torch.empty((tokens, len(self.auxiliary_layers) * hidden_width), dtype=delta.dtype,
            device=delta.device) if self.auxiliary_layers else None)
        records = None
        if self.replay is not None:
            records = KDARecords(self.record_layers, torch.zeros(self.record_shape, dtype=torch.float32, device=delta.device),
                torch.zeros(self.kg_shape, dtype=torch.bfloat16, device=delta.device), self.record_A_log, self.record_dt_bias)
        for index, (w, attention, mlp, reduction) in enumerate(self.layers):
            committed, boundary = -(-index // size), index % size == 0
            mixture = None
            if index in self.auxiliary_layers:
                column = self.auxiliary_layers.index(index) * hidden_width
                mixture = auxiliary[:, column:column + hidden_width]
            hidden = residual.mix(delta, mode, sources=committed, write=index // size if boundary else -1,
                score=w["attention_res_score"], norm=w["input_layernorm"], eps=eps, output_eps=eps, mixture=mixture)
            attended = reduction(attention(hidden, d, records=None if records is None else records.layer(index))
                if self.weights.config.layers[index] == "kda" else attention(hidden, m))
            hidden = residual.mix(attended, REPLACE if boundary else ADD, sources=committed + boundary, write=-1,
                score=w["mlp_res_score"], norm=w["post_attention_layernorm"], eps=eps, output_eps=eps,
                output=torch.empty_like(hidden))
            delta, mode = reduction(mlp(hidden)), ADD
        hidden = residual.mix(delta, mode, sources=-(-len(self.layers) // size), write=-1,
            score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps)
        if self.scoring is not None:
            self.scoring(hidden)
        sampled = hidden[m.sample_indices]
        return sampled, self.head(sampled), auxiliary, records
