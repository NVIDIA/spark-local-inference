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
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.column_pair_gather import PreparedColumnPairGather
from ignition.components.recurrent import DeltaMetadata, _commit_convolution
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.kda_speculation import KDARecords, PreparedKDAReplay
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy
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
    def allocate(cls, tokens, rows, config, world_size, device, *, examples=None):
        def tensor(role, shape):
            return declaration_tensor(examples, role, shape, torch.bfloat16, device)
        return cls(tensor("kimi.hidden_input", (tokens, config.hidden)),
                   tensor("kimi.hidden", (rows, config.hidden)),
                   tensor("kimi.local_logits", (rows, config.vocabulary // world_size)),
                   tensor("kimi.logits", (rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.hidden[:rows], self.local_logits[:rows], self.logits[:rows])


class PreparedMLP:
    """SiTU projections with invocation-local intermediates and explicit side-stream joins."""

    def __init__(self, weights, source, *, session, name, workspace=None, parallel_plans=(), examples=None):
        rows, device = source.shape[0], source.device
        width = (weights["gate_up"].shape[0] if isinstance(weights["gate_up"], torch.Tensor)
                 else weights["gate_up"].out_features)
        examples = {} if examples is None else examples
        def tensor(role, columns):
            key = "kimi.hidden_output" if role == "output" else "kimi.mlp." + role
            return declaration_tensor(examples, key, (rows, columns), source.dtype, device)
        intermediate = tensor("intermediate", width // 2)
        self.gate_up = prepare_projection(source, weights["gate_up"], tensor("gate_up", width),
            session=session, name=name + ".gate_up", workspace=workspace)
        self.down = prepare_projection(intermediate, weights["down"], tensor("output", source.shape[1]),
            session=session, name=name + ".down", workspace=workspace)
        self.intermediate_width = width // 2
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
            register_finalizer(session, self.refresh)
        self.gate_up.functional(session=session)
        self.down.functional(session=session)

    def refresh(self):
        self.workspace_nbytes = max(p.plan.memory_requirements().scratch_nbytes
                                   for p in (self.gate_up, self.down))

    def __call__(self, source):
        gate_up = self.gate_up(source)
        intermediate = torch.empty((source.shape[0], self.intermediate_width), dtype=source.dtype, device=source.device)
        situ(gate_up, intermediate)
        return self.down(intermediate)

    def start(self, source):
        outputs = [torch.empty((source.shape[0], width), dtype=source.dtype, device=source.device)
                   for width in (2 * self.intermediate_width, self.intermediate_width, source.shape[1])]
        workspace = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=source.device)
        _start_shared_expert(source, self.weights, self.scales, outputs, workspace,
            [self.gate_up.plan.handle, self.down.plan.handle], self.stream_handle)
        return source, outputs, workspace

    def finish(self, pending, routed):
        source, outputs, workspace = pending
        _join_shared_expert(source, routed, outputs, workspace, self.stream_handle)
        return outputs[-1]


class PreparedLatentMoE:
    def __init__(self, weights, source, config, shard, *, runtime, router_runtime, coordinator,
                 session, name, workspace, storage, route_runtime=None, prefill_route_runtime=None, examples=None):
        tokens, hidden = source.shape
        c, device = config.text, source.device
        latent = c["routed_expert_hidden_size"]
        examples = {} if examples is None else examples
        def buffer(role, width, dtype=torch.bfloat16):
            key = "kimi.hidden_output" if role == "output" else "kimi.moe." + role
            return declaration_tensor(examples, key, (tokens, width), dtype, device)
        self.hidden_columns, self.latent_begin = storage.hidden_columns, storage.latent_begin
        self.world, self.latent_width = shard.world_size, latent
        self.latent_norm, self.eps = weights["latent_norm"], c["rms_norm_eps"]
        latent_example, expert_output = buffer("latent", latent), buffer("expert_output", latent)
        up_input = buffer("up_input", latent // shard.world_size)
        self.up = prepare_projection(up_input, weights["latent_up"], buffer("output", hidden),
            session=session, name=name + ".latent_up", workspace=workspace).functional(session=session)
        self.expert_reduce = PreparedChunkedAllReduce(expert_output, buffer("latent_sum", latent),
            runtime=runtime, coordinator=coordinator, session=session,
            name=name + ".expert_reduce").functional(session=session)
        local_logits = buffer("local_router", weights["router"].shape[0], torch.float32)
        self.router_projection = UnquantizedProjection(source, weights["router"], local_logits,
            session=session, name=name + ".router").functional(session=session)
        top_k = c["num_experts_per_token"]
        routes = RouterScratch(buffer("selected_logits", top_k, torch.float32),
            buffer("route_ids", top_k, torch.int32),
            buffer("route_weights", top_k, torch.float32).fill_(1 / top_k))
        self.route_gather = self.prefill_route_gather = None
        if route_runtime is not None and tokens <= ROUTE_ROWS:
            down_output = buffer("local_down", latent // shard.world_size)
            self.down = prepare_projection(source, weights["latent_down_rows"], down_output,
                session=session, name=name + ".latent_down_rows", workspace=workspace)
            self.route_gather = PreparedRouteGather(down_output, local_logits, weights["correction"],
                latent_example, routes.weights, routes.ids, runtime=route_runtime, session=session,
                coordinator=coordinator, name=name + ".route_gather")
        else:
            logits = buffer("router", c["num_experts"], torch.float32)
            if prefill_route_runtime is not None:
                down_output = buffer("local_down", latent // shard.world_size)
                self.down = prepare_projection(source, weights["latent_down_rows"], down_output,
                    session=session, name=name + ".latent_down_rows", workspace=workspace)
                self.prefill_route_gather = PreparedColumnPairGather(down_output, local_logits, latent_example, logits,
                    runtime=prefill_route_runtime, session=session, coordinator=coordinator,
                    name=name + ".latent_router_gather",
                    buffers=examples.setdefault("kimi.moe.paired_columns", {})).functional(session=session)
            else:
                self.down = prepare_projection(buffer("source", hidden // shard.world_size), weights["latent_down"],
                    buffer("down", latent), session=session, name=name + ".latent_down", workspace=workspace)
                self.down_reduce = PreparedChunkedAllReduce(self.down.output, latent_example, runtime=runtime,
                    coordinator=coordinator, session=session, name=name + ".down_reduce").functional(session=session)
                self.local_experts = local_logits.shape[1]
                self.router_width = (self.local_experts + 31) // 32 * 32
                router_input = buffer("router_input", self.router_width, torch.float32)
                gathered = buffer("gathered_router", self.router_width * shard.world_size, torch.float32)
                self.router_gather = PreparedChunkedAllGather(router_input.view(torch.bfloat16), gathered.view(torch.bfloat16),
                    runtime=router_runtime, coordinator=coordinator, session=session,
                    name=name + ".router_gather").functional(session=session)
            self.router = PreparedRouter(logits, routes, score_func="sigmoid", correction_bias=weights["correction"],
                renormalize=True, routed_scaling_factor=c["routed_scaling_factor"], session=session,
                name=name + ".route").functional(session=session)
        self.experts = PreparedRoutedExperts(latent_example, weights["experts"], routes.ids, routes.weights,
            expert_output, workspace=workspace, session=session, name=name + ".experts").functional(session=session)
        self.shared = PreparedMLP(weights["shared"], source, session=session, name=name + ".shared", workspace=workspace,
            parallel_plans=(self.router_projection.plan, self.down.plan, self.experts.plan, self.up.plan)
            if self.route_gather is not None else (), examples=examples)
        self.concurrent_down = (ConcurrentProjectionChain((self.down,), parallel_plans=(self.router_projection.plan,),
            session=session, workspace=workspace).functional(session=session)
            if self.route_gather is not None and workspace is not None and workspace.overlap else None)
        if self.concurrent_down is None:
            self.down.functional(session=session)

    def __call__(self, source):
        pending_down = None if self.concurrent_down is None else self.concurrent_down.start(source)
        logits = self.router_projection(source)
        if self.route_gather is not None:
            down = (self.down(source) if self.concurrent_down is None else
                    self.concurrent_down.finish(pending_down, logits))
        pending_shared = self.shared.start(source) if self.shared.concurrent else None
        if self.route_gather is not None:
            latent, ids, route_weights = self.route_gather(down, logits)
        else:
            if self.prefill_route_gather is not None:
                latent, logits = self.prefill_route_gather(self.down(source), logits)
            else:
                router_input = torch.zeros((source.shape[0], self.router_width), dtype=torch.float32, device=source.device)
                router_input[:, :self.local_experts].copy_(logits)
                gathered = self.router_gather(router_input.view(torch.bfloat16)).view(torch.float32)
                logits = gathered.view(source.shape[0], self.world, self.router_width)[:, :, :self.local_experts].reshape(source.shape[0], -1)
                latent = self.down_reduce(self.down(source.index_select(1, self.hidden_columns)))
            routes = self.router(logits)
            ids, route_weights = routes.ids, routes.weights
        latent_sum = self.expert_reduce(self.experts(latent, ids, route_weights))
        up_input = torch.empty((source.shape[0], self.latent_width // self.world), dtype=source.dtype, device=source.device)
        rms_norm_columns(latent_sum, self.latent_norm, up_input, self.eps, begin=self.latent_begin)
        output = self.up(up_input)
        shared = self.shared.finish(pending_shared, output) if self.shared.concurrent else self.shared(source)
        return output + shared


class PreparedDecoder:
    """Compose Kimi layers with local residuals and verification records."""

    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, vocabulary_runtime,
                 embedding_runtime, router_runtime, vision_state, coordinator, session, name,
                 workspace, buffers, kda_records=None, route_runtime=None, context=None,
                 latent_runtimes=(), scoring=None, prefill_route_runtime=None, examples=None):
        if mode not in ("prefill", "decode", "verify") or mode == "verify" and kda_records is None:
            raise ValueError("Kimi verification requires candidate recording storage")
        self.weights, self.scratch, self.pool = weights, scratch, pool
        self.mode = mode
        self.vision_state = vision_state if mode == "prefill" else None
        config, source = weights.config, scratch.activations
        tokens, hidden = source.shape
        self.tokens, self.hidden = tokens, hidden
        self.sequences = metadata.starts.numel()
        self.verification_columns = 8 if mode == "verify" else 1
        examples = {} if examples is None else examples
        local_embedding = declaration_tensor(examples, "embedding", (tokens, weights.embedding.shape[1]), source.dtype, source.device)
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, local_embedding, source,
            runtime=embedding_runtime, coordinator=coordinator, num_rows=metadata.live_counts[:1],
            session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        delta = DeltaMetadata.allocate(self.sequences, source.device, verification_columns=self.verification_columns)
        self.layers = []
        self.residual_blocks = -(-len(weights.layers) // config.text["attn_res_block_size"])
        self.auxiliary_layers = config.draft.auxiliary_layers if weights.draft is not None else ()
        self.auxiliary_stream = config.draft.auxiliary_stream if self.auxiliary_layers else None
        self.auxiliary = (None if not self.auxiliary_layers else declaration_tensor(examples, "auxiliary",
            (tokens, len(self.auxiliary_layers) * hidden), source.dtype, source.device))
        for index, w in enumerate(weights.layers):
            prefix, kind = f"{name}.layer{index}", config.layers[index]
            kwargs = dict(mode=mode, session=session, name=prefix + ".attention", workspace=workspace,
                          buffers=examples)
            if kind == "kda":
                attention = PreparedKDA(w["attention"], source, pool.recurrent(index), delta,
                    eps=config.text["rms_norm_eps"], records=kda_records.layer(index) if mode == "verify" else None,
                    **kwargs)
            else:
                attention_type, extra = PreparedMLA, {}
                if kind == "dsa":
                    from ignition.recipes.kimi31.attention import PreparedDSA
                    attention_type, extra = PreparedDSA, dict(index_cache=pool.indexer_pages(index))
                attention = attention_type(w["attention"], source, metadata, pool.attention(index),
                    eps=config.mla_norm_eps, context=context,
                    peers=None if context is None else context.peers[index], coordinator=coordinator,
                    latents=next((r for r in latent_runtimes if tokens <= r.max_batch_size),
                                 latent_runtimes[-1] if latent_runtimes else None), **extra, **kwargs)
            if "mlp" in w:
                mlp = PreparedMLP(w["mlp"], source, session=session, name=prefix + ".mlp", workspace=workspace, examples=examples)
            else:
                mlp = PreparedLatentMoE(w, source, config, weights.shard, runtime=runtime,
                    router_runtime=router_runtime, route_runtime=route_runtime if mode != "prefill" else None,
                    prefill_route_runtime=prefill_route_runtime if mode == "prefill" else None,
                    coordinator=coordinator, session=session, name=prefix + ".moe", workspace=workspace, storage=buffers, examples=examples)
            total = declaration_tensor(examples, "kimi.hidden_output", source.shape, source.dtype, source.device)
            reduce = PreparedChunkedAllReduce(source, total, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".reduce").functional(session=session)
            self.layers.append((w, attention, mlp, reduce))
        self.scoring = None if scoring is None else scoring.prepare(source, weights.head,
            vocabulary=config.text["vocab_size"], session=session, coordinator=coordinator, name=name + ".prompt")
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head").functional(session=session)
        self.replay = None
        if mode == "verify":
            indices = kda_records.layers
            states = [pool.recurrent(i).recurrent[:kda_records.corrections.shape[1]] for i in indices]
            self.record_layers = indices
            self.record_shapes = (tuple(kda_records.corrections.shape), tuple(kda_records.keys_and_gates.shape))
            self.record_constants = (kda_records.A_log, kda_records.dt_bias)
            self.replay = PreparedKDAReplay(pool.backing, states, kda_records, self.layers[indices[0]][1].plan,
                session=session).functional()
            regions = [pool.layout.region(f"layer{i}.convolution") for i in indices]
            self.convolution_offsets = persistent_copy(torch.tensor([r.offset // 2 for r in regions],
                dtype=torch.int64, device=source.device))
            self.convolution_shape = regions[0].shape

    def release_outputs(self):
        self.scratch = self.auxiliary = None

    def commit(self, counts, verification, metadata):
        records, delta = verification
        self.replay(counts, records, delta)
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)

    def __call__(self, metadata):
        m, c = metadata, self.weights.config.text
        d = DeltaMetadata.allocate(self.sequences, m.token_ids.device, verification_columns=self.verification_columns)
        active = m.lengths > 0
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        d.fresh.copy_(m.starts == 0)
        d.state_indices[:, 0].copy_(m.state_rows)
        d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        d.final_state_indices.copy_(m.state_rows)
        d.num_tokens.copy_(m.live_counts[:1])
        d.num_seqs.copy_(m.live_counts[1:])
        records = None
        if self.replay is not None:
            records = KDARecords(self.record_layers,
                torch.zeros(self.record_shapes[0], dtype=torch.float32, device=m.token_ids.device),
                torch.zeros(self.record_shapes[1], dtype=torch.bfloat16, device=m.token_ids.device), *self.record_constants)
        local = self.embedding.lookup(m.token_ids, num_rows=m.live_counts[:1])
        if self.vision_state is not None:
            self.vision_state.inject(m, local)
        size, eps = c["attn_res_block_size"], c["rms_norm_eps"]
        delta, mode = self.embedding.gather(local), REPLACE
        prefix = torch.zeros_like(delta)
        blocks = torch.zeros((self.tokens, self.residual_blocks, self.hidden), dtype=delta.dtype, device=delta.device)
        residual = ResidualStream(prefix, blocks, torch.empty_like(delta))
        auxiliary = (None if not self.auxiliary_layers else
            torch.empty((self.tokens, len(self.auxiliary_layers) * self.hidden), dtype=delta.dtype, device=delta.device))
        for index, (w, attention, mlp, reduce) in enumerate(self.layers):
            committed, boundary = -(-index // size), index % size == 0
            captured = None
            if index in self.auxiliary_layers:
                column = self.auxiliary_layers.index(index) * self.hidden
                captured = auxiliary[:, column:column + self.hidden]
            hidden = residual.mix(delta, mode, sources=committed, write=index // size if boundary else -1,
                score=w["attention_res_score"], norm=w["input_layernorm"], eps=eps, output_eps=eps,
                mixture=captured if self.auxiliary_stream == "attn_res" else None, output=torch.empty_like(delta))
            if captured is not None and self.auxiliary_stream == "prefix":
                captured.copy_(residual.prefix)
            if self.weights.config.layers[index] == "kda":
                attended = attention(hidden, d, records=None if records is None else records.layer(index))
            else:
                attended = attention(hidden, m)
            attended = reduce(attended)
            hidden = residual.mix(attended, REPLACE if boundary else ADD, sources=committed + boundary, write=-1,
                score=w["mlp_res_score"], norm=w["post_attention_layernorm"], eps=eps, output_eps=eps,
                output=torch.empty_like(delta))
            delta, mode = reduce(mlp(hidden)), ADD
        hidden = residual.mix(delta, mode, sources=-(-len(self.layers) // size), write=-1,
            score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps,
            output=torch.empty_like(delta))
        if self.scoring is not None:
            self.scoring(hidden)
        sampled = hidden[m.sample_indices]
        return sampled, self.head(sampled), auxiliary, (records, d)
