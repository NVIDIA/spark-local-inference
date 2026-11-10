# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 attention residuals and latent MoE composed into complete forwards."""

from dataclasses import dataclass, field

import torch
import triton

from ignition.components.collectives import PreparedChunkedAllGather, PreparedChunkedAllReduce
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.recurrent import DeltaMetadata, _commit_convolution
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.kda_speculation import PreparedKDAReplay
from ignition.preparation.scratch import buffer_rows
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
    scratch: dict
    hidden_columns: torch.Tensor
    latent_begin: torch.Tensor
    allocator: object = None
    stage_allocators: dict = field(default_factory=dict)

    @classmethod
    def allocate(cls, config, shard, device, *, scratch=None, allocator=None):
        # Rank-specific offsets stay in tensors so every rank replays rank zero's compiled forward.
        return cls({} if scratch is None else scratch, torch.arange(*shard.partition(config.hidden), device=device),
            torch.tensor(shard.partition(config.text["routed_expert_hidden_size"])[:1], dtype=torch.int32, device=device),
            allocator)

    def scratch_tensors(self):
        return tuple(self.scratch.values())


    def stage_allocator(self, name):
        """Overlay serial stage storage after the decoder's persistent values."""
        if self.allocator is None:
            return None
        if name not in self.stage_allocators:
            self.stage_allocators[name] = self.allocator.fork()
        return self.stage_allocators[name]


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, world_size, device, *, activations=None, buffers=None):
        def zeros(shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))
        return cls(zeros((tokens, config.hidden)) if activations is None else activations, zeros((rows, config.hidden)),
                   zeros((rows, config.vocabulary // world_size)), zeros((rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.hidden[:rows], self.local_logits[:rows], self.logits[:rows])


class PreparedMLP:
    """A SiTU MLP; with an overlapping workspace, ``start`` and ``finish`` run it on the auxiliary stream."""

    def __init__(self, weights, source, *, session, name, buffers, workspace=None, allocator=None,
                 parallel_plans=()):
        rows, device = source.shape[0], source.device
        def buffer(field, width):
            if field not in buffers or buffers[field].shape[0] < rows:
                buffers[field] = (torch.zeros((rows, width), dtype=source.dtype, device=device) if allocator is None
                                  else allocator.zeros((rows, width), source.dtype))
            return buffer_rows(buffers, field, rows)
        width = (weights["gate_up"].shape[0] if isinstance(weights["gate_up"], torch.Tensor)
                 else weights["gate_up"].out_features)
        self.input = buffer("input", width // 2)
        self.gate_up = prepare_projection(source, weights["gate_up"], buffer("gate_up", width),
                                          session=session, name=name + ".gate_up", workspace=workspace)
        self.down = prepare_projection(self.input, weights["down"], buffer("output", source.shape[1]),
                                       session=session, name=name + ".down", workspace=workspace)
        self.concurrent = bool(parallel_plans) and workspace is not None and workspace.overlap
        if self.concurrent:
            self.arena, self.parallel_plans = workspace, tuple(parallel_plans)
            self.stream_handle = workspace.auxiliary_stream.cuda_stream
            self.weights, self.scales = [], []
            for projection in (self.gate_up, self.down):
                if isinstance(projection, UnquantizedProjection):
                    self.weights.append(projection.weight)
                    self.scales.append(None)
                else:
                    values, scales, _ = projection.parts
                    self.weights.append(values)
                    self.scales.append(scales.view(torch.uint8))
                for main in self.parallel_plans:
                    workspace.register_parallel(main, projection.plan)
            self.outputs = [self.gate_up.output, self.input, self.down.output]
            register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        main_bytes = max(plan.memory_requirements().scratch_nbytes for plan in self.parallel_plans)
        side_bytes = max(p.plan.memory_requirements().scratch_nbytes for p in (self.gate_up, self.down))
        _, self.workspace = self.arena.views(((main_bytes,), torch.uint8), ((side_bytes,), torch.uint8))

    def __call__(self, source):
        situ(self.gate_up(source), self.input)
        return self.down(self.input)

    def start(self, source):
        _start_shared_expert(source, self.weights, self.scales, self.outputs, self.workspace,
                             [self.gate_up.plan.handle, self.down.plan.handle], self.stream_handle)

    def finish(self, source, routed):
        _join_shared_expert(source, routed, self.outputs, self.workspace, self.stream_handle)
        return self.down.output


class PreparedLatentMoE:
    def __init__(self, weights, source, config, shard, *, runtime, router_runtime, coordinator, session, name, workspace, buffers, storage,
                 route_runtime=None, allocator=None):
        tokens, hidden = source.shape
        c, device = config.text, source.device
        latent = c["routed_expert_hidden_size"]

        def buffer(field, width, dtype=torch.bfloat16):
            # Decode-only fields first appear in a decoder of any row count, so each holds every route row.
            if field not in buffers or buffers[field].shape[0] < tokens:
                shape = (max(tokens, ROUTE_ROWS), width)
                buffers[field] = (torch.zeros(shape, dtype=dtype, device=device) if allocator is None
                                  else allocator.zeros(shape, dtype))
            return buffer_rows(buffers, field, tokens)

        self.source = buffer("source", hidden // shard.world_size)
        self.hidden_columns, self.latent_begin = storage.hidden_columns, storage.latent_begin
        self.latent = buffer("latent", latent)
        # The routed experts write where prefill reduces the latent down-projection from, and their sum
        # replaces the latent rows they consumed.
        self.expert_output = buffer("down", latent)
        self.latent_sum = self.latent
        self.up_input = buffer("up_input", latent // shard.world_size)
        self.output = buffer("output", hidden)
        self.up = prepare_projection(self.up_input, weights["latent_up"], self.output,
                                      session=session, name=name + ".latent_up", workspace=workspace)
        # Each rank normalizes the full latent row and keeps the columns its up-projection shard reads.
        self.latent_norm, self.eps = weights["latent_norm"], c["rms_norm_eps"]
        self.expert_reduce = PreparedChunkedAllReduce(self.expert_output, self.latent_sum, runtime=runtime,
                                                     coordinator=coordinator, session=session, name=name + ".expert_reduce")
        local_logits = buffer("local_router", weights["router"].shape[0], torch.float32)
        self.router_projection = UnquantizedProjection(source, weights["router"], local_logits, session=session, name=name + ".router")
        if "routes" not in buffers:
            buffers["routes"] = RouterScratch.allocate(tokens, device, experts=c["num_experts"], top_k=c["num_experts_per_token"])
        routes = buffers["routes"].rows(tokens)
        self.route_gather = None
        if route_runtime is not None and tokens <= ROUTE_ROWS:
            # Decode rows gather row-sharded latent and router shards and select experts in one exchange.
            self.down = prepare_projection(source, weights["latent_down_rows"],
                buffer("local_down", latent // shard.world_size), session=session, name=name + ".latent_down_rows",
                workspace=workspace)
            self.route_gather = PreparedRouteGather(self.down.output, local_logits, weights["correction"],
                self.latent, routes.weights, routes.ids, runtime=route_runtime, session=session,
                coordinator=coordinator, name=name + ".route_gather")
        else:
            self.down = prepare_projection(self.source, weights["latent_down"], buffer("down", latent),
                                            session=session, name=name + ".latent_down", workspace=workspace)
            self.down_reduce = PreparedChunkedAllReduce(self.down.output, self.latent, runtime=runtime,
                                                       coordinator=coordinator, session=session, name=name + ".down_reduce")
            logits = buffer("router", c["num_experts"], torch.float32)
            self.router_logits = logits
            local_experts = local_logits.shape[1]
            router_width = (local_experts + 31) // 32 * 32
            self.router_input = buffer("router_input", router_width, torch.float32)
            self.gathered_logits = buffer("gathered_router", router_width * shard.world_size, torch.float32)
            self.router_columns = self.gathered_logits.view(tokens, shard.world_size, router_width)[:, :, :local_experts]
            self.router_output = logits.view(tokens, shard.world_size, local_experts)
            # Gathering BF16 views transports the FP32 logits without conversion or arithmetic.
            self.router_gather = PreparedChunkedAllGather(self.router_input.view(torch.bfloat16), self.gathered_logits.view(torch.bfloat16),
                runtime=router_runtime, coordinator=coordinator, session=session, name=name + ".router_gather")
            self.router = PreparedRouter(logits, routes, score_func="sigmoid", correction_bias=weights["correction"],
                renormalize=True, routed_scaling_factor=c["routed_scaling_factor"], session=session, name=name + ".route")
        self.experts = PreparedRoutedExperts(self.latent, weights["experts"], routes.ids, routes.weights,
            self.expert_output, workspace=workspace, session=session, name=name + ".experts")
        # Decode and verification rows run the shared experts beside routing and the routed experts.
        self.shared = PreparedMLP(weights["shared"], source, session=session, name=name + ".shared",
                                  buffers=buffers.setdefault("shared", {}), allocator=allocator,
                                  workspace=workspace, parallel_plans=(self.router_projection.plan, self.down.plan,
                                      self.experts.plan, self.up.plan) if self.route_gather is not None else ())
        self.concurrent_down = (ConcurrentProjectionChain((self.down,),
            parallel_plans=(self.router_projection.plan,), session=session, workspace=workspace)
            if self.route_gather is not None and workspace is not None and workspace.overlap else None)

    def __call__(self, source):
        if self.concurrent_down is not None:
            self.concurrent_down.start(source)
        logits = self.router_projection(source)
        if self.route_gather is not None:
            if self.concurrent_down is not None:
                self.concurrent_down.finish(source, logits)
            else:
                self.down(source)
        if self.shared.concurrent:
            self.shared.start(source)
        if self.route_gather is not None:
            self.route_gather()
        else:
            self.router_input[:, :logits.shape[1]].copy_(logits)
            self.router_gather(self.router_input.view(torch.bfloat16))
            self.router_output.copy_(self.router_columns)
            self.router(self.router_logits)
            self.source.copy_(source.index_select(1, self.hidden_columns))
            self.down_reduce(self.down(self.source))
        self.expert_reduce(self.experts())
        rms_norm_columns(self.latent_sum, self.latent_norm, self.up_input, self.eps, begin=self.latent_begin)
        self.up(self.up_input)
        shared = self.shared.finish(source, self.output) if self.shared.concurrent else self.shared(source)
        self.output.add_(shared)
        return self.output


class PreparedDecoder:
    """Kimi K3's decoder over packed rows.

    ``verify`` rows are speculative candidates: KDA layers record them without advancing committed states
    until ``commit`` replays each request's accepted prefix. With a draft model, every forward also keeps
    the draft's configured residual stream entering its auxiliary layers in ``auxiliary``.
    """

    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, vocabulary_runtime, embedding_runtime, router_runtime, vision_state,
                 coordinator, session, name, workspace, buffers, kda_records=None, route_runtime=None, context=None,
                 latent_runtimes=(), scoring=None):
        # Verification records its candidate rows; other modes ignore records.
        if mode not in ("prefill", "decode", "verify") or mode == "verify" and kda_records is None:
            raise ValueError("Kimi K3 verification decodes recorded candidate rows")
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.vision_state = vision_state if mode == "prefill" else None
        config, source = weights.config, scratch.activations
        allocator = buffers.allocator
        if ("local_embedding" not in buffers.scratch
                or buffers.scratch["local_embedding"].shape[0] < source.shape[0]):
            shape = (source.shape[0], weights.embedding.shape[1])
            buffers.scratch["local_embedding"] = (torch.zeros(shape, dtype=source.dtype, device=source.device)
                if allocator is None else allocator.zeros(shape, source.dtype))
        self.local_embedding = buffers.scratch["local_embedding"][:source.shape[0]]
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, self.local_embedding, source,
            runtime=embedding_runtime, coordinator=coordinator,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.delta = DeltaMetadata.allocate(metadata.starts.numel(), source.device,
            verification_columns=8 if mode == "verify" else 1)
        self.buffers, self.layers = buffers.scratch, []
        def zeros(shape):
            return (torch.zeros(shape, dtype=source.dtype, device=source.device) if allocator is None
                    else allocator.zeros(shape, source.dtype))

        # Residual blocks stay live across layers; reserve them before smaller layer intermediates.
        tokens, hidden = source.shape
        if "residual" not in self.buffers:
            committed = -(-len(weights.layers) // config.text["attn_res_block_size"])
            self.buffers["residual"] = dict(blocks=zeros((tokens, committed, hidden)), prefix=zeros(source.shape))
        if "total" not in self.buffers:
            self.buffers["total"] = zeros(source.shape)
        self.auxiliary = None
        # Auxiliary features remain live until draft context updates consume them.
        self.auxiliary_layers = config.draft.auxiliary_layers if weights.draft is not None else ()
        self.auxiliary_stream = config.draft.auxiliary_stream if self.auxiliary_layers else None
        if self.auxiliary_layers:
            if "auxiliary" not in self.buffers:
                self.buffers["auxiliary"] = zeros((tokens, len(self.auxiliary_layers) * hidden))
            self.auxiliary = self.buffers["auxiliary"][:tokens]
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            kind = config.layers[index]
            kwargs = dict(mode=mode, session=session, name=prefix + ".attention", workspace=workspace,
                          buffers=self.buffers.setdefault(kind, {}), allocator=buffers.stage_allocator(kind))
            if kind == "kda":
                attention = PreparedKDA(w["attention"], source, pool.recurrent(index), self.delta,
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
                    latents=next((r for r in latent_runtimes if source.shape[0] <= r.max_batch_size),
                                 latent_runtimes[-1] if latent_runtimes else None), **extra, **kwargs)
            if "mlp" in w:
                # The dense MLP writes the MoE layers' output, which its reduction consumes before the next mixture.
                dense = self.buffers.setdefault("dense", {})
                if "output" in self.buffers.get("moe", {}):
                    dense.setdefault("output", self.buffers["moe"]["output"])
                mlp = PreparedMLP(w["mlp"], source, session=session, name=prefix + ".mlp", buffers=dense,
                                  allocator=buffers.stage_allocator("dense"))
            else:
                # Shared experts write the sum buffer, which the preceding mixture consumed and the MoE
                # reduction overwrites only after their output is added.
                if "total" in self.buffers:
                    self.buffers.setdefault("moe", {}).setdefault("shared", {}).setdefault("output", self.buffers["total"])
                mlp = PreparedLatentMoE(w, source, config, weights.shard, runtime=runtime,
                    router_runtime=router_runtime, route_runtime=route_runtime if mode != "prefill" else None,
                    coordinator=coordinator, session=session, name=prefix + ".moe", workspace=workspace,
                    buffers=self.buffers.setdefault("moe", {}), storage=buffers, allocator=buffers.stage_allocator("moe"))
            # The sum buffer is consumed before the next attention or MLP reduction.
            total = self.buffers["total"][:source.shape[0]]
            reduce = PreparedChunkedAllReduce(source, total, runtime=runtime, coordinator=coordinator,
                                              session=session, name=prefix + ".reduce")
            self.layers.append((w, attention, mlp, reduce))
        if allocator is not None:
            allocator.join(*buffers.stage_allocators.values())
        # The running block sum and committed blocks persist through one forward; every forward
        # rewrites them from its embeddings before reading them.
        r = self.buffers["residual"]
        # The normalized mixture entering attention takes the MoE output, which the previous reduction
        # consumed; the one entering the MLP takes the attention output, which its reduction consumed.
        self.mlp_input = self.buffers["kda"]["output"][:tokens]
        self.residual = ResidualStream(r["prefix"][:tokens], r["blocks"][:tokens], self.buffers["moe"]["output"][:tokens])
        self.scoring = None if scoring is None else scoring.prepare(self.residual.output, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator, name=name + ".prompt")
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")
        self.replay = None
        if mode == "verify":
            indices = kda_records.layers
            states = [pool.recurrent(i).recurrent[:kda_records.corrections.shape[1]] for i in indices]
            self.replay = PreparedKDAReplay(pool.backing, states, kda_records, self.layers[indices[0]][1].plan,
                self.delta.state_indices[:, 0], session=session, workspace=workspace)
            regions = [pool.layout.region(f"layer{i}.convolution") for i in indices]
            self.convolution_offsets = torch.tensor([r.offset // 2 for r in regions], dtype=torch.int64, device=source.device)
            self.convolution_shape = regions[0].shape

    def commit(self, counts):
        self.replay(counts)
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, self.metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)

    def __call__(self):
        m, d, c = self.metadata, self.delta, self.weights.config.text
        active = m.lengths > 0
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        d.fresh.copy_(m.starts == 0)
        d.state_indices[:, 0].copy_(m.state_rows)
        d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        d.final_state_indices.copy_(m.state_rows)
        d.num_tokens.copy_(m.live_counts[:1])
        d.num_seqs.copy_(m.live_counts[1:])
        local = self.embedding.lookup()
        if self.vision_state is not None:
            self.vision_state.inject(m, local)
        # Each layer's residual source is the running sum of its block plus every committed block;
        # a block commits its running sum when the next block starts.
        size, eps = c["attn_res_block_size"], c["rms_norm_eps"]
        delta, mode = self.embedding.gather(local), REPLACE
        for index, (w, attention, mlp, reduce) in enumerate(self.layers):
            committed, boundary = -(-index // size), index % size == 0
            auxiliary = None
            if index in self.auxiliary_layers:
                column = self.auxiliary_layers.index(index) * self.residual.prefix.shape[1]
                auxiliary = self.auxiliary[:, column:column + self.residual.prefix.shape[1]]
            hidden = self.residual.mix(delta, mode, sources=committed, write=index // size if boundary else -1,
                score=w["attention_res_score"], norm=w["input_layernorm"], eps=eps, output_eps=eps,
                mixture=auxiliary if self.auxiliary_stream == "attn_res" else None)
            if auxiliary is not None and self.auxiliary_stream == "prefix":
                auxiliary.copy_(self.residual.prefix)
            attended = reduce(attention(hidden))
            hidden = self.residual.mix(attended, REPLACE if boundary else ADD, sources=committed + boundary, write=-1,
                score=w["mlp_res_score"], norm=w["post_attention_layernorm"], eps=eps, output_eps=eps,
                output=self.mlp_input)
            delta, mode = reduce(mlp(hidden)), ADD
        hidden = self.residual.mix(delta, mode, sources=-(-len(self.layers) // size), write=-1,
            score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps)
        if self.scoring is not None:
            self.scoring()
        self.scratch.hidden.copy_(hidden[m.sample_indices])
        self.head(self.scratch.hidden)
