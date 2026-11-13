# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM head-sharded latent attention with replicated C4 index and KV caches."""

import itertools
import weakref
from dataclasses import dataclass, replace
from functools import partial

import torch
from b12x.attention import sparse_mla
from flashinfer.triton.norm import rms_norm

from ignition.components.linear import (
    BatchedQuantizedProjection, BatchedUnquantizedProjection, QuantizedProjection, prepare_projection,
)
from ignition.components import sm103
from ignition.components.block_fp8 import BlockFP8Projection, _workspace_bytes
from ignition.components.sparse_mla import PreparedLatentCacheWriter, PreparedSparseMLA
from ignition.components.sparse_mla_sm103 import Sm103PrefillSparseMLA, Sm103SparseMLA
from ignition.preparation.tuning import register_finalizer

from .indexer import IndexMetadata, IndexScratch, IndexState, PreparedGlmIndexer


@dataclass(frozen=True)
class MLAState:
    cache: torch.Tensor
    index: IndexState
    # Every rank's latent cache address when ranks share the context; prefill reads peer records through it.
    peers: torch.Tensor | None = None

    @classmethod
    def allocate(cls, pages, slots, device, *, ring=4):
        index = IndexState.allocate(pages, slots, device, ring=ring)
        return cls(torch.zeros((pages * 4, 64, 528), dtype=torch.uint8, device=device), index)


@dataclass(frozen=True)
class MLAScratch:
    projected_input: torch.Tensor
    query_input: torch.Tensor
    query_residual: torch.Tensor
    latent_input: torch.Tensor
    latent: torch.Tensor
    query: torch.Tensor
    absorbed_query: torch.Tensor
    attended: torch.Tensor
    expanded: torch.Tensor
    output: torch.Tensor
    visible: torch.Tensor
    index: IndexScratch
    # Context-parallel decode exchanges every rank's heads for one slice of rows; context-parallel
    # prefill gathers every rank's records of the whole prepared context into the slab.
    gathered: torch.Tensor
    partial: torch.Tensor
    lse: torch.Tensor
    slab: torch.Tensor
    # Ranks on separate hosts stage the head shards each peer combines, and every rank's packs.
    exchange_send: torch.Tensor
    exchange_received: torch.Tensor
    # Prefill across hosts gathers one window of every rank's latent records at a time, [world, window * 528]
    # bytes, with each rank's byte count in the window.
    records: torch.Tensor
    record_counts: torch.Tensor

    @classmethod
    def allocate(cls, tokens, heads, blocks, mode, device, *, context_ranks=1, remote=False):
        if heads not in (16, 32, 64):
            raise ValueError("GLM MLA scratch requires TP1, TP2 or TP4 head partitions")
        from .dcp import PREFILL_SLICE_ROWS, SLICE_ROWS

        index = IndexScratch.allocate(tokens, blocks, mode, device, context_ranks=context_ranks)
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        def buffer(*shape):
            return zeros(tokens, *shape)
        # Fused layer-input projection: query 1536, latent 512, index key 128 and index head weights 32.
        projected = buffer(2208)
        prefill = context_ranks > 1 and mode == "prefill"
        query = buffer(heads, 256)
        # CP prefill consumes each query slice before expanding values into the same rows.
        expanded = query if prefill else buffer(heads, 256)
        exchange = min(tokens, SLICE_ROWS) if context_ranks > 1 and not prefill else 0
        # Context-parallel rows absorb, attend and expand one slice at a time; otherwise every row at once.
        absorbed = exchange or (min(tokens, PREFILL_SLICE_ROWS) if prefill else tokens)
        # One 528-byte record per global position: `blocks` pages of 256 records on every rank.
        positions = blocks * 256 * context_ranks if prefill else 0
        from ignition.components.head_exchange import pack_width

        width = pack_width(context_ranks, heads * context_ranks, 512) if remote and exchange else 0
        from .dcp import WINDOW_RECORDS

        window = min(WINDOW_RECORDS, blocks * 256) if remote and prefill else 0
        return cls(projected, projected[:, :1536], buffer(1536), projected[:, 1536:2048], buffer(512),
            query, zeros(absorbed, heads, 512), zeros(absorbed, heads, 512), expanded,
            buffer(4096), zeros(tokens, dtype=torch.int32), index,
            zeros(exchange, heads * context_ranks, 512), zeros(exchange, heads * context_ranks, 512),
            zeros(exchange, heads * context_ranks, dtype=torch.float32), zeros(positions, 528, dtype=torch.uint8),
            zeros(exchange if width else 0, width), zeros(exchange if width else 0, width * context_ranks),
            zeros(context_ranks if window else 0, window * 528, dtype=torch.uint8),
            zeros(context_ranks if window else 0, dtype=torch.int64))

    def rows(self, tokens, mode):
        if type(tokens) is not int or not 0 < tokens <= self.output.shape[0]:
            raise ValueError("GLM MLA rows exceed scratch capacity")
        return MLAScratch(self.projected_input[:tokens], self.query_input[:tokens],
            self.query_residual[:tokens], self.latent_input[:tokens],
            self.latent[:tokens], self.query[:tokens], self.absorbed_query[:tokens], self.attended[:tokens],
            self.expanded[:tokens], self.output[:tokens], self.visible[:tokens], self.index.rows(tokens, mode),
            self.gathered[:tokens], self.partial[:tokens], self.lse[:tokens], self.slab,
            self.exchange_send[:tokens], self.exchange_received[:tokens], self.records, self.record_counts)


_CHAINS = weakref.WeakValueDictionary()
_CHAIN_HANDLES = itertools.count()


@torch.library.custom_op("flare::start_glm_query_chain", mutates_args=("query", "absorbed", "gathered", "workspace", "encoded"))
def _start_query_chain(source: torch.Tensor, query: torch.Tensor, absorbed: torch.Tensor, gathered: torch.Tensor,
                       workspace: torch.Tensor, encoded: list[torch.Tensor], handle: int, stream: int) -> None:
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        _CHAINS[handle].run(source, query, absorbed, gathered, workspace, encoded)


@_start_query_chain.register_fake
def _fake_start_query_chain(source, query, absorbed, gathered, workspace, encoded, handle, stream):
    pass


@torch.library.custom_op("flare::join_glm_query_chain", mutates_args=("query", "absorbed", "gathered", "workspace", "encoded"))
def _join_query_chain(source: torch.Tensor, dependency: torch.Tensor, query: torch.Tensor,
                      absorbed: torch.Tensor, gathered: torch.Tensor, workspace: torch.Tensor,
                      encoded: list[torch.Tensor], stream: int) -> None:
    torch.cuda.current_stream(query.device).wait_stream(torch.cuda.ExternalStream(stream, device=query.device))


@_join_query_chain.register_fake
def _fake_join_query_chain(source, dependency, query, absorbed, gathered, workspace, encoded, stream):
    pass


@dataclass(frozen=True)
class PendingQueryChain:
    source: torch.Tensor
    query: torch.Tensor
    absorbed: torch.Tensor
    gathered: torch.Tensor
    workspace: torch.Tensor
    encoded: list[torch.Tensor]


class QueryChain:
    """Project and absorb decode queries while the main stream selects records.

    The chain reads only the normalized low-rank query and writes the query, absorbed query and gathered
    heads, which the main stream's latent writes and record selection never touch. Its projection uses
    a disjoint arena suffix and the absorption BMM none. PCIe head exchange uses a separate channel on
    the auxiliary stream. RoCE gathers run after the join on the main stream because the runtime shares
    one ordered transport channel with the model's other collectives.
    """

    def __init__(self, projection, absorb, source, query,
                 absorbed, gather=None, gathered=None, *, session, workspace, parallel_plans):
        from b12x.gemm.blockscaled._a16 import _weight_parts

        if projection.bias is not None:
            raise ValueError("the query chain requires an unbiased query projection")
        self.projection, self.absorb, self.gather = projection, absorb, gather
        self.gather_on_main = gather is not None and not gather.single and gather.roce
        if isinstance(projection, QuantizedProjection):
            values, scales, gain, fp4 = _weight_parts(projection.weight)
            if gain is not None or fp4:
                raise ValueError("the query chain requires BF16, block-FP8, MXFP8 or MXFP4 weights")
            self.values, self.scales = values, scales.view(torch.uint8)
        self.source, self.query, self.absorbed = source, query, absorbed
        self.gathered = gathered if gathered is not None else absorbed
        self.query_layout = (tuple(query.shape), tuple(query.stride()), query.dtype)
        self.absorbed_layout = (tuple(absorbed.shape), tuple(absorbed.stride()), absorbed.dtype)
        self.gathered_layout = (tuple(self.gathered.shape), tuple(self.gathered.stride()), self.gathered.dtype)
        self.arena, self.parallel_plans = workspace, tuple(parallel_plans)
        for main in self.parallel_plans:
            workspace.register_parallel(main, projection.plan)
        self.stream_handle = workspace.auxiliary_stream.cuda_stream
        self.handle = next(_CHAIN_HANDLES)
        _CHAINS[self.handle] = self
        register_finalizer(session, partial(self.refresh, self.parallel_plans, workspace), workspace=workspace)

    def refresh(self, parallel_plans, arena):
        main_bytes = max(plan.memory_requirements().scratch_nbytes for plan in parallel_plans)
        side_bytes = self.projection.plan.memory_requirements().scratch_nbytes
        _, self.workspace = arena.views(((main_bytes,), torch.uint8), ((side_bytes,), torch.uint8))

    def functional(self, *, session):
        self.projection.functional(session=session)
        self.absorb.functional(session=session)
        if self.gather is not None:
            self.gather.functional(session=session)
        register_finalizer(session, self._finalize_functional, workspace=self.arena)
        return self

    def _finalize_functional(self):
        plans = [self.projection.plan]
        if getattr(self.projection, "quantizer_plan", None) is not None:
            plans.append(self.projection.quantizer_plan)
        self.workspace_nbytes = max(plan.memory_requirements().scratch_nbytes for plan in plans)
        self.encoded_layouts = ()
        if isinstance(self.projection, BlockFP8Projection):
            rows, columns = self.query_layout[0][0], self.projection.weight.in_features
            self.encoded_layouts = (((rows, columns), torch.float8_e4m3fn),
                                   ((rows, columns // 128), torch.float32))
            if self.projection.sm103 and rows <= 32:
                self.encoded_layouts += (((columns // 128, rows), torch.float32),
                                        (tuple(self.projection.weight.scales.T.shape), torch.float32))
            self.workspace_nbytes = _workspace_bytes(self.projection.plan)
        elif getattr(self.projection, "dense", False):
            from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE

            self.workspace_nbytes = DEFAULT_WORKSPACE_SIZE
        self.source = self.query = self.absorbed = self.gathered = None
        self.arena = self.workspace = self.parallel_plans = None

    def run(self, source, query, absorbed, gathered, workspace, encoded):
        from ignition.components.sm103_projection import packed_out_workspace, _dense_out_workspace

        output = query.view(source.shape[0], -1)
        if isinstance(self.projection, BlockFP8Projection):
            from ignition.components.block_fp8 import _project_out, _quantize_out

            _quantize_out(source, encoded[0], encoded[1])
            activation_scales, weight_scales = encoded[1], self.projection.weight.scales
            if len(encoded) == 4:
                encoded[2].copy_(activation_scales.T)
                encoded[3].copy_(weight_scales.T)
                activation_scales, weight_scales = encoded[2:]
            _project_out(encoded[0], activation_scales, self.projection.weight.values, weight_scales,
                         output, workspace, self.projection.plan.handle)
        elif isinstance(self.projection, QuantizedProjection):
            packed_out_workspace(source, self.values, self.scales, None, None, output, workspace, "none",
                                 None, self.projection.plan.handle, "any")
        elif self.projection.dense:
            _dense_out_workspace(source, self.projection.weight, output, workspace, self.projection.plan.handle)
        else:
            from b12x.gemm import bf16_gemv

            bf16_gemv.mm(source, self.projection.weight, out=output, plan=self.projection.plan,
                         bias=self.projection.bias)
        self.absorb(query.transpose(0, 1), output=absorbed.transpose(0, 1))
        if self.gather is not None and not self.gather_on_main:
            self.gather(absorbed.view(absorbed.shape[0], -1), output=gathered.view(gathered.shape[0], -1))

    def start(self, source, query, absorbed, gathered):
        if self.source is None:
            workspace = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=source.device)
        else:
            workspace = self.workspace
        encoded = [torch.empty(shape, dtype=dtype, device=source.device) for shape, dtype in self.encoded_layouts]
        _start_query_chain(source, query, absorbed, gathered, workspace, encoded, self.handle, self.stream_handle)
        return PendingQueryChain(source, query, absorbed, gathered, workspace, encoded)

    def finish(self, pending, dependency):
        source, query, absorbed, gathered, workspace, encoded = (
            pending.source, pending.query, pending.absorbed, pending.gathered, pending.workspace, pending.encoded)
        _join_query_chain(source, dependency, query, absorbed, gathered, workspace, encoded, self.stream_handle)
        if self.gather_on_main:
            self.gather(absorbed.view(absorbed.shape[0], -1), output=gathered.view(gathered.shape[0], -1))
        return gathered if self.gather is not None else absorbed


class PreparedGlmMLA:
    """One rank's sparse attention before the row-parallel output reduction.

    Each rank holds identical low-rank projections, latent records, and C4
    indexing state. Query and value expansion partition the 64 heads. Packed
    KV weights retain BF16 or MXFP8 storage and supply key/value views to native BMMs.
    Physical model pages contain 256 tokens, or four 64-record latent pages.
    """

    def __init__(self, module, weights, source, scratch: MLAScratch, state: MLAState,
                 metadata: IndexMetadata, *, mode, session, name, workspace, max_cache_pages=None, context=None):
        rows, heads = scratch.query.shape[:2]
        if (source.shape != (rows, 4096) or source.dtype != torch.bfloat16
                or source.device != scratch.query.device or scratch.projected_input.shape[1] != 2208
                or scratch.latent_input.shape[1] != 512):
            raise ValueError("GLM MLA requires BF16 [token_capacity, 4096] input on the scratch device")
        if (module.num_heads, module.q_lora_rank, module.kv_lora_rank, module.qk_nope_head_dim,
                module.qk_rope_head_dim, module.v_head_dim) != (64, 1536, 512, 256, 0, 256):
            raise ValueError("canonical MLA geometry differs from GLM 5.3 Flash")
        if module.indexer is None or module.skip_topk or heads not in (16, 32, 64):
            raise ValueError("GLM MLA requires a full indexer and TP1, TP2 or TP4 head partitions")
        # A plain BF16 tensor states its rows in its shape; a packed weight states them in out_features.
        def output_rows(weight):
            return weight.shape[0] if isinstance(weight, torch.Tensor) else weight.out_features

        if (state.cache.shape != (state.index.cache.shape[0] * 4, 64, 528)
                or output_rows(weights["q_b_proj"]) != heads * 256
                or output_rows(weights["kv_b_proj"]) != heads * 512):
            raise ValueError("GLM MLA cache or head partition geometry mismatch")
        self.module, self.scratch, self.state, self.cache, self.metadata = module, scratch, state, state.cache, metadata
        # Decode projects and absorbs queries while records are selected; shared RoCE gathers follow the join.
        self.query_chain = None
        for key in ("q_a_layernorm", "kv_a_layernorm"):
            module.get_submodule(key).weight = torch.nn.Parameter(weights[f"{key}.weight"], requires_grad=False)
        self.projections = {}
        for key, input, output in (
            ("qkv_a_proj", source, scratch.projected_input),
            ("q_b_proj", scratch.query_residual, scratch.query.view(rows, -1)),
            ("o_proj", scratch.expanded.view(rows, -1), scratch.output),
        ):
            self.projections[key] = prepare_projection(input, weights[key], output,
                session=session, name=f"{name}.{key}", workspace=workspace, activation_mode="auto")
        if isinstance(weights["kv_b_proj"], torch.Tensor):
            values = weights["kv_b_proj"].view(heads, 512, 512)
            self.absorb_weights, self.expand_weights = values[:, :256], values[:, 256:]
            batched_projection = BatchedUnquantizedProjection
        else:
            packed = weights["kv_b_proj"].weight
            values, scales = packed.values.view(heads, 512, 512), packed.scale_rows.view(heads, 512, 16)
            self.absorb_weights, self.expand_weights = (values[:, :256], scales[:, :256]), (values[:, 256:], scales[:, 256:])
            batched_projection = BatchedQuantizedProjection
        self.mode = mode
        self.heads, self.device, self.blocks = heads, source.device, metadata.block_table.shape[1]
        self.context_ranks = 1 if context is None else context.shards.world
        self.remote = context is not None and context.remote
        self.sm103_prefill = context is None and mode == "prefill" and sm103.is_sm103(source.device)
        if context is None:
            self.query_absorption = batched_projection(scratch.query.transpose(0, 1),
                self.absorb_weights, scratch.absorbed_query.transpose(0, 1), major="n",
                session=session, name=f"{name}.absorb_query")
            self.value_expansion = batched_projection(scratch.attended.transpose(0, 1),
                self.expand_weights, scratch.expanded.transpose(0, 1), major="k",
                session=session, name=f"{name}.expand_value")
        self.indexer = PreparedGlmIndexer(module.indexer, weights, source, scratch.query_residual,
            scratch.projected_input[:, 2048:2176], scratch.projected_input[:, 2176:2208],
            scratch.index, state.index, metadata, mode=mode, session=session, name=f"{name}.indexer",
            workspace=workspace, max_cache_pages=max_cache_pages, context=context)
        self.writer = PreparedLatentCacheWriter(scratch.latent, state.cache, metadata.token_slots,
            session=session, name=f"{name}.cache_write")
        self.context = context
        ranks = 1 if context is None else context.shards.world
        slices = [(0, rows)] if context is None else context.slices(rows)
        caps = sparse_mla.Caps(device=source.device, num_q_heads=heads * ranks, max_q_rows=slices[0][1],
            max_width=2051, softmax_scale=module.scaling, max_batch=slices[0][1],
            max_kv_rows=max(state.cache.shape[0], 4 * (max_cache_pages or 0)) * 64,
            kv_dtype=torch.uint8, head_dim=512, v_head_dim=512, page_size=64,
            model_type=sparse_mla.ModelType.GLM_NEXT, cache_record_bytes=528,
            mode="decode" if mode == "decode" else "extend",
            **({} if context is None else {"return_lse": True, "lse_scale": "natural"}))
        self.end = None
        if context is None and mode == "prefill" and sm103.is_sm103(source.device):
            # SM103 prefill dequantizes the request's records into a slab and attends to selected positions.
            self.end = torch.zeros(1, dtype=torch.int64, device=source.device)
            self.attention = Sm103PrefillSparseMLA(scratch.absorbed_query, state.cache, scratch.index.physical,
                scratch.index.selected_lengths, metadata.block_table[0], self.end, scratch.attended, caps=caps,
                session=session, name=f"{name}.attention", workspace=workspace)
            return
        if context is None:
            # SM103 decode attends through Triton or Cake; b12x's sparse MLA serves SM120/SM121.
            attend = Sm103SparseMLA if sm103.is_sm103(source.device) else PreparedSparseMLA
            self.attention = attend(scratch.absorbed_query, state.cache, scratch.index.physical,
                scratch.visible, scratch.index.selected_lengths, scratch.attended, caps=caps,
                session=session, name=f"{name}.attention", workspace=workspace)
            if mode != "prefill" and workspace.overlap:
                self.query_chain = QueryChain(self.projections["q_b_proj"], self.query_absorption,
                    scratch.query_residual, scratch.query, scratch.absorbed_query, session=session,
                    workspace=workspace, parallel_plans=self.indexer.scratch_plans)
            return
        if mode == "prefill":
            # Prefill gathers every rank's records of the context into the slab, where a record's index is its
            # global position, and attends with this rank's heads, so neither queries nor outputs cross ranks.
            from .dcp import PREFILL_SLICE_ROWS

            step = min(rows, PREFILL_SLICE_ROWS)
            slab_caps = sparse_mla.Caps(device=source.device, num_q_heads=heads, max_q_rows=step, max_width=2051,
                softmax_scale=module.scaling, max_batch=step, max_kv_rows=scratch.slab.shape[0],
                kv_dtype=torch.uint8, head_dim=512, v_head_dim=512, page_size=64,
                model_type=sparse_mla.ModelType.GLM_NEXT, cache_record_bytes=528, mode="extend")
            if scratch.slab.shape[0] != metadata.block_table.shape[1] * context.shards.page_tokens:
                raise ValueError("GLM prefill slab must hold every position of the prepared context")
            self.windows = ()
            if context.remote:
                # Ranks on separate hosts gather their records window by window instead of reading peers'
                # caches through CUDA IPC; windows past the context's end carry no bytes.
                from ignition.components.collectives import PreparedVarlenGather

                window = scratch.records.shape[1] // 528
                self.windows = tuple((first, PreparedVarlenGather(scratch.records, scratch.record_counts,
                    runtime=context.heads, session=session, coordinator=context.coordinator,
                    name=f"{name}.gather_records{first}"))
                    for first in range(0, scratch.slab.shape[0] // context.shards.world, window))
            self.slices = []
            for start in range(0, rows, step):
                end = min(start + step, rows)
                count = end - start
                absorbed, attended = scratch.absorbed_query[:count], scratch.attended[:count]
                absorb = batched_projection(scratch.query[start:end].transpose(0, 1), self.absorb_weights,
                    absorbed.transpose(0, 1), major="n", session=session, name=f"{name}.absorb_query{start}")
                attention = PreparedSparseMLA(absorbed, scratch.slab.view(-1, 64, 528),
                    scratch.index.physical[start:end], scratch.visible[start:end],
                    scratch.index.selected_lengths[start:end], attended, caps=slab_caps, session=session,
                    name=f"{name}.attention{start}", workspace=workspace)
                expand = batched_projection(attended.transpose(0, 1), self.expand_weights,
                    scratch.expanded[start:end].transpose(0, 1), major="k", session=session,
                    name=f"{name}.expand_value{start}")
                self.slices.append((start, end, absorb, attention, expand))
            return
        # Each slice gathers every rank's query heads, attends to local records, and keeps this rank's heads.
        from ignition.components.collectives import PreparedAllGather, PreparedLseReduceScatter

        self.slices = []
        for start, end in slices:
            count = end - start
            gathered, partial, lse = scratch.gathered[:count], scratch.partial[:count], scratch.lse[:count]
            absorbed, attended = scratch.absorbed_query[:count], scratch.attended[:count]
            absorb = batched_projection(scratch.query[start:end].transpose(0, 1), self.absorb_weights,
                absorbed.transpose(0, 1), major="n", session=session, name=f"{name}.absorb_query{start}")
            expand = batched_projection(attended.transpose(0, 1), self.expand_weights,
                scratch.expanded[start:end].transpose(0, 1), major="k", session=session,
                name=f"{name}.expand_value{start}")
            gather = PreparedAllGather(absorbed.view(count, -1), gathered.view(count, -1),
                runtime=context.heads, session=session, coordinator=context.coordinator,
                name=f"{name}.gather_heads{start}")
            attention = PreparedSparseMLA(gathered, state.cache, scratch.index.physical[start:end],
                scratch.visible[start:end], scratch.index.selected_lengths[start:end], partial, caps=caps,
                session=session, name=f"{name}.attention{start}", workspace=workspace, lse=lse)
            remote = {} if not context.remote else {"rank": context.shards.index,
                "staging": (scratch.exchange_send[:count], scratch.exchange_received[:count])}
            combine = PreparedLseReduceScatter(partial, lse, attended, runtime=context.heads,
                session=session, coordinator=context.coordinator, name=f"{name}.combine{start}", **remote)
            self.slices.append((start, end, absorb, gather, attention, combine, expand))
        if len(self.slices) == 1 and workspace.overlap:
            start, end, absorb, gather = self.slices[0][:4]
            self.query_chain = QueryChain(self.projections["q_b_proj"], absorb, scratch.query_residual,
                scratch.query, scratch.absorbed_query[:end - start], gather, scratch.gathered[:end - start],
                session=session, workspace=workspace, parallel_plans=self.indexer.scratch_plans)

    def attend_gathered_context(self, s, m, query, physical, selected_lengths):
        """Attend one prefill request over its context gathered from every rank, one query slice at a time."""
        from .dcp import pull_context

        end = (m.positions.max() + 1).reshape(1)
        shards = self.context.shards
        if self.windows:
            from .dcp import pack_records, scatter_records, window_counts

            cache = self.state.cache.view(torch.int32).view(-1, 132)
            window = s.records.shape[1] // 528
            for first, gather in self.windows:
                pack_records(cache, m.block_table[0], end, s.records, shards, first)
                s.record_counts.copy_(window_counts(end, shards.world, first, window))
                gather(output=s.records, counts=s.record_counts)
                scatter_records(s.records, end, s.slab.view(torch.int32), shards, first)
        else:
            # The indexer's candidate gather orders this layer's peer cache writes before these reads.
            pull_context(self.state.peers, m.block_table[0], end, s.slab.view(torch.int32), shards)
        # Selected indices are global positions, which index the slab directly.
        s.visible.copy_(m.positions + 1)
        for start, end, absorb, attention, expand in self.slices:
            absorbed = absorb(query[start:end].transpose(0, 1)).transpose(0, 1)
            attended = attention(absorbed, physical[start:end], s.visible[start:end],
                selected_lengths[start:end], cache=s.slab.view(-1, 64, 528))
            expanded = expand(attended.transpose(0, 1)).transpose(0, 1)
            s.expanded[start:end].copy_(expanded)

    def rebind_pages(self):
        """Write and attend through the state's current page pools after they were replaced."""
        self.cache = self.state.cache
        self.writer.cache = self.state.cache
        self.indexer.rebind_pages()
        if self.context is None:
            self.attention.rebind_cache(self.state.cache)
        elif self.mode != "prefill":
            for slice in self.slices:
                slice[4].rebind_cache(self.state.cache)

    def functional(self, *, session):
        if self.query_chain is not None:
            self.query_chain.functional(session=session)
        for projection in self.projections.values():
            projection.functional(session=session)
        if hasattr(self, "query_absorption"):
            self.query_absorption.functional(session=session)
            self.value_expansion.functional(session=session)
        self.indexer.functional(session=session)
        self.writer.functional(session=session)
        if hasattr(self, "attention"):
            self.attention.functional(session=session)
        for item in getattr(self, "slices", ()):
            for operator in item[2:]:
                if hasattr(operator, "functional"):
                    operator.functional(session=session)
        for _, gather in getattr(self, "windows", ()):
            gather.functional(session=session)
        self.scratch = self.metadata = self.end = None
        return self

    def __call__(self, source, metadata):
        output, _ = self.forward(source, metadata)
        return output

    def forward(self, source, metadata):
        s = MLAScratch.allocate(source.shape[0], self.heads, self.blocks, self.mode, self.device,
            context_ranks=self.context_ranks, remote=self.remote)
        p = self.projections
        projected = p["qkv_a_proj"](source)
        s.projected_input.copy_(projected)
        rms_norm(s.query_input, self.module.q_a_layernorm.weight, s.query_residual,
            self.module.q_a_layernorm.variance_epsilon, round_before_weight=True)
        chain = self.query_chain
        chain_pending = None
        if chain is not None:
            chain_pending = chain.start(s.query_residual, s.query, s.absorbed_query, s.gathered)
            query = s.query
        else:
            query = p["q_b_proj"](s.query_residual).view(source.shape[0], self.heads, 256)
            if self.context is None:
                absorbed_query = self.query_absorption(query.transpose(0, 1)).transpose(0, 1)
                s.absorbed_query.copy_(absorbed_query)
        rms_norm(s.latent_input, self.module.kv_a_layernorm.weight, s.latent,
            self.module.kv_a_layernorm.variance_epsilon, round_before_weight=True)
        self.writer(s.latent, metadata.token_slots)
        selection, index_scratch = self.indexer.forward(source, s.query_residual,
            s.projected_input[:, 2048:2176], s.projected_input[:, 2176:2208], metadata, s.index)
        physical, selected_lengths, page_table = selection
        s = replace(s, index=index_scratch, query=query)
        chain_result = chain.finish(chain_pending, physical) if chain is not None else None
        if self.context is None:
            s.visible.copy_(metadata.positions + 1)
            if self.sm103_prefill:
                end = metadata.positions.max().reshape(1) + 1
                attended = self.attention(chain_result if chain_result is not None else s.absorbed_query,
                    metadata.positions, selected_lengths, page_table[0], end, cache=self.cache)
            else:
                attended = self.attention(chain_result if chain_result is not None else s.absorbed_query,
                    physical, s.visible, selected_lengths, cache=self.cache)
            expanded = self.value_expansion(attended.transpose(0, 1)).transpose(0, 1)
            s.expanded.copy_(expanded)
        elif self.mode == "prefill":
            self.attend_gathered_context(s, metadata, query, physical, selected_lengths)
        else:
            s.visible.copy_(self.context.shards.local_tokens(metadata.positions + 1))
            for start, end, absorb, gather, attention, combine, expand in self.slices:
                count = end - start
                absorbed = s.absorbed_query[start:end] if chain is not None else absorb(query[start:end].transpose(0, 1)).transpose(0, 1)
                gathered = (chain_result[start:end] if chain is not None else
                            gather(absorbed.view(count, -1)).view(count, self.heads * self.context_ranks, 512))
                # The head exchange uses its own channel, so it reads the partial output and LSE in place.
                partial, lse = attention(gathered, physical[start:end], s.visible[start:end],
                    selected_lengths[start:end], cache=self.cache)
                attended = combine(partial, lse)
                expanded = expand(attended.transpose(0, 1)).transpose(0, 1)
                s.expanded[start:end].copy_(expanded)
        output = p["o_proj"](s.expanded.view(source.shape[0], -1))
        return output, replace(s, output=output)
