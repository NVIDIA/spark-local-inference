# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Windowed MLA, source-shared compression, and two-level sparse selection."""

from dataclasses import dataclass

import torch
from b12x.quantization import mla_norm_mxfp8, mxfp8
import triton
import triton.language as tl
from b12x.attention import mla_compress
from b12x.preparation import PreparedCall

from ignition.components import sm103_projection
from ignition.components.collectives import PreparedGatherShards
from ignition.components.compressed_mla import PreparedCompressedCacheWriter, PreparedCompressedMLA
from ignition.components.attention_output import PreparedGroupedOutputProjection
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.mla_norm_quant import PreparedMLANormQuant
from ignition.components.mxfp4_indexer import PreparedMXFP4Indexer
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.state import gather_rows, gather_rows_three, save_hidden
from ignition.components.window_indices import PreparedWindowIndices, WindowIndices
from ignition.preparation.tuning import declare, register_finalizer
from ignition.preparation.scratch import ScratchArena

from ignition.components.rotary import CompressedRotary as Rotary
from ignition.components.parallel import TensorParallelShard

from .pool import PAGE_TOKENS


# Bound indexer workspace while amortizing prefill selection launches.
INDEXER_QUERY_ROWS = 128


@dataclass(frozen=True)
class CompressionGeometry:
    slots: torch.Tensor
    lengths: torch.Tensor
    index_lengths: torch.Tensor
    active_width: torch.Tensor
    position: torch.Tensor


@dataclass(frozen=True)
class AttentionGeometry(WindowIndices):
    position: torch.Tensor
    ratios: dict

    def compressed(self, ratio):
        return self.ratios[ratio]


class AttentionIndices(PreparedWindowIndices):
    """Window and compression addresses shared by layers with identical metadata."""

    def __init__(self, metadata, *, mode, tokens, page_size, width, ratios=(), visible=None,
                 replay_start=None):
        super().__init__(metadata, mode=mode, tokens=tokens, page_size=page_size, width=width,
                         visible=visible, replay_start=replay_start)
        self.ratio_ids = tuple(sorted(set(ratios)))
        if any(type(ratio) is not int or ratio not in (1, 2) for ratio in self.ratio_ids):
            raise ValueError("DeepSeek V4.1 compression ratios must be one or two")
        self.position = torch.zeros_like(metadata.positions)
        self.ratios = {ratio: CompressionGeometry(
            slots=torch.full_like(metadata.positions, -1),
            lengths=torch.zeros(tokens, dtype=torch.int32, device=metadata.positions.device),
            index_lengths=torch.zeros(tokens, dtype=torch.int32, device=metadata.positions.device),
            active_width=torch.zeros(1, dtype=torch.int32, device=metadata.positions.device),
            position=torch.zeros_like(metadata.positions)) for ratio in self.ratio_ids}

    def compressed(self, ratio):
        if ratio not in self.ratios:
            raise ValueError("compression geometry must be declared before preparing attention")
        return self.ratios[ratio]

    def functional(self):
        super().functional()
        self.position = self.ratios = None
        return self

    def __call__(self, metadata, visible=None, replay_start=None):
        window = super().__call__(metadata, visible=visible, replay_start=replay_start)
        m = metadata
        live, position = m.positions >= 0, m.positions.clamp_min(0)
        ratios = {}
        if self.ratio_ids:
            pages = m.page_table[m.requests, (position // PAGE_TOKENS).clamp(max=m.page_table.shape[1] - 1)]
        for ratio in self.ratio_ids:
            lengths = torch.where(live, (position + 1) // ratio, 0).to(torch.int32)
            ratios[ratio] = CompressionGeometry(lengths=lengths, index_lengths=lengths.clamp(max=512),
                active_width=lengths.amax().view(1),
                slots=torch.where(live, pages.long() * (PAGE_TOKENS // ratio) + (position % PAGE_TOKENS) // ratio, -1),
                position=(position + 1) // ratio * ratio - ratio)
        return AttentionGeometry(window.window_slots, window.window_indices, window.window_lengths,
                                 window.window_page_size, position, ratios)


def normalize(source, weight, out, eps):
    value = source.float()
    out.copy_((value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)) * weight.float())


@dataclass(frozen=True)
class AttentionScratch:
    projected: torch.Tensor
    query_low: torch.Tensor
    query: torch.Tensor
    window_values: torch.Tensor
    attended: torch.Tensor
    output: torch.Tensor
    selected: torch.Tensor
    candidates: torch.Tensor
    candidate_counts: torch.Tensor
    index_query: torch.Tensor
    index_weights: torch.Tensor
    index_keys: torch.Tensor
    normalized_keys: torch.Tensor

    @classmethod
    def allocate(cls, tokens, world, device, *, candidate_tokens=None, shard_rows=None):
        """Attention buffers for full rows, or for one rank's shard of a row-sharded layer.

        A shard keeps every head, so its query side is shard-sized and holds all 64 heads; the two
        index-key planes stay full-row because the compressor still runs over every row, and the
        gathered KV latent then lands in the caller's full-size window buffer.
        """
        candidate_tokens = tokens if candidate_tokens is None else candidate_tokens
        if type(candidate_tokens) is not int or not 0 < candidate_tokens <= tokens:
            raise ValueError("candidate row capacity must fit the attention token capacity")
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        first, last = TensorParallelShard(0, world).padded_partition(8)
        heads = (last - first) * 8
        if shard_rows is not None:
            # Every rank's shard covers the whole head axis, which is the eight head blocks of the
            # checkpoint layout rather than this rank's partition of them.
            rows, candidates, heads = shard_rows, min(candidate_tokens, shard_rows), 8 * (last - first) * world
        else:
            rows, candidates = tokens, candidate_tokens
        return cls(zeros((rows, 1792)),
            zeros((rows, 1280)), zeros((rows, heads, 512)),
            zeros((rows, 512)), zeros((rows, heads, 512)),
            zeros((rows, 5120)), zeros((rows, 512), torch.int32).fill_(-1),
            zeros((candidates, 16384), torch.int32).fill_(-1), zeros(candidates, torch.int32),
            zeros((rows, 32, 128)), zeros((rows, 32)), zeros((tokens, 128)), zeros((tokens, 128)))

    def rows(self, count, tokens):
        """This scratch narrowed to `count` query rows, keeping index keys at `tokens` rows."""
        if not 0 < count <= self.projected.shape[0]:
            raise ValueError("attention scratch rows must fit the allocated query rows")
        if not 0 < tokens <= self.index_keys.shape[0]:
            raise ValueError("attention scratch tokens must fit the allocated index keys")
        return type(self)(self.projected[:count], self.query_low[:count], self.query[:count],
            self.window_values[:count], self.attended[:count], self.output[:count], self.selected[:count],
            self.candidates[:count], self.candidate_counts[:count], self.index_query[:count],
            self.index_weights[:count], self.index_keys[:tokens], self.normalized_keys[:tokens])


class PreparedCompressor:
    def __init__(self, layer, weights, source, metadata, pool, slots, *, session, name, deferred=False,
                 row_shard=None, runtime=None, coordinator=None):
        self.pool, self.metadata, self.layer = pool, metadata, layer
        self.deferred = deferred
        # The kernel call reads the norm directly; `self.arguments` is the declaration the preparation
        # binds, and a forward does not read it.
        self.weight = weights["norm"]
        ratio = pool.config.text["compress_ratios"][layer]
        self.ratio = ratio
        # The kernel still compresses every row, so its operands stay full-row; only the two input
        # projections run on the shard, and a gather returns their products to all rows.
        tokens, rows = source.shape[0], metadata.starts.numel()
        total = tokens if row_shard is None else metadata.positions.numel()
        self.total = total
        device = source.device
        dtype = torch.float32 if ratio == 2 else torch.bfloat16
        def zeros(shape, dtype):
            return torch.zeros(shape, device=device, dtype=dtype)
        self.values = zeros((total, 512), dtype)
        self.local_values = self.values if row_shard is None else zeros((tokens, 512), dtype)
        self.values_gather = None if row_shard is None else PreparedGatherShards(
            self.local_values.view(torch.bfloat16), self.values.view(torch.bfloat16), runtime=runtime,
            session=session, coordinator=coordinator, name=f"{name}.values_gather")
        self.project_values = UnquantizedProjection(source, weights["values"], self.local_values,
            session=session, name=f"{name}.values").functional(session=session)
        self.output = zeros((total, 512), torch.bfloat16)
        self.emitted = zeros(total, torch.bool)
        self.slots = zeros(total, torch.int64).fill_(-1)
        self.state_ids = torch.arange(rows, dtype=torch.int64, device=device)
        self.arguments = dict(values=self.values, weight=weights["norm"], query_start_loc=metadata.query_start_loc,
            positions=metadata.starts, state_ids=self.state_ids, destination_slots=slots,
            live_counts=metadata.live_counts, out=self.output, emitted=self.emitted, emitted_slots=self.slots)
        if ratio == 2:
            self.gates = zeros(self.values.shape, self.values.dtype)
            self.local_gates = self.gates if row_shard is None else zeros(self.local_values.shape, dtype)
            self.gates_gather = None if row_shard is None else PreparedGatherShards(
                self.local_gates.view(torch.bfloat16), self.gates.view(torch.bfloat16), runtime=runtime,
                session=session, coordinator=coordinator, name=f"{name}.gates_gather")
            self.project_gates = UnquantizedProjection(source, weights["gates"], self.local_gates,
                session=session, name=f"{name}.gates").functional(session=session)
            self.pending = {key: zeros((rows, 512) if key != "pending_position" else (rows,),
                torch.float32 if key != "pending_position" else torch.int64)
                for key in ("pending_values", "pending_gates", "pending_position")}
            self.arguments.update(gates=self.gates, **self.pending)
        self.plan = mla_compress.plan(mla_compress.Caps(device=device, max_tokens=total,
            max_requests=rows, max_states=rows, ratio=ratio))

        declaration = self.arguments

        def prepare(state):
            arguments = {key: value if key in ("weight", "state_ids") else torch.zeros_like(value)
                         for key, value in declaration.items()}
            arguments["query_start_loc"].copy_(torch.arange(rows + 1, device=device) * total // rows)
            arguments["live_counts"].copy_(torch.tensor([total, rows], dtype=torch.int32, device=device))
            arguments["destination_slots"].copy_(torch.arange(total, device=device))
            if ratio == 2:
                arguments["pending_position"].fill_(-1)
            binding = state.bind(**arguments)
            return PreparedCall(run=lambda: state.run(binding))

        declare(session, (self.plan.request(name=name, prepare_call=prepare),))
    def functional(self, *, session):
        register_finalizer(session, self.release_examples)
        if self.values_gather is not None:
            self.values_gather.functional(session=session)
        if self.ratio == 2 and self.gates_gather is not None:
            self.gates_gather.functional(session=session)
        return self

    def release_examples(self):
        self.values = self.output = self.emitted = self.slots = self.state_ids = self.arguments = None
        self.local_values = None
        if self.ratio == 2:
            self.gates = self.pending = self.local_gates = None

    def __call__(self, source, metadata, slots):
        from b12x.attention.mla_compress._impl import _run_ratio1, _run_ratio2

        m = metadata
        total = self.total
        output = torch.zeros((total, 512), dtype=torch.bfloat16, device=source.device)
        emitted = torch.zeros(total, dtype=torch.bool, device=source.device)
        emitted_slots = torch.full((total,), -1, dtype=torch.int64, device=source.device)
        state_ids = torch.arange(m.starts.numel(), dtype=torch.int64, device=source.device)
        gates = None
        if self.ratio == 2:
            pending = {key: torch.zeros((m.starts.numel(), 512) if key != "pending_position" else (m.starts.numel(),),
                dtype=torch.float32 if key != "pending_position" else torch.int64, device=source.device)
                for key in ("pending_values", "pending_gates", "pending_position")}
            gather_rows_three(tuple(self.pool.region(f"layer{self.layer}.{key}") for key in pending),
                              m.state_rows, tuple(pending.values()))
            gates = self.project_gates(source)
            if self.gates_gather is not None:
                gates = self.gates_gather(gates.view(torch.bfloat16)).view(torch.float32)
            values = self.project_values(source)
            if self.values_gather is not None:
                values = self.values_gather(values.view(torch.bfloat16)).view(torch.float32)
            _run_ratio2(self.plan.handle, values, gates, self.weight, m.query_start_loc, m.starts,
                state_ids, slots, m.live_counts, *pending.values(), output, emitted, emitted_slots)
            if not self.deferred:
                for key, value in pending.items():
                    save_hidden(value.view(value.shape[0], -1), self.pool.region(f"layer{self.layer}.{key}"),
                                m.state_rows, m.lengths > 0)
        else:
            values = self.project_values(source)
            if self.values_gather is not None:
                values = self.values_gather(values.view(torch.bfloat16))
            _run_ratio1(self.plan.handle, values, self.weight, m.query_start_loc, m.starts,
                       state_ids, slots, m.live_counts, output, emitted, emitted_slots)
        return output, emitted_slots, (values, gates)

    def commit(self, counts, records, metadata):
        if self.ratio != 2:
            return
        m = metadata
        values, gates = records
        last = (m.query_start_loc[:-1] + counts - 1).long().clamp(0, self.total - 1)
        position = m.starts + counts - 1
        pending = (counts > 0) & (position % 2 == 0)
        for key, source in (("pending_values", values), ("pending_gates", gates)):
            save_hidden(source[last], self.pool.region(f"layer{self.layer}.{key}"), m.state_rows, pending)
        tags = torch.where(position % 2 == 0, position, -1).view(-1, 1)
        save_hidden(tags, self.pool.region(f"layer{self.layer}.pending_position"), m.state_rows, counts > 0)


class PreparedGlobalKV:
    """Project and store every source row before decoder queries are packed.

    A row-sharded layer computes the two compressor projections on its own rows and gathers them, so
    the compression kernel, the index keys and the page writes stay full-row and hold every row.
    """

    def __init__(self, layer, weights, source, scratch, metadata, pool, rotary, *,
                 session, name, indices, deferred=False, row_shard=None, runtime=None, coordinator=None):
        self.rotary = rotary
        self.ratio = pool.config.text["compress_ratios"][layer]
        total = source.shape[0] if row_shard is None else metadata.positions.numel()
        if indices.metadata is not metadata or indices.position.shape != (total,):
            raise ValueError("global KV indices must describe every source row")
        slots = indices.compressed(self.ratio).slots
        self.compressor = PreparedCompressor(layer, weights["compressor"], source, metadata,
            pool, slots, session=session, name=f"{name}.compress", deferred=deferred, row_shard=row_shard,
            runtime=runtime, coordinator=coordinator)
        self.index_key = UnquantizedProjection(self.compressor.output, weights["index_k"], scratch.index_keys,
            session=session, name=f"{name}.index_key").functional(session=session)
        self.index_norm = PreparedRMSNorm(scratch.index_keys, weights["index_norm"], scratch.normalized_keys,
            eps=pool.config.text["rms_norm_eps"], session=session, name=name + ".index_norm").functional()
        self.writer = PreparedCompressedCacheWriter(self.compressor.output,
            pool.pages[f"layer{layer}.latent_page"], self.compressor.slots,
            page_size=PAGE_TOKENS // self.ratio, kind="indexed", session=session,
            name=f"{name}.compressed_write").functional(session=session)
        self.compressor.functional(session=session)

    def __call__(self, source, indexer, metadata, geometry):
        latent, slots, records = self.compressor(source, metadata, geometry.slots)
        keys = self.index_norm(self.index_key(latent))
        self.rotary.apply(keys, geometry.position, compressed=True)
        indexer.write_keys(keys, slots)
        self.rotary.apply(latent, geometry.position, compressed=True)
        self.writer(latent, slots)
        return records


@triton.jit
def _commit_window(Staged, Window, State, Starts, Lengths, STRIDE: tl.constexpr,
                   STAGED_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    item = tl.program_id(0)
    request = tl.program_id(1)
    length, start, row = tl.load(Lengths + request), tl.load(Starts + request), tl.load(State + request)
    token = tl.maximum(0, length - 128) + item
    offset = tl.arange(0, BLOCK)
    valid = (token < length) & (row > 0) & (offset < 528)
    value = tl.load(Staged + request * STAGED_STRIDE + (128 + token) * 528 + offset, valid, other=0)
    tl.store(Window + row * STRIDE + ((start + token) % 128) * 528 + offset, value, valid)


class PreparedAttention:
    def __init__(self, layer, weights, source, scratch, metadata, pool, rotary, *, mode,
                 world, session, name, workspace, replay_start=None, window_staging=None,
                 mla_compute_mode="auto", window_metadata=None, examples=None, row_shard=None,
                 window_target=None, runtime=None, coordinator=None):
        self.layer, self.weights, self.scratch, self.metadata = layer, weights, scratch, metadata
        self.pool, self.rotary, self.mode = pool, rotary, mode
        self.replay_start = replay_start
        config = pool.config
        draft = mode == "draft"
        self.ratio = 0 if draft else config.text["compress_ratios"][layer]
        self.source = None if draft else config.source(layer)
        self.eps = config.text["rms_norm_eps"]
        # `source` holds this rank's query rows: every row, or one shard when attention is sharded.
        tokens, device = source.shape[0], source.device
        total = tokens if row_shard is None else metadata.positions.numel()
        self.rows_index = None if row_shard is None else row_shard.rows_index
        self.total = total
        rows = metadata.starts.numel()
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)
        self.window_page_size = 128 if mode == "decode" else 128 + metadata.max_query_tokens
        self.window = pool.window(layer, draft=draft)
        if window_staging is not None:
            if (mode != "verify" or window_staging.shape != (rows, self.window_page_size * 528)
                    or window_staging.dtype != torch.uint8 or window_staging.device != device
                    or not window_staging.is_contiguous()):
                raise ValueError("verification window staging must match the contiguous cache geometry")
            self.window_cache = window_staging
        else:
            self.window_cache = self.window if mode == "decode" else zeros(
                (rows, self.window_page_size * 528), torch.uint8)
        window_width = self.window_page_size if draft else 128
        self.owns_window_metadata = window_metadata is None
        if window_metadata is None:
            window_metadata = AttentionIndices(metadata, mode=mode, tokens=tokens,
                page_size=self.window_page_size, width=window_width, replay_start=replay_start,
                ratios=(self.ratio,) if self.ratio else ())
        elif (window_metadata.metadata is not metadata or window_metadata.mode != mode
                or window_metadata.window_page_size != self.window_page_size
                or window_metadata.window_indices.shape != (total, window_width)
                or window_metadata.replay_start is not replay_start or window_metadata.visible is not None
                or not isinstance(window_metadata, AttentionIndices)):
            raise ValueError("shared window indices must match the attention metadata and cache geometry")
        self.window_metadata = window_metadata
        # Slots and the cache itself span every row; everything a query row reads is this rank's slice.
        self.window_slots = window_metadata.window_slots
        def rank_rows(full):
            return full if row_shard is None else full[:tokens]
        self.window_indices = rank_rows(window_metadata.window_indices)
        self.window_lengths = rank_rows(window_metadata.window_lengths)
        self.token_offsets = torch.arange(tokens, device=device)
        geometry = window_metadata.compressed(self.ratio) if self.ratio else None
        self.lengths = zeros(tokens, torch.int32) if geometry is None else rank_rows(geometry.lengths)
        self.active_width = zeros(1, torch.int32) if geometry is None else geometry.active_width
        self.index_lengths = zeros(tokens, torch.int32) if geometry is None else rank_rows(geometry.index_lengths)
        self.page_table = rank_rows(metadata.token_page_table)
        self.projections = {}
        # A shard owns every attention head and every output group, so it reads the replicated
        # projections while a full-row layer keeps the tensor-parallel partition.
        sharded = row_shard is not None
        if sharded and not {"q_b_full", "output_full", "sink_full"} <= set(weights):
            raise ValueError("row-sharded attention requires replicated attention weights")
        q_b_key, output_key, sink_key = (("q_b_full", "output_full", "sink_full") if sharded
                                         else ("q_b", "output", "sink"))
        use_b12x_mxfp8 = not sm103_projection.is_sm103(device)
        paired_mxfp8 = use_b12x_mxfp8 and all(isinstance(getattr(weights[key], "weight", None), mxfp8.MXFP8Rows)
                           for key in ("qkv_a", q_b_key))
        self.norm_quant = (PreparedMLANormQuant(scratch.projected,
            (weights["q_norm"], weights["kv_norm"]), scratch.query_low, scratch.window_values,
            eps=self.eps, session=session, name=name + ".qkv_norm",
            release_dependents=tokens <= 8, examples=examples)
            if paired_mxfp8 and self.eps == 1e-20 and mla_norm_mxfp8.is_supported(device) else None)
        for key, name_key, values, out in (
            ("qkv_a", "qkv_a", source, scratch.projected),
            ("q_b", q_b_key, scratch.query_low, scratch.query.view(tokens, -1)),
        ):
            packed_mxfp8 = use_b12x_mxfp8 and isinstance(
                getattr(weights[name_key], "weight", None), mxfp8.MXFP8Rows)
            self.projections[key] = QuantizedProjection(values, weights[name_key], out,
                session=session, name=f"{name}.{key}", workspace=workspace, activation_mode="auto",
                mxfp8_min_amax=(1e-4 if packed_mxfp8 and (key != "q_b" or self.norm_quant is None)
                                else None),
                encoded=self.norm_quant.encoded if key == "q_b" and self.norm_quant is not None else None,
                rhs_prefetch_ready=packed_mxfp8,
                allow_bf16_atomic_reduction=not packed_mxfp8,
                examples=examples, example_role="attention." + key)
        self.query_norm = (PreparedRMSNorm(scratch.projected[:, :1280], weights["q_norm"], scratch.query_low,
            eps=self.eps, session=session, name=name + ".q_norm") if self.norm_quant is None else None)
        self.window_norm = (PreparedRMSNorm(scratch.projected[:, 1280:], weights["kv_norm"], scratch.window_values,
            eps=self.eps, session=session, name=name + ".kv_norm") if self.norm_quant is None else None)
        # The output projection sums all of this rank's groups in one kernel, so a shard needs no
        # cross-rank reduction of the attention output.
        self.output_projection = PreparedGroupedOutputProjection(scratch.attended.view(tokens, -1, 4096),
            weights[output_key], scratch.output, session=session, name=name + ".output", workspace=workspace)
        # Every rank's caches must hold every row, so a shard gathers the KV latent it computed with
        # this rank's global slots before the write, leaving the cache bytes full-row.
        window_values = scratch.window_values if row_shard is None else window_target
        if row_shard is None:
            if window_target is not None:
                raise ValueError("a full-row attention writes its own KV latent")
            self.window_gather = None
        else:
            if window_target is None or window_target.shape != (total, 512):
                raise ValueError("row-sharded attention gathers its KV latent into a full-row buffer")
            self.window_gather = PreparedGatherShards(scratch.window_values, window_target,
                runtime=runtime, session=session, coordinator=coordinator, name=name + ".window_gather")
        self.window_writer = PreparedCompressedCacheWriter(window_values, self.window_cache,
            self.window_slots, page_size=self.window_page_size, kind="swa", session=session,
            name=f"{name}.window_write",
            history=self.window if mode in ("verify", "draft") else None,
            history_rows=metadata.state_rows if mode in ("verify", "draft") else None,
            snapshot_tokens=128 if mode in ("verify", "draft") else 0)
        self.global_kv = None
        self.selectors = []
        if "compressor" in weights:
            self.global_kv = PreparedGlobalKV(layer, weights, source, scratch, metadata, pool, rotary,
                session=session, name=name, indices=window_metadata, deferred=mode == "verify",
                row_shard=row_shard, runtime=runtime, coordinator=coordinator)
        if "index_q" in weights:
            self.index_query = QuantizedProjection(scratch.query_low, weights["index_q"],
                scratch.index_query.view(tokens, -1), session=session, name=f"{name}.index_query",
                workspace=workspace, activation_mode="auto",
                examples=examples, example_role="attention.index_query")
            self.index_weight = UnquantizedProjection(source, weights["index_weights"], scratch.index_weights,
                session=session, name=f"{name}.index_weight")
            candidate_source = config.text["candidate_source_layer_id"]
            if layer >= candidate_source and scratch.candidates.shape[0] < tokens:
                raise ValueError("hierarchical selection exceeds candidate row capacity")
            for first in range(0, tokens, INDEXER_QUERY_ROWS):
                last = min(tokens, first + INDEXER_QUERY_ROWS)
                candidates = {}
                if layer == candidate_source:
                    candidates = dict(candidates_out=scratch.candidates[first:last], counts_out=scratch.candidate_counts[first:last])
                elif layer > candidate_source:
                    candidates = dict(candidates=scratch.candidates[first:last], counts=scratch.candidate_counts[first:last])
                selector = PreparedMXFP4Indexer(scratch.index_query[first:last], scratch.index_weights[first:last],
                    pool.pages[f"layer{self.source.layer}.index_page"],
                    self.page_table[first:last],
                    self.lengths[first:last], self.active_width, scratch.selected[first:last],
                    page_size=128 // self.ratio, session=session, name=f"{name}.index{first}", workspace=workspace,
                    mode="prefill" if mode == "prefill" else "decode", **candidates)
                self.selectors.append(selector)
        indexed = {} if not self.ratio else dict(indexed=pool.pages[f"layer{self.source.layer}.latent_page"],
            indexed_indices=scratch.selected, indexed_lengths=self.index_lengths,
            page_table=self.page_table,
            indexed_page_size=128 // self.ratio)
        self.attention = PreparedCompressedMLA(scratch.query, self.window_cache, self.window_indices,
            self.window_lengths, weights[sink_key], scratch.attended,
            window_page_size=self.window_page_size,
            # Speculative rows use explicit indices into staged caches. Their
            # prepared row bound belongs to the decode contract, including
            # multi-request verification and noncausal draft blocks.
            mode="decode" if mode in ("decode", "verify", "draft") else "extend",
            decode_row_capacity=tokens if mode in ("decode", "verify", "draft") else None,
            session=session, name=f"{name}.attention",
            workspace=workspace, compute_mode=mla_compute_mode, **indexed)
        self.concurrent_query = None
        if (mode in ("decode", "verify", "draft") and not (mode == "decode" and tokens == 1)
                and isinstance(workspace, ScratchArena) and workspace.overlap):
            parallel_plans = [self.window_writer.plan]
            if self.window_norm is not None:
                parallel_plans.append(self.window_norm.plan)
            if self.global_kv is not None:
                global_kv = self.global_kv
                compressor = global_kv.compressor
                parallel_plans.extend((compressor.project_values.plan, compressor.plan,
                    global_kv.index_key.plan, global_kv.index_norm.plan, global_kv.writer.plan))
                if hasattr(compressor, "project_gates"):
                    parallel_plans.append(compressor.project_gates.plan)
            if self.selectors:
                parallel_plans.extend((self.index_query.plan, self.index_weight.plan))
                parallel_plans.extend(selector.plan for selector in self.selectors)
            self.concurrent_query = ConcurrentProjectionChain((self.projections["q_b"],),
                parallel_plans=parallel_plans, session=session, workspace=workspace)

        self.query_shape = tuple(scratch.query.shape)
        self.candidate_source = config.text["candidate_source_layer_id"]
        self.projections["qkv_a"].functional(session=session)
        if self.concurrent_query is None:
            self.projections["q_b"].functional(session=session)
        else:
            self.concurrent_query.functional(session=session)
        if self.norm_quant is not None:
            self.norm_quant.functional(session=session)
        else:
            self.query_norm.functional()
            self.window_norm.functional()
        self.output_projection.functional(session=session)
        if self.window_gather is not None:
            self.window_gather.functional(session=session)
        self.window_writer.functional(session=session, cache=None)
        self.attention.functional(session=session, window=None)
        if self.selectors:
            self.index_query.functional(session=session)
            self.index_weight.functional(session=session)
            for selector in self.selectors:
                selector.functional(session=session)
        if self.owns_window_metadata:
            self.window_metadata.functional()
        register_finalizer(session, self.release_examples)

    def release_examples(self):
        self.scratch = self.metadata = self.replay_start = self.window_cache = None
        self.window_slots = self.window_indices = self.window_lengths = self.token_offsets = None
        self.lengths = self.active_width = self.index_lengths = self.page_table = None

    def __call__(self, source, metadata, selection, *, indices=None, replay_start=None, window_cache=None):
        output, record, selection, _ = self._run(source, metadata, selection, indices=indices,
                                               replay_start=replay_start, window_cache=window_cache)
        return output, record, selection

    def _run(self, source, metadata, selection, *, indices=None, replay_start=None, window_cache=None):
        m = metadata
        indices = (self.window_metadata(m, replay_start=replay_start)
                   if self.owns_window_metadata else indices)
        # A shard computes the geometry for every row, because its caches hold every row, and reads
        # this rank's slice of the row-indexed operands through the row numbers the graph has.
        rows_index = self.rows_index
        def rank_rows(full):
            return full if rows_index is None else full.index_select(0, rows_index)
        position = rank_rows(indices.position)
        window_indices = rank_rows(indices.window_indices)
        window_lengths = rank_rows(indices.window_lengths)
        page_table = rank_rows(m.token_page_table)
        if window_cache is None:
            window_cache = self.window if self.mode == "decode" else torch.zeros(
                (m.starts.numel(), self.window_page_size * 528), dtype=torch.uint8, device=source.device)
        if self.mode == "prefill":
            gather_rows(self.window, m.state_rows, window_cache[:, :128 * 528])
        qkv = self.projections["qkv_a"]
        qkv_encoded_bases = (None if qkv.encoded_layout is None else tuple(
            torch.zeros(shape, dtype=dtype, device=source.device) for shape, dtype in qkv.encoded_layout))
        projected = qkv(source, encoded_bases=qkv_encoded_bases)
        encoded_bases = None
        if self.norm_quant is None:
            query_low = self.query_norm(projected[:, :1280])
        else:
            query_low, window_values, encoded_bases = self.norm_quant(projected)
        pending = None
        if self.concurrent_query is not None:
            pending = self.concurrent_query.start(query_low, encoded_bases=encoded_bases)
        if self.window_norm is not None:
            window_values = self.window_norm(projected[:, 1280:])
        if self.concurrent_query is None:
            query = self.projections["q_b"](query_low, encoded_bases=encoded_bases).view(self.query_shape)
            self.rotary.apply(query, position, compressed=bool(self.ratio))
        self.rotary.apply(window_values, position, compressed=bool(self.ratio))
        if self.window_gather is not None:
            window_values = self.window_gather(window_values)
        self.window_writer(window_values, indices.window_slots,
                           history_rows=m.state_rows if self.mode in ("verify", "draft") else None, cache=window_cache)
        dependency = window_cache
        selected, candidates, candidate_counts = selection
        records = None
        if self.ratio:
            geometry = indices.compressed(self.ratio)
            lengths = rank_rows(geometry.lengths)
            index_lengths = rank_rows(geometry.index_lengths)
            if self.global_kv is not None:
                records = self.global_kv(source, self.selectors[0], m, geometry)
            if self.selectors:
                index_query = self.index_query(query_low).view(source.shape[0], 32, 128)
                self.rotary.apply(index_query, position, compressed=True)
                index_weights = self.index_weight(source)
                index_weights.mul_(128 ** -.5 * 32 ** -.5)
                outputs, candidate_outputs, count_outputs = [], [], []
                for ordinal, selector in enumerate(self.selectors):
                    first = ordinal * INDEXER_QUERY_ROWS
                    last = min(source.shape[0], first + INDEXER_QUERY_ROWS)
                    result = selector(index_query[first:last], index_weights[first:last],
                        page_table[first:last], lengths[first:last], geometry.active_width,
                        candidates=None if self.layer <= self.candidate_source else candidates[first:last],
                        counts=None if self.layer <= self.candidate_source else candidate_counts[first:last])
                    if self.layer == self.candidate_source:
                        result, next_candidates, next_counts = result
                        candidate_outputs.append(next_candidates)
                        count_outputs.append(next_counts)
                    outputs.append(result)
                selected = torch.cat(outputs, dim=0)
                if self.layer == self.candidate_source:
                    candidates = torch.cat(candidate_outputs, dim=0)
                    candidate_counts = torch.cat(count_outputs, dim=0)
                dependency = selected
        if self.concurrent_query is not None:
            query = self.concurrent_query.finish(pending, dependency).view(self.query_shape)
            self.rotary.apply(query, position, compressed=bool(self.ratio))
        attended = self.attention(query, window_indices, window_lengths,
            indexed_indices=selected if self.ratio else None,
            indexed_lengths=index_lengths if self.ratio else None,
            page_table=page_table if self.ratio else None, window=window_cache)
        self.rotary.apply(attended, position, compressed=bool(self.ratio), inverse=True)
        output = self.output_projection(attended.view(source.shape[0], -1, 4096))
        record = window_cache, records
        if self.mode == "prefill":
            self.commit(m.lengths, record, m)
        observations = dict(projected=projected, query_low=query_low, query=query,
                            window_values=window_values, qkv_encoded_bases=qkv_encoded_bases,
                            query_encoded_bases=encoded_bases, selected=selected, attended=attended, output=output)
        return output, record, (selected, candidates, candidate_counts), observations

    def commit(self, counts, records, metadata):
        window_cache, _ = records
        torch.library.wrap_triton(_commit_window)[(128, metadata.starts.numel())](window_cache, self.window,
            metadata.state_rows, metadata.starts, counts, self.window.stride(0), window_cache.stride(0), 1024)
        self.commit_pending(counts, records, metadata)

    def commit_pending(self, counts, records, metadata):
        if self.mode == "verify" and self.global_kv is not None:
            self.global_kv.compressor.commit(counts, records[1], metadata)
