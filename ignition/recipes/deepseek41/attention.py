# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Windowed MLA, source-shared compression, and two-level sparse selection."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from b12x.quantization import mla_norm_mxfp8, mxfp8
import triton
import triton.language as tl
from b12x.attention import mla_compress
from b12x.preparation import PreparedCall

from ignition.components import sm103_projection
from ignition.components.compressed_mla import PreparedCompressedCacheWriter, PreparedCompressedMLA
from ignition.components.attention_output import PreparedGroupedOutputProjection
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.mla_norm_quant import PreparedMLANormQuant
from ignition.components.mxfp4_indexer import PreparedMXFP4Indexer
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.state import gather_rows, gather_rows_three, save_hidden
from ignition.components.window_indices import PreparedWindowIndices
from ignition.preparation.tuning import declare, register_finalizer
from ignition.preparation.scratch import ScratchArena

from ignition.components.rotary import CompressedRotary as Rotary
from ignition.components.parallel import TensorParallelShard

from .pool import PAGE_TOKENS


# Bound indexer workspace while amortizing prefill selection launches.
INDEXER_QUERY_ROWS = 128


class AttentionIndices(PreparedWindowIndices):
    """Window and compression geometry shared by layers with identical query metadata.

    Allocate every compression ratio before per-layer scratch cursors fork.
    Compressor outputs, pending state, and cache payloads remain layer-owned.
    """

    def __init__(self, metadata, *, mode, tokens, page_size, width, ratios=(), visible=None,
                 replay_start=None, buffers=None):
        super().__init__(metadata, mode=mode, tokens=tokens, page_size=page_size, width=width,
                         visible=visible, replay_start=replay_start, buffers=buffers)
        tokens = metadata.positions.numel()

        def zeros(shape, dtype):
            return (torch.zeros(shape, device=metadata.positions.device, dtype=dtype)
                    if buffers is None else buffers.zeros(shape, dtype))

        self.position = zeros((tokens,), torch.int64)
        self.ratios = {}
        for ratio in sorted(set(ratios)):
            if type(ratio) is not int or ratio not in (1, 2):
                raise ValueError("DeepSeek V4.1 compression ratios must be one or two")
            self.ratios[ratio] = SimpleNamespace(
                slots=zeros((tokens,), torch.int64).fill_(-1),
                lengths=zeros((tokens,), torch.int32),
                index_lengths=zeros((tokens,), torch.int32),
                active_width=zeros((1,), torch.int32),
                position=zeros((tokens,), torch.int64))

    def compressed(self, ratio):
        if ratio not in self.ratios:
            raise ValueError("compression geometry must be declared before layer scratch is allocated")
        return self.ratios[ratio]

    def __call__(self):
        super().__call__()
        m = self.metadata
        live = m.positions >= 0
        self.position.copy_(m.positions.clamp_min(0))
        if self.ratios:
            pages = m.page_table[m.requests, (self.position // PAGE_TOKENS).clamp(max=m.page_table.shape[1] - 1)]
        for ratio, shared in self.ratios.items():
            shared.lengths.copy_(torch.where(live, (self.position + 1) // ratio, 0))
            shared.index_lengths.copy_(shared.lengths.clamp(max=512))
            shared.active_width.copy_(shared.lengths.amax().view(1))
            shared.slots.copy_(torch.where(live,
                pages.long() * (PAGE_TOKENS // ratio) + (self.position % PAGE_TOKENS) // ratio, -1))
            shared.position.copy_((self.position + 1) // ratio * ratio - ratio)


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
    def allocate(cls, tokens, world, device, *, buffers=None, candidate_tokens=None):
        candidate_tokens = tokens if candidate_tokens is None else candidate_tokens
        if type(candidate_tokens) is not int or not 0 < candidate_tokens <= tokens:
            raise ValueError("candidate row capacity must fit the attention token capacity")
        def zeros(shape, dtype=torch.bfloat16):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        first, last = TensorParallelShard(0, world).padded_partition(8)
        heads = (last - first) * 8
        return cls(zeros((tokens, 1792)),
            zeros((tokens, 1280)), zeros((tokens, heads, 512)),
            zeros((tokens, 512)), zeros((tokens, heads, 512)),
            zeros((tokens, 5120)), zeros((tokens, 512), torch.int32).fill_(-1),
            zeros((candidate_tokens, 16384), torch.int32).fill_(-1), zeros(candidate_tokens, torch.int32),
            zeros((tokens, 32, 128)), zeros((tokens, 32)), zeros((tokens, 128)), zeros((tokens, 128)))


class PreparedCompressor:
    def __init__(self, layer, weights, source, metadata, pool, slots, *, session, name, deferred=False,
                 buffers=None):
        self.pool, self.metadata, self.layer = pool, metadata, layer
        self.deferred = deferred
        ratio = pool.config.text["compress_ratios"][layer]
        self.ratio = ratio
        tokens, rows = source.shape[0], metadata.starts.numel()
        device = source.device
        def zeros(shape, dtype):
            return (torch.zeros(shape, device=device, dtype=dtype) if buffers is None
                    else buffers.zeros(shape, dtype))
        self.values = zeros((tokens, 512), torch.float32 if ratio == 2 else torch.bfloat16)
        self.project_values = UnquantizedProjection(source, weights["values"], self.values,
            session=session, name=f"{name}.values")
        self.output = zeros((tokens, 512), torch.bfloat16)
        self.emitted = zeros(tokens, torch.bool)
        self.slots = zeros(tokens, torch.int64).fill_(-1)
        self.state_ids = torch.arange(rows, dtype=torch.int64, device=device)
        self.arguments = dict(values=self.values, weight=weights["norm"], query_start_loc=metadata.query_start_loc,
            positions=metadata.starts, state_ids=self.state_ids, destination_slots=slots,
            live_counts=metadata.live_counts, out=self.output, emitted=self.emitted, emitted_slots=self.slots)
        if ratio == 2:
            self.gates = zeros(self.values.shape, self.values.dtype)
            self.project_gates = UnquantizedProjection(source, weights["gates"], self.gates,
                session=session, name=f"{name}.gates")
            self.pending = {key: zeros((rows, 512) if key != "pending_position" else (rows,),
                torch.float32 if key != "pending_position" else torch.int64)
                for key in ("pending_values", "pending_gates", "pending_position")}
            self.arguments.update(gates=self.gates, **self.pending)
        self.plan = mla_compress.plan(mla_compress.Caps(device=device, max_tokens=tokens,
            max_requests=rows, max_states=rows, ratio=ratio))

        def prepare(state):
            arguments = {key: value if key in ("weight", "state_ids") else torch.zeros_like(value)
                         for key, value in self.arguments.items()}
            arguments["query_start_loc"].copy_(torch.arange(rows + 1, device=device) * tokens // rows)
            arguments["live_counts"].copy_(torch.tensor([tokens, rows], dtype=torch.int32, device=device))
            arguments["destination_slots"].copy_(torch.arange(tokens, device=device))
            if ratio == 2:
                arguments["pending_position"].fill_(-1)
            binding = state.bind(**arguments)
            return PreparedCall(run=lambda: state.run(binding))

        declare(session, (self.plan.request(name=name, prepare_call=prepare),))
        register_finalizer(session, lambda: setattr(self, "binding", mla_compress.bind(self.plan, **self.arguments)))

    def __call__(self, source):
        m = self.metadata
        if self.ratio == 2:
            gather_rows_three(tuple(self.pool.region(f"layer{self.layer}.{key}") for key in self.pending),
                              m.state_rows, tuple(self.pending.values()))
            self.project_gates(source)
        self.project_values(source)
        mla_compress.run(self.binding)
        if self.ratio == 2 and not self.deferred:
            for key, value in self.pending.items():
                save_hidden(value.view(value.shape[0], -1), self.pool.region(f"layer{self.layer}.{key}"),
                            m.state_rows, m.lengths > 0)
        return self.output

    def commit(self, counts):
        if self.ratio != 2:
            return
        m = self.metadata
        last = (m.query_start_loc[:-1] + counts - 1).long().clamp(0, self.values.shape[0] - 1)
        position = m.starts + counts - 1
        pending = (counts > 0) & (position % 2 == 0)
        for key, values in (("pending_values", self.values), ("pending_gates", self.gates)):
            save_hidden(values[last], self.pool.region(f"layer{self.layer}.{key}"), m.state_rows, pending)
        tags = torch.where(position % 2 == 0, position, -1).view(-1, 1)
        save_hidden(tags, self.pool.region(f"layer{self.layer}.pending_position"), m.state_rows, counts > 0)


class PreparedGlobalKV:
    """Project and store every source row before decoder queries are packed."""

    def __init__(self, layer, weights, source, scratch, metadata, pool, rotary, *,
                 session, name, indices, deferred=False, buffers=None):
        self.layer, self.metadata, self.pool, self.rotary = layer, metadata, pool, rotary
        self.ratio = pool.config.text["compress_ratios"][layer]
        if indices.metadata is not metadata or indices.position.shape != (source.shape[0],):
            raise ValueError("global KV indices must describe every source row")
        self.geometry = indices.compressed(self.ratio)
        self.slots = self.geometry.slots
        self.compressor = PreparedCompressor(layer, weights["compressor"], source, metadata,
            pool, self.slots, session=session, name=f"{name}.compress", deferred=deferred, buffers=buffers)
        self.keys, self.normalized_keys = scratch.index_keys, scratch.normalized_keys
        self.index_key = UnquantizedProjection(self.compressor.output, weights["index_k"], self.keys,
            session=session, name=f"{name}.index_key")
        self.index_norm = PreparedRMSNorm(self.keys, weights["index_norm"], self.normalized_keys,
            eps=pool.config.text["rms_norm_eps"], session=session, name=name + ".index_norm")
        self.writer = PreparedCompressedCacheWriter(self.compressor.output,
            pool.pages[f"layer{layer}.latent_page"], self.compressor.slots,
            page_size=PAGE_TOKENS // self.ratio, kind="indexed", session=session,
            name=f"{name}.compressed_write")

    def __call__(self, source, indexer):
        latent = self.compressor(source)
        self.index_key(latent)
        self.index_norm(self.keys)
        compressed_position = self.geometry.position
        self.rotary.apply(self.normalized_keys, compressed_position, compressed=True)
        indexer.write_keys(self.normalized_keys, self.compressor.slots)
        self.rotary.apply(latent, compressed_position, compressed=True)
        self.writer(latent)


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
                 mla_compute_mode="auto", buffers=None, window_metadata=None,
                 preprojected=False, external_output=False):
        self.layer, self.weights, self.scratch, self.metadata = layer, weights, scratch, metadata
        self.pool, self.rotary, self.mode = pool, rotary, mode
        self.preprojected = preprojected
        self.replay_start = replay_start
        config = pool.config
        draft = mode == "draft"
        self.ratio = 0 if draft else config.text["compress_ratios"][layer]
        self.source = None if draft else config.source(layer)
        self.eps = config.text["rms_norm_eps"]
        tokens, device = source.shape[0], source.device
        rows = metadata.starts.numel()
        def zeros(shape, dtype):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
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
                ratios=(self.ratio,) if self.ratio else (), buffers=buffers)
        elif (window_metadata.metadata is not metadata or window_metadata.mode != mode
                or window_metadata.window_page_size != self.window_page_size
                or window_metadata.window_indices.shape != (tokens, window_width)
                or window_metadata.replay_start is not replay_start or window_metadata.visible is not None
                or not isinstance(window_metadata, AttentionIndices)):
            raise ValueError("shared window indices must match the attention metadata and cache geometry")
        self.window_metadata = window_metadata
        self.window_slots = window_metadata.window_slots
        self.window_indices = window_metadata.window_indices
        self.window_lengths = window_metadata.window_lengths
        self.token_offsets = torch.arange(tokens, device=device)
        geometry = window_metadata.compressed(self.ratio) if self.ratio else None
        self.lengths = zeros(tokens, torch.int32) if geometry is None else geometry.lengths
        self.active_width = zeros(1, torch.int32) if geometry is None else geometry.active_width
        self.index_lengths = zeros(tokens, torch.int32) if geometry is None else geometry.index_lengths
        self.page_table = metadata.token_page_table
        self.projections = {}
        use_b12x_mxfp8 = not sm103_projection.is_sm103(device)
        paired_mxfp8 = use_b12x_mxfp8 and all(isinstance(getattr(weights[key], "weight", None), mxfp8.MXFP8Rows)
                           for key in ("qkv_a", "q_b"))
        self.norm_quant = (PreparedMLANormQuant(scratch.projected,
            (weights["q_norm"], weights["kv_norm"]), scratch.query_low, scratch.window_values,
            eps=self.eps, session=session, name=name + ".qkv_norm", buffers=buffers,
            release_dependents=tokens <= 8)
            if not preprojected and paired_mxfp8 and self.eps == 1e-20 and mla_norm_mxfp8.is_supported(device) else None)
        for key, values, out in (
            ("qkv_a", source, scratch.projected), ("q_b", scratch.query_low, scratch.query.view(tokens, -1)),
        ):
            if preprojected:
                break
            packed_mxfp8 = use_b12x_mxfp8 and isinstance(
                getattr(weights[key], "weight", None), mxfp8.MXFP8Rows)
            self.projections[key] = QuantizedProjection(values, weights[key], out,
                session=session, name=f"{name}.{key}", workspace=workspace, activation_mode="auto",
                mxfp8_min_amax=(1e-4 if packed_mxfp8 and (key != "q_b" or self.norm_quant is None)
                                else None),
                encoded=self.norm_quant.encoded if key == "q_b" and self.norm_quant is not None else None,
                rhs_prefetch_ready=packed_mxfp8,
                allow_bf16_atomic_reduction=not packed_mxfp8,
                buffers=buffers)
        self.query_norm = (PreparedRMSNorm(scratch.projected[:, :1280], weights["q_norm"], scratch.query_low,
            eps=self.eps, session=session, name=name + ".q_norm") if not preprojected and self.norm_quant is None else None)
        self.window_norm = (PreparedRMSNorm(scratch.projected[:, 1280:], weights["kv_norm"], scratch.window_values,
            eps=self.eps, session=session, name=name + ".kv_norm") if not preprojected and self.norm_quant is None else None)
        self.output_projection = (None if external_output else PreparedGroupedOutputProjection(
            scratch.attended.view(tokens, -1, 4096), weights["output"], scratch.output,
            session=session, name=name + ".output", workspace=workspace))
        self.window_writer = PreparedCompressedCacheWriter(scratch.window_values, self.window_cache,
            self.window_slots, page_size=self.window_page_size, kind="swa", session=session,
            name=f"{name}.window_write",
            history=self.window if mode in ("verify", "draft") else None,
            history_rows=metadata.state_rows if mode in ("verify", "draft") else None,
            snapshot_tokens=128 if mode in ("verify", "draft") else 0)
        self.global_kv = None
        self.selectors = []
        if "compressor" in weights:
            self.global_kv = PreparedGlobalKV(layer, weights, source, scratch, metadata, pool, rotary,
                session=session, name=name, indices=window_metadata, deferred=mode == "verify", buffers=buffers)
        if "index_q" in weights:
            self.index_query = QuantizedProjection(scratch.query_low, weights["index_q"],
                scratch.index_query.view(tokens, -1), session=session, name=f"{name}.index_query",
                workspace=workspace, activation_mode="auto")
            self.index_weight = UnquantizedProjection(source, weights["index_weights"], scratch.index_weights,
                session=session, name=f"{name}.index_weight")
            candidate_source = config.text["candidate_source_layer_id"]
            if layer >= candidate_source and scratch.candidates.shape[0] < tokens:
                raise ValueError("hierarchical selection exceeds candidate row capacity")
            selector_cursors = []
            for first in range(0, tokens, INDEXER_QUERY_ROWS):
                last = min(tokens, first + INDEXER_QUERY_ROWS)
                candidates = {}
                if layer == candidate_source:
                    candidates = dict(candidates_out=scratch.candidates[first:last], counts_out=scratch.candidate_counts[first:last])
                elif layer > candidate_source:
                    candidates = dict(candidates=scratch.candidates[first:last], counts=scratch.candidate_counts[first:last])
                cursor = None if buffers is None else buffers.fork()
                if cursor is not None:
                    selector_cursors.append(cursor)
                selector = PreparedMXFP4Indexer(scratch.index_query[first:last], scratch.index_weights[first:last],
                    pool.pages[f"layer{self.source.layer}.index_page"],
                    self.page_table[first:last],
                    self.lengths[first:last], self.active_width, scratch.selected[first:last],
                    page_size=128 // self.ratio, session=session, name=f"{name}.index{first}", workspace=workspace,
                    buffers=cursor, mode="prefill" if mode == "prefill" else "decode", **candidates)
                self.selectors.append(selector)
            if buffers is not None:
                buffers.join(*selector_cursors)
        indexed = {} if not self.ratio else dict(indexed=pool.pages[f"layer{self.source.layer}.latent_page"],
            indexed_indices=scratch.selected, indexed_lengths=self.index_lengths,
            page_table=self.page_table,
            indexed_page_size=128 // self.ratio)
        self.attention = PreparedCompressedMLA(scratch.query, self.window_cache, self.window_indices,
            self.window_lengths, weights["sink"], scratch.attended,
            window_page_size=self.window_page_size,
            # Speculative rows use explicit indices into staged caches. Their
            # prepared row bound belongs to the decode contract, including
            # multi-request verification and noncausal draft blocks.
            mode="decode" if mode in ("decode", "verify", "draft") else "extend",
            decode_row_capacity=tokens if mode in ("decode", "verify", "draft") else None,
            session=session, name=f"{name}.attention",
            workspace=workspace, compute_mode=mla_compute_mode, **indexed)
        self.concurrent_query = None
        if (not preprojected and mode in ("decode", "verify", "draft") and not (mode == "decode" and tokens == 1)
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

    def __call__(self, source):
        m, s = self.metadata, self.scratch
        if self.owns_window_metadata:
            self.window_metadata()
        position = self.window_metadata.position
        if self.mode == "prefill":
            gather_rows(self.window, m.state_rows, self.window_cache[:, :128 * 528])
        if self.preprojected:
            self.rotary.apply(s.query, position, compressed=bool(self.ratio))
        else:
            self.projections["qkv_a"](source)
            if self.norm_quant is None:
                self.query_norm(s.projected[:, :1280])
            else:
                self.norm_quant(s.projected)
            if self.concurrent_query is not None:
                self.concurrent_query.start(s.query_low)
            if self.window_norm is not None:
                self.window_norm(s.projected[:, 1280:])
            if self.concurrent_query is None:
                self.projections["q_b"](s.query_low)
                self.rotary.apply(s.query, position, compressed=bool(self.ratio))
        self.rotary.apply(s.window_values, position, compressed=bool(self.ratio))
        self.window_writer(s.window_values)
        dependency = self.window_cache
        if self.ratio:
            if self.global_kv is not None:
                self.global_kv(source, self.selectors[0])
            if self.selectors:
                self.index_query(s.query_low)
                self.rotary.apply(s.index_query, position, compressed=True)
                self.index_weight(source)
                s.index_weights.mul_(128 ** -.5 * 32 ** -.5)
                for selector in self.selectors:
                    dependency = selector()
        if self.concurrent_query is not None:
            # The indexer only reads query_low. Its output and all main-stream
            # scratch uses must complete before the query branch rejoins.
            self.concurrent_query.finish(s.query_low, dependency)
            self.rotary.apply(s.query, position, compressed=bool(self.ratio))
        self.attention()
        self.rotary.apply(s.attended, position, compressed=bool(self.ratio), inverse=True)
        if self.output_projection is not None:
            self.output_projection(s.attended.view(s.attended.shape[0], -1, 4096))
        if self.mode == "prefill":
            self.commit(m.lengths)
        return s.output if self.output_projection is not None else s.attended

    def commit(self, counts):
        m = self.metadata
        torch.library.wrap_triton(_commit_window)[(128, m.starts.numel())](self.window_cache, self.window,
            m.state_rows, m.starts, counts, self.window.stride(0), self.window_cache.stride(0), 1024)
        self.commit_pending(counts)

    def commit_pending(self, counts):
        if self.mode == "verify" and self.global_kv is not None:
            self.global_kv.compressor.commit(counts)
