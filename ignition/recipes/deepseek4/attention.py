# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sliding windows, overlapping C4 retrieval, and dense C128 history."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from ignition.components.attention_output import PreparedGroupedOutputProjection
from ignition.components.compressed_mla import (
    PreparedCompressedCacheWriter,
    PreparedCompressedMLA,
)
from ignition.components.gated_compression import PreparedGatedCompressor
from ignition.components.compressor_overlap import ConcurrentGatedCompression
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.fp8_indexer import PreparedFP8Indexer
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.rotary import PreparedRotaryAngles
from ignition.components.window_indices import PreparedWindowIndices
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import register_finalizer

from .config import Compressor
from .pool import PAGE_TOKENS, latent_page_bytes

# Select a 4096-token prefill in one call; sequential layers share the planned FP8 workspace.
INDEX_ROWS = 4096
SHARED_ROTARY_ROWS = 32


@triton.jit
def _transfer_window(
    Source,
    Destination,
    StateRows,
    Starts,
    Counts,
    SOURCE_STRIDE: tl.constexpr,
    DEST_STRIDE: tl.constexpr,
    STAGED_TOKENS: tl.constexpr,
    COMMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request, item = tl.program_id(0), tl.program_id(1)
    row = tl.load(StateRows + request)
    offset = tl.arange(0, BLOCK)
    if COMMIT:
        length = tl.load(Counts + request)
        token = tl.maximum(0, length - 128) + item
        live = (row > 0) & (token < length)
        source_row, source_token = request, 128 + token
        dest_row, dest_token = row, (tl.load(Starts + request) + token) % 128
        source_extent: tl.constexpr = STAGED_TOKENS
        dest_extent: tl.constexpr = 128
    else:
        live = row > 0
        source_row, source_token = row, item
        dest_row, dest_token = request, item
        source_extent: tl.constexpr = 128
        dest_extent: tl.constexpr = STAGED_TOKENS
    source_offset = tl.where(
        offset < 576, source_token * 576 + offset, source_extent * 576 + source_token * 8 + offset - 576
    )
    dest_offset = tl.where(
        offset < 576, dest_token * 576 + offset, dest_extent * 576 + dest_token * 8 + offset - 576
    )
    value = tl.load(Source + source_row * SOURCE_STRIDE + source_offset, live & (offset < 584), other=0)
    tl.store(Destination + dest_row * DEST_STRIDE + dest_offset, value, live & (offset < 584))


def transfer_window(source, destination, metadata, staged_tokens, *, counts=None):
    torch.library.wrap_triton(_transfer_window)[(metadata.starts.numel(), 128)](
        source,
        destination,
        metadata.state_rows,
        metadata.starts,
        metadata.lengths if counts is None else counts,
        source.stride(0),
        destination.stride(0),
        staged_tokens,
        counts is not None,
        1024,
    )


@triton.jit
def _hadamard(Values, Output, ROWS: tl.constexpr):
    row = tl.program_id(0)
    dim = tl.arange(0, 128)
    value = tl.load(Values + row * 128 + dim)
    value = value.to(tl.float32)
    for bit in tl.static_range(7):
        other = tl.gather(value, dim ^ (1 << bit), axis=0)
        value = tl.where((dim & (1 << bit)) == 0, value + other, other - value)
    tl.store(Output + row * 128 + dim, value * (128**-0.5))


@triton.jit
def _normalize_rotate(Query, Positions, Frequencies, ROW_STRIDE, HEAD_STRIDE, EPS, WIDTH: tl.constexpr,
                      Angles=None, SHARED_ANGLES: tl.constexpr=False):
    """RMS-normalize one row's head with BF16 rounding at every step, then rotate its last 64 values."""
    row, head = tl.program_id(0), tl.program_id(1)
    base = Query + row * ROW_STRIDE + head * HEAD_STRIDE
    offsets = tl.arange(0, WIDTH)
    value = tl.load(base + offsets).to(tl.float32)
    square = (value * value).to(tl.bfloat16).to(tl.float32)
    mean = (tl.sum(square, axis=0) / WIDTH).to(tl.bfloat16).to(tl.float32)
    scale = tl.rsqrt((mean + EPS).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(base + offsets, (value * scale).to(tl.bfloat16), mask=offsets < WIDTH - 64)
    pair = tl.arange(0, 32)
    if SHARED_ANGLES:
        cos = tl.load(Angles + row * 64 + 2 * pair)
        sin = tl.load(Angles + row * 64 + 2 * pair + 1)
    else:
        angle = tl.load(Positions + row).to(tl.float32) * tl.load(Frequencies + pair)
        cos, sin = tl.cos(angle), tl.sin(angle)
    tail = base + WIDTH - 64 + 2 * pair
    a = (tl.load(tail).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    b = (tl.load(tail + 1).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    tl.store(tail, (a * cos - b * sin).to(tl.bfloat16))
    tl.store(tail + 1, (a * sin + b * cos).to(tl.bfloat16))


def normalize_rotate(query, positions, frequencies, eps, *, angles=None):
    """Normalize and rotate (rows, heads, width) BF16 queries in place in one launch."""
    torch.library.wrap_triton(_normalize_rotate)[(query.shape[0], query.shape[1])](
        query, positions, frequencies, query.stride(0), query.stride(1), eps, query.shape[2],
        angles, angles is not None
    )


def rotate_index(source, destination):
    torch.library.wrap_triton(_hadamard)[(source.numel() // 128,)](source, destination, source.numel() // 128)


@dataclass(frozen=True)
class AttentionScratch:
    projected: torch.Tensor
    query_low: torch.Tensor
    query: torch.Tensor
    window_values: torch.Tensor
    attended: torch.Tensor
    output: torch.Tensor
    selected: torch.Tensor
    dense_indices: torch.Tensor
    index_query: torch.Tensor
    index_weights: torch.Tensor

    @classmethod
    def allocate(cls, tokens, world, device, *, blocks=8192):
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((tokens, 1536)),
            zeros((tokens, 1024)),
            zeros((tokens, 64 // world, 512)),
            zeros((tokens, 512)),
            zeros((tokens, 64 // world, 512)),
            zeros((tokens, 4096)),
            zeros((tokens, 512), torch.int32).fill_(-1),
            zeros((tokens, blocks * (PAGE_TOKENS // 128)), torch.int32).fill_(-1),
            zeros((tokens, 64, 128)),
            zeros((tokens, 64)),
        )


class CompressionGeometry(NamedTuple):
    slots: torch.Tensor
    lengths: torch.Tensor
    index_lengths: torch.Tensor
    active_width: torch.Tensor
    position: torch.Tensor
    emitted: torch.Tensor
    emitted_slots: torch.Tensor


class AttentionGeometry(NamedTuple):
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    window_lengths: torch.Tensor
    ratios: dict[int, CompressionGeometry]
    dense: torch.Tensor | None
    angles: dict[tuple[bool, int], torch.Tensor]


class AttentionIndices(PreparedWindowIndices):
    """Window, compression, and rotary inputs shared by layers of one invocation."""

    def __init__(self, metadata, *, mode, tokens, visible=None):
        page_size = 128 if mode == "decode" else 128 + metadata.max_query_tokens
        width = page_size if mode == "draft" else 128 + (384 if visible is not None else 0)
        super().__init__(metadata, mode=mode, tokens=tokens, page_size=page_size, width=width, visible=visible)
        self.ratios, self.dense, self.angles = {}, None, {}

    def rotary_angles(self, rotary, *, compressed, ratio=0):
        key = compressed, ratio
        if key not in self.angles:
            positions = self.metadata.positions if ratio == 0 else self.compressed(ratio).position
            self.angles[key] = PreparedRotaryAngles(positions, rotary.frequencies[compressed])
        return self.angles[key]

    def compressed(self, ratio):
        if ratio not in self.ratios:
            device, tokens = self.window_slots.device, self.window_slots.shape[0]
            def zeros(dtype):
                return torch.zeros((tokens,), dtype=dtype, device=device)
            self.ratios[ratio] = CompressionGeometry(slots=zeros(torch.int64), lengths=zeros(torch.int32),
                index_lengths=zeros(torch.int32), active_width=torch.zeros(1, dtype=torch.int32, device=device),
                position=zeros(torch.int64), emitted=zeros(torch.bool), emitted_slots=zeros(torch.int64))
        return self.ratios[ratio]

    def dense_pages(self, target):
        if self.dense is None:
            self.dense = target
        return self.dense

    def functional(self):
        self.compression_ratios = tuple(self.ratios)
        self.has_dense = self.dense is not None
        self.ratios = self.dense = None
        for operation in self.angles.values():
            operation.functional()
        super().functional()
        return self

    def __call__(self, metadata, visible=None):
        base = super().__call__(metadata, visible=visible)
        m = metadata
        live, position = m.positions >= 0, m.positions.clamp_min(0)
        ratios = {}
        for ratio in self.compression_ratios:
            lengths = torch.where(live, (position + 1) // ratio, 0).to(torch.int32)
            pages = m.page_table[m.requests, (position // PAGE_TOKENS).clamp(max=m.page_table.shape[1] - 1)]
            slots = torch.where(live, pages.long() * (PAGE_TOKENS // ratio) + (position % PAGE_TOKENS) // ratio, -1)
            emitted = live & ((m.positions + 1) % ratio == 0)
            ratios[ratio] = CompressionGeometry(slots=slots, lengths=lengths,
                index_lengths=lengths.clamp(max=512) if ratio == 4 else lengths,
                active_width=lengths.amax().view(1), position=((position + 1) // ratio * ratio - ratio).clamp_min(0),
                emitted=emitted, emitted_slots=torch.where(emitted, slots, -1))
        dense = None
        if self.has_dense:
            pages = m.token_page_table[:, :, None]
            records = PAGE_TOKENS // 128
            dense = torch.where(pages >= 0, pages * records + torch.arange(records, device=pages.device), -1).flatten(1).to(torch.int32)
        angles = {key: operation(m.positions if key[1] == 0 else ratios[key[1]].position)
                  for key, operation in self.angles.items()}
        return AttentionGeometry(window_slots=base.window_slots, window_indices=base.window_indices,
            window_lengths=base.window_lengths, ratios=ratios, dense=dense, angles=angles)


class PreparedAttention:
    def __init__(
        self,
        layer,
        weights,
        source,
        scratch,
        metadata,
        pool,
        rotary,
        *,
        mode,
        world,
        session,
        name,
        workspace,
        visible=None,
        indices=None,
        examples=None,
    ):
        self.pool = pool
        self.heads = scratch.query.shape[1]
        self.rotary, self.mode = rotary, mode
        self.ratio = 0 if mode == "draft" else pool.config.text["compress_ratios"][layer]
        self.eps = pool.config.text["rms_norm_eps"]
        tokens, rows, device = source.shape[0], metadata.starts.numel(), source.device
        self.window = pool.window(layer, draft=mode == "draft")
        # Layers of one forward share their window and compression indices; a standalone layer owns them.
        self.owns_indices = indices is None
        if indices is None:
            indices = AttentionIndices(metadata, mode=mode, tokens=tokens, visible=visible)
        self.indices = indices
        self.window_page_size = indices.window_page_size
        self.window_shape = (rows, latent_page_bytes(self.window_page_size))
        # Staged windows belong to one invocation; only decode writes the persistent window directly.
        self.window_cache = (
            self.window
            if mode == "decode"
            else declaration_tensor(examples, "attention.window",
                                    (rows, latent_page_bytes(self.window_page_size)), torch.uint8, device)
        )
        self.window_slots, self.window_indices = indices.window_slots, indices.window_indices
        self.window_lengths = indices.window_lengths
        shared = indices.compressed(self.ratio) if self.ratio else None
        # Small graphs share repeated trigonometry; large prefills avoid table-read traffic.
        share_angles = tokens <= SHARED_ROTARY_ROWS
        self.angles = indices.rotary_angles(rotary, compressed=bool(self.ratio)) if share_angles else None
        self.compressed_angles = (indices.rotary_angles(rotary, compressed=True, ratio=self.ratio)
                                  if self.ratio and share_angles else None)
        if shared is not None:
            self.compressed_slots, self.lengths = shared.slots, shared.lengths
            self.active_width, self.index_lengths = shared.active_width, shared.index_lengths
        self.overlap_compression = self.ratio == 4 and mode == "verify" and tokens <= 8
        # Small C4 verification overlaps compression. Quantized projections leave room for that
        # branch; tuning the projections in isolation does not measure their combined latency.
        activation_mode = "quantized" if self.overlap_compression else "auto"
        self.qkv = QuantizedProjection(
            source,
            weights["qkv_a"],
            scratch.projected,
            session=session,
            name=name + ".qkv_a",
            workspace=workspace,
            activation_mode=activation_mode,
            examples=examples, example_role="attention.qkv",
        )
        self.qb = QuantizedProjection(
            scratch.query_low,
            weights["q_b"],
            scratch.query.view(tokens, -1),
            session=session,
            name=name + ".q_b",
            workspace=workspace,
            activation_mode=activation_mode,
            examples=examples, example_role="attention.query",
        )
        self.qnorm = PreparedRMSNorm(
            scratch.projected[:, :1024],
            weights["q_norm"],
            scratch.query_low,
            eps=self.eps,
            session=session,
            name=name + ".q_norm",
        )
        self.kvnorm = PreparedRMSNorm(
            scratch.projected[:, 1024:],
            weights["kv_norm"],
            scratch.window_values,
            eps=self.eps,
            session=session,
            name=name + ".kv_norm",
        )
        self.output_projection = PreparedGroupedOutputProjection(
            scratch.attended.view(tokens, 8 // world, 4096),
            weights["output"],
            scratch.output,
            session=session,
            name=name + ".output",
            workspace=workspace,
        )

        def writer(values, cache, slots, size, kind, suffix):
            return PreparedCompressedCacheWriter(
                values,
                cache,
                slots,
                page_size=size,
                kind=kind,
                session=session,
                name=name + suffix,
                cache_format="deepseek_v4",
            )

        self.window_writer = writer(
            scratch.window_values,
            self.window_cache,
            self.window_slots,
            self.window_page_size,
            "swa",
            ".window_write",
        )
        self.compressor = self.index_compressor = None
        self.selectors = []
        self.indexed_indices = None
        indexed = {}
        self.compression = None
        if self.ratio:
            specs = [Compressor(layer, self.ratio, 512)]
            if self.ratio == 4:
                specs.append(Compressor(layer, 4, 128, index=True))
            widths = [spec.state_shape[1] for spec in specs]
            self.compressed = declaration_tensor(examples, "attention.compression",
                                                 (tokens, 2 * sum(widths)), torch.float32, device)
            self.compression = UnquantizedProjection(
                source, weights["compression"], self.compressed, session=session, name=name + ".compression"
            )
            projected, offset = [], 0
            for width in widths:
                projected.append((self.compressed[:, offset:offset + width],
                                  self.compressed[:, offset + width:offset + 2 * width]))
                offset += 2 * width
            self.compressor = PreparedGatedCompressor(
                specs[0],
                weights["compressor"],
                source,
                metadata,
                pool,
                self.compressed_slots,
                session=session,
                name=name + ".compress",
                deferred=mode == "verify",
                examples=examples, example_role="attention.compressor",
                emitted=shared.emitted,
                emitted_slots=shared.emitted_slots,
                projected=projected[0],
            )
            self.compressed_writer = writer(
                self.compressor.output,
                pool.pages[f"layer{layer}.latent_page"],
                self.compressor.slots,
                PAGE_TOKENS // self.ratio,
                "indexed",
                ".compressed_write",
            )
            if self.ratio == 4:
                self.index_compressor = PreparedGatedCompressor(
                    specs[1],
                    weights["index_compressor"],
                    source,
                    metadata,
                    pool,
                    self.compressed_slots,
                    session=session,
                    name=name + ".index_compress",
                    deferred=mode == "verify",
                    examples=examples, example_role="attention.index_compressor",
                    emitted=shared.emitted,
                    emitted_slots=shared.emitted_slots,
                    projected=projected[1],
                )
                self.index_query = QuantizedProjection(
                    scratch.query_low,
                    weights["index_q"],
                    scratch.index_query.view(tokens, -1),
                    session=session,
                    name=name + ".index_query",
                    workspace=workspace,
                    activation_mode=activation_mode,
                    examples=examples, example_role="attention.index_query",
                )
                self.index_weight = UnquantizedProjection(
                    source,
                    weights["index_weights"],
                    scratch.index_weights,
                    session=session,
                    name=name + ".index_weight",
                )
                for first in range(0, tokens, INDEX_ROWS):
                    last = min(tokens, first + INDEX_ROWS)
                    self.selectors.append(
                        PreparedFP8Indexer(
                            scratch.index_query[first:last],
                            scratch.index_weights[first:last],
                            pool.pages[f"layer{layer}.index_page"],
                            metadata.token_page_table[first:last],
                            self.lengths[first:last],
                            self.active_width,
                            scratch.selected[first:last],
                            mode="prefill" if mode == "prefill" else "decode",
                            session=session,
                            name=f"{name}.index{first}",
                            workspace=workspace,
                            examples=examples, example_role="attention.indexer",
                            prefill_records_per_call=131072,
                        )
                    )
                selected = scratch.selected
            else:
                selected = indices.dense_pages(scratch.dense_indices[:,
                    : metadata.page_table.shape[1] * (PAGE_TOKENS // 128)])
            self.indexed_indices = selected
            indexed = {
                "indexed": pool.pages[f"layer{layer}.latent_page"],
                "indexed_indices": selected,
                "indexed_lengths": self.index_lengths,
                "indexed_page_size": PAGE_TOKENS // self.ratio,
            }
        self.concurrent_compression = None
        self.concurrent_index = None
        if self.overlap_compression and workspace.overlap:
            self.concurrent_index = ConcurrentProjectionChain(
                (self.index_query,), parallel_plans=(self.kvnorm.plan, self.qb.plan, self.window_writer.plan),
                session=session, workspace=workspace)
            compressors = [self.compressor]
            if self.index_compressor is not None:
                compressors.append(self.index_compressor)
            self.concurrent_compression = ConcurrentGatedCompression(
                self.compression, compressors, self.compressed_writer,
                indexer=self.selectors[0] if self.selectors else None,
                index_weight=self.index_weight if self.selectors else None,
                metadata=metadata, positions=self.indices.compressed(self.ratio).position,
                frequencies=self.rotary.frequencies[True],
                angles=self.compressed_angles.values,
                parallel_plans=(self.qkv.plan, self.qnorm.plan, self.kvnorm.plan,
                                self.qb.plan, self.window_writer.plan),
                session=session, workspace=workspace)
        self.attention = PreparedCompressedMLA(
            scratch.query,
            self.window_cache,
            self.window_indices,
            self.window_lengths,
            weights["sink"],
            scratch.attended,
            window_page_size=self.window_page_size,
            # Draft indices expose the complete noncausal block. The decode kernel follows
            # those explicit lists; its tiling does not impose a causal attention mask.
            mode="decode" if mode in ("decode", "verify") or mode == "draft" and tokens <= 32 else "extend",
            session=session,
            name=name + ".attention",
            workspace=workspace,
            cache_format="deepseek_v4",
            **indexed,
        )

        if self.concurrent_index is not None:
            self.concurrent_index.functional(session=session)
            self.concurrent_compression.functional(session=session)
        else:
            if self.ratio == 4:
                self.index_query.functional(session=session)
                self.index_weight.functional(session=session)
            if self.compression is not None:
                self.compression.functional(session=session)
        self.qkv.functional(session=session)
        self.qb.functional(session=session)
        self.qnorm.functional()
        self.kvnorm.functional()
        self.output_projection.functional(session=session)
        self.window_writer.functional(session=session, cache=None if mode != "decode" else self.window)
        self.attention.functional(session=session, window=None if mode != "decode" else self.window)
        if self.compressor is not None:
            self.compressor.functional(session=session)
            self.compressed_writer.functional(session=session)
        if self.index_compressor is not None:
            self.index_compressor.functional(session=session)
        for selector in self.selectors:
            selector.functional(session=session)
        register_finalizer(session, self._release_examples)
        if self.owns_indices:
            self.indices.functional()

    def _release_examples(self):
        self.window_cache = self.window_slots = self.window_indices = self.window_lengths = None
        self.compressed_slots = self.lengths = self.active_width = self.index_lengths = None
        self.compressed = self.indexed_indices = None
        if self.concurrent_index is not None:
            self.index_query = self.index_weight = self.compression = None

    def rotate(self, tensor, metadata, indices, *, compression=False, inverse=False):
        operation = self.compressed_angles if compression else self.angles
        if operation is not None:
            key = (True, self.ratio) if compression else (bool(self.ratio), 0)
            operation.apply(tensor, indices.angles[key], inverse=inverse)
        else:
            position = indices.ratios[self.ratio].position if compression else metadata.positions.clamp_min(0)
            self.rotary.apply(tensor, position, compressed=compression or bool(self.ratio), inverse=inverse)

    def __call__(self, source, metadata, indices=None):
        m = metadata
        if self.owns_indices:
            indices = self.indices(m)
        window = self.window if self.mode == "decode" else torch.empty(self.window_shape, dtype=torch.uint8, device=source.device)
        if self.mode != "decode":
            transfer_window(self.window, window, m, self.window_page_size)
        shared = indices.ratios[self.ratio] if self.ratio else None
        compressed_pending = None
        if self.concurrent_compression is not None:
            compressed_pending = self.concurrent_compression.start(source, m, shared.position, shared.emitted_slots,
                angles=indices.angles[(True, self.ratio)])
        projected = self.qkv(source)
        query_low = self.qnorm(projected[:, :1024])
        index_pending = self.concurrent_index.start(query_low) if self.concurrent_index is not None else None
        window_values = self.kvnorm(projected[:, 1024:])
        query = self.qb(query_low).view(source.shape[0], self.heads, 512)
        angles = None if self.angles is None else indices.angles[(bool(self.ratio), 0)]
        normalize_rotate(query, m.positions if angles is not None else m.positions.clamp_min(0),
            self.rotary.frequencies[bool(self.ratio)], self.eps, angles=angles)
        self.rotate(window_values, m, indices)
        self.window_writer(window_values, indices.window_slots, cache=window)
        compressed_records = index_records = None
        index_query = index_weights = None
        if compressed_pending is not None:
            index_query = self.concurrent_index.finish(index_pending, query).view(source.shape[0], 64, 128)
            self.rotate(index_query, m, indices)
            outputs = self.concurrent_compression.finish(compressed_pending, [query, window, index_query])
            projected = outputs[0]
            width = self.compressor.spec.state_shape[1]
            index_width = self.index_compressor.spec.state_shape[1]
            compressed_records = (projected[:, :width], projected[:, width:2*width])
            index_records = (projected[:, 2*width:2*width+index_width], projected[:, 2*width+index_width:])
            index_weights = outputs[-1]
        selected = indices.dense if self.ratio else None
        if self.ratio:
            if compressed_pending is None:
                projected = self.compression(source)
                width = self.compressor.spec.state_shape[1]
                latent, values, scores, _, slots = self.compressor(source, m, shared.slots,
                    projected=(projected[:, :width], projected[:, width:2*width]),
                    emitted=shared.emitted, emitted_slots=shared.emitted_slots)
                compressed_records = values, scores
                self.rotate(latent, m, indices, compression=True)
                self.compressed_writer(latent, slots)
            if self.index_compressor is not None:
                if compressed_pending is None:
                    offset = 2 * self.compressor.spec.state_shape[1]
                    width = self.index_compressor.spec.state_shape[1]
                    keys, values, scores, _, slots = self.index_compressor(source, m, shared.slots,
                        projected=(projected[:, offset:offset+width], projected[:, offset+width:offset+2*width]),
                        emitted=shared.emitted, emitted_slots=shared.emitted_slots)
                    index_records = values, scores
                    self.rotate(keys, m, indices, compression=True)
                    self.selectors[0].write_keys(keys, slots)
                    index_query = self.index_query(query_low).view(source.shape[0], 64, 128)
                    self.rotate(index_query, m, indices)
                    index_weights = self.index_weight(source)
                selected = torch.cat([selector(index_query[first:first+INDEX_ROWS], index_weights[first:first+INDEX_ROWS],
                    m.token_page_table[first:first+INDEX_ROWS], shared.lengths[first:first+INDEX_ROWS], shared.active_width)
                    for first, selector in zip(range(0, source.shape[0], INDEX_ROWS), self.selectors, strict=True)], dim=0)
                logical = selected.long()
                records_per_page = PAGE_TOKENS // 4
                physical = m.token_page_table.gather(1, (logical // records_per_page).clamp(0, m.page_table.shape[1] - 1))
                selected = torch.where((logical >= 0) & (physical >= 0),
                    physical * records_per_page + logical % records_per_page, -1).to(selected.dtype)
        attended = self.attention(query, indices.window_indices, indices.window_lengths,
            indexed_indices=selected, indexed_lengths=None if shared is None else shared.index_lengths, window=window)
        self.rotate(attended, m, indices, inverse=True)
        output = self.output_projection(attended.view(source.shape[0], -1, 4096))
        records = window, compressed_records, index_records
        if self.mode == "prefill":
            self.commit(m.lengths, records, m)
        return output, records

    def commit(self, counts, records, metadata):
        window, compressed, index = records
        transfer_window(window, self.window, metadata, self.window_page_size, counts=counts)
        if self.mode == "verify":
            if self.compressor is not None:
                self.compressor.commit(counts, compressed, metadata)
            if self.index_compressor is not None:
                self.index_compressor.commit(counts, index, metadata)
