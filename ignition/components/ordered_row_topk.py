# SPDX-License-Identifier: Apache-2.0

"""Prepared score selection with deterministic cutoff ties and input-position order."""

import torch
import triton
import triton.language as tl

from ignition.components.row_topk import PreparedRowTopK


_TILE = 512


@triton.jit
def _ordered_key(bits):
    bits = bits.to(tl.uint32)
    return tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)


@triton.jit
def _radix_cutoff_counts(Scores, Indices, Lengths, Cutoffs, Greater, Ties, OutScores, OutIndices,
                         WIDTH: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr,
                         BLOCK: tl.constexpr, ROW_BLOCK: tl.constexpr, BITS: tl.constexpr):
    """Resolve the K-th ordered key and per-tile counts in one program per row."""
    row = tl.program_id(0)
    col = tl.arange(0, ROW_BLOCK)
    value = tl.load(Scores + row * WIDTH + col, mask=col < WIDTH, other=0.0)
    length = tl.load(Lengths + row)
    kcol = tl.arange(0, K)
    if length > K:
        live = (col < length) & (col < WIDTH)
        bits = value.to(tl.uint32, bitcast=True)
        key = _ordered_key(bits)
        bins = tl.arange(0, 1 << BITS).to(tl.int64)
        remaining = length * 0 + K
        prefix = (length * 0).to(tl.uint32)
        resolved = length * 0
        for p in tl.static_range(32 // BITS):
            if resolved == 0:
                if p == 0:
                    match = live
                else:
                    match = live & ((key >> (32 - p * BITS)) == prefix)
                digit = ((key >> (32 - (p + 1) * BITS)) & ((1 << BITS) - 1)).to(tl.int32)
                hist = tl.histogram(digit, 1 << BITS, mask=match)
                at_least = tl.cumsum(hist, 0, reverse=True)
                packed = tl.max(tl.where(at_least >= remaining,
                                         (bins << 40) | ((at_least - hist).to(tl.int64) << 20) | hist.to(tl.int64),
                                         -1), 0)
                above = ((packed >> 20) & 0xFFFFF).to(tl.int32)
                remaining = remaining - above
                prefix = (prefix << BITS) | (packed >> 40).to(tl.uint32)
                if (packed & 0xFFFFF).to(tl.int32) == remaining:
                    resolved = length * 0 + (p + 1) * BITS
        if resolved == 0:
            resolved = length * 0 + 32
        lower = prefix << (32 - resolved).to(tl.uint32)
        cutoff = tl.min(tl.where(live & (key >= lower), key, 0xffffffff), 0)
        tl.store(Cutoffs + row, cutoff.to(tl.int32))
        greater = tl.sum(tl.reshape((live & (key > cutoff)).to(tl.int32), (ROW_BLOCK // BLOCK, BLOCK)), 1)
        ties = tl.sum(tl.reshape((live & (key == cutoff)).to(tl.int32), (ROW_BLOCK // BLOCK, BLOCK)), 1)
        tiles = tl.arange(0, ROW_BLOCK // BLOCK)
        tl.store(Greater + row * TILES + tiles, greater, mask=tiles < TILES)
        tl.store(Ties + row * TILES + tiles, ties, mask=tiles < TILES)
    else:
        underfull = (kcol < length) & (kcol < WIDTH)
        ids = tl.load(Indices + row * WIDTH + kcol, mask=underfull, other=-1)
        values = tl.load(Scores + row * WIDTH + kcol, mask=underfull, other=-float('inf'))
        tl.store(OutScores + row * K + kcol, values)
        tl.store(OutIndices + row * K + kcol, tl.maximum(ids, -1))


@triton.jit
def _cutoff(Selected, Lengths, Cutoffs, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cutoff = tl.full((), 0, tl.uint32)
    if tl.load(Lengths + row) > K:
        col = tl.arange(0, BLOCK)
        bits = tl.load(Selected + row * K + col, col < K, other=0)
        keys = tl.where(col < K, _ordered_key(bits), 0xffffffff)
        cutoff = tl.min(keys, 0)
    tl.store(Cutoffs + row, cutoff.to(tl.int32))


@triton.jit
def _counts(Scores, Lengths, Cutoffs, Greater, Ties,
            WIDTH: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    length = tl.load(Lengths + row)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    live = (length > K) & (col < length) & (col < WIDTH)
    bits = tl.load(Scores + row * WIDTH + col, live, other=0)
    cutoff = tl.load(Cutoffs + row).to(tl.uint32)
    keys = _ordered_key(bits)
    tl.store(Greater + row * TILES + tile, tl.sum((live & (keys > cutoff)).to(tl.int32), 0))
    tl.store(Ties + row * TILES + tile, tl.sum((live & (keys == cutoff)).to(tl.int32), 0))


@triton.jit
def _emit(Scores, Indices, Lengths, Cutoffs, Greater, Ties, OutScores, OutIndices,
          WIDTH: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr,
          BLOCK: tl.constexpr, COUNT_BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    length = tl.load(Lengths + row)
    if length > K:
        bins = tl.arange(0, COUNT_BLOCK)
        greater = tl.load(Greater + row * TILES + bins, bins < TILES, other=0)
        ties = tl.load(Ties + row * TILES + bins, bins < TILES, other=0)
        greater_before = tl.sum(tl.where(bins < tile, greater, 0), 0)
        ties_before = tl.sum(tl.where(bins < tile, ties, 0), 0)
        needed = K - tl.sum(greater, 0)
        cutoff = tl.load(Cutoffs + row).to(tl.uint32)
        col = tile * BLOCK + tl.arange(0, BLOCK)
        live = (col < length) & (col < WIDTH)
        bits = tl.load(Scores + row * WIDTH + col, live, other=0)
        keys = _ordered_key(bits)
        gt, eq = live & (keys > cutoff), live & (keys == cutoff)
        tie_prefix = ties_before + tl.cumsum(eq.to(tl.int32), 0)
        accepted = gt | (eq & (tie_prefix <= needed))
        prefix = greater_before + tl.minimum(ties_before, needed)
        destination = prefix + tl.cumsum(accepted.to(tl.int32), 0) - 1
        ids = tl.load(Indices + row * WIDTH + col, accepted, other=-1)
        # The selector cutoff guarantees G <= K <= G + E, hence exactly K
        # disjoint destinations. Negative IDs consume slots and map to -1.
        bounded = accepted & (destination >= 0) & (destination < K)
        tl.store(OutScores + row * K + destination, bits, bounded)
        tl.store(OutIndices + row * K + destination, tl.where(ids >= 0, ids, -1), bounded)


def validate_ordered_topk(scores, lengths, indices, output_scores, output_indices, cutoffs, greater, ties):
    if scores.ndim != 2 or output_indices.ndim != 2:
        raise ValueError('ordered row top-k requires score and output matrices')
    rows, width = scores.shape
    topk = output_indices.shape[1]
    if min(rows, width) <= 0 or topk not in (512, 1024, 2048) or max(rows * width, rows * topk) >= 2**31:
        raise ValueError('ordered row top-k requires supported positive int32 geometry')
    tiles = triton.cdiv(width, _TILE)
    expected = ((scores, (rows, width), torch.float32), (lengths, (rows,), torch.int32),
        (indices, (rows, width), torch.int32), (output_scores, (rows, topk), torch.float32),
        (output_indices, (rows, topk), torch.int32), (cutoffs, (rows,), torch.int32),
        (greater, (rows, tiles), torch.int32), (ties, (rows, tiles), torch.int32))
    for value, shape, dtype in expected:
        if value.shape != shape or value.dtype != dtype or value.device != scores.device or not value.is_contiguous():
            raise ValueError('ordered row top-k tensors require matching contiguous shapes, dtypes and devices')
    spans = [(value.untyped_storage()._cdata, value.storage_offset() * value.element_size(),
              value.numel() * value.element_size()) for value, _, _ in expected]
    for index, (storage, begin, size) in enumerate(spans[3:], 3):
        if any(storage == other and begin < offset + extent and offset < begin + size
               for other, offset, extent in spans[:index]):
            raise ValueError('ordered row top-k outputs and metadata must be disjoint from all live operands')


class PreparedOrderedRowTopK:
    """Select positions by FP32 bit-key rank, breaking cutoff ties by input position.

    Selected score bits and mapped IDs are emitted together in input order.
    Lengths must lie in [0,width]. IDs do not affect ranking; selected negative
    IDs map to -1. Signed zeros and NaN payloads retain the selector's ordered
    UInt32 keys. Rows with length <= K retain the selector's input-order copy
    and (-inf,-1) padding. Replicas agree when their input bytes are identical.

    The caller owns all metadata through ``buffers``. Serial invocations may
    reuse it; overlapping invocations require separate storage.
    """

    def __init__(self, scores, lengths, indices, output_scores, output_indices, *, session, name, buffers, radix=None):
        rows, width = scores.shape
        self.rows, self.width, self.topk = rows, width, output_indices.shape[1]
        self.radix = width <= 32768 if radix is None else radix
        if type(self.radix) is not bool or self.radix and width > 32768:
            raise ValueError("radix ordered top-k requires widths through 32768")
        self.tiles = triton.cdiv(width, _TILE)
        self.cutoffs = buffers.zeros((rows,), torch.int32)
        self.greater = buffers.zeros((rows, self.tiles), torch.int32)
        self.ties = buffers.zeros((rows, self.tiles), torch.int32)
        validate_ordered_topk(scores, lengths, indices, output_scores, output_indices,
                              self.cutoffs, self.greater, self.ties)
        self.scores, self.lengths, self.indices = scores, lengths, indices
        self.output_scores, self.output = output_scores, output_indices
        self.selection = PreparedRowTopK(scores, lengths, indices, output_scores, output_indices,
                                         session=session, name=name)
        self.plan = self.selection.plan

    def functional(self, *, session):
        self.selection.functional(session=session)
        self.scores = self.lengths = self.indices = self.output_scores = self.output = None
        self.cutoffs = self.greater = self.ties = None
        return self

    def __call__(self, scores=None, lengths=None, indices=None):
        return self._run(scores, lengths, indices)[0]

    def _run(self, scores=None, lengths=None, indices=None):
        if scores is None:
            scores, lengths, indices = self.scores, self.lengths, self.indices
            output_scores, output = self.output_scores, self.output
            cutoffs, greater, ties = self.cutoffs, self.greater, self.ties
        else:
            output_scores = torch.empty((self.rows, self.topk), dtype=torch.float32, device=scores.device)
            output = torch.empty_like(output_scores, dtype=torch.int32)
            cutoffs = torch.empty((self.rows,), dtype=torch.int32, device=scores.device)
            greater = torch.empty((self.rows, self.tiles), dtype=torch.int32, device=scores.device)
            ties = torch.empty_like(greater)
        if self.radix:
            torch.library.wrap_triton(_radix_cutoff_counts)[(self.rows,)](
                scores, indices, lengths, cutoffs, greater, ties, output_scores, output,
                WIDTH=self.width, K=self.topk, TILES=self.tiles, BLOCK=_TILE,
                ROW_BLOCK=triton.next_power_of_2(self.width), BITS=8, num_warps=16)
        else:
            self.selection(scores, lengths, indices, output_scores, output)
            torch.library.wrap_triton(_cutoff)[(self.rows,)](
                output_scores.view(torch.int32), lengths, cutoffs,
                K=self.topk, BLOCK=triton.next_power_of_2(self.topk), num_warps=4)
            torch.library.wrap_triton(_counts)[(self.rows, self.tiles)](
                scores.view(torch.int32), lengths, cutoffs, greater, ties,
                WIDTH=self.width, K=self.topk, TILES=self.tiles, BLOCK=_TILE, num_warps=4)
        torch.library.wrap_triton(_emit)[(self.rows, self.tiles)](
            scores.view(torch.int32), indices, lengths, cutoffs, greater, ties,
            output_scores.view(torch.int32), output, WIDTH=self.width, K=self.topk,
            TILES=self.tiles, BLOCK=_TILE, COUNT_BLOCK=triton.next_power_of_2(self.tiles), num_warps=4)
        return output, output_scores
