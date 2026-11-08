# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unpack candidate pairs, compacting only ordinary rows that fit the selection budget."""

import torch
import triton
import triton.language as tl


_BLOCK = 512


@triton.jit
def _count(Packed, Active, Counts, Exceptional,
           WIDTH: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    present = (column < WIDTH) & (tl.load(Active + row) != 0)
    offset = row * WIDTH * 2 + column * 2
    index = tl.load(Packed.to(tl.pointer_type(tl.int32)) + offset + 1, mask=present, other=-1)
    value = tl.load(Packed + offset, mask=present, other=-float("inf"))
    valid = present & (index >= 0)
    nonfinite = (value != value) | (tl.abs(value) == float("inf"))
    malformed_padding = present & (index < 0) & ((index != -1) | (value != -float("inf")))
    tl.store(Counts + row * TILES + tile, tl.sum(valid.to(tl.int32), 0))
    tl.store(Exceptional + row * TILES + tile,
             tl.sum(((valid & nonfinite) | malformed_padding).to(tl.int32), 0))


@triton.jit
def _unpack(Packed, Active, Counts, Exceptional, Values, Indices, Lengths,
            WIDTH: tl.constexpr, TILES: tl.constexpr, TOPK: tl.constexpr,
            BLOCK: tl.constexpr, COUNT_BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    active = tl.load(Active + row) != 0
    count_column = tl.arange(0, COUNT_BLOCK)
    counts = tl.load(Counts + row * TILES + count_column, mask=count_column < TILES, other=0)
    exceptional = tl.load(Exceptional + row * TILES + count_column,
                          mask=count_column < TILES, other=0)
    total = tl.sum(counts, 0)
    compact = (total <= TOPK) & (tl.sum(exceptional, 0) == 0)
    present = (column < WIDTH) & active
    offset = row * WIDTH * 2 + column * 2
    value = tl.load(Packed + offset, mask=present, other=-float("inf"))
    index = tl.load(Packed.to(tl.pointer_type(tl.int32)) + offset + 1, mask=present, other=-1)
    if active & compact:
        valid = present & (index >= 0)
        prefix = tl.sum(tl.where(count_column < tile, counts, 0), 0)
        destination = prefix + tl.cumsum(valid.to(tl.int32), 0) - 1
        tl.store(Values + row * WIDTH + destination, value, mask=valid)
        tl.store(Indices + row * WIDTH + destination, index, mask=valid)
        # Compacted stores end before total; tail initialization cannot race them.
        tl.store(Values + row * WIDTH + column, -float("inf"), mask=(column < WIDTH) & (column >= total))
        tl.store(Indices + row * WIDTH + column, -1, mask=(column < WIDTH) & (column >= total))
    else:
        tl.store(Values + row * WIDTH + column, value, mask=column < WIDTH)
        tl.store(Indices + row * WIDTH + column, index, mask=column < WIDTH)
    if tile == 0:
        tl.store(Lengths + row, tl.where(active, tl.where(compact, total, WIDTH), 0))


def validate_candidate_rows(packed, active, values, indices, lengths, counts, exceptional, *, topk):
    """Validate caller-owned data and metadata without reading their contents."""
    if packed.ndim != 3 or packed.shape[2] != 2:
        raise ValueError("candidate input must contain rows of FP32 score and int32 ID pairs")
    rows, width, _ = packed.shape
    if (min(rows, width) <= 0 or rows * width * 2 >= 2**31
            or type(topk) is not int or not 0 < topk <= width):
        raise ValueError("candidate dimensions and selection budget must fit positive int32 extents")
    expected = (("packed", packed, (rows, width, 2), torch.float32),
                ("active", active, (rows,), torch.bool),
                ("values", values, (rows, width), torch.float32),
                ("indices", indices, (rows, width), torch.int32),
                ("lengths", lengths, (rows,), torch.int32),
                ("counts", counts, (rows, triton.cdiv(width, _BLOCK)), torch.int32),
                ("exceptional", exceptional, (rows, triton.cdiv(width, _BLOCK)), torch.int32))
    for name, tensor, shape, dtype in expected:
        if tuple(tensor.shape) != shape or tensor.device != packed.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous with shape {shape} on {packed.device}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")

    def span(tensor):
        return (tensor.untyped_storage()._cdata, tensor.storage_offset() * tensor.element_size(),
                tensor.numel() * tensor.element_size())

    # Storage identities also describe aliases during metadata-only compilation.
    spans = [(name, *span(tensor)) for name, tensor, _, _ in expected]
    for index, (name, storage, begin, size) in enumerate(spans[2:], 2):
        if any(storage == other and begin < offset + extent and offset < begin + size
               for _, other, offset, extent in spans[:index]):
            raise ValueError(f"candidate output or scratch {name} overlaps another operand")


class PreparedCandidateRows:
    """Retain the selector input for overfull or exceptional rows.

    Valid IDs are nonnegative; ordinary padding is the pair (-inf, -1). A row
    with at most topk valid, finite scores is compacted in input order. Other
    active rows are copied bit-for-bit at their original width, including ties
    and nonfinite scores. Inactive rows have zero length and padding outputs.
    Compaction preserves the selected set, not the selector's atomic emission
    order. Attention reductions and blockwise probability encodings can depend
    on that order and require numerical qualification.
    """

    def __init__(self, packed, active, values, indices, lengths, *, topk, buffers):
        if packed.ndim != 3:
            raise ValueError("candidate input must have three dimensions")
        rows, width = packed.shape[:2]
        tiles = triton.cdiv(width, _BLOCK)
        self.counts = buffers.zeros((rows, tiles), torch.int32)
        self.exceptional = buffers.zeros((rows, tiles), torch.int32)
        validate_candidate_rows(packed, active, values, indices, lengths,
                                self.counts, self.exceptional, topk=topk)
        self.packed, self.active = packed, active
        self.values, self.indices, self.lengths = values, indices, lengths
        self.rows, self.width, self.tiles, self.topk = rows, width, tiles, topk

    def __call__(self):
        grid = (self.rows, self.tiles)
        torch.library.wrap_triton(_count)[grid](self.packed, self.active, self.counts, self.exceptional,
            WIDTH=self.width, TILES=self.tiles, BLOCK=_BLOCK, num_warps=4)
        torch.library.wrap_triton(_unpack)[grid](self.packed, self.active, self.counts, self.exceptional,
            self.values, self.indices, self.lengths, WIDTH=self.width, TILES=self.tiles,
            TOPK=self.topk, BLOCK=_BLOCK, COUNT_BLOCK=triton.next_power_of_2(self.tiles), num_warps=4)
