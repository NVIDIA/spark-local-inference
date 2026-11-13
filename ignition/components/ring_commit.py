# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Commit accepted token prefixes from packed staging into multiple byte rings."""

import torch
import triton
import triton.language as tl

from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.persistent import persistent_copy


@triton.jit
def _commit(Staged, Backing, Offsets, Rows, Starts, Lengths, ROW_STRIDE: tl.constexpr,
            STAGED_STRIDE: tl.constexpr, REGION_STRIDE: tl.constexpr,
            ROWS_STRIDE: tl.constexpr, STARTS_STRIDE: tl.constexpr, LENGTHS_STRIDE: tl.constexpr,
            WINDOW: tl.constexpr, RECORD: tl.constexpr, BLOCK: tl.constexpr):
    item, request, region = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(Lengths + request * LENGTHS_STRIDE)
    start, row = tl.load(Starts + request * STARTS_STRIDE), tl.load(Rows + request * ROWS_STRIDE)
    token = tl.maximum(0, length - WINDOW) + item
    byte = tl.arange(0, BLOCK)
    live = (token < length) & (row > 0) & (byte < RECORD)
    value = tl.load(Staged + region * REGION_STRIDE + request * STAGED_STRIDE
                    + (WINDOW + token) * RECORD + byte, live, other=0)
    offset = tl.load(Offsets + region)
    tl.store(Backing + row * ROW_STRIDE + offset + ((start + token) % WINDOW) * RECORD + byte,
             value, live)


class PreparedRingCommit:
    """Commit per-region staging produced inside a speculative forward.

    Admission validates distinct positive destination rows and bounds accepted
    counts by token_capacity. Row zero and unaccepted ring records are preserved.
    The complete backing tensor exposes mutations to forward compilation.
    """

    def __init__(self, backing, offsets, rows, starts, *, token_capacity, window_tokens, record_bytes, staging=None):
        if (backing.device.type != "cuda" or backing.dtype != torch.uint8 or backing.ndim != 2
                or backing.stride(1) != 1 or backing.stride(0) < backing.shape[1]
                or rows.ndim != 1 or rows.numel() == 0 or starts.shape != rows.shape
                or rows.dtype != torch.int64 or starts.dtype != torch.int64
                or any(value.device != backing.device for value in (rows, starts))):
            raise ValueError("ring commits require byte rows and matching int64 row/position vectors")
        if any(type(value) is not int or value <= 0 for value in (token_capacity, window_tokens, record_bytes)):
            raise ValueError("ring and staging capacities must be positive integers")
        offsets = tuple(offsets)
        size = window_tokens * record_bytes
        if (not offsets or any(type(offset) is not int or offset < 0 or offset + size > backing.shape[1]
                               for offset in offsets)
                or any(first + size > second for first, second in zip(sorted(offsets), sorted(offsets)[1:]))):
            raise ValueError("ring regions must be distinct non-overlapping fields within each backing row")
        self.backing, self.rows, self.starts = backing, rows, starts
        self.offsets = persistent_copy(torch.tensor(offsets, dtype=torch.int64, device=backing.device))
        record_numeric_tensor(self.offsets)
        from ignition.preparation.scratch import bind_scratch

        shape = (len(offsets), rows.numel(), (window_tokens + token_capacity) * record_bytes)
        self.staging_shape, self.rows_shape = shape, tuple(rows.shape)
        self.staging = bind_scratch({"staging": (shape, torch.uint8)}, device=backing.device,
            workspace=None if staging is None else {"staging": staging})["staging"]
        self.staging.zero_()
        self.window_tokens, self.record_bytes = window_tokens, record_bytes
        self.grid = (min(token_capacity, window_tokens), rows.numel(), len(offsets))

    def functional(self):
        self.staging = self.rows = self.starts = None
        return self

    def __call__(self, counts, *, staging, rows, starts):
        if counts.shape != self.rows_shape or counts.dtype not in (torch.int32, torch.int64) or counts.device != self.backing.device:
            raise ValueError("accepted counts must match the ring commit's request rows")
        torch.library.wrap_triton(_commit)[self.grid](
            staging, self.backing, self.offsets, rows, starts, counts,
            self.backing.stride(0), staging.stride(1), staging.stride(0),
            rows.stride(0), starts.stride(0), counts.stride(0),
            self.window_tokens, self.record_bytes, triton.next_power_of_2(self.record_bytes))
