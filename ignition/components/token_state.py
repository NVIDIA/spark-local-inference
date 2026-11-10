# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed token consumption and optional previous-token context for captured forwards."""

import torch
import triton
import triton.language as tl

from ignition.components.sampling import DeviceRequestState, _check_tensor, _compile


@triton.jit
def _gather(Slots, Epochs, Live, Generation, Status, Pending, Consumed, History,
            PackedPending, PackedPositions, PackedHistory, PackedActive, ROWS: tl.constexpr, CS: tl.constexpr, HS: tl.constexpr, HISTORY: tl.constexpr, PAD: tl.constexpr, SLOTS: tl.constexpr = 4):
    row = tl.program_id(0)
    slot = tl.load(Slots + row)
    valid = (row < tl.load(Live)) & (slot >= 0) & (slot < SLOTS)
    valid &= tl.load(Generation + slot, valid, other=-1) == tl.load(Epochs + row)
    valid &= tl.load(Status + slot, valid, other=0) == 1
    tl.store(PackedPending + row, tl.load(Pending + slot, valid, other=-1))
    tl.store(PackedPositions + row, tl.load(Consumed + slot * CS, valid, other=0))
    tl.store(PackedActive + row, valid.to(tl.int32))
    if HISTORY:
        history_index = tl.arange(0, 2)
        history = tl.load(History + slot * HS + history_index, valid, other=PAD)
        tl.store(PackedHistory + row * 2 + history_index, history)


@triton.jit
def _consume(Tokens, CuSeq, Slots, Epochs, Live, Generation, Status, Error, Pending, Consumed,
             History, ROWS: tl.constexpr, CAPACITY: tl.constexpr, CS: tl.constexpr, HS: tl.constexpr, HISTORY: tl.constexpr, SLOTS: tl.constexpr = 4):
    slot = tl.program_id(0)
    rows = tl.arange(0, SLOTS)
    matches = (rows < ROWS) & (rows < tl.load(Live))
    matches &= tl.load(Slots + rows, rows < ROWS, other=-1) == slot
    matches &= tl.load(Epochs + rows, rows < ROWS, other=-1) == tl.load(Generation + slot)
    count = tl.sum(matches.to(tl.int32), 0)
    if (count > 0) & (tl.load(Status + slot) == 1):
        if count > 1:
            tl.store(Status + slot, 3)
            tl.store(Error + slot, 4)
        else:
            row = tl.max(tl.where(matches, rows, 0), 0)
            start, end = tl.load(CuSeq + row), tl.load(CuSeq + row + 1)
            length = end - start
            pending = tl.load(Pending + slot)
            first = tl.load(Tokens + start, (start >= 0) & (start < CAPACITY), other=-1)
            valid = (start >= 0) & (end <= CAPACITY) & (end >= start)
            valid &= (length == 0) | (pending < 0) | ((length == 1) & (first == pending))
            if not valid:
                tl.store(Status + slot, 3)
                tl.store(Error + slot, 3)
            elif length > 0:
                consumed = tl.load(Consumed + slot * CS)
                if HISTORY:
                    old_last = tl.load(History + slot * HS + 1)
                    last = tl.load(Tokens + end - 1)
                    previous = tl.load(Tokens + end - 2, length >= 2, other=0)
                    previous = tl.where(length >= 2, previous, old_last)
                    tl.store(History + slot * HS, previous)
                    tl.store(History + slot * HS + 1, last)
                final = consumed + length
                tl.store(Consumed + slot * CS, final)
                tl.store(Pending + slot, -1)


class PreparedTokenState:
    """Gather packed decode inputs and commit consumed prefill/decode spans.

    Call ``gather`` before forward and ``consume`` after forward. ``token_ids``
    is packed by ``cu_seqlens``; each live row must refer to a unique physical
    slot and its epoch. A pending sample can only be consumed as a one-token
    span containing that sample. Zero-token spans do not change state. Neither
    operation changes penalty counts or RNG state. Recipes may supply a strided
    two-token history for previous-token embeddings. Without history, these
    kernels only gather and commit counters, pending tokens, and active flags.
    """

    def __init__(self, state: DeviceRequestState, token_ids, cu_seqlens, row_slot_ids, row_generations, live_rows,
                 *, history: torch.Tensor | None = None, history_pad_id: int = 0, buffers=None):
        rows, device = row_slot_ids.numel(), state.generation.device
        if device.type != "cuda" or not 1 <= rows <= state.capacity or token_ids.ndim != 1 or not token_ids.numel():
            raise ValueError("token state requires CUDA storage, rows within request capacity and a nonempty packed token capacity")
        for tensor, shape, dtype, name in ((row_slot_ids, (rows,), torch.int32, "row_slot_ids"),
                (row_generations, (rows,), torch.int64, "row_generations"), (live_rows, (1,), torch.int32, "live_rows"),
                (token_ids, (token_ids.numel(),), torch.int64, "token_ids"),
                (cu_seqlens, (rows + 1,), torch.int32, "cu_seqlens")):
            _check_tensor(tensor, shape, dtype, device, name)
        if history is not None:
            _check_tensor(history, (state.capacity, 2), torch.int64, device, "history", contiguous=False)
            if history.stride(0) <= 0 or history.stride(1) != 1:
                raise ValueError("history requires positive row stride and contiguous columns")
        self.state = state
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)
        self.pending_tokens = empty((rows,), torch.int64)
        self.positions = empty((rows,), torch.int64)
        self.history = None if history is None else empty((rows, 2), torch.int64)
        self.active = empty((rows,), torch.int32)
        self._gather_args = (row_slot_ids, row_generations, live_rows, state.generation, state.status,
            state.pending_token, state.consumed_tokens, history,
            self.pending_tokens, self.positions, self.history, self.active, rows,
            state.consumed_tokens.stride(0), 0 if history is None else history.stride(0), history is not None, history_pad_id, state.capacity)
        self._consume_args = (token_ids, cu_seqlens, row_slot_ids, row_generations, live_rows, state.generation,
            state.status, state.error_code, state.pending_token, state.consumed_tokens,
            history, rows, token_ids.numel(),
            state.consumed_tokens.stride(0), 0 if history is None else history.stride(0), history is not None, state.capacity)
        self._rows = rows
        self._token_capacity = token_ids.numel()
        self._prepared = False

    def prepare(self) -> None:
        self._gather = _compile(_gather, self._gather_args, (self._rows, 1, 1))
        self._consume = _compile(_consume, self._consume_args, (self.state.capacity, 1, 1))
        self._prepared = True

    def functional(self):
        """Keep prepared kernels while supplying packed storage at each invocation."""
        if not self._prepared:
            raise RuntimeError("token state prepare() must complete before functional execution")
        if self._gather_args is not None:
            self._gather_inputs, self._gather_constants = self._gather_args[:8], self._gather_args[12:]
            self._consume_inputs = self._consume_args[2:]
            self._gather_args = self._consume_args = None
            self._gather.args = self._consume.args = None
            self.pending_tokens = self.positions = self.history = self.active = None
        return self

    def gather(self):
        if not self._prepared:
            raise RuntimeError("token state prepare() must complete before capture")
        if self._gather_args is None:
            from ignition.preparation.triton import launch_triton

            device = self.state.generation.device
            pending = torch.empty((self._rows,), dtype=torch.int64, device=device)
            positions = torch.empty_like(pending)
            history = (torch.empty((self._rows, 2), dtype=torch.int64, device=device)
                       if self._gather_constants[3] else None)
            active = torch.empty((self._rows,), dtype=torch.int32, device=device)
            args = (*self._gather_inputs, pending, positions, history, active, *self._gather_constants)
            launch_triton(self._gather.kernel, self._gather.compiled, args, self._gather.grid,
                          **self._gather.options)
            return pending, positions, history, active
        self._gather()

    def consume(self, token_ids=None, cu_seqlens=None) -> None:
        if not self._prepared:
            raise RuntimeError("token state prepare() must complete before capture")
        if self._consume_args is None:
            from ignition.preparation.triton import launch_triton

            if token_ids is None or cu_seqlens is None:
                raise ValueError("functional token consumption requires tokens and packed row boundaries")
            device = self.state.generation.device
            _check_tensor(token_ids, (self._token_capacity,), torch.int64, device, "token_ids")
            _check_tensor(cu_seqlens, (self._rows + 1,), torch.int32, device, "cu_seqlens")
            args = (token_ids, cu_seqlens, *self._consume_inputs)
            launch_triton(self._consume.kernel, self._consume.compiled, args, self._consume.grid,
                          **self._consume.options)
            return
        if token_ids is not None or cu_seqlens is not None:
            raise ValueError("runtime token operands require functional token state")
        self._consume()
