# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Aligned slot storage for immutable continuation states and live model state."""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class StateRegion:
    name: str
    offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True)
class StateLayout:
    regions: tuple[StateRegion, ...]
    row_bytes: int

    @classmethod
    def from_specs(cls, specifications):
        regions, names, offset = [], set(), 0
        for name, shape, dtype in specifications:
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("state region names must be nonempty and unique")
            shape = tuple(shape)
            if not shape or any(type(value) is not int or value <= 0 for value in shape):
                raise ValueError("state region extents must be positive integers")
            if not isinstance(dtype, torch.dtype):
                raise TypeError("state regions require a tensor dtype")
            offset = (offset + 255) // 256 * 256
            region = StateRegion(name, offset, shape, dtype)
            regions.append(region)
            names.add(name)
            offset += region.nbytes
        if not regions:
            raise ValueError("state storage requires at least one region")
        return cls(tuple(regions), (offset + 255) // 256 * 256)

    def region(self, name):
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)

    def snapshot_capacity(self, budget_bytes=4 << 30):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("snapshot budget must be a nonnegative byte count")
        return budget_bytes // self.row_bytes


@dataclass(frozen=True)
class StateStorage:
    backing: torch.Tensor
    layout: StateLayout
    memory_capacity: torch.Tensor

    @classmethod
    def allocate(cls, layout, slots, device, *, storage=None, maximum_slots=None, backed_slots=None):
        if type(slots) is not int or slots < 2:
            raise ValueError("state storage requires null and live slots")
        backed_slots = slots if backed_slots is None else backed_slots
        if type(backed_slots) is not int or not 2 <= backed_slots <= slots or (backed_slots != slots and storage is None):
            raise ValueError("partial state backing requires dynamic storage and two through all declared slots")
        if maximum_slots is None and storage is not None:
            maximum_slots = torch.cuda.mem_get_info(device)[1] // layout.row_bytes
        if maximum_slots is not None and (type(maximum_slots) is not int or maximum_slots < slots):
            raise ValueError("state virtual extent must cover prepared slots")
        backing = (torch.zeros((slots, layout.row_bytes), dtype=torch.uint8, device=device) if storage is None else
            storage.device_allocate("continuations", (slots, layout.row_bytes), torch.uint8, group="STATE_ROWS",
                                    maximum=maximum_slots, units=backed_slots))
        return cls(backing, layout, torch.tensor([backed_slots, 0], dtype=torch.int64, device=device))

    def region(self, name):
        region = self.layout.region(name)
        return self.backing[:, region.offset:region.offset + region.nbytes].view(
            region.dtype).view(self.backing.shape[0], *region.shape)


def gather_rows(source, rows, destination):
    """Read selected rows without materializing the full strided state view."""
    source = source.view(source.shape[0], -1)
    output = destination.view(destination.shape[0], -1)
    if source.is_cuda:
        torch.library.wrap_triton(_gather_rows)[(output.shape[0], triton.cdiv(output.shape[1], 1024))](
            source, rows, output, source.stride(0), output.stride(0), rows.stride(0), output.shape[1], 1024)
    else:
        output.copy_(source[rows, :output.shape[1]])
    return destination


@triton.jit
def _gather_rows(Source, Rows, Output, SS: tl.constexpr, OS: tl.constexpr,
                 RS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    selected = tl.load(Rows + row * RS)
    value = tl.load(Source + selected * SS + columns, columns < WIDTH, other=0)
    tl.store(Output + row * OS + columns, value, columns < WIDTH)


def gather_rows_three(sources, rows, destinations):
    """Read three typed state fields with one row map and separate destinations."""
    first, second, third = (value.view(value.shape[0], -1) for value in sources)
    out_first, out_second, out_third = (value.view(value.shape[0], -1) for value in destinations)
    if first.is_cuda:
        width = max(out_first.shape[1], out_second.shape[1], out_third.shape[1])
        torch.library.wrap_triton(_gather_rows_three)[(out_first.shape[0], triton.cdiv(width, 1024))](
            first, second, third, rows, out_first, out_second, out_third,
            first.stride(0), second.stride(0), third.stride(0),
            out_first.stride(0), out_second.stride(0), out_third.stride(0), rows.stride(0),
            out_first.shape[1], out_second.shape[1], out_third.shape[1], 1024)
    else:
        for source, output in zip((first, second, third), (out_first, out_second, out_third), strict=True):
            gather_rows(source, rows, output)


@triton.jit
def _gather_rows_three(First, Second, Third, Rows, OutFirst, OutSecond, OutThird,
                       FS: tl.constexpr, SS: tl.constexpr, TS: tl.constexpr,
                       FO: tl.constexpr, SO: tl.constexpr, TO: tl.constexpr, RS: tl.constexpr,
                       FW: tl.constexpr, SW: tl.constexpr, TW: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    selected = tl.load(Rows + row * RS)
    first = tl.load(First + selected * FS + columns, columns < FW, other=0)
    second = tl.load(Second + selected * SS + columns, columns < SW, other=0)
    third = tl.load(Third + selected * TS + columns, columns < TW, other=0)
    tl.store(OutFirst + row * FO + columns, first, columns < FW)
    tl.store(OutSecond + row * SO + columns, second, columns < SW)
    tl.store(OutThird + row * TO + columns, third, columns < TW)


def save_hidden(source, destination, slots, active):
    """Save packed final hidden outputs to the requests' continuation rows.

    Admission must reject duplicate active rows before forward execution.
    Inactive destinations retain their bytes.
    """
    if destination.is_cuda:
        torch.library.wrap_triton(_save_hidden)[(source.shape[0],)](source, destination, slots, active, source.stride(0),
            destination.stride(0), source.shape[1], triton.next_power_of_2(source.shape[1]))
    else:
        selected = active.bool()
        destination[slots[selected]] = source[selected]


def scatter_rows(source, rows, destination, active):
    """Store active rows at unique request slots, preserving all other rows."""
    source = source.view(source.shape[0], -1)
    output = destination.view(destination.shape[0], -1)
    if source.is_cuda:
        torch.library.wrap_triton(_scatter_rows)[(source.shape[0], triton.cdiv(source.shape[1], 1024))](
            source, rows, output, active, source.stride(0), source.stride(1), output.stride(0),
            source.shape[1], output.shape[0], 1024)
    else:
        selected = active.bool() & (rows >= 0) & (rows < output.shape[0])
        output[rows[selected], :source.shape[1]] = source[selected]


@triton.jit
def _scatter_rows(Source, Rows, Output, Active, SS: tl.constexpr, CS: tl.constexpr,
                  OS: tl.constexpr, WIDTH: tl.constexpr, SLOTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    selected = tl.load(Rows + row)
    active = (tl.load(Active + row) != 0) & (selected >= 0) & (selected < SLOTS)
    values = tl.load(Source + row * SS + columns * CS, active & (columns < WIDTH), other=0)
    tl.store(Output + selected * OS + columns, values, active & (columns < WIDTH))


@triton.jit
def _save_hidden(Source, Destination, Slots, Active, SS: tl.constexpr, DS: tl.constexpr,
                 WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(Slots + row)
    active = (tl.load(Active + row) != 0) & (slot > 0)
    columns = tl.arange(0, BLOCK)
    value = tl.load(Source + row * SS + columns, active & (columns < WIDTH), other=0)
    tl.store(Destination + slot * DS + columns, value, active & (columns < WIDTH))


class PreparedRequestHistory:
    """Transfer request-local counters and token history at continuation boundaries."""

    def __init__(self, pool, requests, commands, *, snapshot):
        from .sampling import _compile

        history = getattr(requests, "ple_history", None)
        saved_history = pool.region("ple_history") if history is not None else None
        self.args = (requests.consumed_tokens, history, pool.region("consumed_tokens"), saved_history,
            commands.request_slots, commands.source_slots, commands.destination_slots,
            commands.num_ops, commands.errors, pool.memory_capacity, requests.capacity,
            requests.consumed_tokens.stride(0), pool.region("consumed_tokens").stride(0),
            0 if history is None else history.stride(0),
            0 if saved_history is None else saved_history.stride(0), history is not None, snapshot)
        self.run = _compile(_request_history, self.args, (requests.capacity, 1, 1))

    def __call__(self):
        self.run()


@triton.jit
def _request_history(Counts, History, SavedCounts, SavedHistory, Requests, Sources, Destinations,
                     Count, Errors, MemoryCapacity, REQUESTS: tl.constexpr,
                     CS: tl.constexpr, SS: tl.constexpr, HS: tl.constexpr, SHS: tl.constexpr,
                     HAS_HISTORY: tl.constexpr, SNAPSHOT: tl.constexpr):
    command = tl.program_id(0)
    slot = tl.load(Requests + command)
    row = tl.load(Sources + command) if SNAPSHOT else tl.load(Destinations + command)
    valid = (command < tl.load(Count)) & (tl.load(Errors + command) == 0)
    valid &= (slot >= 0) & (slot < REQUESTS) & (row > 0) & (row < tl.load(MemoryCapacity))
    if SNAPSHOT:
        value = tl.load(Counts + slot * CS, valid, other=0)
        tl.store(SavedCounts + row * SS, value, valid)
    else:
        value = tl.load(SavedCounts + row * SS, valid, other=0)
        tl.store(Counts + slot * CS, value, valid)
    if HAS_HISTORY:
        columns = tl.arange(0, 2)
        if SNAPSHOT:
            value = tl.load(History + slot * HS + columns, valid, other=0)
            tl.store(SavedHistory + row * SHS + columns, value, valid)
        else:
            value = tl.load(SavedHistory + row * SHS + columns, valid, other=0)
            tl.store(History + slot * HS + columns, value, valid)
