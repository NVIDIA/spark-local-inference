# SPDX-License-Identifier: Apache-2.0

"""Copy interleaved paged records through fixed caller-owned peer buffers."""

import torch
import triton
import triton.language as tl

from ignition.components.collectives import PreparedAllReduce


@triton.jit
def _pack(Cache, Table, End, Rank, Completed, Stage, Signal,
          COLUMNS: tl.constexpr, CACHE_PAGES: tl.constexpr, PAGE: tl.constexpr,
          WORLD: tl.constexpr, MAXIMUM: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    words = tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    local_end = tl.maximum(end - tl.load(Rank) + WORLD - 1, 0) // WORLD
    first = tl.program_id(0).to(tl.int64) * 4
    if tl.load(Completed) == 0:
        while first < local_end:
            row = (first + tl.arange(0, 4)).to(tl.int64)
            live = row < local_end
            column = row // PAGE
            page = tl.load(Table + column, mask=live & (column < COLUMNS), other=-1).to(tl.int64)
            valid = live & (page >= 0) & (page < CACHE_PAGES)
            data = tl.load(Cache + (page[:, None] * PAGE + row[:, None] % PAGE) * WORDS + words[None, :],
                           mask=valid[:, None] & (words[None, :] < WORDS), other=0)
            tl.store(Stage + row[:, None] * WORDS + words[None, :], data,
                     mask=live[:, None] & (words[None, :] < WORDS))
            first += tl.num_programs(0) * 4
    if tl.program_id(0) == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


@triton.jit
def _unpack(Addresses, End, Ready, Output, Signal,
            WORLD: tl.constexpr, MAXIMUM: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    words = tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(End), 0), MAXIMUM).to(tl.int64)
    first = tl.program_id(0).to(tl.int64) * 8
    if tl.load(Ready) == 0:
        while first < end:
            row = (first + tl.arange(0, 8)).to(tl.int64)
            live = row < end
            source = tl.load(Addresses + row % WORLD).to(tl.pointer_type(tl.int32))
            data = tl.load(source[:, None] + (row[:, None] // WORLD) * WORDS + words[None, :],
                           mask=live[:, None] & (words[None, :] < WORDS), other=0)
            tl.store(Output + row[:, None] * WORDS + words[None, :], data,
                     mask=live[:, None] & (words[None, :] < WORDS))
            first += tl.num_programs(0) * 8
    if tl.program_id(0) == 0:
        tl.store(Signal + tl.arange(0, 8), 0)


def validate_history(cache, table, end, rank, stage, addresses, output, signals):
    """Check transfer geometry without inspecting device values."""
    if cache.ndim != 3 or output.ndim != 2 or stage.ndim != 2:
        raise ValueError("history transfer requires paged source and matrix staging/output")
    pages, page, width = cache.shape
    maximum, record = output.shape
    world = addresses.numel()
    if (min(pages, page, width, maximum, world) <= 0 or width != record or width % 16
            or maximum >= 2**31 or maximum % world or stage.shape != (maximum // world, width)):
        raise ValueError("history transfer requires whole interleaved records and 16-byte widths")
    if cache.dtype not in (torch.uint8, torch.float8_e4m3fn) or stage.dtype != torch.uint8 or output.dtype != cache.dtype:
        raise TypeError("history transfer preserves uint8 or E4M3 bytes through uint8 staging")
    if any(value.storage_offset() % 4 for value in (cache, stage, output)):
        raise ValueError("history records require word-aligned storage offsets")
    if (table.ndim != 1 or table.numel() * page < maximum // world or table.dtype != torch.int32
            or end.shape != (1,) or end.dtype != torch.int64 or rank.shape != (1,) or rank.dtype != torch.int64
            or addresses.ndim != 1 or addresses.dtype != torch.int64):
        raise ValueError("history transfer requires a complete int32 page table and int64 addressing")
    operands = (cache, table, end, rank, stage, addresses, output, *signals)
    if any(value.device != cache.device or not value.is_contiguous() for value in operands):
        raise ValueError("history transfer operands must be contiguous on one device")
    if len(signals) != 4 or any(value.shape != (1, 8) or value.dtype != torch.bfloat16 for value in signals):
        raise ValueError("history transfer requires four BF16 collective signal rows")
    spans = [(value.untyped_storage()._cdata, value.storage_offset() * value.element_size(),
              value.numel() * value.element_size()) for value in (cache, stage, output, *signals)]
    for index, (storage, begin, size) in enumerate(spans):
        if any(storage == other and begin < offset + extent and offset < begin + size
               for other, offset, extent in spans[:index]):
            raise ValueError("history source, destination and signals must not overlap")


class PreparedInterleavedHistory:
    """Publish local history, copy peer bytes, then release the shared stage.

    Signal inputs are written by the copy kernels. Their collective outputs are
    copy inputs, so compilation preserves both publication and reuse boundaries.
    The caller retains all storage and maps every address before preparation.
    """

    def __init__(self, cache, table, end, rank, stage, addresses, output, signals, *,
                 runtime, session, coordinator, name):
        validate_history(cache, table, end, rank, stage, addresses, output, signals)
        self.cache, self.table, self.end, self.rank = cache, table, end, rank
        self.stage, self.addresses, self.output = stage, addresses, output
        self.packed, self.ready, self.copied, self.completed = signals
        self.world, self.maximum = addresses.numel(), output.shape[0]
        self.publish = PreparedAllReduce(self.packed, self.ready, runtime=runtime, session=session,
            coordinator=coordinator, name=name + ".ready")
        self.consume = PreparedAllReduce(self.copied, self.completed, runtime=runtime, session=session,
            coordinator=coordinator, name=name + ".consumed")

    def functional(self):
        self.output_layout = tuple(self.output.shape), self.output.dtype
        self.table = self.end = self.output = None
        return self

    def __call__(self, table=None, end=None):
        if table is None:
            table, end, output = self.table, self.end, self.output
        else:
            shape, dtype = self.output_layout
            output = torch.empty(shape, dtype=dtype, device=self.cache.device)
        words = self.cache.shape[2] // 4
        # Bound empty-CTA work by the launch grid, not the prepared history
        # capacity. Grid-stride copies stop at the device-visible prefix length.
        torch.library.wrap_triton(_pack)[(min(1024, triton.cdiv(self.stage.shape[0], 4)),)](
            self.cache.view(torch.int32), table, end, self.rank, self.completed,
            self.stage.view(torch.int32), self.packed, COLUMNS=table.numel(),
            CACHE_PAGES=self.cache.shape[0], PAGE=self.cache.shape[1], WORLD=self.world,
            MAXIMUM=self.maximum, WORDS=words, BLOCK=triton.next_power_of_2(words), num_warps=4)
        self.publish(self.packed)
        torch.library.wrap_triton(_unpack)[(min(1024, triton.cdiv(self.maximum, 8)),)](
            self.addresses, end, self.ready, output.view(torch.int32), self.copied,
            WORLD=self.world, MAXIMUM=self.maximum, WORDS=words,
            BLOCK=triton.next_power_of_2(words), num_warps=4)
        self.consume(self.copied)
        return output
