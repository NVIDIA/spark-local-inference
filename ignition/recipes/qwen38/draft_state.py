# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed target-feedback inputs and metadata for prepared MTP execution."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .attention import QSAMetadata

HIDDEN = 10240
HIDDEN_CHUNK = 1024


@triton.jit
def _pack_draft_metadata(Slots, Lengths, Starts, TokenIds, token_stride, Hidden, hidden_stride, Rope, rope_row_stride,
                         rope_column_stride, Tables, OutTokens, OutHidden, RequestIds, QueryPositions, RopePositions,
                         RawSlots, SequenceLengths, QueryStartLoc, NumAccepted, IsPrefilling, MainTable, CompressedTable,
                         SlotMapping, OutputIndices, source_tokens, table_rows, table_stride,
                         ROWS: tl.constexpr, ROWS_P2: tl.constexpr, CAPACITY: tl.constexpr, PAGES: tl.constexpr,
                         HIDDEN: tl.constexpr, CHUNK: tl.constexpr, PAGE_BLOCK: tl.constexpr):
    token, chunk = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, ROWS_P2)
    lengths = tl.load(Lengths + rows, rows < ROWS, other=0)
    ends = tl.cumsum(lengths, 0)
    total = tl.max(tl.where(rows < ROWS, ends, 0), 0)
    row = tl.minimum(tl.sum(((rows < ROWS) & (ends <= token)).to(tl.int32), 0), ROWS - 1)
    length = tl.sum(tl.where(rows == row, lengths, 0), 0)
    local = token - (tl.sum(tl.where(rows == row, ends, 0), 0) - length)
    active = (token < total) & (length > 0)
    source = tl.minimum(tl.maximum(row * CAPACITY + local, 0), source_tokens - 1)
    row_slot = tl.load(Slots + row)
    row_table = tl.minimum(tl.maximum(row_slot, 0), table_rows - 1).to(tl.int64) * table_stride

    columns = chunk * CHUNK + tl.arange(0, CHUNK)
    hidden = tl.load(Hidden + source * hidden_stride + columns, columns < HIDDEN, other=0.)
    tl.store(OutHidden + token * HIDDEN + columns, tl.where(active, hidden, 0.), columns < HIDDEN)

    if token < ROWS:
        own_slot = tl.load(Slots + token)
        own_table = tl.minimum(tl.maximum(own_slot, 0), table_rows - 1).to(tl.int64) * table_stride
        pages = chunk * PAGE_BLOCK + tl.arange(0, PAGE_BLOCK)
        entries = tl.load(Tables + own_table + pages, pages < PAGES)
        tl.store(MainTable + token * PAGES + pages, entries, pages < PAGES)
        tl.store(CompressedTable + token * PAGES + pages, entries, pages < PAGES)

    if chunk == 0:
        logical = tl.load(Starts + row) + local
        tl.store(OutTokens + token, tl.load(TokenIds + source * token_stride))
        tl.store(RequestIds + token, tl.where(active, row, -1).to(tl.int64))
        tl.store(QueryPositions + token, tl.where(active, logical, -1))
        axes = tl.arange(0, 4)
        rope = tl.load(Rope + source * rope_row_stride + axes * rope_column_stride, axes < 3)
        tl.store(RopePositions + token * 3 + axes, tl.where(active, rope, -1), axes < 3)
        page_index = tl.where(logical < 0, -((63 - logical) // 64), logical // 64)
        page_index = tl.minimum(tl.maximum(page_index, 0), PAGES - 1)
        page = tl.load(Tables + row_table + page_index).to(tl.int64)
        offset = logical - (logical // 64) * 64
        offset = tl.where(offset < 0, offset + 64, offset)
        tl.store(SlotMapping + token, tl.where(active, page * 64 + offset, -1))
        if token < ROWS:
            own_length = tl.load(Lengths + token)
            own_end = tl.sum(tl.where(rows == token, ends, 0), 0)
            tl.store(QueryStartLoc + token + 1, own_end.to(tl.int32))
            if token == 0:
                tl.store(QueryStartLoc, 0)
            tl.store(RawSlots + token, tl.where(own_length > 0, tl.load(Slots + token).to(tl.int64) + 1, -1))
            tl.store(SequenceLengths + token,
                     tl.where(own_length > 0, tl.load(Starts + token) + own_length, 0).to(tl.int32))
            tl.store(NumAccepted + token, 1)
            tl.store(IsPrefilling + token, 1)
            tl.store(OutputIndices + token, tl.maximum(own_end - 1, 0))

DRAFT_RAW_FIELDS = ("raw_k_ring", "raw_logical_positions", "raw_rope_positions", "raw_interval_start_positions")


class DraftMetadata:
    """Caller-owned feedback rows and QSA metadata for one prepared MTP shape."""

    def __init__(self, *, tokens, rows, context, device, hidden=None):
        self.tokens, self.rows, self.context, self.device = tokens, rows, context, device
        self.qsa = QSAMetadata.allocate(tokens=tokens, batch=rows, context=context, device=device)
        self.token_ids = torch.zeros(tokens, dtype=torch.int64, device=device)
        self.hidden = (torch.zeros((tokens, 10240), dtype=torch.bfloat16, device=device)
                       if hidden is None else hidden)
        if (self.hidden.shape != (tokens, HIDDEN) or self.hidden.dtype != torch.bfloat16
                or self.hidden.device != self.token_ids.device or not self.hidden.is_contiguous()):
            raise ValueError("MTP feedback requires contiguous BF16 [tokens, 10240] storage on the model device")
        self.output_indices = torch.zeros(rows, dtype=torch.int64, device=device)
        self.indices = torch.arange(tokens, device=device)
        self.row_indices = torch.arange(rows, device=device)

    def functional(self):
        self.qsa = self.token_ids = self.hidden = self.output_indices = None
        self.indices = self.row_indices = None
        return self

    def _set_eager(self, slots, lengths, starts, token_ids, hidden, rope, tables, *, capacity):
        """Pack fixed-stride source rows into b12x's dense request intervals."""
        m = self.qsa
        ends = lengths.cumsum(0)
        begins = ends - lengths
        m.query_start_loc[0].zero_()
        m.query_start_loc[1:].copy_(ends)
        row = (self.indices[:, None] >= ends[None, :]).sum(1).clamp_max(self.rows - 1)
        local = self.indices - begins[row]
        active = (self.indices < ends[-1]) & (lengths[row] > 0)
        source = (row * capacity + local).clamp(0, token_ids.numel() - 1)
        self.token_ids.copy_(token_ids.reshape(-1)[source])
        self.hidden.copy_(hidden.reshape(-1, 10240)[source])
        self.hidden.masked_fill_(~active[:, None], 0)
        logical = starts[row] + local
        m.request_ids.copy_(torch.where(active, row, -1))
        m.query_positions.copy_(torch.where(active, logical, -1))
        m.rope_positions.copy_(rope.reshape(-1, 3)[source])
        m.rope_positions.masked_fill_(~active[:, None], -1)
        m.raw_state_slot_ids.copy_(torch.where(lengths > 0, slots.long() + 1, -1))
        m.sequence_lengths.copy_(torch.where(lengths > 0, starts + lengths, 0))
        m.num_accepted_tokens.fill_(1)
        m.is_prefilling.fill_(True)
        safe_slots = slots.long().clamp(0, tables.shape[0] - 1)
        m.main_block_table.copy_(tables[safe_slots])
        m.compressed_block_table.copy_(m.main_block_table)
        page = m.main_block_table[row, (logical // 64).clamp(0, tables.shape[1] - 1)]
        m.slot_mapping.copy_(torch.where(active, page.long() * 64 + logical % 64, -1))
        self.output_indices.copy_((ends - 1).clamp_min(0))

    def set(self, slots, lengths, starts, token_ids, hidden, rope, tables, *, capacity):
        """Pack fixed-stride source rows into b12x's dense request intervals in one launch."""
        if self.token_ids is None:
            local = DraftMetadata(tokens=self.tokens, rows=self.rows, context=self.context, device=self.device)
            local.set(slots, lengths, starts, token_ids, hidden, rope, tables, capacity=capacity)
            return local
        if not self.token_ids.is_cuda:
            return self._set_eager(slots, lengths, starts, token_ids, hidden, rope, tables, capacity=capacity)
        m = self.qsa
        pages = m.main_block_table.shape[1]
        if tables.shape[1] != pages or tables.stride(1) != 1 or self.rows > 64:
            raise ValueError("draft page tables must match QSA metadata and at most 64 rows")
        token_ids, hidden, rope = token_ids.reshape(-1), hidden.reshape(-1, HIDDEN), rope.reshape(-1, 3)
        chunks = triton.cdiv(HIDDEN, HIDDEN_CHUNK)
        args = (slots, lengths, starts, token_ids, token_ids.stride(0), hidden, hidden.stride(0), rope,
                rope.stride(0), rope.stride(1), tables, self.token_ids, self.hidden, m.request_ids,
                m.query_positions, m.rope_positions, m.raw_state_slot_ids, m.sequence_lengths, m.query_start_loc,
                m.num_accepted_tokens, m.is_prefilling, m.main_block_table, m.compressed_block_table, m.slot_mapping,
                self.output_indices, token_ids.numel(), tables.shape[0], tables.stride(0), self.rows,
                triton.next_power_of_2(self.rows), capacity, pages, HIDDEN, HIDDEN_CHUNK,
                triton.next_power_of_2(triton.cdiv(pages, chunks)))
        grid = (self.tokens, chunks, 1)
        if torch.compiler.is_compiling() or get_proxy_mode() is not None:
            torch.library.wrap_triton(_pack_draft_metadata)[grid](*args, num_warps=4)
        else:
            _pack_draft_metadata[grid](*args, num_warps=4)
