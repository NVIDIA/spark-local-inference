# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-width target verification metadata and accepted hybrid checkpoints."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ignition.components.capacity import BATCH_CAPACITIES
from ignition.components.recurrent import DeltaMetadata
from ignition.components.token_state import PreparedTokenState

from .attention import QSAMetadata
from .decoder_state import compile_kernel
from .ple import PLEMetadata


@triton.jit
def _validate_round(Slots, Epochs, Live, Generation, Status, Error, Pending, Consumed,
                    Limits, LIMITED: tl.constexpr,
                    ROWS: tl.constexpr, WIDTH: tl.constexpr, CONTEXT: tl.constexpr,
                    CONSUMED_STRIDE: tl.constexpr, SLOTS: tl.constexpr = 4):
    slot = tl.program_id(0)
    rows = tl.arange(0, SLOTS)
    match = (rows < ROWS) & (rows < tl.load(Live))
    match &= tl.load(Slots + rows, rows < ROWS, other=-1) == slot
    match &= tl.load(Epochs + rows, rows < ROWS, other=-1) == tl.load(Generation + slot)
    count = tl.sum(match.to(tl.int32), 0)
    if (count > 0) & (tl.load(Status + slot) == 1):
        position = tl.load(Consumed + slot * CONSUMED_STRIDE)
        valid = (count == 1) & (tl.load(Live) <= ROWS)
        length = WIDTH
        if LIMITED:
            length = tl.sum(tl.where(match, tl.load(Limits + rows, rows < ROWS, other=0), 0), 0)
        valid &= (length >= 1) & (length <= WIDTH)
        valid &= (position >= 0) & (position + length <= CONTEXT) & (tl.load(Pending + slot) >= 0)
        if not valid:
            tl.store(Status + slot, 3)
            tl.store(Error + slot, 34)


@triton.jit
def _verification_metadata(Slots, Live, Active, Positions, Pending, Candidates, PreviousAccepted,
                           Delta, Tokens, Cu, ConvSlots, Fresh, StateIndices, Accepted,
                           NumSeqs, NumTokens, RequestIds, QueryPositions, Rope, SequenceLengths,
                           SlotMapping, PhysicalPages, PackedPages, Limits, StateRows, RawSlots, LIMITED: tl.constexpr,
                           ROWS: tl.constexpr, WIDTH: tl.constexpr, CANDIDATE_STRIDE: tl.constexpr,
                           PAGES: tl.constexpr, DELTA_STRIDE: tl.constexpr,
                           BLOCK: tl.constexpr, SLOTS: tl.constexpr = 4):
    row = tl.program_id(0)
    slot = tl.load(Slots + row)
    live = tl.minimum(tl.load(Live), ROWS)
    active = (row < live) & (tl.load(Active + row) != 0) & (slot >= 0) & (slot < SLOTS)
    start = row * WIDTH
    position = tl.load(Positions + row)
    model_slot = tl.load(StateRows + slot * 8, active, other=-1)
    tl.store(Cu + row, tl.minimum(row, live) * WIDTH)
    if row == 0:
        tl.store(Cu + ROWS, live * WIDTH)
        tl.store(NumSeqs, live)
        tl.store(NumTokens, live * WIDTH)
    tl.store(ConvSlots + row, model_slot)
    tl.store(RawSlots + row, tl.where(active, slot + 1, -1))
    tl.store(Fresh + row, False)
    previous = tl.load(PreviousAccepted + slot, active, other=1)
    tl.store(Accepted + row, previous)
    columns = tl.arange(0, 8)
    # Verification resumes from and commits into the request's live state row; later columns are unused.
    indices = tl.where(columns == 0, tl.load(StateRows + slot * 8, active, other=0), 0)
    tl.store(StateIndices + row * 8 + columns, tl.where(active, indices, 0))
    length = WIDTH
    if LIMITED:
        length = tl.load(Limits + row)
    tl.store(SequenceLengths + row, tl.where(active, position + length, 0))
    offsets = tl.arange(0, triton.next_power_of_2(WIDTH))
    valid = offsets < WIDTH
    token = tl.load(Candidates + row * CANDIDATE_STRIDE + offsets, active & valid, other=248044)
    token = tl.where(offsets == 0, tl.load(Pending + row), token)
    tl.store(Tokens + start + offsets, tl.where(active, token, 248044), valid)
    logical = position + offsets
    in_context = active & (offsets < length)
    page = tl.load(PhysicalPages + slot * PAGES + logical // 64, in_context & valid, other=-1)
    tl.store(RequestIds + start + offsets, tl.where(in_context, row, -1), valid)
    tl.store(QueryPositions + start + offsets, tl.where(in_context, logical, -1), valid)
    tl.store(SlotMapping + start + offsets, tl.where(in_context, page * 64 + logical % 64, -1), valid)
    delta = tl.load(Delta + model_slot * DELTA_STRIDE, active, other=0)
    for axis in tl.static_range(3):
        tl.store(Rope + (start + offsets) * 3 + axis, tl.where(in_context, logical + delta, -1), valid)
    for block in range(tl.cdiv(PAGES, BLOCK)):
        page_indices = block * BLOCK + tl.arange(0, BLOCK)
        pages = tl.load(PhysicalPages + slot * PAGES + page_indices, active & (page_indices < PAGES), other=-1)
        tl.store(PackedPages + row * PAGES + page_indices, pages, page_indices < PAGES)


@triton.jit
def _commit_hidden(Hidden, Residual, SavedHidden, SavedResidual, Rope, SavedRope, Slots, Count,
                   PreviousAccepted, StateRows, WIDTH: tl.constexpr, HS: tl.constexpr, RS: tl.constexpr,
                   RPS: tl.constexpr, BLOCK: tl.constexpr, SLOTS: tl.constexpr = 4):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot, count = tl.load(Slots + row), tl.load(Count + row)
    state_row = tl.load(StateRows + slot * 8, (slot >= 0) & (slot < SLOTS), other=0)
    active = (count > 0) & (slot >= 0) & (slot < SLOTS)
    channel = tile * BLOCK + tl.arange(0, BLOCK)
    offset = row * WIDTH + count - 1
    residual = tl.load(Residual + offset * 10240 + channel, active & (channel < 10240), other=0)
    tl.store(SavedResidual + state_row * RS + channel, residual, active & (channel < 10240))
    hidden = tl.load(Hidden + offset * 2560 + channel, active & (channel < 2560), other=0)
    tl.store(SavedHidden + state_row * HS + channel, hidden, active & (channel < 2560))
    if tile == 0:
        tl.store(PreviousAccepted + slot, count, active)
        axes = tl.arange(0, 4)
        rope = tl.load(Rope + (row * WIDTH + count - 1) * 3 + axes, active & (axes < 3), other=-1)
        tl.store(SavedRope + state_row * RPS + axes, rope, active & (axes < 3))



class PreparedVerificationState:
    """Bind all widths to verification metadata that resumes from each request's live state row.

    Model evaluation leaves counters, PLE history and prefix publication private.
    The acceptance operation supplies clipped consumed counts before commit.
    """

    def __init__(self, inputs, requests, pool, candidates, previous_accepted,
                 *, width: int, rows: int, token_limits=None):
        if type(width) is not int or not 1 <= width <= 8 or (rows not in BATCH_CAPACITIES or rows > requests.capacity):
            raise ValueError("verification requires width 1 through 8 and a power-of-two batch within request capacity")
        if candidates.dtype != torch.int64 or candidates.shape != (requests.capacity, 8):
            raise ValueError("verification candidates require int64 [capacity,8]")
        self.inputs, self.requests, self.pool = inputs, requests, pool
        self.tokens, self.rows, self.mode, self.width = rows * width, rows, "decode", width
        self.previous_accepted = previous_accepted
        device = inputs.token_ids.device
        self.token_ids = torch.full((self.tokens,), 248044, dtype=torch.int64, device=device)
        self.gdn = DeltaMetadata.allocate(rows, device, verification_columns=8)
        self.qsa = QSAMetadata.allocate(tokens=self.tokens, batch=rows,
            context=inputs.page_tables.shape[1] * 64, device=device)
        self.qsa = QSAMetadata(self.qsa.main_block_table, self.qsa.main_block_table,
            self.qsa.raw_state_slot_ids, self.qsa.request_ids, self.qsa.query_positions,
            self.qsa.rope_positions, self.qsa.sequence_lengths, self.gdn.query_start_loc,
            self.gdn.accepted_tokens, self.qsa.is_prefilling, self.qsa.slot_mapping)
        self.token_state = PreparedTokenState(requests, self.token_ids, self.gdn.query_start_loc,
            inputs.row_slot_ids[:rows], inputs.row_generations[:rows], inputs.live_rows, history=requests.ple_history, history_pad_id=248044)
        self.token_state.prepare()
        self.ple = PLEMetadata(self.token_ids, self.gdn.query_start_loc, self.token_state.history,
            self.gdn.convolution_slots, self.gdn.fresh, self.gdn.accepted_tokens,
            self.gdn.num_seqs, self.gdn.num_tokens)
        self._validate = compile_kernel(_validate_round,
            (inputs.row_slot_ids, inputs.row_generations, inputs.live_rows, requests.generation,
             requests.status, requests.error_code, requests.pending_token, requests.consumed_tokens,
             inputs.query_lengths if token_limits is None else token_limits, token_limits is not None,
             rows, width, inputs.page_tables.shape[1] * 64, requests.consumed_tokens.stride(0), requests.capacity), (requests.capacity, 1, 1))
        g, q, t = self.gdn, self.qsa, self.token_state
        self._metadata = compile_kernel(_verification_metadata,
            (inputs.row_slot_ids, inputs.live_rows, t.active, t.positions, t.pending_tokens,
             candidates, previous_accepted, pool.rope_delta, self.token_ids, g.query_start_loc,
             g.convolution_slots, g.fresh, g.state_indices, g.accepted_tokens, g.num_seqs, g.num_tokens,
             q.request_ids, q.query_positions, q.rope_positions, q.sequence_lengths, q.slot_mapping,
             inputs.page_tables, q.main_block_table,
             inputs.query_lengths if token_limits is None else token_limits, inputs.request_state_rows, q.raw_state_slot_ids, token_limits is not None,
             rows, width, candidates.stride(0),
             inputs.page_tables.shape[1], pool.rope_delta.stride(0), 128, requests.capacity), (rows, 1, 1))

    def prepare_hidden(self, hidden, sample):
        if hidden.shape != sample.shape:
            raise ValueError("verification must retain every target row for the vocabulary projection")
        self.hidden, self.sample = hidden, sample

    def prepare_residual(self, residual):
        self.residual = residual

    def functional(self, *, dynamic_inputs=False):
        if self.gdn is None:
            return self
        from .decoder_state import PreparedDecoderState
        examples = {f"gdn.{name}": getattr(self.gdn, name) for name in self.gdn.__dataclass_fields__}
        examples.update({f"qsa.{name}": getattr(self.qsa, name) for name in self.qsa.__dataclass_fields__})
        examples.update({name: getattr(self.token_state, name)
                         for name in ("pending_tokens", "positions", "history", "active")})
        examples.update(token_ids=self.token_ids, candidates=self._metadata.args[5])
        if dynamic_inputs:
            examples.update({f"input.{name}": getattr(self.inputs, name) for name in
                             ("row_slot_ids", "row_generations", "live_rows")})
            examples["token_limits"] = self._metadata.args[23]
        commits = getattr(self, "_commit", ())
        if commits:
            examples.update(hidden=commits[0].args[0], residual=commits[0].args[1],
                            emitted=commits[0].args[7])
        def detach(call):
            slots = {index: name for index, value in enumerate(call.args)
                     for name, example in examples.items() if value is not None and value is example}
            args = tuple(None if index in slots else value for index, value in enumerate(call.args))
            call.args = ()
            return call.kernel, call.compiled, args, call.grid, call.options, slots
        self._runtime_metadata = detach(self._metadata)
        self._runtime_commit = tuple(detach(call) for call in commits)
        self._runtime_validate = detach(self._validate) if dynamic_inputs else None
        self._launch_runtime = PreparedDecoderState._launch_runtime
        self.token_state.functional(dynamic_inputs=dynamic_inputs)
        if dynamic_inputs:
            self.inputs = None
        self.gdn = self.qsa = self.ple = self.token_ids = self.hidden = self.sample = self.residual = None
        return self

    def begin_runtime(self, candidates, inputs=None):
        inputs = self.inputs if inputs is None else inputs
        if inputs is None:
            raise ValueError("functional verification requires runtime role inputs")
        input_values = {f"input.{name}": getattr(inputs, name) for name in
                        ("row_slot_ids", "row_generations", "live_rows")}
        input_values["token_limits"] = inputs.query_lengths[:self.rows]
        if self._runtime_validate is None:
            self._validate()
            pending, positions, history, active = self.token_state.gather()
        else:
            self._launch_runtime(self._runtime_validate, input_values)
            pending, positions, history, active = self.token_state.gather(inputs.row_slot_ids[:self.rows],
                inputs.row_generations[:self.rows], inputs.live_rows)
        device = candidates.device
        token_ids = torch.full((self.tokens,), 248044, dtype=torch.int64, device=device)
        gdn = DeltaMetadata.allocate(self.rows, device, verification_columns=8)
        q = QSAMetadata.allocate(tokens=self.tokens, batch=self.rows,
            context=inputs.page_tables.shape[1] * 64, device=device)
        qsa = QSAMetadata(q.main_block_table, q.main_block_table, q.raw_state_slot_ids,
            q.request_ids, q.query_positions, q.rope_positions, q.sequence_lengths,
            gdn.query_start_loc, gdn.accepted_tokens, q.is_prefilling, q.slot_mapping)
        ple = PLEMetadata(token_ids, gdn.query_start_loc, history, gdn.convolution_slots,
            gdn.fresh, gdn.accepted_tokens, gdn.num_seqs, gdn.num_tokens)
        values = {f"gdn.{name}": getattr(gdn, name) for name in gdn.__dataclass_fields__}
        values.update({f"qsa.{name}": getattr(qsa, name) for name in qsa.__dataclass_fields__})
        values.update(pending_tokens=pending, positions=positions, history=history, active=active,
                      token_ids=token_ids, candidates=candidates)
        values.update(input_values)
        self._launch_runtime(self._runtime_metadata, values)
        from .decoder_state import DecoderRuntime
        return DecoderRuntime(gdn, qsa, ple, values, inputs)

    def finish_runtime(self, runtime, hidden, residual):
        return hidden

    def commit_runtime(self, runtime, hidden, residual, emitted):
        values = {**runtime.values, "hidden": hidden, "residual": residual, "emitted": emitted}
        for operation in self._runtime_commit:
            self._launch_runtime(operation, values)

    def prepare_commit(self, emitted_count, consumed_before, residual, hidden):
        pool, slots = self.pool, self.inputs.row_slot_ids
        self._commit = []
        saved_residual = pool.region("target_residual")
        saved_rope = pool.region("target_residual_position")
        self._commit.append(compile_kernel(_commit_hidden,
            (hidden, residual, pool.last_hidden, saved_residual, self.qsa.rope_positions, saved_rope, slots,
             emitted_count, self.previous_accepted, self.inputs.request_state_rows, self.width,
             pool.last_hidden.stride(0), saved_residual.stride(0), saved_rope.stride(0), 1024, self.requests.capacity),
            (self.rows, 10, 1)))
