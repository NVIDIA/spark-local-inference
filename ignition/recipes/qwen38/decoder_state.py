# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed device inputs and captured metadata for the hybrid Qwen decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ignition.components.capacity import BATCH_CAPACITIES
from ignition.components.recurrent import DeltaMetadata
from ignition.components.token_state import PreparedTokenState
from ignition.preparation.triton import prepare_triton

from .attention import QSAMetadata
from .ple import PLEMetadata
from .sampling import DeviceRequestState
from .state import HybridStatePool


@dataclass
class DecoderRuntime:
    gdn: object
    qsa: object
    ple: object
    values: dict
    inputs: object
    injection_errors: object = None
    capture_metadata: object = None


def compile_kernel(kernel, args, grid):
    return prepare_triton(kernel, args, grid, num_warps=4)


@triton.jit
def _validate(Slots, Epochs, Live, Lengths, Generation, Status, Error, Pending, Consumed,
              ROWS: tl.constexpr, TOKENS: tl.constexpr, CONTEXT: tl.constexpr,
              CONSUMED_STRIDE: tl.constexpr, PREFILL: tl.constexpr, SLOTS: tl.constexpr = 4):
    slot = tl.program_id(0)
    rows = tl.arange(0, SLOTS)
    match = (rows < ROWS) & (rows < tl.load(Live))
    match &= tl.load(Slots + rows, rows < ROWS, other=-1) == slot
    match &= tl.load(Epochs + rows, rows < ROWS, other=-1) == tl.load(Generation + slot)
    count = tl.sum(match.to(tl.int32), 0)
    if (count > 0) & (tl.load(Status + slot) == 1):
        row = tl.max(tl.where(match, rows, 0), 0)
        length = tl.load(Lengths + row) if PREFILL else 1
        consumed = tl.load(Consumed + slot * CONSUMED_STRIDE)
        valid = (count == 1) & (tl.load(Live) <= ROWS) & (length > 0) & (length <= TOKENS)
        valid &= (consumed >= 0) & (consumed + length <= CONTEXT)
        if PREFILL:
            valid &= tl.load(Pending + slot) < 0
        else:
            valid &= tl.load(Pending + slot) >= 0
        if not valid:
            tl.store(Status + slot, 3)
            tl.store(Error + slot, 32)


@dataclass(frozen=True)
class DecoderInputs:
    request_state_rows: torch.Tensor
    row_slot_ids: torch.Tensor
    row_generations: torch.Tensor
    live_rows: torch.Tensor
    query_lengths: torch.Tensor
    token_ids: torch.Tensor
    prefill_positions: torch.Tensor
    page_tables: torch.Tensor
    sample_enabled: torch.Tensor
    vision_token_indices: torch.Tensor
    prefill_row: torch.Tensor
    vision_feature_slot: torch.Tensor

    @classmethod
    def allocate(cls, device, *, context: int, tokens: int = 512, capacity: int = 4):
        if capacity not in BATCH_CAPACITIES:
            raise ValueError("decoder request capacity must be a power of two through 64")
        if (type(context) is not int or type(tokens) is not int
                or context <= 0 or context % 64 or tokens <= 0):
            raise ValueError("decoder capacities require positive tokens and a 64-aligned context")
        return cls(torch.zeros((capacity, 8), dtype=torch.int64, device=device),
            torch.full((capacity,), -1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device),
            torch.full((tokens,), 248044, dtype=torch.int64, device=device),
            torch.zeros((tokens, 3), dtype=torch.int64, device=device),
            torch.full((capacity, context // 64), -1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device),
            torch.full((tokens,), -1, dtype=torch.int32, device=device),
            torch.full((1,), -1, dtype=torch.int32, device=device),
            torch.full((1,), -1, dtype=torch.int32, device=device),
        )

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@triton.jit
def _metadata(Slots, Live, Lengths, Pending, Positions, Active, Delta, Tokens, InputRope,
              Cu, ConvSlots, Fresh, StateIndices, Initial, Final, Checkpoint, Offset,
              NumSeqs, NumTokens, RequestIds, QueryPositions, Rope, SequenceLengths,
              SlotMapping, PhysicalPages, PackedPages, IsPrefilling, StateRows, RawSlots,
              ROWS: tl.constexpr, TOKENS: tl.constexpr, PAGES: tl.constexpr,
              DELTA_STRIDE: tl.constexpr, PREFILL: tl.constexpr, BLOCK: tl.constexpr, SLOTS: tl.constexpr = 4):
    row = tl.program_id(0)
    rows = tl.arange(0, SLOTS)
    live = tl.minimum(tl.load(Live), ROWS)
    active = tl.load(Active + rows, rows < ROWS, other=0) != 0
    lengths = tl.load(Lengths + rows, rows < ROWS, other=0) if PREFILL else tl.full((SLOTS,), 1, tl.int32)
    lengths = tl.where((rows < live) & active, lengths, 0)
    begin = tl.sum(tl.where(rows < row, lengths, 0), 0)
    length = tl.sum(tl.where(rows == row, lengths, 0), 0)
    total = tl.sum(lengths, 0)
    slot = tl.load(Slots + row)
    valid = (row < live) & (length > 0) & (slot >= 0) & (slot < SLOTS)
    position = tl.load(Positions + row)
    model_slot = tl.load(StateRows + slot * 8, valid, other=-1)
    end = position + length
    tl.store(Cu + row, begin)
    if row == 0:
        tl.store(Cu + ROWS, total)
        tl.store(NumTokens, total)
        tl.store(NumSeqs, live)
    tl.store(ConvSlots + row, model_slot)
    tl.store(RawSlots + row, tl.where(valid, slot + 1, -1))
    tl.store(Fresh + row, valid & (position == 0))
    tl.store(StateIndices + row, tl.where(valid, model_slot, 0))
    tl.store(Initial + row, tl.where(valid & (position > 0), model_slot, 0))
    tl.store(Final + row, tl.where(valid, model_slot, 0))
    tl.store(Checkpoint + row, 0)
    tl.store(Offset + row, 0)
    tl.store(SequenceLengths + row, tl.where(valid, end, 0))
    tl.store(IsPrefilling + row, PREFILL & valid)
    if not PREFILL:
        tl.store(Tokens + begin, tl.load(Pending + row), valid)
    delta = tl.load(Delta + model_slot * DELTA_STRIDE, valid, other=0)
    for block in range(tl.cdiv(TOKENS, BLOCK)):
        local = block * BLOCK + tl.arange(0, BLOCK)
        index = begin + local
        mask = valid & (local < length) & (index < TOKENS)
        logical = position + local
        page = tl.load(PhysicalPages + slot * PAGES + logical // 64, mask, other=-1)
        tl.store(RequestIds + index, row, mask)
        tl.store(QueryPositions + index, logical, mask)
        tl.store(SlotMapping + index, page * 64 + logical % 64, mask)
        for axis in tl.static_range(3):
            rope = tl.load(InputRope + index * 3 + axis, mask, other=0) if PREFILL else logical + delta
            tl.store(Rope + index * 3 + axis, rope, mask)
    for block in range(tl.cdiv(PAGES, BLOCK)):
        page_indices = block * BLOCK + tl.arange(0, BLOCK)
        pages = tl.load(PhysicalPages + slot * PAGES + page_indices, valid & (page_indices < PAGES), other=-1)
        tl.store(PackedPages + row * PAGES + page_indices, pages, page_indices < PAGES)


@triton.jit
def _residual_checkpoint(Source, Saved, Rope, SavedRope, Slots, Active, Cu,
                         StateRows, SAVED_STRIDE: tl.constexpr, ROPE_STRIDE: tl.constexpr, BLOCK: tl.constexpr, SLOTS: tl.constexpr = 4):
    row, tile = tl.program_id(0), tl.program_id(1)
    channel = tile * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + row)
    state_row = tl.load(StateRows + slot * 8, (slot >= 0) & (slot < SLOTS), other=0)
    begin, end = tl.load(Cu + row), tl.load(Cu + row + 1)
    active = (tl.load(Active + row) != 0) & (slot >= 0) & (slot < SLOTS) & (end > begin)
    value = tl.load(Source + (end - 1) * 10240 + channel, active & (channel < 10240), other=0)
    tl.store(Saved + state_row * SAVED_STRIDE + channel, value, active & (channel < 10240))
    if tile == 0:
        axes = tl.arange(0, 4)
        rope = tl.load(Rope + (end - 1) * 3 + axes, active & (axes < 3), other=-1)
        tl.store(SavedRope + state_row * ROPE_STRIDE + axes, rope, active & (axes < 3))



@triton.jit
def _hidden_checkpoint(Hidden, Saved, Delta, Sample, Slots, Active, Cu, Positions, Rope,
                       StateRows, SAVED_STRIDE: tl.constexpr, DELTA_STRIDE: tl.constexpr,
                       PREFILL: tl.constexpr, POSITION_BLOCK: tl.constexpr, BLOCK: tl.constexpr, SLOTS: tl.constexpr = 4):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    slot = tl.load(Slots + row)
    state_row = tl.load(StateRows + slot * 8, (slot >= 0) & (slot < SLOTS), other=0)
    valid = (tl.load(Active + row) != 0) & (slot >= 0) & (slot < SLOTS)
    begin, end = tl.load(Cu + row), tl.load(Cu + row + 1)
    valid &= end > begin
    value = tl.load(Hidden + (end - 1) * 2560 + channel, valid & (channel < 2560), other=0)
    tl.store(Sample + row * 2560 + channel, value, channel < 2560)
    tl.store(Saved + state_row * SAVED_STRIDE + channel, value, valid & (channel < 2560))
    if PREFILL:
        position = tl.load(Positions + row)
        old_delta = tl.load(Delta + state_row * DELTA_STRIDE, valid, other=0)
        old_next = position + old_delta
        indices = tl.arange(0, POSITION_BLOCK)
        rope = tl.load(Rope + indices, valid & (indices // 3 >= begin) & (indices // 3 < end), other=-1)
        next_position = tl.maximum(old_next, tl.max(rope, 0) + 1)
        tl.store(Delta + state_row * DELTA_STRIDE, next_position - (position + end - begin), valid)


@triton.jit
def _resume_hidden(Saved, Sample, Slots, Epochs, Live, Generation, Status, Error, Consumed, Pending,
                   StateRows, SAVED_STRIDE: tl.constexpr, CONSUMED_STRIDE: tl.constexpr, BLOCK: tl.constexpr, SLOTS: tl.constexpr = 4):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    slot = tl.load(Slots + row)
    state_row = tl.load(StateRows + slot * 8, (slot >= 0) & (slot < SLOTS), other=0)
    valid = (row < tl.load(Live)) & (slot >= 0) & (slot < SLOTS)
    valid &= tl.load(Generation + slot, valid, other=-1) == tl.load(Epochs + row)
    valid &= tl.load(Status + slot, valid, other=0) == 1
    consumed = tl.load(Consumed + slot * CONSUMED_STRIDE, valid, other=0)
    pending = tl.load(Pending + slot, valid, other=-1)
    ready = (consumed > 0) & (pending < 0)
    if valid & ~ready:
        tl.store(Status + slot, 3)
        tl.store(Error + slot, 33)
    value = tl.load(Saved + state_row * SAVED_STRIDE + channel,
                    valid & ready & (channel < 2560), other=0)
    tl.store(Sample + row * 2560 + channel, value, channel < 2560)


class PreparedDecoderState:
    """Derive model metadata from persistent generation state inside each graph.

    Prefill has one sequence and consumes explicit token/position uploads.
    Decode packs the pending GPU samples from physical request slots. Epoch
    and terminal-state checks precede any model-state mutation. The scheduler
    validates lengths, vocabulary IDs, context bounds and physical page IDs.
    """

    def __init__(self, inputs: DecoderInputs, requests: DeviceRequestState, pool: HybridStatePool,
                 *, tokens: int, rows: int, mode: str, maintain_draft_state=True):
        if mode not in ("prefill", "decode") or type(rows) is not int or not 1 <= rows <= inputs.row_slot_ids.numel():
            raise ValueError("decoder mode requires prefill/decode and rows within request capacity")
        if (type(tokens) is not int or tokens <= 0
                or any(tokens > tensor.shape[0] for tensor in
                       (inputs.token_ids, inputs.prefill_positions, inputs.vision_token_indices))):
            raise ValueError("decoder token capacity exceeds its input buffers")
        if (mode == "prefill" and rows != 1) or (mode == "decode" and tokens != rows):
            raise ValueError("prefill has one row; decode has one token per row")
        self.inputs, self.requests, self.pool = inputs, requests, pool
        self.tokens, self.rows, self.mode = tokens, rows, mode
        self.maintain_draft_state = maintain_draft_state
        device = inputs.token_ids.device
        self.gdn = DeltaMetadata.allocate(rows, device)
        self.qsa = QSAMetadata.allocate(tokens=tokens, batch=rows,
            context=inputs.page_tables.shape[1] * 64, device=device)
        self.qsa = QSAMetadata(self.qsa.main_block_table, self.qsa.main_block_table,
            self.qsa.raw_state_slot_ids, self.qsa.request_ids, self.qsa.query_positions,
            self.qsa.rope_positions, self.qsa.sequence_lengths, self.gdn.query_start_loc,
            self.gdn.accepted_tokens, self.qsa.is_prefilling, self.qsa.slot_mapping)
        self.token_state = PreparedTokenState(requests, inputs.token_ids[:tokens], self.gdn.query_start_loc,
            inputs.row_slot_ids[:rows], inputs.row_generations[:rows], inputs.live_rows, history=requests.ple_history, history_pad_id=248044)
        self.token_state.prepare()
        self._validate = compile_kernel(_validate,
            (inputs.row_slot_ids, inputs.row_generations, inputs.live_rows, inputs.query_lengths,
             requests.generation, requests.status, requests.error_code, requests.pending_token,
             requests.consumed_tokens, rows, tokens, inputs.page_tables.shape[1] * 64,
             requests.consumed_tokens.stride(0), mode == "prefill", requests.capacity), (requests.capacity, 1, 1))
        self.ple = PLEMetadata(inputs.token_ids[:tokens], self.gdn.query_start_loc,
            self.token_state.history, self.gdn.convolution_slots, self.gdn.fresh,
            self.gdn.accepted_tokens, self.gdn.num_seqs, self.gdn.num_tokens)
        g, q, t = self.gdn, self.qsa, self.token_state
        args = (inputs.row_slot_ids, inputs.live_rows, inputs.query_lengths,
            t.pending_tokens, t.positions, t.active, pool.rope_delta, inputs.token_ids, inputs.prefill_positions,
            g.query_start_loc, g.convolution_slots, g.fresh, g.state_indices, g.initial_state_indices,
            g.final_state_indices, g.checkpoint_state_indices, g.checkpoint_offsets, g.num_seqs, g.num_tokens,
            q.request_ids, q.query_positions, q.rope_positions, q.sequence_lengths, q.slot_mapping,
            inputs.page_tables, q.main_block_table, q.is_prefilling, inputs.request_state_rows, q.raw_state_slot_ids,
            rows, tokens, inputs.page_tables.shape[1], pool.rope_delta.stride(0), mode == "prefill", 128, requests.capacity)
        self._metadata = compile_kernel(_metadata, args, (rows, 1, 1))
        self._residual = None

    def prepare_hidden(self, hidden, sample):
        self._hidden = compile_kernel(_hidden_checkpoint,
            (hidden, self.pool.last_hidden, self.pool.rope_delta, sample, self.inputs.row_slot_ids,
             self.token_state.active, self.gdn.query_start_loc, self.token_state.positions, self.qsa.rope_positions,
             self.inputs.request_state_rows, self.pool.last_hidden.stride(0), self.pool.rope_delta.stride(0),
             self.mode == "prefill", triton.next_power_of_2(self.tokens * 3), 4096, self.requests.capacity), (self.rows, 1, 1))

    def prepare_residual(self, residual):
        if self.pool.layout.speculative_tokens and self.maintain_draft_state:
            saved = self.pool.region("target_residual")
            rope = self.pool.region("target_residual_position")
            self._residual = compile_kernel(_residual_checkpoint,
                (residual, saved, self.qsa.rope_positions, rope, self.inputs.row_slot_ids, self.token_state.active,
                 self.gdn.query_start_loc, self.inputs.request_state_rows, saved.stride(0), rope.stride(0), 1024, self.requests.capacity), (self.rows, 10, 1))

    def functional(self, *, dynamic_inputs=False):
        if self.gdn is None:
            return self
        examples = {f"gdn.{name}": getattr(self.gdn, name) for name in self.gdn.__dataclass_fields__}
        examples.update({f"qsa.{name}": getattr(self.qsa, name) for name in self.qsa.__dataclass_fields__})
        examples.update({name: getattr(self.token_state, name)
                         for name in ("pending_tokens", "positions", "history", "active")})
        if dynamic_inputs:
            examples.update({f"input.{name}": getattr(self.inputs, name) for name in
                ("row_slot_ids", "row_generations", "live_rows", "query_lengths", "token_ids", "prefill_positions")})
        examples.update(hidden=self._hidden.args[0], sample=self._hidden.args[3])
        if self._residual is not None:
            examples["residual"] = self._residual.args[0]
        def detach(call):
            if call is None:
                return None
            slots = {index: name for index, value in enumerate(call.args)
                     for name, example in examples.items() if value is not None and value is example}
            args = tuple(None if index in slots else value for index, value in enumerate(call.args))
            call.args = ()
            return call.kernel, call.compiled, args, call.grid, call.options, slots
        self._runtime_metadata = detach(self._metadata)
        self._runtime_hidden = detach(self._hidden)
        self._runtime_residual = detach(self._residual)
        self._runtime_validate = detach(self._validate) if dynamic_inputs else None
        self.token_state.functional(dynamic_inputs=dynamic_inputs)
        if dynamic_inputs:
            self.inputs = None
        self.gdn = self.qsa = self.ple = None
        return self

    @staticmethod
    def _launch_runtime(call, values):
        from ignition.preparation.triton import launch_triton
        kernel, compiled, args, grid, options, slots = call
        args = tuple(values[slots[index]] if index in slots else value for index, value in enumerate(args))
        launch_triton(kernel, compiled, args, grid, **options)

    def begin_runtime(self, inputs=None):
        inputs = self.inputs if inputs is None else inputs
        if inputs is None:
            raise ValueError("functional decoder state requires runtime role inputs")
        input_values = {f"input.{name}": getattr(inputs, name) for name in
            ("row_slot_ids", "row_generations", "live_rows", "query_lengths", "token_ids", "prefill_positions")}
        if self._runtime_validate is None:
            self._validate()
            pending, positions, history, active = self.token_state.gather()
        else:
            self._launch_runtime(self._runtime_validate, input_values)
            pending, positions, history, active = self.token_state.gather(inputs.row_slot_ids[:self.rows],
                inputs.row_generations[:self.rows], inputs.live_rows)
        device = inputs.token_ids.device
        gdn = DeltaMetadata.allocate(self.rows, device)
        q = QSAMetadata.allocate(tokens=self.tokens, batch=self.rows,
            context=inputs.page_tables.shape[1] * 64, device=device)
        qsa = QSAMetadata(q.main_block_table, q.main_block_table, q.raw_state_slot_ids,
            q.request_ids, q.query_positions, q.rope_positions, q.sequence_lengths,
            gdn.query_start_loc, gdn.accepted_tokens, q.is_prefilling, q.slot_mapping)
        qsa.request_ids.fill_(-1)
        qsa.query_positions.fill_(-1)
        qsa.rope_positions.fill_(-1)
        qsa.slot_mapping.fill_(-1)
        ple = PLEMetadata(inputs.token_ids[:self.tokens], gdn.query_start_loc, history,
            gdn.convolution_slots, gdn.fresh, gdn.accepted_tokens, gdn.num_seqs, gdn.num_tokens)
        values = {f"gdn.{name}": getattr(gdn, name) for name in gdn.__dataclass_fields__}
        values.update({f"qsa.{name}": getattr(qsa, name) for name in qsa.__dataclass_fields__})
        values.update(pending_tokens=pending, positions=positions, history=history, active=active)
        values.update(input_values)
        self._launch_runtime(self._runtime_metadata, values)
        return DecoderRuntime(gdn, qsa, ple, values, inputs)

    def finish_runtime(self, runtime, hidden, residual):
        sample = torch.empty((self.rows, 2560), dtype=hidden.dtype, device=hidden.device)
        values = {**runtime.values, "hidden": hidden, "sample": sample, "residual": residual}
        self._launch_runtime(self._runtime_hidden, values)
        if self._runtime_residual is not None:
            self._launch_runtime(self._runtime_residual, values)
        inputs = runtime.inputs
        if self._runtime_validate is None:
            self.token_state.consume(inputs.token_ids[:self.tokens], runtime.gdn.query_start_loc)
        else:
            self.token_state.consume(inputs.token_ids[:self.tokens], runtime.gdn.query_start_loc,
                row_slot_ids=inputs.row_slot_ids[:self.rows], row_generations=inputs.row_generations[:self.rows],
                live_rows=inputs.live_rows)
        return sample



class PreparedResume:
    """Sample directly from an exact restored prefix's final hidden vector."""

    def __init__(self, inputs, requests, pool, sample_hidden, logits, lm_head_weight,
                 completion, frame_index, submission_sequence, *, session, name,
                 local_logits=None, vocabulary_runtime=None, coordinator=None, sampling_workspace=None):
        from ignition.components.linear import UnquantizedProjection
        from ignition.components.sampling import PreparedSampling

        rows = sample_hidden.shape[0]
        if not 1 <= rows <= requests.capacity or logits.shape != (rows, requests.vocab_size):
            raise ValueError("resume hidden/logit row capacities must agree")
        self._gather = compile_kernel(_resume_hidden,
            (pool.last_hidden, sample_hidden, inputs.row_slot_ids, inputs.row_generations, inputs.live_rows,
             requests.generation, requests.status, requests.error_code, requests.consumed_tokens,
             requests.pending_token, inputs.request_state_rows, pool.last_hidden.stride(0), requests.consumed_tokens.stride(0), 4096, requests.capacity),
            (rows, 1, 1))
        self.sample_hidden = sample_hidden
        if vocabulary_runtime is None:
            self.projection = UnquantizedProjection(sample_hidden, lm_head_weight, logits, session=session, name=name)
        else:
            from ignition.components.vocabulary import PreparedVocabularyProjection

            self.projection = PreparedVocabularyProjection(sample_hidden, lm_head_weight, local_logits, logits,
                runtime=vocabulary_runtime, session=session, coordinator=coordinator, name=name)
        self.sampler = PreparedSampling(logits, requests, inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows], inputs.live_rows, completion, frame_index, submission_sequence,
            sample_enabled=inputs.sample_enabled[:rows], workspace=sampling_workspace)
        self.sampler.prepare()
        self.sampler.prepare_broadcast(runtime=vocabulary_runtime, session=session,
            coordinator=coordinator, name=name + ".sampling")

    def functional(self, *, session):
        self.hidden_layout = (tuple(self.sample_hidden.shape), tuple(self.sample_hidden.stride()), self.sample_hidden.dtype)
        args = list(self._gather.args)
        args[1] = None
        self._gather.args = tuple(args)
        self.sample_hidden = None
        self.projection.functional(session=session)
        self.sampler.functional(session=session)
        return self

    def __call__(self):
        from ignition.preparation.triton import launch_triton
        shape, stride, dtype = self.hidden_layout
        hidden = torch.empty_strided(shape, stride, dtype=dtype, device=self.sampler.state.generation.device)
        args = list(self._gather.args)
        args[1] = hidden
        launch_triton(self._gather.kernel, self._gather.compiled, tuple(args), self._gather.grid,
                      **self._gather.options)
        self.sampler.run(self.projection(hidden))
