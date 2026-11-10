# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Request epochs and rank agreement around immutable continuation transfers."""

import torch
import triton
import triton.language as tl

from .capacity import BATCH_CAPACITIES
from .sampling import (
    PreparedAdmission,
    _check_tensor,
    _compile,
    _completion_fields,
    _write_completion,
)


@triton.jit
def _stage(Slots, Epochs, Count, Sources, Destinations, Operations, StateCount, TransferErrors,
           AdmissionErrors, Generation, Status, Validation, StateRows, RequestSlots, MODE: tl.constexpr, CAPACITY: tl.constexpr, SLOTS: tl.constexpr = 4):
    row = tl.program_id(0)
    count, state_count = tl.load(Count), tl.load(StateCount)
    source, destination, operation = tl.load(Sources + row), tl.load(Destinations + row), tl.load(Operations + row)
    slot, epoch = tl.load(Slots + row), tl.load(Epochs + row)
    present = (row < count) | (row < state_count)
    valid_slot = (slot >= 0) & (slot < SLOTS)
    valid = (count >= 0) & (count <= CAPACITY) & (count == state_count) & valid_slot & (epoch > 0)
    state = tl.load(StateRows + slot * 8, valid_slot, other=0)
    valid &= (state > 0) & (tl.load(RequestSlots + row) == slot)
    if MODE == 2:
        valid &= (operation == 0) & (source == state) & ((destination > 0) & (destination != source))
        valid &= epoch == tl.load(Generation + slot, valid_slot, other=0)
        status = tl.load(Status + slot, valid_slot, other=0)
        valid &= (status == 1) | (status == 2)
        others = tl.arange(0, SLOTS)
        duplicates = (others < count) & (tl.load(Slots + others) == slot)
        valid &= tl.sum(duplicates.to(tl.int32), 0) == 1
    else:
        valid &= (destination == state) & (operation == (1 if MODE == 0 else 0))
        valid &= (source == -1) if MODE == 0 else ((source > 0) & (source != destination))
        valid &= tl.load(AdmissionErrors + row) == 0
    bad = (tl.load(TransferErrors + row) != 0) | (present & ~valid)
    tl.store(Validation + row, bad.to(tl.bfloat16))
    tl.store(Validation + SLOTS + row, present.to(tl.bfloat16))


@triton.jit
def _agree(Totals, Errors, AdmissionErrors, WORLD: tl.constexpr, ADMIT: tl.constexpr, SLOTS: tl.constexpr = 4):
    row = tl.arange(0, SLOTS)
    bad, present = tl.load(Totals + row), tl.load(Totals + SLOTS + row)
    bad = (bad > 0) | ((present > 0) & (present != WORLD))
    error = tl.where(bad, 8, 0)
    tl.store(Errors + row, error)
    if ADMIT:
        tl.store(AdmissionErrors + row, error << 8)


@triton.jit
def _complete(Slots, Epochs, Count, Errors, Generation, Consumed, Generated, Status, StateError,
                       Records, Frame, Sequence, CAPACITY: tl.constexpr, CS: tl.constexpr, RECORD_FIELDS: tl.constexpr, SLOTS: tl.constexpr = 4):
    slot = tl.program_id(0)
    rows = tl.arange(0, SLOTS)
    matches = (rows < CAPACITY) & (rows < tl.load(Count)) & (tl.load(Slots + rows) == slot)
    active = tl.sum(matches.to(tl.int32), 0) > 0
    row = tl.min(tl.where(matches, rows, SLOTS), 0)
    epoch, consumed, generated = tl.full((), 0, tl.int64), tl.full((), 0, tl.int64), tl.full((), 0, tl.int64)
    status, code = tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)
    if active:
        epoch = tl.load(Epochs + row)
        error = tl.load(Errors + row)
        if epoch == tl.load(Generation + slot):
            consumed, generated = tl.load(Consumed + slot * CS), tl.load(Generated + slot)
            status, code = tl.load(Status + slot), tl.load(StateError + slot)
        if error != 0:
            status, code = 3, 64 + error
            consumed, generated = tl.full((), 0, tl.int64), tl.full((), 0, tl.int64)
    _write_completion(Records, Frame, Sequence, slot, epoch, consumed, generated,
                      -1, status, code, active, RECORD_FIELDS, SLOTS)


class PreparedMaintenance:
    """Compose reset/restore admission or snapshot publication with row transfers.

    validate() returns two BF16 flags per command: error and participation.
    The caller reduces them across all TP ranks and writes validation_sum before
    apply(). Rejection or disagreement in row participation suppresses that
    command on every rank.
    Failed commands return code 72 with zero completion counters and preserve
    model and request state. Native ownership remains responsible for snapshot
    reservation and full-page sharing.
    """

    def __init__(self, mode, requests, transfer, slots, epochs, count, completion, frame, sequence, *,
                 world_size, capacity=4, admission_buffers=None, buffers=None):
        if mode not in ("init", "restore", "snapshot") or (type(world_size) is not int or world_size < 1) or capacity not in BATCH_CAPACITIES or capacity > requests.capacity:
            raise ValueError("maintenance requires a supported transition, rank count, and row capacity")
        self.slots = requests.capacity
        device = transfer.pool.backing.device
        for value, shape, dtype, name in ((slots, (self.slots,), torch.int32, "slots"),
                (epochs, (self.slots,), torch.int64, "epochs"), (count, (1,), torch.int32, "count"),
                (frame, (1,), torch.int32, "frame"), (sequence, (1,), torch.int64, "sequence")):
            _check_tensor(value, shape, dtype, device, name)
        records = completion.records
        fields = _completion_fields(records, device, self.slots)
        self.transfer, self.requests, self.admission = transfer, requests, None
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)
        self.validation_input = zeros((1, max(8, 2 * self.slots)), torch.bfloat16)
        self.validation_sum = zeros(self.validation_input.shape, torch.bfloat16)
        if mode != "snapshot":
            if admission_buffers is None:
                raise ValueError("reset and restore require request admission buffers")
            for value, expected in ((slots, admission_buffers.admission_slots),
                    (epochs, admission_buffers.admission_generations), (count, admission_buffers.admission_count)):
                if value.data_ptr() != expected.data_ptr():
                    raise ValueError("maintenance and admission must share packed slot commands")
            self.admission = PreparedAdmission(requests, admission_buffers, completion, frame, sequence, capacity=capacity)
        self.admission_errors = self.admission.errors if self.admission is not None else torch.zeros(self.slots, dtype=torch.int32, device=device)
        c, r = transfer.commands, requests
        self.stage_args = (slots, epochs, count, c.source_slots, c.destination_slots, c.operation, c.num_ops, c.errors,
            self.admission_errors, r.generation, r.status, self.validation_input, transfer.state_rows, c.request_slots, ("init", "restore", "snapshot").index(mode), capacity, self.slots)
        self.agree_args = (self.validation_sum, c.errors, self.admission_errors, world_size, self.admission is not None, self.slots)
        self.complete_args = (slots, epochs, count, c.errors, r.generation, r.consumed_tokens, r.generated_tokens,
            r.status, r.error_code, records, frame, sequence, capacity, r.consumed_tokens.stride(0), fields, self.slots)
        from .state import PreparedRequestHistory

        self.history = PreparedRequestHistory(transfer.pool, requests, c, snapshot=mode == "snapshot")
        self.snapshot = mode == "snapshot"
        self._stage = None
        self.local_metadata = False

    def prepare(self):
        self.transfer.prepare()
        if self.admission is not None:
            self.admission.prepare()
        self._stage = _compile(_stage, self.stage_args, (self.slots, 1, 1))
        self._agree = _compile(_agree, self.agree_args, (1, 1, 1))
        self._complete = _compile(_complete, self.complete_args, (self.slots, 1, 1))

    def functional(self):
        if self._stage is None:
            raise RuntimeError("maintenance prepare() must complete before functional conversion")
        self._stage.args = tuple(None if i in (8, 11) else value
                                 for i, value in enumerate(self._stage.args))
        self._agree.args = tuple(None if i in (0, 2) else value
                                 for i, value in enumerate(self._agree.args))
        self.stage_args = self.agree_args = ()
        if self.admission is not None:
            self.admission.functional()
        self.validation_input = self.validation_sum = self.admission_errors = None
        self.local_metadata = True
        return self

    def validate(self):
        if self._stage is None:
            raise RuntimeError("maintenance prepare() must complete before capture")
        self.transfer.validate()
        if self.local_metadata:
            from ignition.preparation.triton import launch_triton

            device = self.requests.generation.device
            validation = torch.zeros((1, max(8, 2 * self.slots)), dtype=torch.bfloat16, device=device)
            admission_errors = (self.admission.validate() if self.admission is not None
                                else torch.zeros(self.slots, dtype=torch.int32, device=device))
            args = list(self._stage.args)
            args[8], args[11] = admission_errors, validation
            launch_triton(self._stage.kernel, self._stage.compiled, tuple(args), self._stage.grid,
                          **self._stage.options)
            return validation, admission_errors
        else:
            if self.admission is not None:
                self.admission.validate()
            self._stage()
            return self.validation_input

    def apply(self, totals=None, *, admission_errors=None):
        if self._stage is None:
            raise RuntimeError("maintenance prepare() must complete before capture")
        if self.local_metadata:
            from ignition.preparation.triton import launch_triton

            if totals is None or admission_errors is None:
                raise ValueError("functional maintenance requires validation totals and admission errors")
            args = list(self._agree.args)
            args[0], args[2] = totals, admission_errors
            launch_triton(self._agree.kernel, self._agree.compiled, tuple(args), self._agree.grid,
                          **self._agree.options)
        else:
            self._agree()
        if self.snapshot:
            self.history()
        self.transfer.apply()
        if not self.snapshot:
            self.history()
        if self.admission is not None:
            if self.local_metadata:
                self.admission.apply(admission_errors)
            else:
                self.admission.apply()
        self._complete()
