# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved hidden-output gathering for sampling an exact cached prefix."""

import torch

from .sampling import ERROR, RUNNING, _check_tensor
from .state import gather_rows


class PreparedResumeState:
    """Validate rank participation, then gather live-slot hidden outputs.

    A cached prefix must contain consumed tokens and have no pending sample.
    Sampling state belongs to the admitted request; resume never consumes a
    token or modifies the cached continuation. The caller reduces validation
    flags across ranks before gather(), using the same world size as preparation.
    """

    def __init__(self, requests, saved_hidden, slots, epochs, count, *, world_size, state_rows, buffers=None):
        self.state_rows = state_rows
        self.capacity = requests.capacity
        rows, device = slots.numel(), requests.generation.device
        if device.type != "cuda" or (type(world_size) is not int or world_size < 1) or not 1 <= rows <= self.capacity:
            raise ValueError("resume requires CUDA request state, TP1/2/4, and rows within request capacity")
        if saved_hidden.ndim != 2 or saved_hidden.shape[0] < 2 or saved_hidden.shape[1] <= 0:
            raise ValueError("resume requires saved hidden outputs for all physical request slots")
        _check_tensor(saved_hidden, saved_hidden.shape, torch.bfloat16, device, "saved_hidden", contiguous=False)
        if saved_hidden.stride(1) != 1 or saved_hidden.stride(0) < saved_hidden.shape[1]:
            raise ValueError("saved hidden rows must be disjoint with contiguous channels")
        for value, shape, dtype, name in ((slots, (rows,), torch.int32, "slots"),
                (epochs, (rows,), torch.int64, "epochs"), (count, (1,), torch.int32, "count")):
            _check_tensor(value, shape, dtype, device, name)
        self.requests, self.saved_hidden = requests, saved_hidden
        self.slots, self.epochs, self.count, self.world_size = slots, epochs, count, world_size
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)
        self.validation_input = zeros((1, max(8, 2 * self.capacity)))
        self.validation_sum = zeros(self.validation_input.shape)
        self.hidden = zeros((rows, saved_hidden.shape[1]))
        self.hidden_shape = (rows, saved_hidden.shape[1])

    def validate(self):
        r, slots = self.requests, self.slots.long()
        row_ids = torch.arange(self.slots.numel(), device=slots.device)
        slot_ids = torch.arange(self.capacity, device=slots.device)
        validation = torch.zeros((1, max(8, 2 * self.capacity)), dtype=torch.bfloat16, device=slots.device)
        safe = slots.clamp(0, self.capacity - 1)
        present = (row_ids < self.count) & (slots >= 0) & (slots < self.capacity)
        present &= (self.epochs == r.generation[safe]) & (r.status[safe] == RUNNING)
        duplicate = ((slots[:, None] == slots[None, :]) & present[:, None] & present[None, :]).sum(1) > 1
        valid = (self.count <= self.slots.numel()) & ~duplicate
        valid &= (r.consumed_tokens[safe] > 0) & (r.pending_token[safe] < 0)
        slot_rows = slot_ids[:, None] == slots[None, :]
        validation[0, :self.capacity].copy_((slot_rows & (present & ~valid)[None, :]).any(1))
        present = (slot_rows & present[None, :]).any(1)
        validation[0, self.capacity:2 * self.capacity].copy_(present)
        return validation, present

    def functional(self):
        self.hidden = None
        self.validation_input = self.validation_sum = None
        return self

    def gather(self, totals, *, present):
        r = self.requests
        participants = totals[0, self.capacity:2 * self.capacity]
        mismatch = (participants > 0) & (participants != self.world_size)
        bad = present & ((totals[0, :self.capacity] > 0) | mismatch)
        r.status.copy_(torch.where(bad, ERROR, r.status))
        r.error_code.copy_(torch.where(bad, 33, r.error_code))
        safe = self.slots.long().clamp(0, self.capacity - 1)
        row_ids = torch.arange(self.slots.numel(), device=self.slots.device)
        active = (row_ids < self.count) & (self.slots >= 0) & (self.slots < self.capacity)
        active &= (self.epochs == r.generation[safe]) & (r.status[safe] == RUNNING)
        hidden = torch.empty(self.hidden_shape, dtype=torch.bfloat16, device=self.saved_hidden.device)
        gather_rows(self.saved_hidden, self.state_rows[safe, 0], hidden)
        hidden.masked_fill_(~active[:, None], 0)
        return hidden
