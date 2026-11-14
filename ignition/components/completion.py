# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Persistent two-frame generation outputs and their request flags."""

from dataclasses import dataclass, field

import torch

from ignition.components.capacity import completion_token_capacity
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.persistent import persistent_zeros

MAX_TOP_LOGPROBS = 20
MAX_LOGPROB_TOKEN_IDS = 128
COMPLETION_FRAMES = 2
TOKEN_FIELD = 4
EMITTED_COUNT_FIELD = 8
VECTOR_TOKENS_FIELD = 16
# Each token/logprob pair occupies two int32 words, starting with the chosen token.
LOGPROB_PAIR_FIELDS = 2
LOGPROB_FIELDS = LOGPROB_PAIR_FIELDS * (1 + MAX_LOGPROB_TOKEN_IDS)


@dataclass(frozen=True)
class CompletionOutputs:
    """Own persistent outputs shared by the serial graph catalog.

    ``records`` is int64 [2, slots, 16 + token_capacity]. The runtime
    alternates frames so readback can retain one while the next graph writes.
    ``logprobs`` is int32 [2, slots, token_capacity, 258]: the chosen token followed by up to
    128 requested alternatives, each packed as [token ID, FP32 logprob bits]. Per-slot
    int32 ``requested`` flags gate reductions; only committed positions are valid.
    Mixed-role outputs remain disjoint until their active-frame merge.
    """
    records: torch.Tensor
    logprobs: torch.Tensor
    requested: torch.Tensor
    token_ids: torch.Tensor
    token_id_counts: torch.Tensor
    speculative_survival: torch.Tensor | None = None
    _private: list = field(default_factory=list, compare=False, repr=False)

    @classmethod
    def allocate(cls, shape, *, device):
        if len(shape) != 3:
            raise ValueError("completion records require [2,slots,16+token_capacity]")
        frames, slots, fields = shape
        if (type(frames) is not int or frames != COMPLETION_FRAMES or type(slots) is not int or slots < 1
                or type(fields) is not int or fields <= VECTOR_TOKENS_FIELD):
            raise ValueError("completion records require [2,slots,16+token_capacity] with positive slots and token capacity")
        tokens = completion_token_capacity(fields - VECTOR_TOKENS_FIELD)
        return cls(torch.zeros(shape, dtype=torch.int64, device=device),
            torch.zeros((frames, slots, tokens, LOGPROB_FIELDS), dtype=torch.int32, device=device),
            torch.zeros(slots, dtype=torch.int32, device=device),
            torch.zeros((slots, MAX_LOGPROB_TOKEN_IDS), dtype=torch.int32, device=device),
            torch.zeros(slots, dtype=torch.int32, device=device),
            torch.full((frames, slots, tokens - 1), -1., dtype=torch.float32, device=device) if tokens > 1 else None)

    @property
    def token_capacity(self):
        return completion_record_capacity(self.records)

    def buffers(self):
        result = dict(completion_records=self.records, completion_logprobs=self.logprobs,
                      logprob_requested=self.requested, logprob_token_ids=self.token_ids,
                      logprob_token_id_counts=self.token_id_counts)
        if self.speculative_survival is not None:
            result["completion_speculative_survival"] = self.speculative_survival
        return result

    @property
    def nbytes(self):
        return sum(t.nbytes for t in self.buffers().values()) + sum(
            t.nbytes for child in self._private for t in (child.records, child.logprobs, child.speculative_survival) if t is not None)

    def zero_ranges(self):
        return storage_ranges([self.buffers(), *(child.buffers() for child in self._private)])

    def private(self):
        """Allocate disjoint mixed-role outputs while sharing request flags."""
        device = self.records.device
        child = CompletionOutputs(persistent_zeros(self.records.shape, torch.int64, device),
            persistent_zeros(self.logprobs.shape, torch.int32, device), self.requested,
            self.token_ids, self.token_id_counts,
            persistent_zeros(self.speculative_survival.shape, torch.float32, device).fill_(-1.)
            if self.speculative_survival is not None else None)
        self._private.append(child)
        return child

    def merge(self, prompt, decode, prompt_slot, frame):
        for value in (prompt, decode):
            if value.token_capacity != self.token_capacity:
                raise ValueError("mixed completion roles must share a token capacity")
        tokens = self.token_capacity
        slots = torch.arange(self.records.shape[1], device=self.records.device)
        role = slots == prompt_slot
        index = frame.long()
        for output, a, b in ((self.records, prompt.records, decode.records),
                             (self.speculative_survival, prompt.speculative_survival, decode.speculative_survival)):
            if output is None:
                continue
            mask = role.view(1, -1, *([1] * (output.ndim - 2)))
            output.index_copy_(0, index, torch.where(mask,
                -1. if output is self.speculative_survival else a.index_select(0, index), b.index_select(0, index)))
        from .logprobs import _merge
        torch.library.wrap_triton(_merge)[(self.records.shape[1], tokens)](
            self.logprobs, prompt.logprobs, decode.logprobs, self.token_id_counts,
            prompt_slot, frame, self.records.shape[1], LOGPROB_FIELDS,
            tokens, MAX_TOP_LOGPROBS, 512,
        )


def completion_record_capacity(records, capacity=None):
    shape = tuple(records.shape)
    if (len(shape) != 3 or shape[0] != COMPLETION_FRAMES or shape[1] < 1
            or (capacity is not None and shape[1] != capacity) or shape[2] <= VECTOR_TOKENS_FIELD
            or str(records.dtype) != "torch.int64"):
        raise ValueError("completion_records requires int64 [2,slots,16+token_capacity]")
    return completion_token_capacity(shape[2] - VECTOR_TOKENS_FIELD)


def validate_completion_buffers(buffers, capacity, token_capacity=None):
    tokens = completion_record_capacity(buffers["completion_records"], capacity)
    if token_capacity is not None and completion_token_capacity(token_capacity) != tokens:
        raise ValueError("completion token capacity disagrees with completion_records geometry")
    if (survival := buffers.get("completion_speculative_survival")) is not None:
        if tokens == 1:
            raise ValueError("completion_speculative_survival must be omitted at token capacity one")
        shape = (COMPLETION_FRAMES, capacity, tokens - 1)
        if tuple(survival.shape) != shape or str(survival.dtype) != "torch.float32":
            raise ValueError(f"completion_speculative_survival: expected shape {shape}, dtype torch.float32")
    names = ("completion_logprobs", "logprob_requested", "logprob_token_ids", "logprob_token_id_counts")
    present = [name in buffers for name in names]
    if not any(present):
        return tokens
    required = names if any(present[2:]) else names[:2]
    for name in required:
        if name not in buffers:
            raise ValueError(f"generation logprobs missing buffer {name}")
    fields = buffers[names[0]].shape[-1]
    if (any(present[2:]) and fields != LOGPROB_FIELDS) or (
            not any(present[2:]) and fields not in range(4, 2 * (1 + MAX_TOP_LOGPROBS) + 1, 2)):
        raise ValueError("completion_logprobs: expected shape with a valid alternative count and dtype torch.int32; "
                         f"got shape {tuple(buffers[names[0]].shape)}, dtype {buffers[names[0]].dtype}")
    for name, shape in ((names[0], (COMPLETION_FRAMES, capacity, tokens, fields)),
                        (names[1], (capacity,)), (names[2], (capacity, MAX_LOGPROB_TOKEN_IDS)),
                        (names[3], (capacity,))):
        if name not in buffers:
            continue
        tensor = buffers[name]
        if tuple(tensor.shape) != shape or str(tensor.dtype) != "torch.int32":
            raise ValueError(f"{name}: expected shape {shape}, dtype torch.int32; "
                             f"got shape {tuple(tensor.shape)}, dtype {tensor.dtype}")

    return tokens
