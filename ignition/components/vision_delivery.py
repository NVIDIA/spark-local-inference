# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish bounded image features and complete a scheduled vision graph."""

import torch
import triton
import triton.language as tl

from .sampling import _completion_fields, _write_completion


@triton.jit
def _publish(Output, Arena, Slot, Offset, Count, Error, FEATURES: tl.constexpr, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = (index < tl.load(Count) * HIDDEN) & (tl.load(Error) == 0)
    values = tl.load(Output + index, valid, other=0)
    target = (tl.load(Slot).to(tl.int64) * FEATURES + tl.load(Offset)) * HIDDEN + index
    tl.store(Arena + target, values, valid)


@triton.jit
def _finish(
    Slot,
    Epoch,
    Count,
    Offset,
    Error,
    FeatureEpoch,
    FeatureCount,
    Generation,
    Consumed,
    Generated,
    Status,
    StateError,
    Records,
    Frame,
    Sequence,
    CS: tl.constexpr,
    RECORD_FIELDS: tl.constexpr,
    SLOTS: tl.constexpr,
):
    slot = tl.program_id(0)
    active = tl.load(Slot) == slot
    epoch = tl.load(Epoch)
    current = active & (epoch == tl.load(Generation + slot))
    error = tl.load(Error)
    status, code = tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)
    if active:
        if error == 0:
            tl.store(FeatureEpoch + slot, epoch)
            tl.store(FeatureCount + slot, tl.load(Offset) + tl.load(Count))
            status = 1
        else:
            status, code = 3, 128 + error
            if current & (tl.load(Status + slot) == 1):
                tl.store(Status + slot, status)
                tl.store(StateError + slot, code)
    consumed = tl.load(Consumed + slot * CS, current, other=0)
    generated = tl.load(Generated + slot, current, other=0)
    _write_completion(
        Records,
        Frame,
        Sequence,
        slot,
        epoch,
        consumed,
        generated,
        -1,
        status,
        code,
        active,
        RECORD_FIELDS,
        SLOTS,
    )


class PreparedFeaturePublication:
    def __init__(self, state, inputs, completion, frame, sequence):
        records = completion.records
        self.state, self.inputs = state, inputs
        self.records, self.frame, self.sequence = records, frame, sequence
        self.record_fields = _completion_fields(records, records.device, state.requests.capacity)

    def __call__(self, output, count, errors):
        s, i = self.state, self.inputs
        r = s.requests
        torch.library.wrap_triton(_publish)[(triton.cdiv(output.numel(), 1024),)](
            output, s.arena, s.feature_slot, i.vision_feature_offset, count, errors,
            s.arena.shape[1], s.arena.shape[2], 1024)
        torch.library.wrap_triton(_finish)[(r.capacity,)](
            i.vision_slot, i.vision_generation, count, i.vision_feature_offset, errors,
            s.generation, s.count, r.generation, r.consumed_tokens, r.generated_tokens,
            r.status, r.error_code, self.records, self.frame, self.sequence,
            r.consumed_tokens.stride(0), self.record_fields, r.capacity)
