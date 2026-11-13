# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen request defaults and PLE history alongside shared device sampling state."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ignition.components.sampling import (
    _check_tensor,
)
from ignition.components.sampling import (
    DeviceRequestState as SamplingState,
)
from ignition.components.sampling import (
    RequestAdmission as SamplingAdmission,
)

VOCAB_SIZE = 248320


@dataclass(frozen=True)
class RequestAdmission(SamplingAdmission):
    eos_token_ids: tuple[int, ...] = (248046, 248044)

    def validate(self, vocab_size: int = VOCAB_SIZE) -> None:
        super().validate(vocab_size)


@dataclass
class DeviceRequestState(SamplingState):
    ple_history: torch.Tensor

    @classmethod
    def allocate(cls, device, *, vocab_size: int = VOCAB_SIZE,
                 consumed_tokens: torch.Tensor | None = None,
                 ple_history: torch.Tensor | None = None, capacity: int = 4) -> DeviceRequestState:
        if type(vocab_size) is not int or not 1 <= vocab_size <= VOCAB_SIZE:
            raise ValueError("vocab_size must be in [1, 248320]")
        state = SamplingState.allocate(device, vocab_size=vocab_size, consumed_tokens=consumed_tokens, capacity=capacity)
        history = torch.full((state.capacity, 2), 248044, dtype=torch.int64, device=device) if ple_history is None else ple_history
        _check_tensor(history, (state.capacity, 2), torch.int64, state.generation.device, "ple_history", contiguous=False)
        if history.stride(0) <= 0 or history.stride(1) != 1:
            raise ValueError("ple_history requires positive row stride and contiguous history columns")
        return cls(**{field.name: getattr(state, field.name) for field in fields(SamplingState)}, ple_history=history)

    def admit(self, slot: int, admission: RequestAdmission, prompt_ids, *, consumed_tokens: int = 0,
              ple_history: tuple[int, int] = (248044, 248044)) -> None:
        if len(ple_history) != 2 or any(type(x) is not int or not 0 <= x < VOCAB_SIZE for x in ple_history):
            raise ValueError("PLE history requires two token IDs")
        super().admit(slot, admission, prompt_ids, consumed_tokens=consumed_tokens)
        self.ple_history[slot].copy_(torch.tensor(ple_history))
