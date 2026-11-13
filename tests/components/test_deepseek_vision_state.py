# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-owned image features preserve request bounds and sentinel injection."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.deepseek_vision import PrefillVisionInputs, VisionState


def state_with_arena(arena):
    inputs = SimpleNamespace(token_ids=torch.tensor([7, 101, 100]),
                             row_slot_ids=torch.tensor([1]), row_generations=torch.tensor([12]))
    weights = {name: torch.full((4,), value, dtype=torch.bfloat16)
               for name, value in (("image_start", 11), ("image_newline", 13), ("image_end", 17))}
    return VisionState(weights, inputs, PrefillVisionInputs.allocate(3, "cpu"), SimpleNamespace(capacity=2),
        features=8, token_base=100, feature_type=1, image_slots=1, arena=arena,
        sentinel_names=("image_start", "image_start", "image_newline", "image_end"))


def test_caller_owned_arena_retains_storage_and_feature_injection():
    arena = torch.zeros((1, 8, 4), dtype=torch.bfloat16)
    state = state_with_arena(arena)
    assert state.arena is arena
    state.feature_slot.fill_(0)
    state.generation[1], state.count[1] = 12, 1
    state.prefill.vision_token_indices.copy_(torch.tensor([-1, 0, -1]))
    arena[0, 0].fill_(3)
    embedding = torch.ones((3, 4), dtype=torch.bfloat16)
    state.inject(state.inputs, embedding)
    torch.testing.assert_close(embedding, torch.tensor([[1] * 4, [3] * 4, [11] * 4], dtype=torch.bfloat16))


@pytest.mark.parametrize("arena", [torch.empty((2, 8, 4), dtype=torch.bfloat16),
    torch.empty((1, 8, 4)), torch.empty((1, 4, 8), dtype=torch.bfloat16).transpose(1, 2)])
def test_caller_owned_arena_rejects_incompatible_storage(arena):
    with pytest.raises(ValueError, match="image feature arena"):
        state_with_arena(arena)
