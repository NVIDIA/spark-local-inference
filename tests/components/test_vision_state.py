# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image feature delivery remains scoped to request generations and token positions."""

from types import SimpleNamespace

import torch

from ignition.components.vision_state import PrefillVisionInputs, VisionState


def test_prefill_features_require_matching_generation_and_bounds():
    inputs = SimpleNamespace(token_ids=torch.tensor([7, 99, 99, 0]),
                             row_slot_ids=torch.tensor([1]), row_generations=torch.tensor([12]))
    prefill = PrefillVisionInputs.allocate(4, "cpu")
    state = VisionState(inputs, prefill, SimpleNamespace(capacity=2),
                        image_token=99, features=8, hidden=4)
    state.feature_slot.fill_(1)
    state.generation[1] = 12
    state.count[1] = 2
    prefill.vision_token_indices.copy_(torch.tensor([-1, 0, 1, -1]))
    lengths = torch.tensor([3])
    assert state.validate_prefill(inputs.token_ids, lengths)
    state.generation[1] = 11
    assert not state.validate_prefill(inputs.token_ids, lengths)
    state.generation[1] = 12
    prefill.vision_token_indices[2] = 2
    assert not state.validate_prefill(inputs.token_ids, lengths)
    prefill.vision_token_indices[2] = 1
    prefill.vision_token_indices[0] = 0
    assert not state.validate_prefill(inputs.token_ids, lengths)


def test_injection_preserves_text_and_selects_request_features():
    inputs = SimpleNamespace(token_ids=torch.tensor([7, 99, 99]),
                             row_slot_ids=torch.tensor([1]), row_generations=torch.tensor([12]))
    prefill = PrefillVisionInputs.allocate(3, "cpu")
    state = VisionState(inputs, prefill, SimpleNamespace(capacity=2),
                        image_token=99, features=8, hidden=4)
    state.feature_slot.fill_(1)
    state.arena[0].fill_(100)
    state.arena[1, 0].fill_(3)
    state.arena[1, 1].fill_(5)
    prefill.vision_token_indices.copy_(torch.tensor([-1, 1, 0]))
    embedding = torch.ones((3, 4), dtype=torch.bfloat16)
    state.inject(inputs, embedding)
    torch.testing.assert_close(embedding, torch.tensor([[1] * 4, [5] * 4, [3] * 4], dtype=torch.bfloat16))


def test_feature_slots_are_independent_of_request_slots_and_text_needs_none():
    inputs = SimpleNamespace(token_ids=torch.tensor([7, 99]),
                             row_slot_ids=torch.tensor([7]), row_generations=torch.tensor([12]))
    prefill = PrefillVisionInputs.allocate(2, "cpu")
    state = VisionState(inputs, prefill, SimpleNamespace(capacity=8),
                        image_slots=1, image_token=99, features=8, hidden=4)
    assert state.arena.shape == (1, 8, 4)
    state.generation[7], state.count[7] = 12, 1
    state.feature_slot.fill_(0)
    state.arena[0, 0].fill_(3)
    prefill.vision_token_indices.copy_(torch.tensor([-1, 0]))
    assert state.validate_prefill(inputs.token_ids, torch.tensor([2]))
    embedding = torch.ones((2, 4), dtype=torch.bfloat16)
    state.inject(inputs, embedding)
    torch.testing.assert_close(embedding, torch.tensor([[1] * 4, [3] * 4], dtype=torch.bfloat16))
    for slot in (-1, 1):
        state.feature_slot.fill_(slot)
        assert not state.validate_prefill(inputs.token_ids, torch.tensor([2]))
    inputs.token_ids.fill_(7)
    prefill.vision_token_indices.fill_(-1)
    state.feature_slot.fill_(-1)
    assert state.validate_prefill(inputs.token_ids, torch.tensor([2]))
    state.inject(inputs, embedding)
    torch.testing.assert_close(embedding, torch.tensor([[1] * 4, [3] * 4], dtype=torch.bfloat16))
