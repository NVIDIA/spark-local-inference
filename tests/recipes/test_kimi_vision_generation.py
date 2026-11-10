# SPDX-License-Identifier: Apache-2.0

"""MoonViT validation produces local operands before feature publication."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3 import vision_state


@pytest.mark.parametrize("invalid", [False, True])
def test_vision_generation_sanitizes_local_operands(monkeypatch, invalid):
    captured = {}

    def encoder_call(patches, positions, indices, weights, merge, cu):
        captured["operands"] = patches, positions, indices, weights, merge, cu
        return torch.arange(4, dtype=torch.bfloat16).reshape(1, 4)

    class Encoder:
        capacity = 4
        __call__ = staticmethod(encoder_call)

    encoder = Encoder()
    encoder.inputs = SimpleNamespace(
        vision_slot=torch.tensor([1]), vision_generation=torch.tensor([12]),
        vision_num_patches=torch.tensor([4]), vision_num_features=torch.tensor([1]),
        vision_feature_offset=torch.tensor([0]), vision_cu_seqlens=torch.tensor([0, 4]),
        vision_position_ids=torch.zeros((4, 2), dtype=torch.int64),
        vision_interpolation_indices=torch.zeros((4, 4), dtype=torch.int64),
        vision_interpolation_weights=torch.full((4, 4), .25),
        vision_merge_indices=torch.arange(4).reshape(1, 4),
        vision_pixel_values=torch.full((4, 588), 2., dtype=torch.float32))
    if invalid:
        encoder.inputs.vision_pixel_values[0, 0] = float("nan")
    state = SimpleNamespace(arena=torch.zeros((2, 4, 2), dtype=torch.bfloat16),
        requests=SimpleNamespace(capacity=2, generation=torch.tensor([0, 12]), status=torch.tensor([0, 1])),
        generation=torch.tensor([0, 12]), count=torch.tensor([0, 0]), feature_slot=torch.tensor([1]))

    def publication(*args):
        def publish(features, count, errors):
            captured["publication"] = features, count, errors
        return publish

    monkeypatch.setattr(vision_state, "PreparedFeaturePublication", publication)
    forward = vision_state.PreparedVisionGeneration(encoder, state, None, None, None, feature_range=(1, 3))
    forward()
    patches, positions, indices, weights, merge, cu = captured["operands"]
    features, count, errors = captured["publication"]
    assert patches.dtype == torch.bfloat16
    assert torch.isfinite(patches).all()
    assert errors.tolist() == [int(invalid)]
    assert count.tolist() == [0 if invalid else 1]
    torch.testing.assert_close(features, torch.tensor([[1., 2.]], dtype=torch.bfloat16))
    assert cu.tolist() == ([0, 0] if invalid else [0, 4])
    if invalid:
        for tensor in (patches, positions, indices, weights, merge):
            assert torch.count_nonzero(tensor) == 0
    else:
        assert torch.all(patches == 2)
        torch.testing.assert_close(weights, encoder.inputs.vision_interpolation_weights)
    first_patches = patches
    forward()
    assert captured["operands"][0].data_ptr() != first_patches.data_ptr()
