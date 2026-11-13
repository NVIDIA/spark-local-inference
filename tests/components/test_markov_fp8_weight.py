# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""FP8 Markov checkpoint conversion retains explicit row reconstruction scales."""

import pytest
import torch

from ignition.components.markov_sampling import RowScaledFP8MarkovWeight, fp8_markov_head


def test_markov_fp8_rows_follow_amax_scale_and_preserve_checkpoint_weight():
    weight = torch.stack((torch.linspace(-1, 1, 64), torch.zeros(64),
                          torch.arange(64).float() / 8)).bfloat16()
    saved = weight.clone()
    packed = fp8_markov_head(weight, rows_per_chunk=1)
    expected_scale = weight.float().abs().amax(dim=1).clamp_min(1e-12) / 448.0
    expected_values = (weight.float() / expected_scale[:, None]).to(torch.float8_e4m3fn)
    assert packed.out_features == 3 and packed.in_features == 64
    assert torch.equal(packed.scale, expected_scale)
    assert torch.equal(packed.values.view(torch.uint8), expected_values.view(torch.uint8))
    assert torch.equal(weight, saved)


def test_markov_fp8_rows_reject_invalid_weights_and_storage():
    with pytest.raises(ValueError, match="contiguous BF16"):
        fp8_markov_head(torch.zeros((2, 64), dtype=torch.float32))
    with pytest.raises(ValueError, match="finite checkpoint"):
        fp8_markov_head(torch.full((2, 64), float("nan"), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="row scales"):
        RowScaledFP8MarkovWeight(torch.empty((2, 64), dtype=torch.float8_e4m3fn),
                                torch.empty((2,), dtype=torch.bfloat16))
    for shape in ((0, 64), (2, 0), (2, 32)):
        with pytest.raises(ValueError, match="contiguous BF16"):
            fp8_markov_head(torch.empty(shape, dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="row-scaled FP8"):
            RowScaledFP8MarkovWeight(torch.empty(shape, dtype=torch.float8_e4m3fn),
                                    torch.empty((shape[0],), dtype=torch.float32))


def test_markov_fp8_weight_views_and_numeric_owners_survive_preparation(monkeypatch):
    from ignition.components import markov_sampling
    from ignition.preparation.rebuild_manifest import rebuild_weights_metadata
    from ignition.preparation.tensor_metadata import NumericStorage

    packed = fp8_markov_head(torch.ones((2, 64), dtype=torch.bfloat16))
    source = torch.zeros((1, 64), dtype=torch.bfloat16)
    output = torch.empty((1, 2), dtype=torch.float32)
    monkeypatch.setattr(markov_sampling, "register_finalizer", lambda session, callback: None)
    with NumericStorage() as numeric:
        projection = markov_sampling.PreparedRowScaledFP8MarkovProjection(source, packed, output,
            session=object(), name="markov")
    assert projection.weight is packed
    ranges = numeric.ranges()
    for tensor in (packed.values, packed.scale):
        assert any(start <= tensor.data_ptr() and end >= tensor.data_ptr() + tensor.nbytes
                   for start, end in ranges)
    catalog = rebuild_weights_metadata({"markov": packed})
    views = {item["name"]: item for item in catalog["tensors"]}
    assert views["markov.values"]["dtype"] == "float8_e4m3fn"
    assert views["markov.scale"]["dtype"] == "float32"
    assert not catalog["unavailable"]
