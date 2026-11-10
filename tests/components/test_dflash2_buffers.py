# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 target-feature capture and geometry-only draft scratch."""

from types import SimpleNamespace

import torch

from ignition.components.dflash2 import DraftScratch, PreparedContextCapture
from ignition.components.dflash2_geometry import DraftAttentionGeometry
from ignition.components.parallel import TensorParallelShard


def draft_config():
    return SimpleNamespace(
        attention=DraftAttentionGeometry(8, 4, 128, 128, 16, 10000., 2048),
        text=dict(hidden_size=64, intermediate_size=128, vocab_size=256),
        block_size=8,
    )


def capture_pair(metadata, hidden, *, columns, window, compact):
    bound = PreparedContextCapture(metadata, columns, window, compact_prefill=compact)
    bound.begin()
    functional = bound.functional()
    state = functional.begin(metadata)
    width = hidden[0].shape[-1]
    for index, values in enumerate(hidden):
        column = index * width
        bound(values, column)
        state = functional(state, values, column)
    torch.testing.assert_close(state[0], bound.output, rtol=0, atol=0)
    return functional, state


def test_functional_context_capture_matches_bound_compact_suffix_selection():
    metadata = SimpleNamespace(
        positions=torch.tensor([0, 1, 2, 3, 4, -1, -1, -1], dtype=torch.int64),
        starts=torch.tensor([0], dtype=torch.int64),
        lengths=torch.tensor([5], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
    )
    first = torch.arange(32, dtype=torch.bfloat16).view(8, 4)
    second = first + 100
    _, (features, indices, live) = capture_pair(metadata, (first, second),
        columns=8, window=3, compact=True)

    expected = torch.cat((first[2:5], second[2:5]), dim=-1)
    torch.testing.assert_close(features, expected, rtol=0, atol=0)
    torch.testing.assert_close(indices, torch.tensor([2, 3, 4]), rtol=0, atol=0)
    torch.testing.assert_close(live, torch.tensor([True, True, True]), rtol=0, atol=0)


def test_functional_context_capture_matches_bound_full_rows():
    metadata = SimpleNamespace(
        positions=torch.tensor([4, 5, 0, 1], dtype=torch.int64),
        starts=torch.tensor([4, 0], dtype=torch.int64),
        lengths=torch.tensor([2, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    first = torch.arange(16, dtype=torch.bfloat16).view(4, 4)
    second = first + 40
    _, (features, indices, live) = capture_pair(metadata, (first, second),
        columns=8, window=4, compact=False)

    torch.testing.assert_close(features, torch.cat((first, second), dim=-1), rtol=0, atol=0)
    assert indices is None and live is None


def test_functional_context_capture_keeps_invocation_outputs_independent():
    metadata = SimpleNamespace(
        positions=torch.tensor([0, 1, 2, 3, 4, -1, -1, -1], dtype=torch.int64),
        starts=torch.tensor([0], dtype=torch.int64),
        lengths=torch.tensor([5], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
    )
    functional = PreparedContextCapture(metadata, 4, 3, compact_prefill=True).functional()
    assert functional.metadata is None
    first, second = functional.begin(metadata), functional.begin(metadata)
    assert first[0].data_ptr() != second[0].data_ptr()
    hidden = torch.arange(32, dtype=torch.bfloat16).view(8, 4)
    functional(first, hidden, 0)
    first_value = first[0].clone()
    functional(second, hidden + 200, 0)
    torch.testing.assert_close(first[0], first_value, rtol=0, atol=0)
    torch.testing.assert_close(second[0], hidden[2:5] + 200, rtol=0, atol=0)


def test_functional_context_capture_uses_each_invocations_metadata():
    first_metadata = SimpleNamespace(
        positions=torch.tensor([0, 1, 2, 3, 4, -1, -1, -1], dtype=torch.int64),
        starts=torch.tensor([0], dtype=torch.int64),
        lengths=torch.tensor([5], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
    )
    second_metadata = SimpleNamespace(
        positions=torch.tensor([0, 1, -1, -1, -1, -1, -1, -1], dtype=torch.int64),
        starts=torch.tensor([0], dtype=torch.int64),
        lengths=torch.tensor([2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
    )
    functional = PreparedContextCapture(first_metadata, 4, 3, compact_prefill=True).functional()
    hidden = torch.arange(32, dtype=torch.bfloat16).view(8, 4)
    first = functional.begin(first_metadata)
    second = functional.begin(second_metadata)
    functional(first, hidden, 0)
    functional(second, hidden + 200, 0)

    torch.testing.assert_close(first[0], hidden[2:5], rtol=0, atol=0)
    expected = torch.cat((hidden[:2] + 200, torch.zeros((1, 4), dtype=torch.bfloat16)), dim=0)
    torch.testing.assert_close(second[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(first[1], torch.tensor([2, 3, 4]), rtol=0, atol=0)
    torch.testing.assert_close(second[1], torch.tensor([0, 1, 2]), rtol=0, atol=0)
    torch.testing.assert_close(first[2], torch.tensor([True, True, True]), rtol=0, atol=0)
    torch.testing.assert_close(second[2], torch.tensor([True, True, False]), rtol=0, atol=0)


def test_draft_scratch_tracks_geometry_without_device_buffers():
    config = draft_config()
    scratch = DraftScratch.allocate(config, TensorParallelShard(0, 2), 32, "cpu", block_size=8)
    rows = scratch.rows(16, 16)

    assert scratch.logit_shape == (32, 256)
    assert rows.logit_shape == (16, 256)
    assert rows.hidden_size == scratch.hidden_size == 64
    assert rows.local_hidden_size == scratch.local_hidden_size == 32
    assert rows.local_vocabulary_size == scratch.local_vocabulary_size == 128
    assert rows.block_size == scratch.block_size == 8
    assert rows.window == scratch.window == 2048
    assert scratch.scratch_tensors() == ()


def test_hyperconnection_variants_share_exact_immutable_gain():
    from ignition.recipes.qwen38.hyperconnection import normalization_gain

    weight = torch.linspace(-.25, .25, 10240, dtype=torch.bfloat16)
    module = SimpleNamespace(hc_norm=SimpleNamespace(weight=weight))
    gain = normalization_gain(module)
    assert gain is normalization_gain(module)
    torch.testing.assert_close(gain, 1. + weight.float(), rtol=0, atol=0)
    assert gain.dtype == torch.float32
