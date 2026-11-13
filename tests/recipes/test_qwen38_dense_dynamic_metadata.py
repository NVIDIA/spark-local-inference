# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Dense recurrent metadata follows each invocation's packed inputs."""

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.recurrent import DeltaMetadata
from ignition.recipes.qwen38_dense.decoder import PreparedDecoder


@pytest.mark.parametrize("width", [1, 4])
def test_delta_metadata_matches_prepared_update_for_independent_calls(width):
    decoder = object.__new__(PreparedDecoder)
    decoder.verification_columns = width
    metadata = Metadata.allocate(8, 2, 2, "cpu", max_query_tokens=width)
    for starts, lengths, rows, counts in (
        ([0, 7], [4, 0], [3, 0], [4, 1]),
        ([9, 2], [1, 3], [5, 8], [4, 2]),
    ):
        metadata.starts.copy_(torch.tensor(starts))
        metadata.lengths.copy_(torch.tensor(lengths))
        metadata.state_rows.copy_(torch.tensor(rows))
        metadata.query_start_loc.copy_(torch.tensor([0, lengths[0], sum(lengths)]))
        metadata.live_counts.copy_(torch.tensor(counts))
        expected = DeltaMetadata.allocate(2, "cpu", verification_columns=width)
        active = metadata.lengths > 0
        expected.query_start_loc.copy_(metadata.query_start_loc)
        expected.convolution_slots.copy_(torch.where(active, metadata.state_rows, -1))
        expected.fresh.copy_(metadata.starts == 0)
        expected.state_indices[:, 0].copy_(metadata.state_rows)
        expected.initial_state_indices.copy_(torch.where(metadata.starts > 0, metadata.state_rows, 0))
        expected.final_state_indices.copy_(metadata.state_rows)
        expected.num_tokens.copy_(metadata.live_counts[:1])
        expected.num_seqs.copy_(metadata.live_counts[1:])
        actual = decoder._delta_metadata(metadata)
        for name in DeltaMetadata.__dataclass_fields__:
            actual_field, expected_field = getattr(actual, name), getattr(expected, name)
            assert actual_field.dtype == expected_field.dtype
            assert actual_field.shape == expected_field.shape
            assert torch.equal(actual_field, expected_field), name
