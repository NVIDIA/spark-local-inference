# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Draft proposal construction releases unused vocabulary-wide index storage."""

import pytest
import torch

from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.sampling import DeviceRequestState
from ignition.preparation.tensor_metadata import NumericStorage


@pytest.mark.parametrize("vocabulary", [256, 163840])
def test_draft_proposals_release_unused_numeric_index_storage(vocabulary):
    state = DeviceRequestState.allocate("cpu", vocab_size=vocabulary)
    slots = torch.arange(4, dtype=torch.int32)
    generations = torch.zeros(4, dtype=torch.int64)
    candidates = torch.zeros((4, 7), dtype=torch.int64)
    seed, offset, sampled = (torch.zeros(4, dtype=torch.int64) for _ in range(3))
    initialized = torch.cuda.is_initialized()
    with NumericStorage() as numeric:
        proposal = PreparedDraftDistribution(torch.empty((4, 1, vocabulary)), state,
            slots, generations, torch.tensor([4], dtype=torch.int32), candidates,
            seed=seed, offset=offset, sampled=sampled)
    live_index_bytes = sum(end - start for start, end in numeric.ranges())
    assert live_index_bytes < vocabulary
    assert proposal.probabilities.shape == (4, 1, vocabulary)
    assert torch.cuda.is_initialized() == initialized
