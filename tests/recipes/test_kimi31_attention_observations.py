# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import torch

from tests.recipes.kimi31_attention_observations import (
    _overwrite_query, allocate_observations, copy_observations, poison_observations,
)


def test_observations_copy_invocation_tensors_without_retaining_their_storage():
    source = (torch.arange(6).view(2, 3), (None, torch.tensor([1., 2.])))
    observed = allocate_observations(source)
    poison_observations(observed)
    copy_observations(observed, source)
    source[0].zero_()
    source[1][1].fill_(7)
    torch.testing.assert_close(observed[0], torch.arange(6).view(2, 3), rtol=0, atol=0)
    torch.testing.assert_close(observed[1][1], torch.tensor([1., 2.]), rtol=0, atol=0)
    assert observed[1][0] is None
    copy_observations(observed, source)
    assert not observed[0].count_nonzero()
    torch.testing.assert_close(observed[1][1], torch.tensor([7., 7.]), rtol=0, atol=0)


def test_query_overwrite_mutates_source_and_observation():
    source = torch.ones(1, 6, 576)
    observed = torch.full_like(source, 17)
    _overwrite_query(source, observed)
    assert not source.count_nonzero()
    assert not observed.count_nonzero()
