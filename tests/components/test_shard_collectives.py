# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""A reduce-scatter or shard gather given no runtime serves one rank by copying its source."""

import pytest
import torch

from ignition.components.collectives import PreparedReduceScatter, PreparedShardGather


@pytest.mark.parametrize("component", (PreparedReduceScatter, PreparedShardGather))
def test_single_rank_component_copies_its_source(component):
    source = torch.randn((8, 4608), dtype=torch.float32).bfloat16()
    output = torch.full_like(source, float("nan"))
    assert component.fits(None, source)
    prepared = component(source, output, runtime=None, session=None, coordinator=None, name="single")
    assert prepared(source) is output
    assert torch.equal(output, source)


@pytest.mark.parametrize("component", (PreparedReduceScatter, PreparedShardGather))
def test_single_rank_component_rejects_a_mismatched_output(component):
    source = torch.zeros((8, 4608), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="single-rank"):
        component(source, torch.zeros((4, 4608), dtype=torch.bfloat16), runtime=None, session=None,
                  coordinator=None, name="single")
