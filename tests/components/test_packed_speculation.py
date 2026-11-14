# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared round metadata must outlive both sequential sampling workspaces."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.packed_speculation import PreparedPackedSpeculativeRound


def operands():
    b = 4
    owner = SimpleNamespace(inputs=SimpleNamespace(query_lengths=torch.zeros(b, dtype=torch.int32)),
        requests=SimpleNamespace(capacity=b), pool=SimpleNamespace(backing=torch.empty(0)), weights=SimpleNamespace(shard=SimpleNamespace(world_size=1), draft=SimpleNamespace(confidence_head=None)),
        collectives=dict(runtime=None, vocabulary_runtime=None, coordinator=None), session=None)
    source = SimpleNamespace(batch=b, width=3, owner=owner, active=torch.zeros(b, dtype=torch.bool),
        rows=torch.arange(b),
        validator=SimpleNamespace(validation_input=torch.zeros((1, 8), dtype=torch.bfloat16)))
    target = SimpleNamespace(batch=b, width=6, owner=owner, target_token_capacity=8)
    return source, target


@pytest.mark.parametrize("independent_descriptor", [False, True])
def test_prefix_limits_cannot_alias_mutable_query_lengths(independent_descriptor):
    source, target = operands()
    backing = torch.zeros(5, dtype=torch.int32)
    limits = backing[:4]
    queries = backing[1:]
    if independent_descriptor:
        queries = torch.empty(0, dtype=torch.int32).set_(backing.untyped_storage()[4:], 0, (4,), (1,))
        assert queries.untyped_storage()._cdata != limits.untyped_storage()._cdata
    source.owner.inputs.query_lengths = queries
    with pytest.raises(ValueError, match="must not overlap"):
        PreparedPackedSpeculativeRound(source, target, limits, torch.zeros((2, 4, 2)), name="round")
