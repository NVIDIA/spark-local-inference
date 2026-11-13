# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""A full-support proposal can verify a differently filtered target law."""

from fractions import Fraction as F

import pytest


@pytest.mark.parametrize(
    "proposal,target",
    [
        ([F(1, 2), F(1, 3), F(1, 6)], [F(1, 5), F(3, 10), F(1, 2)]),
        ([F(0), F(1, 4), F(3, 4)], [F(1, 3), F(2, 3), F(0)]),
        ([F(1, 2), F(1, 3), F(1, 6)], [F(0), F(0), F(1)]),
        ([F(1), F(0), F(0)], [F(1), F(0), F(0)]),
    ],
)
def test_acceptance_plus_positive_residual_is_target(proposal, target):
    # The accepted contribution of token i is q_i * min(1,p_i/q_i).
    accepted = [min(p, q) for p, q in zip(target, proposal)]
    rejected = 1 - sum(accepted)
    positive_residual = [max(p - q, 0) for p, q in zip(target, proposal)]
    assert sum(proposal) == sum(target) == 1
    assert rejected == sum(positive_residual)
    result = [
        a + (rejected * r / sum(positive_residual) if rejected else 0) for a, r in zip(accepted, positive_residual)
    ]
    assert result == target
