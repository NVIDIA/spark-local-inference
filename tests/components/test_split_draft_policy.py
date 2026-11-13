# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""A prefix decision must not condition admission on the token it admits."""

from fractions import Fraction
from itertools import product

import pytest

from ignition.components.split_draft import allocate_draft_budget, causal_prefix, lagged_budget


def first_token_probability(selector):
    """Enumerate p=q=(1/2,1/2), where every admitted proposal is accepted."""
    probability = Fraction(0)
    for proposal, bonus in product((0, 1), repeat=2):
        count = selector(proposal)
        emitted = proposal if count else bonus
        if emitted == 0:
            probability += Fraction(1, 4)
    return probability


def test_lookahead_utility_bias_reproducer_and_causal_alternatives():
    costs = (1., 1.7, 2.)
    confidence = lambda proposal: (.6, .9 if proposal == 0 else .1)
    # This intentional misuse of current confidence demonstrates the invalid policy.
    unsafe = lambda proposal: lagged_budget(confidence(proposal), costs) - 1
    assert first_token_probability(unsafe) == Fraction(3, 4)
    for threshold in (.1, .4, .6, .7, 1.):
        assert first_token_probability(lambda proposal: causal_prefix(confidence(proposal), threshold)) == Fraction(1, 2)
    for snapshot in ((.6, .9), (.6, .1)):
        assert first_token_probability(lambda proposal: lagged_budget(snapshot, costs) - 1) == Fraction(1, 2)


def test_causal_admission_never_uses_later_confidence():
    for prefix in ((), (.8,), (.8, .5)):
        for tail in product((.01, .5, 1.), repeat=3 - len(prefix)):
            scores = prefix + tail
            for position in range(1, len(prefix) + 1):
                assert (causal_prefix(scores, .3) >= position) == (causal_prefix(prefix, .3) >= position)


def test_lagged_confidence_requires_fresh_proposal_randomness():
    costs = (1., 1.7, 2.)
    independent, reused = Fraction(0), Fraction(0)
    for discarded, fresh, bonus in product((0, 1), repeat=3):
        width = lagged_budget((.6, .9 if discarded == 0 else .1), costs)
        independent += Fraction(int((fresh if width > 1 else bonus) == 0), 8)
        reused += Fraction(int((discarded if width > 1 else bonus) == 0), 8)
    assert independent == Fraction(1, 2)
    assert reused == Fraction(3, 4)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., 1.1])
def test_invalid_confidence_is_rejected(value):
    with pytest.raises(ValueError):
        causal_prefix([value], .5)
    with pytest.raises(ValueError):
        lagged_budget([value], [1., 2.])


def test_fixed_budget_distributes_work_to_longer_high_confidence_prefixes():
    assert allocate_draft_budget(((.9, .9, .9), (.4, .2, .2)), 4) == (3, 1)
    assert allocate_draft_budget(((1., 1., 1.), (1., 1., 1.)), 3) == (2, 1)


def test_fixed_budget_admission_is_independent_of_its_own_future():
    for budget in range(7):
        for position in range(1, 4):
            prefix = (.8, .5, .2)[:position]
            decisions = {
                allocate_draft_budget((prefix + tail, (.7, .6, .5)), budget)[0] >= position
                for tail in product((0., .5, 1.), repeat=3-position)
            }
            assert len(decisions) == 1


def test_fixed_budget_preserves_exact_first_token_probability():
    for budget in range(5):
        selector = lambda proposal: allocate_draft_budget(
            ((.6, .9 if proposal == 0 else .1), (.7, .5)), budget)[0]
        assert first_token_probability(selector) == Fraction(1, 2)
