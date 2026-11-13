# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact small reference calculations; no dispatch policy is implied."""

from dataclasses import dataclass
from math import isfinite, log

from .model import Graph


def prefill_completion_ms(remaining: int, catalog: tuple[Graph, ...]) -> float:
    """Minimum isolated completion time by finite dynamic programming."""
    if remaining < 0:
        raise ValueError("remaining tokens must be nonnegative")
    graphs = [g for g in catalog if g.prefill_capacity and not g.decode_capacity]
    if not graphs:
        raise ValueError("an isolated prefill reference requires prefill-only graphs")
    best = [0.0] * (remaining + 1)
    for tokens in range(1, remaining + 1):
        best[tokens] = min(g.duration_ms + best[max(0, tokens - g.prefill_capacity)] for g in graphs)
    return best[remaining]


def conditional_remaining(length_mass: dict[int, float], generated: int) -> float:
    """E[L - generated | L > generated] under an explicit discrete prior."""
    if generated < 0 or not length_mass or any(n < 1 or not isfinite(p) or p < 0 for n, p in length_mass.items()):
        raise ValueError("invalid length distribution or attained output")
    surviving = sum(p for n, p in length_mass.items() if n > generated)
    if surviving == 0:
        raise ValueError("the prior assigns zero probability to a still-live request")
    return sum((n - generated) * p for n, p in length_mass.items() if n > generated) / surviving


def output_age_area(age_ms: float, duration_ms: float) -> float:
    """Integral of output age over a graph with no intermediate publication."""
    if not isfinite(age_ms) or not isfinite(duration_ms) or min(age_ms, duration_ms) < 0:
        raise ValueError("age and duration must be finite and nonnegative")
    return age_ms * duration_ms + duration_ms * duration_ms / 2


@dataclass(frozen=True)
class Allocation:
    time_shares: tuple[float, ...]
    rates: tuple[float, ...]
    utility: float
    optimality_gap: float
    iterations: int


def fair_allocation(action_rates: tuple[tuple[float, ...], ...],
                    token_values: tuple[float, ...], fairness_weights: tuple[float, ...],
                    *, tolerance: float = 1e-8, max_iterations: int = 50_000) -> Allocation:
    """Maximize sum(v_i*x_i + w_i*log(x_i)) over mixtures of fixed action rates.

    Shares measure wall time, not launch counts. This is a frozen-workload
    reference, solved by pairwise conditional gradients with exact line search.
    The returned concavity gap bounds utility suboptimality.
    """
    n = len(token_values)
    if not n or not action_rates or len(fairness_weights) != n:
        raise ValueError("allocation requires actions and matching request coefficients")
    if any(not isfinite(v) or v < 0 for v in token_values):
        raise ValueError("token values must be finite and nonnegative")
    if any(not isfinite(w) or w <= 0 for w in fairness_weights):
        raise ValueError("fairness weights must be finite and positive")
    if any(len(row) != n or any(not isfinite(r) or r < 0 for r in row) for row in action_rates):
        raise ValueError("action rates must be finite, nonnegative, and match request count")
    if not isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid allocation solver limits")
    shares = [1 / len(action_rates)] * len(action_rates)

    for iteration in range(max_iterations + 1):
        rates = [sum(share * row[i] for share, row in zip(shares, action_rates)) for i in range(n)]
        if any(x <= 0 for x in rates):
            raise ValueError("every request needs a feasible positive service rate")
        gradient = [v + w / x for v, w, x in zip(token_values, fairness_weights, rates)]
        scores = [sum(q * r for q, r in zip(gradient, row)) for row in action_rates]
        target = max(range(len(shares)), key=scores.__getitem__)
        gap = max(0.0, scores[target] - sum(q * x for q, x in zip(gradient, rates)))
        utility = sum(v * x + w * log(x) for v, w, x in zip(token_values, fairness_weights, rates))
        if gap <= tolerance:
            return Allocation(tuple(shares), tuple(rates), utility, gap, iteration)
        if iteration == max_iterations:
            raise RuntimeError(f"allocation solver did not converge: utility gap {gap:g}")
        source = min((j for j, share in enumerate(shares) if share > 0), key=scores.__getitem__)
        direction = [b - a for a, b in zip(action_rates[source], action_rates[target])]

        def derivative(step):
            moved = [x + step * d for x, d in zip(rates, direction)]
            if any(x <= 0 for x in moved):
                return float("-inf")
            return sum((v + w / x) * d for v, w, x, d
                       in zip(token_values, fairness_weights, moved, direction))

        low, high = 0.0, shares[source]
        if derivative(high) >= 0:
            step = high
        else:
            for _ in range(60):
                midpoint = (low + high) / 2
                if derivative(midpoint) > 0:
                    low = midpoint
                else:
                    high = midpoint
            step = (low + high) / 2
        shares[source] -= step
        shares[target] += step
