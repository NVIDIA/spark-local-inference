# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from itertools import combinations
from math import exp, log
from random import Random

import pytest

from scripts.scheduling_sim.analysis import fair_allocation
from scripts.scheduling_sim.model import Action, Graph, Lane, Request, Value, View, expected_output
from scripts.scheduling_sim.simulator import run
from scripts.scheduling_sim.utility import ServicePrice, UtilityConfig, UtilityScheduler
from scripts.scheduling_sim.workloads import catalog, workloads


def scheduler(*, beta=64, token=4, delivery=4, age=0, adaptation=100):
    return UtilityScheduler(Value(beta), UtilityConfig(token, delivery, adaptation, age))


def test_rate_update_uses_elapsed_time_and_actual_service():
    price = ServicePrice.initial(3, 10, 2)
    price.observe(20, 60, 100)
    assert exp(price.log_fraction) == pytest.approx(exp(-0.2) * 0.5 + (1 - exp(-0.2)) * 0.3)
    before = price.log_price
    price.observe(200, 0, 100)
    assert price.log_price == pytest.approx(before + 2)


def test_equal_rate_intervals_compose_without_launch_count_bias():
    whole = ServicePrice.initial(1, 10, 3)
    pieces = replace(whole)
    whole.observe(60, 120, 100)
    for duration in (3, 7, 19, 31):
        pieces.observe(duration, duration * 2, 100)
    assert whole.log_fraction == pytest.approx(pieces.log_fraction)


def test_cumulative_history_is_the_actual_time_weighted_service_mean():
    price = ServicePrice.initial(3, 10, 2)
    price.observe(20, 60, 100, history="cumulative")
    price.observe(180, 0, 100, history="cumulative")
    assert exp(price.log_fraction) == pytest.approx((100 / 2 + 60 / 10) / (100 + 200))
    assert price.elapsed_ms == 200


def test_cumulative_history_does_not_exponentiate_an_unavoidable_round_of_long_prefills():
    cumulative = ServicePrice.initial(4, 4096 / 66, 16)
    exponential = replace(cumulative)
    cumulative.observe(15 * 66, 0, 100, history="cumulative")
    exponential.observe(15 * 66, 0, 100)
    initial = log(4 * 16)
    assert cumulative.log_price - initial == pytest.approx(log(10.9))
    assert exponential.log_price - initial == pytest.approx(9.9)


def test_service_history_integrates_entropy_dual_descent():
    price = ServicePrice.initial(3, 10, 2)
    price.observe(20, 60, 100, history="service")
    expected_inverse = exp(-0.06) * 2 + (1 - exp(-0.06)) / 0.3
    assert exp(-price.log_fraction) == pytest.approx(expected_inverse)
    before = exp(price.log_price)
    price.observe(200, 0, 100, history="service")
    assert exp(price.log_price) == pytest.approx(before + 6)


def test_service_history_composes_equal_rate_intervals():
    whole = ServicePrice.initial(1, 10, 3)
    pieces = replace(whole)
    whole.observe(60, 120, 100, history="service")
    for duration in (3, 7, 19, 31):
        pieces.observe(duration, duration * 2, 100, history="service")
    assert whole.log_fraction == pytest.approx(pieces.log_fraction)


def test_service_history_prices_a_round_of_long_prefills_linearly():
    price = ServicePrice.initial(4, 4096 / 66, 16)
    price.observe(15 * 66, 0, 100, history="service")
    assert exp(price.log_price) == pytest.approx(4 * (16 + 9.9))


def test_long_unserviced_interval_has_finite_scores_and_recovers_after_service():
    policy = scheduler()
    views = [View("a", 0, 4096, (1.0,) * 7), View("b", 0, 4096, (1.0,) * 7)]
    graphs = (Graph("decode", 30, decode_capacity=1, width=8),)
    policy.prepare(views, graphs)
    policy.observe(100_000, [{"request": "a", "prefill": 0, "emitted": 8}])
    action = policy.choose(views, graphs, 100_000)
    assert action.lanes[0].name == "b"
    assert 0 < policy.score(action) < 1
    policy.observe(30, [{"request": "b", "prefill": 0, "emitted": 8}])
    assert policy.choose(views, graphs, 100_030).lanes[0].name == "a"


@pytest.mark.parametrize("count", [1, 2, 16])
@pytest.mark.parametrize("age_price", [0, 0.01, 1000])
def test_prefill_only_keeps_efficient_full_chunks_without_a_chunk_clock(count, age_price):
    policy = scheduler(age=age_price)
    views = [View(f"p{i}", 32768, 4096, (1.0,) * 7) for i in range(count)]
    for step in range(8):
        action = policy.choose(views, catalog(), step * 66)
        assert action.graph.prefill_capacity == 4096
        lane = action.lanes[0]
        policy.observe(66, [{"request": lane.name, "prefill": 4096, "emitted": 0}])


@pytest.mark.parametrize("remaining", [1, 32, 64])
def test_small_tail_earns_only_useful_work(remaining):
    policy = scheduler(age=1)
    action = policy.choose([View("p", remaining, 4096, (1.0,) * 7)], catalog())
    assert action.graph.prefill_capacity == 64
    assert action.expected_work(Value(64)) == remaining + 64


def test_smaller_full_prefill_can_win_when_its_extra_efficiency_is_too_small_for_decode_wait():
    graphs = (Graph("small", 8, prefill_capacity=512),
              Graph("large", 62, prefill_capacity=4096),
              Graph("decode", 15, decode_capacity=4))
    prefill = View("p", 32768, 4096, ())
    decode = [View(f"d{i}", 0, 4096, ()) for i in range(4)]
    policy = scheduler(beta=1, token=0, delivery=0, age=0.04)
    assert policy.choose([prefill], graphs).graph.name == "large"
    assert policy.choose([prefill, *decode], graphs).graph.name == "small"


def test_age_is_rewarded_once_for_a_burst_not_once_per_token():
    policy = scheduler(token=0, delivery=0, age=0.1)
    views = [View("d", 0, 4096, (1.0,) * 7)]
    graphs = (Graph("one", 30, decode_capacity=1), Graph("eight", 30, decode_capacity=1, width=8))
    policy.prepare(views, graphs, 500)
    actions = [Action(g, (Lane("d", 0, g.width, g.width, 1),)) for g in graphs]
    difference = (policy.score(actions[1]) - policy.score(actions[0])) / exp(-policy.log_scale)
    assert difference == pytest.approx(7 * 64 / 30)


def test_intrinsic_decode_value_changes_phase_choice():
    views = [View("p", 32768, 4096, (1.0,) * 7), View("d", 0, 4096, (1.0,) * 7)]
    assert scheduler(beta=1).choose(views, catalog()).graph.prefill_capacity == 4096
    assert scheduler(beta=1000).choose(views, catalog()).graph.decode_capacity


def test_delivery_history_counts_a_burst_once_and_token_history_counts_accepted_tokens():
    policy = scheduler()
    views = [View("a", 0, 4096, (1.0,) * 7), View("b", 0, 4096, (1.0,) * 7)]
    policy.prepare(views, catalog())
    policy.observe(30, [{"request": "a", "prefill": 0, "emitted": 1},
                        {"request": "b", "prefill": 0, "emitted": 8}])
    a, b = policy.prices.values()
    assert a.deliveries.log_fraction == b.deliveries.log_fraction
    assert a.tokens.log_fraction < b.tokens.log_fraction


def test_prefill_to_decode_preserves_service_history_and_starts_a_delivery_history():
    policy = scheduler()
    policy.prepare([View("p", 64, 100, (1.0,) * 7)], catalog())
    policy.observe(3, [{"request": "p", "prefill": 64, "emitted": 1}])
    price = policy.prices["p"].tokens.value(1, 0)
    policy.prepare([View("p", 0, 99, (1.0,) * 7, last_service_ms=3)], catalog(), 3)
    assert policy.prices["p"].tokens.value(1, 0) == pytest.approx(price)
    assert policy.prices["p"].deliveries is not None
    assert policy.ages["p"][0] == 0


@pytest.mark.parametrize("history", ["exponential", "cumulative", "service"])
def test_normalizer_changes_preserve_the_marginal_service_price(history):
    price = ServicePrice.initial(4, 100, 16)
    price.observe(17, 200, 100, history=history)
    before = price.value(1, 0)
    for rate in (0.01, 1000, 100):
        price.rebase(rate)
        assert price.value(1, 0) == pytest.approx(before)


@pytest.mark.parametrize("count", [1, 2, 5, 16])
def test_delivery_reference_and_initial_price_use_feasible_batch_service(count):
    graphs = catalog()
    streams = [View(f"d{i}", 0, 4096, (1.0,) * 7) for i in range(count)]
    expected = min(count * g.duration_ms / min(count, g.decode_capacity)
                   for g in graphs if g.decode_capacity)
    for extra_prefills in (0, 7):
        policy = scheduler(age=0.1)
        policy.prepare(streams + [View(f"p{i}", 32768, 4096, ())
                                  for i in range(extra_prefills)], graphs)
        assert len(policy.ages) == count
        for stream in streams:
            assert policy.ages[stream.name][1] == pytest.approx(expected)
            assert policy.prices[stream.name].deliveries.value(1, 0) == pytest.approx(4 * expected)


@pytest.mark.parametrize("count", [5, 16])
@pytest.mark.parametrize("accepted", [0, 3, 7])
def test_short_prompt_startup_and_subsequent_decode_gaps(count, accepted):
    requests = [Request(f"r{i:02}", 64, 10000, max_new_tokens=10000,
                        acceptance=tuple(float(j < accepted) for j in range(7)))
                for i in range(count)]
    policy = UtilityScheduler(Value(64), UtilityConfig(16, 4, 100, 0.1, "service"))
    result = run(requests, catalog(), Value(64), scheduler=policy, horizon_ms=5000)
    assert all(row["ttft_ms"] is not None and row["ttft_ms"] < 100 for row in result.requests)
    assert result.summary["max_output_gap_ms"] < 150
    outputs = [row["output_tokens"] for row in result.requests]
    assert min(outputs) > 100
    assert max(outputs) - min(outputs) <= 8


def test_prerequisite_is_valued_with_the_shared_following_batch():
    policy = scheduler(token=0, delivery=0, age=0.1)
    views = [View(f"d{i}", 0, 4096, (1.,) * 7) for i in range(16)]
    graph = Graph("batch", 60, decode_capacity=16, width=8)
    solo = Graph("solo", 30, decode_capacity=1, width=8)
    policy.prepare(views, (solo, graph), 100)
    lanes = tuple(Lane(v.name, 0, 8, 8, 1) for v in views)
    ready = policy.score(Action(graph, lanes[:-1]))
    assert policy.score(Action(solo, lanes[-1:]), preceding_ms=0.1) < ready
    assert policy.score(Action(graph, lanes), preceding_ms=0.1) > ready
    assert policy.score(Action(graph, lanes), preceding_ms=1000) < ready


@pytest.mark.parametrize("seed", range(5))
def test_graph_cohort_ranking_matches_every_subset_with_age_and_fairness(seed):
    rng = Random(seed)
    views = [View(f"d{i}", 0, 100, tuple(rng.random() for _ in range(7)),
                  1 + rng.random(), rng.random() * 100) for i in range(6)]
    policy = scheduler(age=0.04)
    policy.prepare(views, catalog(), 100)
    for price in policy.prices.values():
        price.tokens.log_fraction -= rng.random() * 3
        price.deliveries.log_fraction -= rng.random() * 3
    for graph in (g for g in catalog() if g.decode_capacity):
        action = policy.choose(views, (graph,), 100)
        exhaustive = max(policy.score(Action(graph, tuple(
            Lane(v.name, 0, graph.width, expected_output(v.acceptance, graph.width), v.weight)
            for v in subset)))
            for count in range(1, min(graph.decode_capacity, len(views)) + 1)
            for subset in combinations(views, count))
        assert policy.score(action) == pytest.approx(exhaustive)


@pytest.mark.parametrize("name", list(workloads()))
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("history", ["exponential", "cumulative", "service"])
def test_common_workloads_complete_or_make_progress_to_the_horizon(name, mixed, history):
    requests = workloads()[name]
    policy = scheduler(age=0.01)
    policy.config = replace(policy.config, rate_history=history)
    result = run(requests, catalog(mixed=mixed), Value(64), scheduler=policy, horizon_ms=15000)
    assert result.summary["busy_ms"] == result.summary["elapsed_ms"]
    assert all(row["prefill_tokens"] or row["output_tokens"] for row in result.requests)
    if name != "continuous_decode":
        assert result.summary["unfinished"] == 0
    assert all(e["duration_ms"] > 0 and e["expected_work"] > 0 for e in result.trace)


@pytest.mark.parametrize("history", ["exponential", "cumulative", "service"])
@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_time_units_rescale_preferences_without_changing_graph_choice(history, scale):
    views = [View("p", 32768, 4096, (1.0,) * 7),
             View("d", 0, 4096, (1.0,) * 7, last_service_ms=10)]
    policy = UtilityScheduler(Value(64), UtilityConfig(4, 4, 100, 0.04, history))
    base = policy.choose(views, catalog(mixed=True), 100)
    policy = UtilityScheduler(Value(64), UtilityConfig(
        4 / scale, 4 / scale, 100 * scale, 0.04 / scale**2, history))
    rescaled = policy.choose([replace(v, last_service_ms=v.last_service_ms * scale) for v in views],
                            tuple(replace(g, duration_ms=g.duration_ms * scale)
                                  for g in catalog(mixed=True)), 100 * scale)
    assert rescaled.graph.name == base.graph.name


def test_five_identical_streams_do_not_accumulate_the_tie_break_service_imbalance():
    requests = [Request(f"d{i}", 0, 10000, max_new_tokens=10000, acceptance=(0.0,) * 7) for i in range(5)]
    result = run(requests, catalog(), Value(64), scheduler=scheduler(), horizon_ms=2100)
    outputs = [row["output_tokens"] for row in result.requests]
    assert max(outputs) - min(outputs) <= 1


@pytest.mark.parametrize("rates,values,fairness,durations", [
    (((1., 0.), (0., 7.)), (0., 0.), (1., 3.), (1., 13.)),
    (((1., 0.), (0., 7.)), (2., 4.), (1., 3.), (1., 13.)),
    (((1., 0., 1.), (0., 1., 1.), (.4, .4, .4)), (1., 1., 1.), (1., 1., 1.), (2., 7., 1.)),
])
@pytest.mark.parametrize("history", ["exponential", "service"])
def test_stationary_prices_match_offline_rate_allocation(rates, values, fairness, durations, history):
    optimum = fair_allocation(rates, values, fairness)
    prices = [ServicePrice.initial(f, max(row[i] for row in rates), len(values))
              for i, f in enumerate(fairness)]
    served = [0.] * len(values)
    elapsed = 0
    for step in range(50000):
        choice = max(range(len(rates)), key=lambda k: sum(
            (v + exp(p.log_price) / p.maximum_rate) * r
            for v, p, r in zip(values, prices, rates[k])))
        dt = durations[choice]
        for i, price in enumerate(prices):
            delivered = dt * rates[choice][i]
            price.observe(dt, delivered, 1000, history=history)
            if step >= 25000:
                served[i] += delivered
        if step >= 25000:
            elapsed += dt
    measured = [n / elapsed for n in served]
    utility = sum(v * r + f * log(r) for v, f, r in zip(values, fairness, measured))
    assert 0 <= optimum.utility - utility < 0.0001


def test_age_continuation_satisfies_the_reference_poisson_equation():
    durations = (2., 5., 7.)
    probabilities = (.2, .3, .5)
    served = ((True, False), (False, True), (True, True))
    values = (3., 7., 11.)
    cost = (0.2, 0.7)
    mean = sum(p * t for p, t in zip(probabilities, durations))
    second = sum(p * t * t for p, t in zip(probabilities, durations))
    references = [mean / sum(p * s[i] for p, s in zip(probabilities, served)) for i in range(2)]
    expected_rate = sum(p * v for p, v in zip(probabilities, values)) / mean
    expected_rate -= sum(c * (second / (2 * mean) + ref / mean * sum(
        p * t for p, t, s in zip(probabilities, durations, served) if not s[i]))
        for i, (c, ref) in enumerate(zip(cost, references)))
    for ages in ((0., 0.), (0., 1000.), (53., 7.)):
        before = -sum(c * ref * age for c, ref, age in zip(cost, references, ages))
        reward = 0
        for p, t, s, v in zip(probabilities, durations, served, values):
            immediate = v - sum(c * (age * t + t * t / 2) for c, age in zip(cost, ages))
            after = -sum(c * ref * (0 if selected else age + t)
                         for c, ref, age, selected in zip(cost, references, ages, s))
            reward += p * (immediate + after - before)
        assert reward / mean == pytest.approx(expected_rate)


@pytest.mark.parametrize("config", [(-1, 0, 1, 0), (0, -1, 1, 0), (0, 0, 0, 0), (0, 0, 1, -1)])
def test_invalid_preferences_are_rejected(config):
    with pytest.raises(ValueError):
        UtilityConfig(*config)
