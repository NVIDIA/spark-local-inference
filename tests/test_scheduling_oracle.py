# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from itertools import product
import csv
from random import Random
import subprocess
import sys

import pytest

from scripts.scheduling_sim.model import Graph
from scripts.scheduling_sim.oracle import Decision, Job, Metrics, Objective, Oracle, Progress


def job(name="d", output=4, **kwargs):
    return Job(name, 0, ((output, 1.0),), **kwargs)


def test_completion_oracle_and_rollout_resolve_the_greedy_prefill_tail():
    graphs = (Graph("p64", 2, prefill_capacity=64), Graph("p128", 3, prefill_capacity=128),
              Graph("p256", 4, prefill_capacity=256))
    oracle = Oracle((Job("p", 129, ((1, 1.0),)),), graphs)
    objective = Objective(1)
    greedy = oracle.evaluate(oracle.immediate(objective), objective)
    best = oracle.solve(objective)
    rollout = oracle.evaluate(oracle.rollout(objective, time_value=1), objective)
    assert greedy.metrics.elapsed_ms == 5
    assert best.metrics.elapsed_ms == rollout.metrics.elapsed_ms == 4
    assert best.first.graph.name == rollout.first.graph.name == "p256"
    assert best.metrics.output_tokens == 1
    assert abs(best.residual) < 1e-9


def test_two_prefills_choose_efficient_full_chunks_through_completion():
    graphs = (Graph("p1", 1.5, prefill_capacity=1), Graph("p4", 4, prefill_capacity=4),
              Graph("p8", 7, prefill_capacity=8))
    jobs = tuple(Job(name, 8, ((1, 1.0),)) for name in ("a", "b"))
    best = Oracle(jobs, graphs).solve(Objective(1, prefill_wait_price=0.1))
    assert best.first.graph.name == "p8"
    assert best.metrics.elapsed_ms == 14
    assert best.metrics.prefill_wait_ms == 21
    assert best.metrics.decode_age_ms2 == 0


def test_long_speculative_graph_remains_eligible_with_smaller_alternatives():
    graphs = (Graph("w1", 15, decode_capacity=1), Graph("w3", 17.3, decode_capacity=1, width=3),
              Graph("w8", 30, decode_capacity=1, width=8))
    best = Oracle((job(output=16),), graphs).solve(Objective(1))
    assert best.first.graph.name == "w8"
    assert best.metrics.elapsed_ms == 60
    assert best.metrics.output_tokens == 16


def test_age_price_can_trade_some_rate_for_more_frequent_output():
    graphs = (Graph("w1", 10, decode_capacity=1), Graph("w8", 40, decode_capacity=1, width=8))
    oracle = Oracle((job(output=8),), graphs)
    throughput = oracle.solve(Objective(1))
    interactive = oracle.solve(Objective(1, decode_age_price=0.01))
    assert throughput.first.graph.name == "w8"
    assert throughput.metrics.elapsed_ms == 40
    assert throughput.metrics.decode_age_ms2 == 800
    assert interactive.first.graph.name == "w1"
    assert interactive.metrics.elapsed_ms == 80
    assert interactive.metrics.decode_age_ms2 == 400


def test_mixed_completion_value_can_outweigh_lower_immediate_rate():
    graphs = (Graph("p2", 3, prefill_capacity=2), Graph("p8", 8, prefill_capacity=8),
              Graph("decode", 3, decode_capacity=2, width=2),
              Graph("mixed2", 3.5, prefill_capacity=2, decode_capacity=2, width=2),
              Graph("mixed8", 12, prefill_capacity=8, decode_capacity=2, width=2))
    jobs = (Job("p", 10, ((1, 1.0),)), job("a", 2), job("b", 2))
    oracle = Oracle(jobs, graphs)
    objective = Objective(4)
    greedy = oracle.evaluate(oracle.immediate(objective), objective)
    best = oracle.solve(objective)
    rollout = oracle.evaluate(oracle.rollout(objective, time_value=1), objective)
    assert greedy.first.graph.name == "decode"
    assert greedy.metrics.elapsed_ms == 14
    assert best.first.graph.name in ("p8", "mixed2")
    assert best.metrics.elapsed_ms == rollout.metrics.elapsed_ms == 11.5
    interactive = Objective(4, decode_age_price=0.01)
    assert oracle.solve(interactive).first.graph.name == "mixed2"
    assert oracle.evaluate(oracle.rollout(interactive, time_value=1), interactive).first.graph.name == "mixed2"


def test_eos_is_a_conditioned_belief_not_a_revealed_length():
    graphs = (Graph("w1", 1, decode_capacity=1), Graph("w4", 2.5, decode_capacity=1, width=4))
    oracle = Oracle((Job("d", 0, ((1, 0.5), (4, 0.5))),), graphs)
    best = oracle.solve(Objective(1))
    assert best.first.graph.name == "w1"
    assert best.metrics.elapsed_ms == 2.25
    assert best.metrics.output_tokens == 2.5
    clairvoyant_time = 0.5 * 1 + 0.5 * 2.5
    assert best.metrics.elapsed_ms > clairvoyant_time
    _, branches = oracle.transition(oracle.root, Decision(graphs[0], decodes=(0,)))
    assert sum(p for p, _ in branches) == pytest.approx(1)
    live = next(state for _, state in branches if not state[0].done)
    assert oracle._output(0, live[0], 4) == ((1.0, 3, True),)


def test_acceptance_and_hidden_eos_combine_before_expected_reward():
    graph = Graph("w3", 4, decode_capacity=1, width=3)
    oracle = Oracle((Job("d", 0, ((1, 0.5), (3, 0.5)), acceptance=(0.5, 0.5)),), (graph,))
    metrics, branches = oracle.transition(oracle.root, Decision(graph, decodes=(0,)))
    masses = {(state[0].generated, state[0].done): p for p, state in branches}
    assert masses == {(1, True): 0.5, (1, False): 0.25, (2, False): 0.125, (3, True): 0.125}
    assert metrics.output_tokens == 1.375
    assert metrics.decode_age_ms2 == 8
    assert sum(masses.values()) == 1


def test_batched_observations_have_the_product_distribution():
    graph = Graph("b2w2", 3, decode_capacity=2, width=2)
    oracle = Oracle((job("a", 3, acceptance=(0.5,)), job("b", 3, acceptance=(0.25,))), (graph,))
    metrics, branches = oracle.transition(oracle.root, Decision(graph, decodes=(0, 1)))
    probabilities = {tuple(p.generated for p in state): chance for chance, state in branches}
    assert probabilities == {(1, 1): 0.375, (1, 2): 0.125, (2, 1): 0.375, (2, 2): 0.125}
    assert metrics.output_tokens == 2.75


def test_inactive_and_selected_streams_both_wait_for_graph_completion():
    graph = Graph("b1w2", 10, decode_capacity=1, width=2)
    oracle = Oracle((job("a", 3, age_ms=20), job("b", 3, age_ms=40)), (graph,))
    metrics, branches = oracle.transition(oracle.root, Decision(graph, decodes=(0,)))
    assert metrics.decode_age_ms2 == (20 * 10 + 50) + (40 * 10 + 50)
    assert metrics.unfinished_wait_ms == 20
    assert branches == ((1.0, (Progress(0, 2, 0), Progress(0, 0, 50))),)


def test_prefill_first_output_and_decode_stall_reset_are_distinct():
    graph = Graph("mixed", 4, prefill_capacity=2, decode_capacity=1, width=2)
    jobs = (Job("p", 3, ((1, 1.0),)), job("d", 3, age_ms=10))
    oracle = Oracle(jobs, (graph,))
    metrics, branches = oracle.transition(oracle.root, Decision(graph, prefill=0, decodes=(1,)))
    assert metrics.prefill_tokens == 2
    assert metrics.output_tokens == 2
    assert metrics.prefill_wait_ms == 4
    assert metrics.decode_age_ms2 == 48
    next_metrics, next_branches = oracle.transition(branches[0][1], Decision(graph, prefill=0, decodes=(1,)))
    assert next_metrics.prefill_tokens == 1
    assert next_metrics.output_tokens == 2
    assert all(p.done for p in next_branches[0][1])


def test_objective_is_ratio_of_expectations_not_expectation_of_ratios():
    graph = Graph("w2", 2, decode_capacity=1, width=2)
    oracle = Oracle((Job("d", 0, ((1, 0.5), (3, 0.5))),), (graph,))
    result = oracle.solve(Objective(1))
    assert result.metrics.elapsed_ms == 3
    assert result.metrics.output_tokens == 2
    assert result.rate == pytest.approx(2 / 3)
    assert result.rate != pytest.approx(0.5 * (1 / 2) + 0.5 * (3 / 4))


def exhaustive_policies(oracle, state):
    if all(p.done for p in state):
        return [Metrics()]
    totals = []
    for action in oracle.choices(state):
        immediate, branches = oracle.transition(state, action)
        continuations = [exhaustive_policies(oracle, successor) for _, successor in branches]
        for outcomes in product(*continuations):
            total = immediate
            for (probability, _), metrics in zip(branches, outcomes):
                total = total.plus(metrics, probability)
            totals.append(total)
    return totals


@pytest.mark.parametrize("price", [0, 0.1, 3])
def test_fractional_dynamic_program_matches_exhaustive_policy_trees(price):
    graphs = (Graph("w1", 1, decode_capacity=1), Graph("w2", 1.5, decode_capacity=1, width=2))
    oracle = Oracle((Job("d", 0, ((1, 0.4), (3, 0.6)), acceptance=(0.5,)),), graphs)
    objective = Objective(2, decode_age_price=price)
    expected = max(objective.rate(total) for total in exhaustive_policies(oracle, oracle.root))
    best = oracle.solve(objective)
    assert best.rate == pytest.approx(expected)


@pytest.mark.parametrize("price", [0, 0.1, 2])
def test_multirequest_oracle_matches_exhaustive_schedules(price):
    graphs = (Graph("single", 1, decode_capacity=1), Graph("batch", 1.5, decode_capacity=2))
    oracle = Oracle((job("a", 2), job("b", 2, age_ms=3)), graphs)
    objective = Objective(2, unfinished_wait_price=0.1, decode_age_price=price)
    expected = max(objective.rate(total) for total in exhaustive_policies(oracle, oracle.root))
    assert oracle.solve(objective).rate == pytest.approx(expected)


def test_every_cohort_is_available_to_the_completion_objective():
    graph = Graph("b2", 3, decode_capacity=2)
    oracle = Oracle(tuple(job(name, 2) for name in ("a", "b", "c")), (graph,))
    cohorts = {a.decodes for a in oracle.choices(oracle.root)}
    assert cohorts == {(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)}


def test_uniform_time_and_waiting_price_scaling_preserves_policy():
    graphs = (Graph("w1", 10, decode_capacity=1), Graph("w4", 25, decode_capacity=1, width=4))
    original = Oracle((job(),), graphs).solve(Objective(2, unfinished_wait_price=0.01, decode_age_price=0.01))
    scaled = Oracle((job(),), tuple(replace(g, duration_ms=g.duration_ms * 100) for g in graphs)).solve(
        Objective(2, unfinished_wait_price=0.01 / 100, decode_age_price=0.01 / 100**2))
    assert original.first.graph.name == scaled.first.graph.name
    assert original.rate == pytest.approx(scaled.rate * 100)


def test_missing_work_or_explicit_state_limit_fails_without_a_fallback():
    graph = Graph("decode", 1, decode_capacity=1)
    with pytest.raises(ValueError, match="cannot finish"):
        Oracle((Job("p", 4, ((1, 1.0),)),), (graph,)).solve(Objective(1))
    with pytest.raises(RuntimeError, match="exceeded 1 states"):
        Oracle((job(),), (graph,), max_states=1).solve(Objective(1))


def test_myopic_wait_cost_cannot_distinguish_which_stall_a_graph_clears():
    graph = Graph("single", 10, decode_capacity=1)
    oracle = Oracle((job("fresh", 1), job("stalled", 1, age_ms=100)), (graph,))
    objective = Objective(64, decode_age_price=0.01)
    myopic = oracle.evaluate(oracle.immediate(objective, waiting=True), objective)
    rollout = oracle.evaluate(oracle.rollout(objective, time_value=1), objective)
    best = oracle.solve_priced(objective, time_value=1)
    assert myopic.first.decodes == (0,)
    assert rollout.first.decodes == best.first.decodes == (1,)
    assert myopic.metrics.decode_age_ms2 == 2250
    assert rollout.metrics.decode_age_ms2 == best.metrics.decode_age_ms2 == 1250


def test_myopic_graph_wait_cost_can_make_both_efficiency_and_total_age_worse():
    graphs = (Graph("w1", 15, decode_capacity=1), Graph("w3", 17.3, decode_capacity=1, width=3))
    oracle = Oracle((job(output=3),), graphs)
    objective = Objective(1, decode_age_price=0.1)
    myopic = oracle.evaluate(oracle.immediate(objective, waiting=True), objective)
    best = oracle.solve_priced(objective, time_value=1)
    assert myopic.first.graph.name == "w1"
    assert best.first.graph.name == "w3"
    assert myopic.metrics.elapsed_ms == 45
    assert best.metrics.elapsed_ms == 17.3
    assert myopic.metrics.decode_age_ms2 > best.metrics.decode_age_ms2


def test_normalizing_negative_net_value_by_episode_time_can_reward_waste():
    jobs = (Job("p", 64, ((1, 1.0),)), job("d", 1, age_ms=1000))
    graphs = (Graph("fast", 1, prefill_capacity=64), Graph("slow", 100, prefill_capacity=64),
              Graph("decode", 1, decode_capacity=1))
    objective = Objective(1, decode_age_price=1)
    oracle = Oracle(jobs, graphs)
    ratio = oracle.solve(objective)
    priced = oracle.solve_priced(objective, time_value=1)
    assert ratio.metrics.elapsed_ms == 101
    assert priced.metrics.elapsed_ms == 2
    assert ratio.metrics.decode_age_ms2 == priced.metrics.decode_age_ms2 == 1000.5
    assert ratio.rate > priced.rate


@pytest.mark.parametrize("seed", range(12))
def test_rollout_improves_baseline_surplus_and_is_bounded_by_exact_oracle(seed):
    rng = Random(seed)
    graphs = tuple(Graph(f"b{b}w{w}", rng.uniform(1, 4), decode_capacity=b, width=w)
                   for b in (1, 2) for w in (1, 2))
    jobs = tuple(Job(name, 0, ((1, 0.4), (3, 0.6)), acceptance=(rng.random(),), age_ms=rng.randrange(5))
                 for name in ("a", "b"))
    objective = Objective(2, unfinished_wait_price=rng.random(), decode_age_price=rng.random())
    price = rng.uniform(0.1, 5)
    oracle = Oracle(jobs, graphs)
    greedy = oracle.evaluate(oracle.immediate(objective), objective)
    first_policy = oracle.rollout(objective, time_value=price)
    rollout = oracle.evaluate(first_policy, objective)
    twice = oracle.evaluate(oracle.rollout(objective, time_value=price, baseline=first_policy), objective)
    optimal = oracle.solve_priced(objective, time_value=price)
    def surplus(result):
        return objective.net_work(result.metrics) - price * result.metrics.elapsed_ms
    assert surplus(greedy) <= surplus(rollout) + 1e-8
    assert surplus(rollout) <= surplus(twice) + 1e-8
    assert surplus(twice) <= surplus(optimal) + 1e-8


@pytest.mark.parametrize("time_value", [0.1, 1, 10])
def test_priced_oracle_matches_exhaustive_stochastic_policy_trees(time_value):
    graphs = (Graph("w1", 1, decode_capacity=1), Graph("w2", 1.5, decode_capacity=1, width=2))
    oracle = Oracle((Job("d", 0, ((1, 0.4), (3, 0.6)), acceptance=(0.5,)),), graphs)
    objective = Objective(2, decode_age_price=0.1)
    surplus = lambda metrics: objective.net_work(metrics) - time_value * metrics.elapsed_ms
    expected = max(surplus(total) for total in exhaustive_policies(oracle, oracle.root))
    assert surplus(oracle.solve_priced(objective, time_value=time_value).metrics) == pytest.approx(expected)


def test_intrinsic_token_value_is_constant_across_complete_episode_policies():
    graphs = (Graph("p4", 2, prefill_capacity=4), Graph("b1", 1, decode_capacity=1))
    oracle = Oracle((Job("p", 8, ((1, 1.0),)), job("d", 3)), graphs)
    low = oracle.solve_priced(Objective(1, decode_age_price=0.1), time_value=1)
    high = oracle.solve_priced(Objective(64, decode_age_price=0.1), time_value=1)
    assert low.first == high.first
    assert low.metrics == high.metrics


def test_oracle_sweep_writes_expected_metrics_and_labels_ratio_comparison(tmp_path):
    path = tmp_path / "oracle.csv"
    subprocess.run([sys.executable, "-m", "scripts.scheduling_sim.oracle_sweep", "--case", "penalty_dilution",
                    "--decode-values", "1", "--time-values", "1", "--age-prices", "1", "--include-ratio",
                    "--output", str(path)], check=True, capture_output=True)
    with path.open() as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    assert float(rows["oracle"]["elapsed_ms"]) == 2
    assert float(rows["episode_ratio"]["elapsed_ms"]) == 101
    assert float(rows["rollout"]["surplus_gap"]) == 0
    assert float(rows["rollout_twice"]["surplus_gap"]) == 0
