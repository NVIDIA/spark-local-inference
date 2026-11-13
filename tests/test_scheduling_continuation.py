# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
from random import Random

import pytest

from scripts.scheduling_sim.continuation import SerialReference
from scripts.scheduling_sim.model import Graph
from scripts.scheduling_sim.oracle import Decision, Job, Metrics, Objective, Oracle


@pytest.mark.parametrize("seed", range(12))
def test_composed_serial_moments_match_complete_conditional_evaluation(seed):
    rng = Random(seed)
    jobs = (Job("a", 0, ((1, 0.4), (3, 0.6)), acceptance=(rng.random(),), age_ms=7, weight=2),
            Job("b", 3, ((1, 0.5), (3, 0.5)), acceptance=(rng.random(),)),
            Job("c", 0, ((2, 0.3), (3, 0.7)), acceptance=(rng.random(),), age_ms=11))
    graphs = (Graph("p2", 2, prefill_capacity=2), Graph("p4", 3, prefill_capacity=4),
              Graph("w1", 1, decode_capacity=1), Graph("w2", 1.5, decode_capacity=1, width=2))
    oracle = Oracle(jobs, graphs)
    objective = Objective(2, prefill_wait_price=0.3, unfinished_wait_price=0.2, decode_age_price=0.1)
    reference = SerialReference(oracle, objective, time_value=1)
    explicit = oracle.evaluate(reference.choose_serial, objective)
    assert asdict(reference.metrics(oracle.root)) == pytest.approx(asdict(explicit.metrics))


@pytest.mark.parametrize("seed", range(8))
def test_marginal_composition_matches_joint_acceptance_and_eos_branches(seed):
    rng = Random(seed)
    jobs = tuple(Job(str(i), 0, ((1, 0.2), (3, 0.8)), acceptance=(rng.random(),), age_ms=3 * i, weight=i + 1)
                 for i in range(3))
    graph = Graph("batch", 2, decode_capacity=4, width=2)
    oracle = Oracle(jobs, (graph,))
    reference = SerialReference(oracle, Objective(2), time_value=1)
    for cohort in ((0, 1, 2), (0, 2), (1,)):
        action = Decision(graph, decodes=cohort)
        immediate, branches = oracle.transition(oracle.root, action)
        expected = Metrics()
        for probability, successor in branches:
            expected = expected.plus(reference.metrics(successor), probability)
        assert asdict(reference.expected_after(oracle.root, action)) == pytest.approx(asdict(expected))
        assert asdict(reference.immediate_metrics(oracle.root, action)) == pytest.approx(asdict(immediate))


def test_mixed_first_output_changes_age_accounting_at_the_transition():
    jobs = (Job("p", 3, ((1, 0.5), (3, 0.5))), Job("d", 0, ((2, 1.0),), acceptance=(0.5,), age_ms=10))
    graph = Graph("mixed", 4, prefill_capacity=4, decode_capacity=1, width=2)
    oracle = Oracle(jobs, (graph,))
    reference = SerialReference(oracle, Objective(2), time_value=1)
    action = Decision(graph, prefill=0, decodes=(1,))
    immediate, branches = oracle.transition(oracle.root, action)
    expected = Metrics()
    for probability, successor in branches:
        expected = expected.plus(reference.metrics(successor), probability)
    assert asdict(reference.expected_after(oracle.root, action)) == pytest.approx(asdict(expected))
    assert asdict(reference.immediate_metrics(oracle.root, action)) == pytest.approx(asdict(immediate))


def test_sixteen_stochastic_rows_do_not_enumerate_their_joint_outcomes():
    jobs = tuple(Job(str(i), 0, ((1, 0.25), (4, 0.75)), acceptance=(0.8, 0.6, 0.4)) for i in range(16))
    graph = Graph("b16w4", 20, decode_capacity=16, width=4)
    oracle = Oracle(jobs, (graph,))
    reference = SerialReference(oracle, Objective(64), time_value=1)
    action = Decision(graph, decodes=tuple(range(16)))
    score = reference.score(oracle.root, action)
    assert score > 0
    assert reference.table_entries <= 16 * 4
    assert not oracle._transitions


@pytest.mark.parametrize("seed", range(6))
def test_serial_value_improvement_is_bounded_by_the_oracle(seed):
    rng = Random(seed)
    graphs = tuple(Graph(f"b{b}w{w}", rng.uniform(1, 4), decode_capacity=b, width=w)
                   for b in (1, 2) for w in (1, 2))
    jobs = tuple(Job(name, 0, ((1, 0.5), (3, 0.5)), acceptance=(rng.random(),), age_ms=i * 4)
                 for i, name in enumerate(("a", "b")))
    objective = Objective(2, decode_age_price=0.1)
    oracle = Oracle(jobs, graphs)
    reference = SerialReference(oracle, objective, time_value=1)
    baseline = reference.value(oracle.root)
    improved = oracle.evaluate(reference.choose, objective)
    optimal = oracle.solve_priced(objective, time_value=1)
    value = lambda metrics: objective.net_work(metrics) - metrics.elapsed_ms
    assert baseline <= value(improved.metrics) + 1e-8
    assert value(improved.metrics) <= value(optimal.metrics) + 1e-8
