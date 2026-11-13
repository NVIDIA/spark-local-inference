# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from math import inf, nextafter

import pytest

from scripts.scheduling_sim.learning_policy import PolicyConfig
from scripts.scheduling_sim.learning_run import ExecutionConfig, ExecutionModel, Recovery, run_learning
from scripts.scheduling_sim.model import Graph, Request, View
from scripts.scheduling_sim.utility import UtilityConfig


GRAPHS = (Graph("small", 2, prefill_capacity=64), Graph("large", 8, prefill_capacity=512),
          Graph("decode", 3, decode_capacity=1))
NO_FAIRNESS = UtilityConfig(0, 0, 100, 0, "service")


def test_unknown_first_choice_cannot_see_execution_costs():
    slow_large = tuple(replace(g, duration_ms=1000) if g.name == "large" else g for g in GRAPHS)
    results = [run_learning([Request("p", 512, 1)], graphs, PolicyConfig(policy="greedy"),
                            warmup_ms=0, measurement_ms=100, retain_trace=True,
                            utility_config=NO_FAIRNESS) for graphs in (GRAPHS, slow_large)]
    assert results[0].trace[0]["graph"] == results[1].trace[0]["graph"] == "large"
    assert results[0].trace[0]["decision"] == results[1].trace[0]["decision"]
    assert results[0].trace[0]["duration_ms"] != results[1].trace[0]["duration_ms"]


def test_physical_noise_streams_are_keyed_by_graph_execution_not_policy_calls():
    config = ExecutionConfig(cv=.2, rho=.8, outlier_probability=.01)
    a, b = (ExecutionModel(GRAPHS, config, seed=42) for _ in range(2))
    expected = [a.execute("small", 0, 2).elapsed_ms for _ in range(4)]
    actual = []
    for _ in range(4):
        b.execute("large", 0, 8)
        actual.append(b.execute("small", 0, 2).elapsed_ms)
    assert actual == expected


def test_conditional_mean_does_not_reveal_next_jitter():
    model = ExecutionModel(GRAPHS, ExecutionConfig(cv=.5), seed=2)
    view = View("p", 512, 20, (1,) * 7)
    assert model.conditional_means([view], 0) == {g.name: g.duration_ms for g in GRAPHS}
    assert model.execute("large", 0, 8).elapsed_ms != 8


def test_reference_conditions_on_observed_correlated_noise_without_drawing_future_noise():
    model = ExecutionModel(GRAPHS, ExecutionConfig(cv=.5, rho=.8), seed=2)
    view = View("p", 512, 20, (1,) * 7)
    model.execute("large", 0, 8)
    means = model.conditional_means([view], 0)
    assert means["large"] != 8
    assert model.conditional_means([view], 0) == means
    assert model.counts["large"] == 1
    assert model.nominal_means([view], 0)["large"] == 8


def test_measurement_uses_complete_graphs_and_drain_does_not_enter_throughput():
    result = run_learning([Request("d", 0, 8)], (Graph("decode", 3, decode_capacity=1),),
                          PolicyConfig(policy="known"), warmup_ms=10, measurement_ms=20,
                          replenish=True, retain_trace=True, utility_config=NO_FAIRNESS)
    assert result.summary["measurement_start_ms"] == 12
    assert result.summary["measurement_busy_ms"] == 21
    assert result.summary["measurement_end_ms"] == 33
    assert result.summary["busy_useful_work_per_ms"] == pytest.approx(64 / 3)
    assert result.trace[-1]["drain"]
    assert not result.summary["unfinished"]
    assert not result.summary["censored_decode_gaps"]


def test_idle_is_wall_time_without_execution_or_retention_loss():
    result = run_learning([Request("p", 64, 1)], (GRAPHS[0],), PolicyConfig(policy="greedy"),
                          warmup_ms=0, measurement_ms=100, idle_gaps_ms=(3_600_000,),
                          initial_observations=1000, retain_trace=True, utility_config=NO_FAIRNESS)
    assert [event["category"] for event in result.trace] == ["ordinary", "ordinary"]
    assert result.summary["measurement_busy_ms"] == 4
    assert result.summary["measurement_elapsed_ms"] == 3_600_004
    assert result.summary["policy_state"]["observation_count"] == 1002


def test_ttl_expiry_is_visible_in_resumed_decision_trace():
    result = run_learning([Request("p", 64, 1)], (GRAPHS[0],), PolicyConfig(policy="ttl"),
                          warmup_ms=0, measurement_ms=100, idle_gaps_ms=(31_000,),
                          initial_observations=8, retain_trace=True, utility_config=NO_FAIRNESS)
    assert [event["category"] for event in result.trace] == ["ordinary", "probe"]
    assert result.trace[1]["estimated_ms"] is None


def test_missing_timings_complete_real_work_without_fabricated_observations():
    result = run_learning([Request("p", 512, 1)], GRAPHS, PolicyConfig(policy="thompson"),
                          warmup_ms=0, measurement_ms=100,
                          execution_config=ExecutionConfig(missing_probability=1),
                          retain_trace=True, utility_config=NO_FAIRNESS)
    assert result.requests[0]["output_tokens"] == 1
    assert result.summary["invalid_observations"] == 1
    assert result.summary["policy_state"]["observation_count"] == 0
    assert result.summary["final_credit_ms"] == 0
    assert result.summary["bootstrap_ms"] == 8


def test_short_bursts_cannot_fund_probes_after_positive_idle_credit_is_discarded():
    result = run_learning([Request("p", 64, 1)], GRAPHS, PolicyConfig(policy="epsilon", epsilon=1),
                          warmup_ms=0, measurement_ms=1000, idle_gaps_ms=(1000,) * 20,
                          initial_observations=8, utility_config=NO_FAIRNESS)
    assert result.summary["probe_count"] == 0
    assert result.summary["ordinary_ms"] == 42


def test_censored_tail_cannot_be_reported_as_finished():
    result = run_learning([Request("d0", 0, 10000), Request("d1", 0, 10000)],
                          (Graph("decode", 3, decode_capacity=1),),
                          PolicyConfig(policy="known"), warmup_ms=0, measurement_ms=10,
                          drain_ms=0, utility_config=NO_FAIRNESS)
    assert result.summary["unfinished"] == 2
    assert result.summary["censored_decode_gaps"] == 1
    assert result.requests[0]["unfinished"]


def test_trace_stream_and_retention_are_independent():
    streamed = []
    result = run_learning([Request("p", 64, 1)], GRAPHS, PolicyConfig(policy="known"),
                          warmup_ms=0, measurement_ms=100, trace_sink=streamed.append)
    assert len(streamed) == 1
    assert not result.trace
    assert streamed[0]["useful_work"] == 128


def test_requests_arriving_before_warmup_cannot_escape_unfinished_gate():
    result = run_learning([Request("d0", 0, 10000), Request("d1", 0, 10000)],
                          (Graph("decode", 3, decode_capacity=1),), PolicyConfig(policy="known"),
                          warmup_ms=3, measurement_ms=3, drain_ms=0, utility_config=NO_FAIRNESS)
    assert not any(row["measurement_cohort"] for row in result.requests)
    assert result.summary["unfinished"] == 2
    assert result.summary["starved"] == 1


def test_recovery_requires_sustained_best_exploitation():
    recovery = Recovery("best", 0, window_ms=10)
    recovery.observe(0, 10, "best", "ordinary")
    assert recovery.recovered_at == 10
    recovery.observe(10, 12, "other", "ordinary")
    assert recovery.recovered_at is None
    recovery.observe(12, 22, "best", "ordinary")
    assert recovery.recovered_at == 22
    recovery.observe(22, 24, "other", "probe")
    assert recovery.recovered_at == 22


@pytest.mark.parametrize("best_ms,recovered", [(nextafter(95., 0.), False),
                                               (95., True), (nextafter(95., inf), True)])
def test_recovery_threshold_and_probe_advanced_cutoff(best_ms, recovered):
    recovery = Recovery("best", 0, window_ms=100)
    recovery.observe(0, best_ms, "best", "ordinary")
    recovery.observe(best_ms, 100, "other", "ordinary")
    assert recovery.recovered_at == (100 if recovered else None)
    recovery.observe(100, 101, "other", "probe")
    assert recovery.recovered_at is None
    recovery.observe(101, 200, "other", "probe")
    assert recovery.recovered_at is None
    assert recovery.intervals == []


@pytest.mark.parametrize("configuration", [
    {"cv": -1}, {"rho": 1}, {"outlier_probability": 2},
    {"missing_probability": .6, "zero_probability": .5},
    {"drift_at_busy_ms": 10, "drift_end_busy_ms": 5},
])
def test_invalid_execution_models_are_rejected(configuration):
    with pytest.raises(ValueError):
        ExecutionConfig(**configuration)
