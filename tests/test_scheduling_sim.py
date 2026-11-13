# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from itertools import combinations
import json
from math import prod, sqrt
from random import Random
import subprocess
import sys

import pytest

from scripts.scheduling_sim.analysis import conditional_remaining, fair_allocation, output_age_area, prefill_completion_ms
from scripts.scheduling_sim.model import Graph, Request, Value, View, expected_output
from scripts.scheduling_sim.scheduler import candidates, choose, rank
from scripts.scheduling_sim.simulator import run
from scripts.scheduling_sim.workloads import catalog, workloads


def prefill(tokens, name="p", weight=1):
    return View(name, tokens, 4096, (1.0,) * 7, weight)


def decode(name="d", accepted=7, budget=4096, weight=1):
    return View(name, 0, budget, tuple(float(i < accepted) for i in range(7)), weight)


def test_expected_output_matches_exhaustive_prefix_outcomes():
    probabilities = (0.9, 0.4, 0.7)
    outcomes = sum((i + 1) * prod(probabilities[:i]) * (1 - p) for i, p in enumerate(probabilities))
    outcomes += 4 * prod(probabilities)
    assert expected_output(probabilities, 4) == pytest.approx(outcomes)
    assert expected_output((1, 0), 3) == 2
    assert expected_output((0, 1), 3) == 1


@pytest.mark.parametrize("count", [1, 2, 4, 5, 16])
def test_full_prefills_choose_highest_efficiency_without_a_size_rule(count):
    views = [prefill(32_768, f"p{i}") for i in range(count)]
    action = choose(views, catalog(), Value(64))
    assert action.graph.prefill_capacity == 4096
    assert action.expected_work(Value(64)) == 4096
    assert action.score(Value(64)) == pytest.approx(4096 / 66)


@pytest.mark.parametrize("remaining", [1, 32, 64])
def test_tail_uses_64_graph_by_effective_value(remaining):
    actions = rank([prefill(remaining)], catalog(), Value(64))
    assert actions[0].graph.prefill_capacity == 64
    assert all(a.expected_work(Value(64)) == remaining + 64 for a in actions)


def test_underfilled_larger_graph_can_lose_to_partial_completion():
    action = choose([prefill(576)], catalog(), Value(64))
    assert action.graph.prefill_capacity == 512
    assert action.expected_work(Value(64)) == 512


def test_smaller_graph_can_win_for_full_prompt_if_its_measured_efficiency_is_better():
    graphs = (Graph("small", 1, prefill_capacity=512), Graph("large", 100, prefill_capacity=4096))
    assert choose([prefill(10_000)], graphs, Value(64)).graph.name == "small"


def test_padding_is_not_banned_when_larger_graph_is_actually_faster():
    graphs = (Graph("small", 5, prefill_capacity=64), Graph("large", 1, prefill_capacity=4096))
    action = choose([prefill(64)], graphs, Value(64))
    assert action.graph.name == "large"
    assert action.expected_work(Value(64)) == 128


@pytest.mark.parametrize("accepted,width", [(0, 1), (2, 3), (7, 8)])
@pytest.mark.parametrize("count", [1, 2, 4, 5, 16])
def test_decode_depth_and_batch_compete_by_expected_output(accepted, width, count):
    action = choose([decode(f"d{i}", accepted) for i in range(count)], catalog(), Value(64))
    assert action.graph.width == width
    expected_count = 4 if count == 5 and accepted < 7 else count
    assert len(action.lanes) == expected_count
    assert sum(l.expected_output for l in action.lanes) == expected_count * (accepted + 1)


def test_inactive_batch_rows_and_rejected_drafts_earn_no_work():
    graph = Graph("b8w8", 30, decode_capacity=8, width=8)
    action = choose([decode("a", 0), decode("b", 7)], (graph,), Value(10))
    assert action.expected_work(Value(10)) == 90


def test_explicit_output_budget_caps_expected_work_without_revealing_eos():
    action = choose([decode(budget=2)], catalog(), Value(10))
    assert action.graph.width == 3
    assert action.lanes[0].width == 2
    assert action.lanes[0].expected_output == 2
    assert "output_tokens" not in View.__dataclass_fields__


@pytest.mark.parametrize("scale", [0.001, 1, 1000])
def test_uniform_time_scaling_does_not_change_selection(scale):
    graphs = tuple(replace(g, duration_ms=g.duration_ms * scale) for g in catalog())
    assert choose([decode()], graphs, Value(64)).graph.name == "decode-b1-w8"
    assert choose([prefill(32_768)], graphs, Value(64)).graph.name == "prefill-4096"


def test_breadth_and_depth_are_joint_candidates_not_fixed_priority():
    graphs = (Graph("b2w8", 10, decode_capacity=2, width=8),
              Graph("b8w1", 10, decode_capacity=8))
    high = [decode(f"d{i}") for i in range(8)]
    low = [decode(f"d{i}", 0) for i in range(8)]
    assert choose(high, graphs, Value(1)).graph.name == "b2w8"
    assert choose(low, graphs, Value(1)).graph.name == "b8w1"


def test_cohort_ranking_matches_exhaustive_subsets():
    rng = Random(17)
    views = [View(f"d{i}", 0, 100, tuple(rng.random() for _ in range(7)), 1 + rng.random()) for i in range(6)]
    value = Value(23)
    for graph in catalog():
        if not graph.decode_capacity:
            continue
        expected = max(
            sum(v.weight * value.decode_value * expected_output(v.acceptance, graph.width) for v in subset)
            / graph.duration_ms
            for size in range(1, min(graph.decode_capacity, len(views)) + 1)
            for subset in combinations(views, size)
        )
        assert choose(views, (graph,), value).score(value) == pytest.approx(expected)


def test_age_tie_break_revisits_five_streams_but_does_not_equalize_service():
    requests = [Request(f"d{i}", 0, 4096, acceptance=(0.0,) * 7) for i in range(5)]
    result = run(requests, catalog(), Value(64), horizon_ms=210)
    assert all(e["graph"] == "decode-b4-w1" for e in result.trace)
    assert [r["output_tokens"] for r in result.requests] == [10, 10, 10, 5, 5]
    assert result.summary["max_output_gap_ms"] == 42


def test_decode_value_changes_phase_priority_but_not_prefill_shape_order():
    views = [prefill(32_768), decode()]
    assert choose(views, catalog(), Value(1)).graph.prefill_capacity == 4096
    assert choose(views, catalog(), Value(1000)).graph.decode_capacity == 1
    for value in (Value(1), Value(1000)):
        prefills = [a for a in rank(views, catalog(), value) if a.graph.prefill_capacity]
        assert prefills[0].graph.prefill_capacity == 4096


def test_mixed_capture_can_win_with_smaller_prefill_by_value_alone():
    views = [prefill(32_768), *(decode(f"d{i}") for i in range(4))]
    graphs = (Graph("prefill", 64, prefill_capacity=4096),
              Graph("decode", 30, decode_capacity=4, width=8),
              Graph("mixed-large", 96, prefill_capacity=4096, decode_capacity=4, width=8),
              Graph("mixed-small", 35, prefill_capacity=512, decode_capacity=4, width=8))
    action = choose(views, graphs, Value(64))
    assert action.graph.name == "mixed-small"
    assert action.expected_work(Value(64)) == 512 + 64 * 32
    slower = tuple(replace(g, duration_ms=200) if g.name.startswith("mixed") else g for g in graphs)
    assert choose(views, slower, Value(64)).graph.name == "decode"


def test_cycle_value_is_time_weighted_not_average_of_graph_scores():
    actions = candidates([prefill(32_768), decode()], catalog(), Value(64))
    a = next(a for a in actions if a.graph.name == "prefill-4096")
    b = next(a for a in actions if a.graph.name == "decode-b1-w8")
    cycle = (a.expected_work(Value(64)) + b.expected_work(Value(64))) / (a.graph.duration_ms + b.graph.duration_ms)
    assert min(a.score(Value(64)), b.score(Value(64))) <= cycle <= max(a.score(Value(64)), b.score(Value(64)))


@pytest.mark.parametrize("name", list(workloads()))
@pytest.mark.parametrize("mixed", [False, True])
def test_basic_workloads_execute_without_idle_or_illegal_progress(name, mixed):
    requests = workloads()[name]
    result = run(requests, catalog(mixed=mixed), Value(64), horizon_ms=5000)
    assert result.trace
    assert all(event["end_ms"] > event["start_ms"] for event in result.trace)
    assert all(a["end_ms"] == b["start_ms"] for a, b in zip(result.trace, result.trace[1:]))
    for row, request in zip(result.requests, requests):
        assert 0 <= row["prefill_tokens"] <= request.prompt_tokens
        assert 0 <= row["output_tokens"] <= min(request.output_tokens, request.max_new_tokens)
        if row["output_tokens"]:
            assert row["prefill_tokens"] == request.prompt_tokens
    for event in result.trace:
        actual_prefill = sum(l["prefill"] for l in event["lanes"])
        assert actual_prefill <= event["prefill_capacity"]
        assert len([l for l in event["lanes"] if not l["prefill"]]) <= event["decode_capacity"]
        expected_work = sum(l["weight"] * (l["prefill"] + 64 * l["expected_output"]) for l in event["lanes"])
        assert event["score"] == pytest.approx(expected_work / event["duration_ms"])


def test_two_equal_prefills_take_full_chunks_without_starving_either():
    result = run([Request("a", 32_768, 1), Request("b", 32_768, 1)], catalog(), Value(1))
    events = result.trace
    assert len(events) == 16
    assert all(e["lanes"][0]["prefill"] == 4096 for e in events)
    assert {e["lanes"][0]["request"] for e in events[:2]} == {"a", "b"}
    assert result.summary["unfinished"] == 0


def test_staggered_arrival_waits_for_graph_boundary():
    result = run(workloads()["short_arrival"], catalog(), Value(64))
    assert result.trace[0]["end_ms"] == 66
    assert {l["request"] for l in result.trace[0]["lanes"]} == {"long"}
    assert result.requests[1]["ttft_ms"] >= 65


def test_finishing_decode_removes_its_value_from_future_choices():
    result = run([Request("p", 8192, 1), Request("d", 0, 1)], catalog(), Value(1000))
    assert result.trace[0]["lanes"][0]["request"] == "d"
    assert all(l["request"] != "d" for event in result.trace[1:] for l in event["lanes"])
    assert result.summary["unfinished"] == 0


def test_expected_acceptance_matches_realized_average_over_many_rounds():
    probabilities = (0.8, 0.6, 0.4)
    graph = Graph("verify", 1, decode_capacity=1, width=4)
    result = run([Request("d", 0, 100_000, acceptance=probabilities, max_new_tokens=100_000)],
                 (graph,), Value(1), horizon_ms=10_000, seed=3)
    assert result.summary["output_tokens"] / result.summary["launches"] == pytest.approx(expected_output(probabilities, 4), abs=0.03)


def test_constant_values_do_not_provide_a_fairness_guarantee():
    result = run([Request("p", 32_768, 1), Request("d", 0, 4096)], catalog(), Value(1000), horizon_ms=1000)
    assert result.requests[0]["prefill_tokens"] == 0
    assert result.requests[0]["unfinished"]
    assert result.requests[1]["output_tokens"] > 0


def test_greedy_action_rate_is_not_a_prefill_completion_oracle():
    graphs = (Graph("64", 2, prefill_capacity=64), Graph("128", 3, prefill_capacity=128), Graph("256", 4, prefill_capacity=256))
    result = run([Request("p", 129, 1)], graphs, Value(1))
    assert result.summary["busy_ms"] == 5
    assert prefill_completion_ms(129, graphs) == 4


def test_remaining_output_depends_on_the_conditioned_distribution():
    assert conditional_remaining({100: 1}, 20) == 80
    assert conditional_remaining({100: 1}, 80) == 20
    assert conditional_remaining({1: 0.9, 100: 0.1}, 0) == pytest.approx(10.9)
    assert conditional_remaining({1: 0.9, 100: 0.1}, 1) == 99
    with pytest.raises(ValueError, match="zero probability"):
        conditional_remaining({10: 1}, 10)


def test_age_penalty_counts_waiting_through_execution_and_is_additive():
    assert output_age_area(20, 30) == 20 * 30 + 30 * 30 / 2
    assert output_age_area(20, 30) == output_age_area(20, 10) + output_age_area(30, 20)


def test_decode_age_includes_initial_delay_and_open_stalls():
    graphs = (Graph("prefill", 100, prefill_capacity=4096), Graph("decode", 10, decode_capacity=1))
    result = run([Request("d", 0, 10), Request("p", 4096, 1)], graphs, Value(1), horizon_ms=105)
    row = result.requests[0]
    assert row["max_output_gap_ms"] == 110
    assert row["decode_age_area_ms2"] == 110 * 110 / 2
    stalled = run([Request("d", 0, 10), Request("p", 4096, 1)], graphs, Value(1), horizon_ms=50)
    assert stalled.requests[0]["decode_age_area_ms2"] == 100 * 100 / 2


def test_speculative_burst_clears_one_stall():
    graph = Graph("decode", 30, decode_capacity=1, width=8)
    result = run([Request("d", 0, 16)], (graph,), Value(1))
    assert result.summary["output_tokens"] == 16
    assert result.summary["decode_age_area_ms2"] == 2 * 30 * 30 / 2


def test_prefill_chunk_boundaries_are_not_output_events():
    graph = Graph("prefill", 10, prefill_capacity=64)
    result = run([Request("p", 128, 1)], (graph,), Value(1))
    assert result.requests[0]["ttft_ms"] == 20
    assert result.requests[0]["decode_age_area_ms2"] == 0


def test_proportional_fairness_recovers_weighted_cpu_time_shares():
    result = fair_allocation(((100, 0), (0, 1)), (0, 0), (3, 1))
    assert result.time_shares == pytest.approx((0.75, 0.25))
    assert result.rates == pytest.approx((75, 0.25))
    assert result.optimality_gap <= 1e-8


def test_token_value_and_fairness_have_independent_meanings():
    result = fair_allocation(((10, 0), (0, 1)), (1, 1), (1, 1))
    share = (7 + sqrt(85)) / 18
    assert result.time_shares == pytest.approx((share, 1 - share))
    assert all(rate > 0 for rate in result.rates)


def test_fair_allocation_uses_batch_economy_without_splitting_physical_cost():
    result = fair_allocation(((1, 0), (0, 1), (0.8, 0.8)), (0, 0), (1, 1))
    assert result.time_shares == pytest.approx((0, 0, 1))
    assert result.rates == pytest.approx((0.8, 0.8))


def test_five_decodes_can_share_efficient_batch_four_equally():
    cohorts = tuple(tuple(0 if i == excluded else 1 / 21 for i in range(5)) for excluded in range(5))
    result = fair_allocation((*cohorts, (1 / 29,) * 5), (0,) * 5, (1,) * 5)
    assert result.time_shares[-1] == pytest.approx(0, abs=1e-8)
    assert result.time_shares[:5] == pytest.approx((0.2,) * 5, abs=1e-8)
    assert result.rates == pytest.approx((4 / (5 * 21),) * 5)
    assert all(x > 1 / 29 for x in result.rates)


def test_rate_units_do_not_change_pure_proportional_fair_allocation():
    rates = ((10, 0), (0, 2), (4, 1))
    first = fair_allocation(rates, (0, 0), (1, 2))
    scaled = fair_allocation(tuple((a * 100, b * 0.001) for a, b in rates), (0, 0), (1, 2))
    assert first.time_shares == pytest.approx(scaled.time_shares)


def test_fair_reference_rejects_unserviceable_requests():
    with pytest.raises(ValueError, match="positive service rate"):
        fair_allocation(((1, 0),), (1, 1), (1, 1))


def test_fair_reference_reports_failure_to_converge():
    with pytest.raises(RuntimeError, match="did not converge"):
        fair_allocation(((1, 0), (0, 1), (0.5, 0.7)), (1, 1), (1, 1), max_iterations=1)


def test_cli_writes_scores_and_request_metrics(tmp_path):
    subprocess.run([sys.executable, "-m", "scripts.scheduling_sim", "--workload", "tail_64",
                    "--decode-values", "1,64", "--output", str(tmp_path)], check=True, capture_output=True)
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert trace[0]["graph"] == "prefill-64"
    assert (tmp_path / "requests.csv").exists()
