# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
import csv
import gzip
import io
import json
from math import isfinite
import subprocess
import sys

import pytest

from scripts.scheduling_sim.learning_analysis import (
    HELDOUT_SEEDS, calibration_gate, compare_workload, evaluation_seeds, paired_ratio,
    recovery_gate, stationary_reset_gate,
)
from scripts.scheduling_sim.learning_scenarios import scenarios, selected_scenarios


def measurements(*, throughput=1., ttft=1., gap=1., censored=0, seeds=HELDOUT_SEEDS):
    rows = []
    for seed in seeds:
        for name, multiplier in (("known", 1.), ("candidate", throughput)):
            rows.append({"config_id": name, "seed": seed,
                         "busy_useful_work_per_ms": 100. * multiplier,
                         "p99_ttft_ms": 10. * (ttft if name == "candidate" else 1.),
                         "p99_decode_gap_ms": 20. * (gap if name == "candidate" else 1.),
                         "unfinished": 0, "starved": 0,
                         "measurement_complete": True, "ttft_samples": 200, "decode_gap_samples": 200,
                         "censored_ttft": censored if name == "candidate" else 0,
                         "censored_decode_gaps": 0})
    return rows


def test_stationary_gates_preserve_thresholds_and_pair_whole_seeds():
    result = compare_workload(measurements(throughput=.99, ttft=1.04, gap=1.04),
                              "candidate", configuration_count=3, replicates=100)
    assert result["passed"]
    metric = result["metrics"]["busy_useful_throughput_ratio"]
    assert metric["confidence"] == pytest.approx(1 - .05 / 3)
    assert metric["lower"] == pytest.approx(.99)
    assert not compare_workload(measurements(throughput=.97), "candidate", replicates=100)["passed"]
    assert not compare_workload(measurements(ttft=1.06), "candidate", replicates=100)["passed"]
    assert not compare_workload(measurements(gap=1.06), "candidate", replicates=100)["passed"]


def test_incomplete_censored_or_missing_tail_evidence_cannot_pass():
    assert not compare_workload(measurements(censored=1), "candidate", replicates=100)["passed"]
    rows = measurements()
    rows[-1]["p99_ttft_ms"] = None
    assert not compare_workload(rows, "candidate", replicates=100)["passed"]
    assert not compare_workload(rows[:-1], "candidate", replicates=100)["passed"]
    rows = measurements()
    rows[-1]["unfinished"] = 1
    assert not compare_workload(rows, "candidate", replicates=100)["passed"]
    rows = measurements()
    rows[-1]["measurement_complete"] = False
    assert not compare_workload(rows, "candidate", replicates=100)["passed"]
    rows = measurements()
    rows[-1]["ttft_samples"] = 99
    assert not compare_workload(rows, "candidate", replicates=100)["passed"]


def test_nonapplicable_decode_tail_is_explicit():
    rows = measurements()
    for row in rows:
        row["p99_decode_gap_ms"] = None
    assert compare_workload(rows, "candidate", require_decode=False, replicates=100)["passed"]
    assert not compare_workload(rows, "candidate", replicates=100)["passed"]


def test_paired_bootstrap_preserves_shared_seed_variability():
    reference = {seed: float(1 + seed % 11) for seed in HELDOUT_SEEDS}
    candidate = {seed: 1.02 * value for seed, value in reference.items()}
    interval = paired_ratio(candidate, reference, replicates=100)
    assert interval.lower == pytest.approx(1.02)
    assert interval.upper == pytest.approx(1.02)
    with pytest.raises(ValueError, match="same nonempty seed set"):
        paired_ratio(candidate, {0: 1.})


def test_recovery_requires_61_of_64_including_unrecovered_seeds():
    rows = [{"seed": seed, "recovery_ms": 60_000. if index < 61 else None,
             "measurement_complete": True, "busy_ms": 150_000.}
            for index, seed in enumerate(HELDOUT_SEEDS)]
    assert recovery_gate(rows)["passed"]
    rows[60]["recovery_ms"] = 60_001.
    assert not recovery_gate(rows)["passed"]
    assert not recovery_gate(rows[:60])["passed"]


def test_short_or_invalid_recovery_evidence_cannot_pass():
    rows = [{"seed": seed, "recovery_ms": 10_000., "measurement_complete": True, "busy_ms": 150_000.}
            for seed in HELDOUT_SEEDS]
    assert recovery_gate(rows, change_busy_ms=30_000)["passed"]
    for field, value in (("measurement_complete", False), ("busy_ms", 89_999.)):
        modified = [dict(row) for row in rows]
        modified[-1][field] = value
        assert not recovery_gate(modified, change_busy_ms=30_000)["passed"]
    modified = [dict(row) for row in rows]
    del modified[-1]["measurement_complete"]
    assert not recovery_gate(modified)["passed"]
    for value in (-1., float("nan"), float("inf")):
        modified = [dict(row, recovery_ms=value) for row in rows]
        assert not recovery_gate(modified)["passed"]


def test_stationary_false_switch_gate_is_empirical_zero_of_64():
    rows = [{"seed": seed, "regime_resets": 0} for seed in HELDOUT_SEEDS]
    assert stationary_reset_gate(rows)["passed"]
    rows[-1]["regime_resets"] = 1
    assert not stationary_reset_gate(rows)["passed"]


def test_interval_calibration_checks_fixed_counts_and_width():
    rows = [{"seed": seed, "covered_16": index < 58, "covered_1000": True,
             "width_16": 4., "width_1000": 1.}
            for index, seed in enumerate(HELDOUT_SEEDS)]
    assert calibration_gate(rows)["passed"]
    for row in rows[:33]:
        row["width_1000"] = 2.
    assert not calibration_gate(rows)["passed"]


def test_explicit_evaluation_seed_sets_preserve_all_count_and_pairing_gates():
    seeds = tuple(range(20_000, 20_064))
    assert evaluation_seeds() == HELDOUT_SEEDS
    assert evaluation_seeds(seeds) == seeds
    rows = measurements(seeds=seeds)
    assert compare_workload(rows, "candidate", expected_seeds=seeds, replicates=100)["passed"]
    assert not compare_workload(rows, "candidate", replicates=100)["passed"]
    assert not compare_workload(rows[:-1], "candidate", expected_seeds=seeds, replicates=100)["passed"]
    recovery = [{"seed": seed, "recovery_ms": 10_000., "measurement_complete": True,
                 "busy_ms": 150_000., "regime_resets": 0, "covered_16": True,
                 "covered_1000": True, "width_16": 4., "width_1000": 1.} for seed in seeds]
    for gate in (recovery_gate, stationary_reset_gate, calibration_gate):
        assert gate(recovery, expected_seeds=seeds)["passed"]
        assert not gate(recovery[:-1], expected_seeds=seeds)["passed"]
        assert not gate(recovery)["passed"]
    for invalid in (seeds[:-1], seeds + (20_064,), seeds[:-1] + (seeds[0],), tuple(range(64))):
        with pytest.raises(ValueError):
            evaluation_seeds(invalid)


def test_ttl_comparison_does_not_replace_known_mean_gates():
    rows = measurements(throughput=.9)
    rows.extend(dict(row, config_id="ttl", busy_useful_work_per_ms=80.)
                for row in rows.copy() if row["config_id"] == "known")
    assert not compare_workload(rows, "candidate", replicates=100)["passed"]
    ttl = compare_workload(rows, "candidate", reference_id="ttl", replicates=100)
    assert ttl["passed"] and ttl["reference_id"] == "ttl"
    assert ttl["metrics"]["busy_useful_throughput_ratio"]["lower"] == pytest.approx(1.125)
    assert ttl["metrics"]["busy_useful_throughput_ratio"]["threshold"] == .98
    assert ttl["metrics"]["mean_seed_p99_ttft_ratio"]["threshold"] == 1.05
    rows.pop()
    assert not compare_workload(rows, "candidate", reference_id="ttl", replicates=100)["passed"]


def test_scenarios_cover_fixed_core_noise_and_declared_stress_conditions():
    cases = scenarios()
    for cv in (0., .05, .2, .5):
        for count in (1, 4, 16):
            case = cases[f"prefill-{count}-decoders-cv{cv:g}"]
            assert case.group == "core"
            assert len(case.requests) == count + 1
            assert case.execution["cv"] == cv
    assert cases["mature-1000"].initial_observations == 1000
    assert cases["poisoned-first-10x"].execution["poison_graphs"] == ("prefill-4096", "decode-b1-w8")
    for seconds in (29, 31, 120, 3600):
        case = cases[f"idle-{seconds}s"]
        assert case.idle_gaps_ms[0] == seconds * 1000
        assert case.warmup_ms == 0 and not case.replenish
    assert cases["correlated-rho0.95"].execution["rho"] == .95
    assert cases["rare-10x-outliers"].execution["outlier_probability"] == .001
    assert cases["temporary-global-slowdown"].execution["drift_end_busy_ms"] == 60_000
    assert cases["missing-timings"].execution["missing_probability"] == .1
    assert cases["zero-timings"].execution["zero_probability"] == .1


def test_recovery_choices_have_equal_service_and_twenty_percent_advantage():
    for case in selected_scenarios("recovery"):
        multipliers = dict(case.execution.get("drift_multipliers", ()))
        times = {graph.name: graph.duration_ms * multipliers.get(graph.name, 1.) for graph in case.graphs}
        winner = min(times, key=times.get)
        assert winner == case.recovery_best
        runner_up = min(time for name, time in times.items() if name != winner)
        assert runner_up / times[winner] >= 1.2
        assert all(graph.decode_capacity == graph.width == 1 for graph in case.graphs)
        assert case.utility_config.token_fairness == case.utility_config.delivery_fairness == 0


def test_concurrent_decode_variants_cover_low_and_heterogeneous_acceptance_with_mixed_graphs():
    cases = scenarios()
    low = cases["prefill-4-low-acceptance-decoders-cv0.2"]
    heterogeneous = cases["prefill-4-heterogeneous-decoders-mixed-cv0.2"]
    for case in (low, heterogeneous):
        assert case.group == "core" and case.execution == {"cv": .2}
        assert len(case.requests) == 5 and case.requests[0].prompt_tokens == 8192
        assert all(request.prompt_tokens == 0 for request in case.requests[1:])
    assert all(request.acceptance == (0.,) * 7 for request in low.requests[1:])
    assert [request.acceptance for request in heterogeneous.requests[1:]] == [
        (acceptance,) * 7 for acceptance in (0., .3, .8, 1.)]
    assert not any(graph.prefill_capacity and graph.decode_capacity for graph in low.graphs)
    assert any(graph.prefill_capacity and graph.decode_capacity for graph in heterogeneous.graphs)
    assert len(cases) == 62
    assert sum(case.group == "core" for case in cases.values()) == 35


def test_mature_recovery_retains_one_thousand_observations_before_unused_winner_changes():
    for count in (2, 4):
        case = scenarios()[f"mature-1000-recovery-{count}-slow-fast"]
        assert case.initial_observations == 1000
        assert case.group == "stress"
        assert case.recovery_best == "choice-1"
        assert case.execution["drift_multipliers"] == (("choice-1", .25),)


def test_alternative_catalogs_reverse_prefill_and_width_efficiency_choices():
    from scripts.scheduling_sim.model import Value, View
    from scripts.scheduling_sim.scheduler import choose

    cases = scenarios()
    knee = cases["catalog-prefill-efficiency-knee"]
    view = View("prompt", 8192, 128, (1.,) * 7)
    assert choose([view], knee.graphs, Value(64)).graph.prefill_capacity == 512
    for count, width in ((1, 3), (4, 8)):
        case = cases[f"catalog-width-crossover-{count}"]
        views = [View(f"d{i}", 0, 128, (1.,) * 7) for i in range(count)]
        action = choose(views, case.graphs, Value(64))
        assert action.graph.decode_capacity == count
        assert action.graph.width == width


def test_scenario_selection_is_deterministic_and_rejects_typos():
    assert selected_scenarios("core,tail-64-cv0") == selected_scenarios("core")
    with pytest.raises(ValueError, match="unknown learning scenario"):
        selected_scenarios("typo")


def test_validation_requires_a_frozen_small_configuration_set(tmp_path):
    from scripts.scheduling_sim.learning import load_configurations
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    path = tmp_path / "nominated.json"
    config = {"id": "candidate", "config": asdict(PolicyConfig(policy="thompson"))}
    path.write_text(json.dumps([config]))
    assert load_configurations(path, PolicyConfig)[0].id == "candidate"
    path.write_text(json.dumps([config] * 4))
    with pytest.raises(ValueError, match="one through three"):
        load_configurations(path, PolicyConfig)


def test_comparison_configs_allow_controls_without_adding_nominees(tmp_path):
    from scripts.scheduling_sim.learning import Configuration, add_comparison_configurations, load_configurations
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    known = Configuration("known", PolicyConfig(policy="known", funded=False))
    candidate = Configuration("candidate", PolicyConfig(policy="epsilon"))
    path = tmp_path / "comparisons.json"
    path.write_text(json.dumps([{"id": "ema", "config": asdict(PolicyConfig(policy="ema", funded=False))},
                                {"id": "softmax-control", "config": asdict(PolicyConfig(policy="softmax"))}]))
    controls = load_configurations(path, PolicyConfig, comparison=True)
    assert len(add_comparison_configurations((known, candidate), controls)) == 4
    assert add_comparison_configurations((known,), (known,)) == (known,)
    with pytest.raises(ValueError, match="distinct from nominated"):
        add_comparison_configurations((candidate,), (candidate,))
    with pytest.raises(ValueError, match="distinct from baseline names"):
        load_configurations(path, PolicyConfig)


def test_baseline_recovery_reports_controls_on_the_explicit_evaluation_seed_set():
    from scripts.scheduling_sim.learning import Configuration, baseline_recovery, evaluate
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    seeds = tuple(range(20_000, 20_064))
    case = scenarios()["mature-1000-recovery-4-slow-fast"]
    configurations = tuple(Configuration(name, PolicyConfig(policy=policy, funded=False))
                           for name, policy in (("known", "known"), ("ttl", "ttl"), ("ema", "ema"),
                                                ("softmax-control", "softmax")))
    rows = [{"scenario": case.name, "config_id": configuration.id, "seed": seed,
             "measurement_complete": True, "busy_ms": 150_000., "recovery_ms": 20_000.}
            for configuration in configurations for seed in seeds]
    controls = {"softmax-control"}
    result = baseline_recovery(rows, (case,), configurations, expected_seeds=seeds, comparison_ids=controls)
    assert {entry["config_id"] for entry in result} == {configuration.id for configuration in configurations}
    assert all(entry["passed"] and entry["seeds"] == 64 and entry["group"] == "stress" for entry in result)
    assert evaluate(rows, (case,), configurations, expected_seeds=seeds, comparison_ids=controls) == []


@pytest.mark.parametrize("reference", [None, "known", "ttl"])
def test_validation_cli_uses_explicit_seeds_and_keeps_comparison_gates_separate(tmp_path, monkeypatch, reference):
    import csv
    from types import SimpleNamespace
    from scripts.scheduling_sim import learning
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    case = scenarios()["prefill-8192-cv0"]
    monkeypatch.setattr(learning, "scenarios", lambda: {case.name: case})
    monkeypatch.setattr(learning, "selected_scenarios", lambda selection: (case,))
    def execute(tasks, workers, trace_sink=None):
        rates = {"known": 100., "ttl": 80., "ema": 85., "softmax-control": 95., "candidate": 90.}
        for scenario, configuration, seed, *_ in tasks:
            summary = next(row for row in measurements(seeds=(seed,)) if row["config_id"] == "known")
            summary = {key: value for key, value in summary.items() if key not in ("seed", "config_id")}
            summary.update(busy_useful_work_per_ms=rates[configuration.id], regime_resets=0, probe_ms=0.)
            yield ({"scenario": scenario.name, "group": scenario.group, "config_id": configuration.id,
                    "policy": configuration.config.policy, "seed": seed}, SimpleNamespace(summary=summary, requests=[]))
    monkeypatch.setattr(learning, "_execute_tasks", execute)
    monkeypatch.setattr(learning, "calibrate", lambda config, seeds: [
        {"seed": seed, "covered_16": True, "covered_1000": True, "width_16": 4., "width_1000": 1.}
        for seed in seeds])
    nominated, controls, output = (tmp_path / name for name in ("nominated.json", "controls.json", "results"))
    nominated.write_text(json.dumps([{"id": "candidate", "config": asdict(PolicyConfig(policy="epsilon"))}]))
    controls.write_text(json.dumps([{"id": "ema", "config": asdict(PolicyConfig(policy="ema", funded=False))},
                                   {"id": "softmax-control", "config": asdict(PolicyConfig(policy="softmax"))}]))
    arguments = ["--preset", "validate", "--configs", str(nominated), "--comparison-configs", str(controls),
                 "--seeds", "20000:20064", "--bootstrap-replicates", "100", "--output", str(output)]
    if reference is not None:
        arguments.extend(("--reference", reference))
    assert learning.main(arguments) == 0
    result = json.loads((output / "comparisons.json").read_text())
    accepted = reference == "ttl"
    assert result["reference_id"] == (reference or "known")
    assert result["complete_validation_suite"]
    assert result["accepted_configurations"] == (["candidate"] if accepted else [])
    assert result["selected_configuration"] == ("candidate" if accepted else None)
    assert {gate["config_id"] for gate in result["gates"]} == {"candidate"}
    assert result["gates"][0]["passed"] is accepted
    assert result["gates"][0]["reference_id"] == (reference or "known")
    assert result["gates"][0]["metrics"]["busy_useful_throughput_ratio"]["confidence"] == .95
    comparisons = {gate["reference_id"]: gate for gate in result["comparison_gates"]}
    diagnostic_reference = "known" if accepted else "ttl"
    assert set(comparisons) == {diagnostic_reference, "ema", "softmax-control"}
    assert comparisons[diagnostic_reference]["passed"] is not accepted
    assert not comparisons["softmax-control"]["passed"]
    with (output / "summary.csv").open() as source:
        measured = list(csv.DictReader(source))
    assert len(measured) == 5 * 64
    assert {int(row["seed"]) for row in measured} == set(range(20_000, 20_064))
    with (output / "calibration.csv").open() as source:
        assert {row["config_id"] for row in csv.DictReader(source)} == {"candidate"}


def test_validation_cli_rejects_tuning_overlap_and_non64_seed_ranges(tmp_path):
    from scripts.scheduling_sim.learning import main

    for seeds in ("0:64", "20000:20063", "20000:20065"):
        with pytest.raises(SystemExit) as error:
            main(["--preset", "validate", "--configs", str(tmp_path / "unused.json"), "--seeds", seeds])
        assert error.value.code == 2


def test_smoke_cli_writes_raw_measurements_and_streamed_trace(tmp_path):
    destination = tmp_path / "learning"
    command = [sys.executable, "-m", "scripts.scheduling_sim.learning", "--preset", "smoke",
               "--scenarios", "tail-64-cv0", "--policies", "epsilon", "--seeds", "0:1",
               "--warmup-ms", "0", "--measurement-ms", "100", "--skip-calibration",
               "--bootstrap-replicates", "100", "--trace", "--output", str(destination)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "Synthetic timing experiment" in result.stdout
    assert (destination / "summary.csv").read_text().startswith("scenario,group,config_id,policy,seed,")
    assert (destination / "requests.csv").stat().st_size > 0
    events = [json.loads(line) for line in (destination / "trace.jsonl").read_text().splitlines()]
    assert events and all(event["scenario"] == "tail-64-cv0" for event in events)
    comparison = json.loads((destination / "comparisons.json").read_text())
    assert comparison["accepted_configurations"] == []
    assert comparison["selected_configuration"] is None
    assert not comparison["complete_core_suite"]


def test_calibration_collects_fixed_sample_mean_intervals():
    from scripts.scheduling_sim.learning import calibrate
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    rows = calibrate(PolicyConfig(), (0, 1))
    assert [row["seed"] for row in rows] == [0, 1]
    assert all(isfinite(row["width_16"]) and isfinite(row["width_1000"]) for row in rows)


def test_screen_grid_varies_policy_specific_parameters_and_includes_controls():
    from scripts.scheduling_sim.learning import build_configurations, parser
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    args = parser().parse_args(["--preset", "screen", "--screen-grid"])
    configurations = build_configurations(args, PolicyConfig)
    assert len({c.id for c in configurations}) == len(configurations)
    for mode in ("cumulative", "regime"):
        for funded in (True, False):
            rows = [c.config for c in configurations if c.config.estimator == mode and c.config.funded == funded]
            assert {c.epsilon for c in rows if c.policy == "epsilon"} == {.01, .05, .1}
            assert {c.temperature for c in rows if c.policy == "softmax"} == {.03, .1, .3}
            assert {c.quantile for c in rows if c.policy == "optimism"} == {.01, .05, .1}
            assert {c.anneal for c in rows if c.policy == "softmax"} == {True, False}
            for name in ("epsilon", "softmax", "optimism", "thompson", "predictive"):
                assert {c.refresh_every for c in rows if c.policy == name} == {0, 16}
        assert any(c.config.policy == "greedy" and c.config.estimator == mode for c in configurations)


def test_one_factor_sweeps_do_not_duplicate_irrelevant_policy_variants():
    from scripts.scheduling_sim.learning import build_configurations, parser
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    args = parser().parse_args(["--policies", "epsilon,thompson", "--sweep", "epsilon=0.01,0.1"])
    configurations = build_configurations(args, PolicyConfig)
    assert len([c for c in configurations if c.config.policy == "thompson"]) == 1
    assert {c.config.epsilon for c in configurations if c.config.policy == "epsilon"} == {.01, .05, .1}


def test_targeted_probe_sweeps_leave_other_policies_unchanged():
    from scripts.scheduling_sim.learning import build_configurations, parser
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    args = parser().parse_args(["--policies", "targeted,epsilon", "--estimator", "log-regime",
                               "--probe-extra-ms", "0", "--sweep", "probe_loss_ms=1,5,10",
                               "--sweep", "refresh_busy_ms=5000,15000,30000"])
    configurations = build_configurations(args, PolicyConfig)
    targeted = [c.config for c in configurations if c.config.policy == "targeted"]
    assert len(targeted) == 5 and {c.probe_loss_ms for c in targeted} == {1., 5., 10.}
    assert {c.refresh_busy_ms for c in targeted} == {5000., 15000., 30000.}
    assert len([c for c in configurations if c.config.policy == "epsilon"]) == 1


def test_audit_arbitration_is_explicit_and_sweeps_one_factor():
    from scripts.scheduling_sim.learning import build_configurations, parser
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    strict = build_configurations(parser().parse_args([
        "--policies", "targeted", "--estimator", "log-regime", "--probe-extra-ms", "5"]), PolicyConfig)
    assert strict[-1].config.audit_every == 0
    assert strict[-1].id == "targeted:funded:log-regime:probe_extra_ms=5.0"
    args = parser().parse_args([
        "--policies", "targeted,epsilon", "--estimator", "log-regime", "--probe-extra-ms", "5",
        "--audit-every", "4", "--sweep", "probe_extra_ms=1e12", "--sweep", "audit_every=0,8"])
    configurations = build_configurations(args, PolicyConfig)
    targeted = [c.config for c in configurations if c.config.policy == "targeted"]
    assert {(c.probe_extra_ms, c.audit_every) for c in targeted} == {
        (5., 4), (1e12, 4), (5., 0), (5., 8)}
    assert all(isinstance(c.audit_every, int) for c in targeted)
    assert len([c for c in configurations if c.config.policy == "epsilon"]) == 1


def test_peer_delay_bound_is_optional_and_targeted_only():
    from scripts.scheduling_sim.learning import build_configurations, parser
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    strict = build_configurations(parser().parse_args([
        "--policies", "targeted", "--estimator", "log-regime", "--probe-extra-ms", "5"]), PolicyConfig)
    assert strict[-1].config.probe_harm_ms is None
    assert strict[-1].id == "targeted:funded:log-regime:probe_extra_ms=5.0"
    swept = build_configurations(parser().parse_args([
        "--policies", "targeted,epsilon", "--sweep", "probe_harm_ms=0,5"]), PolicyConfig)
    assert {c.config.probe_harm_ms for c in swept if c.config.policy == "targeted"} == {None, 0., 5.}
    assert [c.config.probe_harm_ms for c in swept if c.config.policy == "epsilon"] == [None]
    explicit = build_configurations(parser().parse_args([
        "--policies", "targeted,epsilon,ema", "--probe-harm-ms", "5"]), PolicyConfig)
    assert [c.config.probe_harm_ms for c in explicit if c.config.policy == "targeted"] == [5.]
    assert [c.config.probe_harm_ms for c in explicit if c.config.policy == "epsilon"] == [None]
    assert [(c.id, c.config.probe_harm_ms) for c in explicit if c.config.policy == "ema"] == [("ema", None)]


def test_soft_peer_credit_and_estimation_options_are_independent():
    from scripts.scheduling_sim.learning import build_configurations, parser
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    defaults = build_configurations(parser().parse_args([
        "--policies", "targeted", "--estimator", "log-regime", "--probe-extra-ms", "5",
        "--refresh-busy-ms", "15000"]), PolicyConfig)[-1]
    assert defaults.config.harm_alpha is None
    assert defaults.config.initial_discovery == "productive"
    assert defaults.config.mean_estimator == "arithmetic"
    assert defaults.id == "targeted:funded:log-regime:probe_extra_ms=5.0:refresh_busy_ms=15000.0"
    varied = build_configurations(parser().parse_args([
        "--policies", "targeted,epsilon", "--initial-discovery", "native", "--mean-estimator", "ema",
        "--harm-alpha", "0.002", "--sweep", "harm_alpha=0.001,0.005"]), PolicyConfig)
    assert {c.config.harm_alpha for c in varied if c.config.policy == "targeted"} == {.001, .002, .005}
    assert [c.config.harm_alpha for c in varied if c.config.policy == "epsilon"] == [None]
    assert all(c.config.initial_discovery == "native" and c.config.mean_estimator == "ema"
               for c in varied if c.config.policy in ("targeted", "epsilon"))
    mean_only = build_configurations(parser().parse_args([
        "--policies", "targeted", "--mean-estimator", "ema"]), PolicyConfig)[-1].config
    assert mean_only.initial_discovery == "productive" and mean_only.harm_alpha is None
    discovery_only = build_configurations(parser().parse_args([
        "--policies", "targeted", "--initial-discovery", "native"]), PolicyConfig)[-1].config
    assert discovery_only.mean_estimator == "arithmetic" and discovery_only.harm_alpha is None
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_configurations(parser().parse_args([
            "--policies", "targeted", "--harm-alpha", "0.002", "--probe-harm-ms", "5"]), PolicyConfig)


def test_partial_block_uncertainty_is_explicit_and_keeps_default_ids():
    from scripts.scheduling_sim.learning import build_configurations, parser
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    defaults = build_configurations(parser().parse_args(["--policies", "targeted"]), PolicyConfig)
    assert defaults[-1].config.partial_blocks is False
    assert defaults[-1].id == "targeted:funded:cumulative"
    enabled = build_configurations(parser().parse_args([
        "--policies", "targeted,softmax", "--partial-blocks"]), PolicyConfig)
    assert all(c.config.partial_blocks is False for c in enabled if c.id in ("known", "ttl"))
    assert all(c.config.partial_blocks is True and c.id.endswith(":partial_blocks=True")
               for c in enabled if c.config.policy in ("targeted", "softmax"))
    with pytest.raises(ValueError, match="sweeps are numeric"):
        build_configurations(parser().parse_args([
            "--policies", "targeted", "--partial-blocks", "--sweep", "partial_blocks=0,1"]), PolicyConfig)
    for value in (None, 0, 1, "true"):
        with pytest.raises(ValueError, match="boolean"):
            PolicyConfig(partial_blocks=value)


def test_partial_block_configuration_reaches_runtime_estimates_and_json(tmp_path):
    from scripts.scheduling_sim.learning import Configuration, _run_task, configuration_id, load_configurations
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    config = PolicyConfig(policy="targeted", partial_blocks=True)
    configuration = Configuration(configuration_id(config), config)
    path = tmp_path / "configurations.json"
    path.write_text(json.dumps([{"id": configuration.id, "config": asdict(config)}]))
    assert load_configurations(path, PolicyConfig) == (configuration,)
    _, result = _run_task((scenarios()["tail-64-cv0"], configuration, 0, 0., 100., 1000))
    estimates = result.summary["policy_state"]["costs"]
    assert estimates and all(estimate["partial_blocks"] is True for estimate in estimates.values())
    _, baseline = _run_task((scenarios()["tail-64-cv0"], Configuration("targeted", PolicyConfig(policy="targeted")),
                            0, 0., 100., 1000))
    assert all("partial_blocks" not in estimate for estimate in baseline.summary["policy_state"]["costs"].values())


def test_calibration_uses_configured_partial_blocks(monkeypatch):
    from scripts.scheduling_sim import timing
    from scripts.scheduling_sim.learning import calibrate
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    constructor = timing.TimingEstimate
    observed = []
    def estimate(*args, **kwargs):
        observed.append(kwargs["partial_blocks"])
        return constructor(*args, **kwargs)
    monkeypatch.setattr(timing, "TimingEstimate", estimate)
    rows = calibrate(PolicyConfig(partial_blocks=True), (0, 1))
    assert observed and all(observed)
    assert [row["seed"] for row in rows] == [0, 1]
    assert all(row["width_1000"] < row["width_16"] for row in rows)


@pytest.mark.parametrize("block_size,mature_boundary", [(8, 1000), (32, 1024)])
def test_partial_block_calibration_reports_initial_and_mature_prior_mismatch(block_size, mature_boundary):
    from scripts.scheduling_sim.learning import calibrate
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    config = PolicyConfig(block_size=block_size, partial_blocks=True)
    row = calibrate(config, (0,))[0]
    diagnostics = row["partial_block_diagnostics"]
    cases = {(entry["prior_location_ms"], entry["true_mean_ms"]) for entry in diagnostics}
    assert cases == {(20., 20.), (1., 20.), (1., 40.)}
    for prior, true_mean in cases:
        samples = [entry for entry in diagnostics
                   if (entry["prior_location_ms"], entry["true_mean_ms"]) == (prior, true_mean)]
        assert {entry["count"] for entry in samples} == {
            count for remainder in range(1, block_size) for count in (remainder, mature_boundary + remainder)}
        assert all(0 < entry["partial_count"] < block_size and isinstance(entry["covered"], bool)
                   and isfinite(entry["width_ms"]) and entry["width_ms"] > 0 for entry in samples)
    assert calibration_gate([row], heldout=False)["checks"].keys() == {
        "seed_set", "coverage_16", "coverage_1000", "mature_width"}
    assert "partial_block_diagnostics" not in calibrate(PolicyConfig(block_size=block_size), (0,))[0]


@pytest.mark.parametrize("estimator", ["cumulative", "regime", "log-regime"])
def test_partial_block_diagnostics_preserve_completed_boundary_calibration(estimator):
    from scripts.scheduling_sim.learning import calibrate
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    baseline = calibrate(PolicyConfig(estimator=estimator), (0, 1))
    enabled = calibrate(PolicyConfig(estimator=estimator, partial_blocks=True), (0, 1))
    assert baseline == [{key: value for key, value in row.items() if key != "partial_block_diagnostics"}
                        for row in enabled]


def test_validation_rejects_partial_block_override(tmp_path, capsys):
    from scripts.scheduling_sim.learning import main

    with pytest.raises(SystemExit) as error:
        main(["--preset", "validate", "--configs", str(tmp_path / "unused.json"), "--partial-blocks"])
    assert error.value.code == 2
    assert "parameters come exclusively from --configs" in capsys.readouterr().err


def test_csv_stream_gzip_preserves_csv_json_and_extra_fields(tmp_path):
    from scripts.scheduling_sim.learning import CsvStream

    rows = [{"scenario": "roundtrip", "text": 'quotes "λ"\nand a newline', "lanes": ["a", "b"]},
            {"scenario": "roundtrip", "text": "tail", "lanes": [], "added": {"count": 3}}]
    plain, compressed = tmp_path / "requests.csv", tmp_path / "requests.csv.gz"
    for path in (plain, compressed):
        writer = CsvStream(path)
        for row in rows:
            writer.write(row)
        writer.close()
    with gzip.open(compressed, "rb") as handle:
        restored = handle.read()
    assert compressed.read_bytes().startswith(b"\x1f\x8b")
    assert restored == plain.read_bytes()
    parsed = list(csv.DictReader(io.StringIO(restored.decode(), newline="")))
    assert parsed[0]["text"] == rows[0]["text"]
    assert json.loads(parsed[0]["lanes"]) == ["a", "b"]
    assert json.loads(parsed[1]["extra_json"]) == {"added": {"count": 3}}


def test_compressed_request_cli_preserves_measurements_and_default_filenames(tmp_path):
    outputs = []
    for compressed in (False, True):
        path = tmp_path / ("compressed" if compressed else "plain")
        command = [sys.executable, "-m", "scripts.scheduling_sim.learning", "--preset", "smoke",
                   "--scenarios", "tail-64-cv0", "--policies", "targeted", "--seeds", "0:2",
                   "--warmup-ms", "0", "--measurement-ms", "100", "--skip-calibration",
                   "--bootstrap-replicates", "100", "--output", str(path)]
        if compressed:
            command.append("--compress-requests")
        subprocess.run(command, check=True, capture_output=True, text=True)
        assert (path / "requests.csv").exists() == (not compressed)
        assert (path / "requests.csv.gz").exists() is compressed
        if compressed:
            with gzip.open(path / "requests.csv.gz", "rb") as handle:
                requests = handle.read()
        else:
            requests = (path / "requests.csv").read_bytes()
        outputs.append(((path / "summary.csv").read_bytes(), requests,
                        (path / "calibration.csv").read_bytes()))
    assert outputs[0] == outputs[1]


def test_compressed_requests_require_an_output_directory(capsys):
    from scripts.scheduling_sim.learning import main

    with pytest.raises(SystemExit) as error:
        main(["--compress-requests"])
    assert error.value.code == 2
    assert "--compress-requests requires --output" in capsys.readouterr().err


def test_parallel_cli_preserves_deterministic_measurements(tmp_path):
    outputs = []
    for workers in (1, 2):
        path = tmp_path / f"workers-{workers}"
        subprocess.run([sys.executable, "-m", "scripts.scheduling_sim.learning", "--preset", "smoke",
                        "--scenarios", "tail-64-cv0", "--policies", "epsilon", "--seeds", "0:2",
                        "--warmup-ms", "0", "--measurement-ms", "100", "--skip-calibration",
                        "--bootstrap-replicates", "100", "--workers", str(workers), "--output", str(path)],
                       check=True, capture_output=True, text=True)
        outputs.append((path / "summary.csv").read_text())
    assert outputs[0] == outputs[1]


def test_training_nomination_excludes_pareto_dominated_configurations_before_ranking():
    from scripts.scheduling_sim.learning import Configuration, nominate
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    configs = tuple(Configuration(name, PolicyConfig(policy="epsilon", epsilon=value))
                    for name, value in (("balanced", .01), ("tail", .05), ("dominated", .1)))
    rows = []
    for name, rate, tail, recovered in (("known", 100., 10., True), ("balanced", 100., 10., True),
                                         ("tail", 99., 9., True), ("dominated", 90., 11., False)):
        rows.extend([
            {"scenario": "prefill-8192-cv0", "group": "core", "config_id": name, "seed": 0,
             "busy_useful_work_per_ms": rate, "p99_ttft_ms": tail, "p99_decode_gap_ms": tail},
            {"scenario": "recovery-2-slow-fast", "group": "recovery", "config_id": name,
             "seed": 0, "recovery_ms": 20_000 if recovered else None}])
    nominees = nominate(rows, configs)
    assert {config.id for config in nominees} == {"balanced", "tail"}


def test_nomination_uses_the_selected_reference_denominator():
    from scripts.scheduling_sim.learning import Configuration, nominate
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    configs = tuple(Configuration(name, PolicyConfig(policy="epsilon", epsilon=value))
                    for name, value in (("prefill-favoring", .01), ("balanced", .05)))
    rows = []
    for scenario, rates in (("prefill-8192-cv0", (100., 100., 100., 90.)),
                            ("decode-cv0", (100., 50., 80., 85.))):
        for name, rate in zip(("known", "ttl", "prefill-favoring", "balanced"), rates):
            rows.append({"scenario": scenario, "group": "core", "config_id": name, "seed": 0,
                         "busy_useful_work_per_ms": rate, "p99_ttft_ms": 10., "p99_decode_gap_ms": 10.})
    assert nominate(rows, configs, limit=1)[0].id == "balanced"
    assert nominate(rows, configs, limit=1, reference_id="known")[0].id == "balanced"
    assert nominate(rows, configs, limit=1, reference_id="ttl")[0].id == "prefill-favoring"


def test_retained_refresh_cli_exposes_only_supported_funding():
    from scripts.scheduling_sim.learning import build_configurations, parser
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    for funding in ("funded", "both"):
        result = build_configurations(parser().parse_args([
            "--policies", "retained-refresh", "--funding", funding, "--alpha", ".002",
            "--credit-cap-ms", "5", "--sweep", "alpha=.005"]), PolicyConfig)
        retained = [item for item in result if item.config.policy == "retained-refresh"]
        assert {item.config.alpha for item in retained} == {.002, .005}
        assert len(retained) == 2 and all(item.config.funded for item in retained)
        assert all(item.config.estimator == "cumulative" and item.config.credit_cap_ms == 5.
                   and item.config.harm_alpha is None and item.config.probe_harm_ms is None for item in retained)
    with pytest.raises(ValueError, match="requires funded"):
        build_configurations(parser().parse_args([
            "--policies", "retained-refresh", "--funding", "unfunded"]), PolicyConfig)
    all_policies = build_configurations(parser().parse_args(["--funding", "both"]), PolicyConfig)
    assert [item.config.funded for item in all_policies if item.config.policy == "retained-refresh"] == [True]
    assert {item.config.funded for item in all_policies if item.config.policy == "thompson"} == {False, True}


@pytest.mark.parametrize("options", [("--estimator", "regime"), ("--estimator", "log-regime"),
                                     ("--funding", "unfunded")])
def test_all_policies_omit_incompatible_retained_refresh_and_explicit_selection_rejects(options):
    from scripts.scheduling_sim.learning import build_configurations, parser
    from scripts.scheduling_sim.learning_policy import PolicyConfig

    configurations = build_configurations(parser().parse_args(options), PolicyConfig)
    assert "retained-refresh" not in {item.config.policy for item in configurations}
    thompson = next(item.config for item in configurations if item.config.policy == "thompson")
    if options[0] == "--estimator":
        assert thompson.estimator == options[1]
    else:
        assert thompson.funded is False
    with pytest.raises(ValueError, match="retained-refresh requires"):
        build_configurations(parser().parse_args(["--policies", "retained-refresh", *options]), PolicyConfig)


def test_accepted_winner_uses_geomean_then_worst_decode_tail_then_probe_expense():
    from scripts.scheduling_sim.learning import rank_accepted_configurations

    gates, rows = [], []
    for name, rates, gap, expense in (("throughput", (1.04, 1.02), 1.03, 100.),
                                       ("lower-tail", (1., 1.), .99, 100.),
                                       ("lower-cost", (1., 1.), .99, 10.),
                                       ("higher-tail", (1., 1.), 1., 0.)):
        for index, rate in enumerate(rates):
            gates.append({"config_id": name, "group": "core", "scenario": str(index),
                          "metrics": {"busy_useful_throughput_ratio": {"ratio": rate},
                                      "mean_seed_p99_decode_gap_ratio": {"ratio": gap}}})
            rows.append({"config_id": name, "group": "core", "probe_ms": expense})
    ranked = rank_accepted_configurations(rows, gates,
                                         ["higher-tail", "lower-tail", "lower-cost", "throughput"])
    assert [row["config_id"] for row in ranked] == ["throughput", "lower-cost", "lower-tail", "higher-tail"]
    assert rank_accepted_configurations(rows, gates, []) == []
