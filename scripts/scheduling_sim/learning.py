# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run synthetic timing-learning experiments without changing the known-cost CLI."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass, fields, replace
import gzip
import json
from math import exp, isfinite, log
from pathlib import Path
from random import Random
from statistics import fmean

from .learning_analysis import (HELDOUT_SEEDS, TRAINING_SEEDS, calibration_gate,
                                compare_workload, evaluation_seeds, recovery_gate,
                                stationary_reset_gate)
from .learning_scenarios import (SCREEN_SCENARIOS, SMOKE_SCENARIOS, scenarios,
                                 selected_scenarios)


POLICIES = ("known", "ttl", "jitter", "ema", "greedy", "epsilon", "softmax",
            "optimism", "thompson", "predictive", "targeted", "retained-refresh")
BASELINES = frozenset(("known", "ttl", "jitter", "ema", "greedy"))
EXPLORATORY = frozenset(POLICIES) - BASELINES
METADATA = ("scenario", "group", "config_id", "policy", "seed")


@dataclass(frozen=True)
class Configuration:
    id: str
    config: object


def seed_range(text):
    try:
        start, end = (int(value) for value in text.split(":"))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("seeds use a half-open START:END range") from error
    if not 0 <= start < end:
        raise argparse.ArgumentTypeError("seed range must be nonnegative and nonempty")
    return tuple(range(start, end))


def configuration_id(config):
    default = type(config)()
    suffix = ["funded" if config.funded else "unfunded", config.estimator]
    for item in fields(config):
        if item.name not in ("policy", "funded", "estimator"):
            value = getattr(config, item.name)
            if value != getattr(default, item.name):
                suffix.append(f"{item.name}={value}")
    return config.policy + ":" + ":".join(suffix)


def load_configurations(path, policy_type, *, comparison=False):
    raw = json.loads(path.read_text())
    if comparison and not isinstance(raw, list):
        raise ValueError("comparison configurations require a JSON array")
    if not comparison and (not isinstance(raw, list) or not 1 <= len(raw) <= 3):
        raise ValueError("validation requires a JSON array of one through three nominated configurations")
    result = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"id", "config"}:
            raise ValueError("each nominated configuration requires exactly id and config")
        identifier = item["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("configuration id must be a nonempty string")
        if not comparison and identifier in BASELINES:
            raise ValueError("nominated configuration ids must be distinct from baseline names")
        config = policy_type(**item["config"])
        if not comparison and config.policy in BASELINES:
            raise ValueError("nominate learning policies; reference baselines are included automatically")
        result.append(Configuration(identifier, config))
    if len({item.id for item in result}) != len(result):
        raise ValueError("nominated configuration ids must be unique")
    return tuple(result)


def add_comparison_configurations(configurations, comparisons):
    result = {configuration.id: configuration for configuration in configurations}
    for configuration in comparisons:
        existing = result.get(configuration.id)
        if existing is not None and (existing != configuration or existing.config.policy not in BASELINES):
            raise ValueError("comparison configuration ids must be distinct from nominated configurations")
        result[configuration.id] = configuration
    return tuple(result.values())


def build_configurations(args, policy_type):
    if args.preset == "validate":
        if args.configs is None:
            raise ValueError("validation requires --configs with configurations frozen before heldout evaluation")
        nominated = load_configurations(args.configs, policy_type)
        return (Configuration("known", policy_type(policy="known", funded=False)),
                Configuration("ttl", policy_type(policy="ttl", funded=False)), *nominated)
    if args.configs is not None:
        raise ValueError("--configs is a heldout validation input; screen uses explicit policy parameters")
    names = POLICIES if args.policies == "all" else tuple(args.policies.split(","))
    if any(name not in POLICIES for name in names):
        raise ValueError("unknown learning policy")
    parameters = {name: getattr(args, name) for name in (
        "epsilon", "temperature", "quantile", "alpha", "credit_cap_ms", "refresh_every",
        "block_size", "estimator", "anneal", "probe_loss_ms", "probe_extra_ms",
        "refresh_busy_ms", "audit_every", "probe_harm_ms", "harm_alpha", "initial_discovery",
        "mean_estimator", "partial_blocks") if getattr(args, name) is not None}
    if args.screen_grid:
        if args.sweep or parameters or args.policies != "all" or args.funding != "funded":
            raise ValueError("--screen-grid defines its policy-specific parameters and both funding variants")
        configurations = [Configuration(name, policy_type(policy=name, funded=False))
                          for name in ("known", "ttl", "jitter", "ema")]
        for estimator in ("cumulative", "regime"):
            greedy = policy_type(policy="greedy", funded=False, estimator=estimator)
            configurations.append(Configuration(configuration_id(greedy), greedy))
            for name in ("epsilon", "softmax", "optimism", "thompson", "predictive"):
                variants = ({"epsilon": value} for value in (.01, .05, .1)) if name == "epsilon" else (
                    ({"temperature": temperature, "anneal": anneal}
                     for temperature in (.03, .1, .3) for anneal in (False, True)) if name == "softmax" else
                    ({"quantile": value} for value in (.01, .05, .1)) if name == "optimism" else ({},))
                for variant in (*variants, {"refresh_every": 0}):
                    for funded in (True, False):
                        config = policy_type(policy=name, funded=funded, estimator=estimator, **variant)
                        if not any(c.config == config for c in configurations):
                            configurations.append(Configuration(configuration_id(config), config))
        return tuple(configurations)
    variants = []
    for expression in args.sweep:
        name, separator, values = expression.partition("=")
        if not separator or name not in parameters.keys() | {
                "epsilon", "temperature", "quantile", "alpha", "credit_cap_ms", "refresh_every", "block_size",
                "probe_loss_ms", "probe_extra_ms", "refresh_busy_ms", "audit_every", "probe_harm_ms",
                "harm_alpha"}:
            raise ValueError("sweep uses a policy parameter followed by =VALUE,VALUE")
        if name in ("estimator", "anneal", "initial_discovery", "mean_estimator", "partial_blocks"):
            raise ValueError("categorical and boolean parameters use explicit flags; sweeps are numeric")
        converter = int if name in ("refresh_every", "block_size", "audit_every") else float
        for value in values.split(","):
            variants.append({name: converter(value)})
    configurations = [Configuration("known", policy_type(policy="known", funded=False)),
                      Configuration("ttl", policy_type(policy="ttl", funded=False))]
    for name in names:
        if name in ("known", "ttl"):
            continue
        if name == "retained-refresh":
            incompatible = ("retained-refresh requires funded exploration" if args.funding == "unfunded" else
                            "retained-refresh requires the cumulative timing estimator"
                            if parameters.get("estimator", "cumulative") != "cumulative" else None)
            if incompatible:
                if args.policies == "all":
                    continue
                raise ValueError(incompatible)
        funding = (True,) if name == "retained-refresh" else (False,) if name in BASELINES else (
            (True, False) if args.funding == "both" else (args.funding == "funded",))
        for funded in funding:
            policy_parameters = {field: value for field, value in parameters.items()
                                 if field not in ("probe_harm_ms", "harm_alpha") or name == "targeted"}
            base = policy_type(policy=name, funded=funded, **policy_parameters)
            for changes in ({}, *variants):
                applicable = {"epsilon": {"epsilon"}, "temperature": {"softmax"},
                              "quantile": {"optimism", "targeted"}, "probe_loss_ms": {"targeted"},
                              "probe_extra_ms": {"targeted"}, "refresh_busy_ms": {"targeted"},
                              "audit_every": {"targeted"}, "probe_harm_ms": {"targeted"},
                              "harm_alpha": {"targeted"}}
                if changes and any(field in applicable and name not in applicable[field] for field in changes):
                    continue
                config = replace(base, **changes)
                identifier = (name if name in BASELINES and not changes and not policy_parameters
                              else configuration_id(config))
                if not any(existing.config == config for existing in configurations):
                    configurations.append(Configuration(identifier, config))
    return tuple(configurations)


def _run_task(task, trace_sink=None):
    from .learning_run import ExecutionConfig, run_learning

    scenario, configuration, seed, warmup, measurement, max_steps = task
    metadata = {"scenario": scenario.name, "group": scenario.group,
                "config_id": configuration.id, "policy": configuration.config.policy, "seed": seed}
    def trace(event):
        trace_sink(metadata | event)
    result = run_learning(list(scenario.requests), scenario.graphs, configuration.config,
        seed=seed, utility_config=scenario.utility_config, execution_config=ExecutionConfig(**scenario.execution),
        warmup_ms=warmup if scenario.warmup_ms is None else scenario.warmup_ms,
        measurement_ms=measurement if scenario.measurement_ms is None else scenario.measurement_ms,
        horizon_ms=scenario.horizon_ms, max_steps=max_steps, replenish=scenario.replenish,
        idle_gaps_ms=scenario.idle_gaps_ms, initial_observations=scenario.initial_observations,
        recovery_best=scenario.recovery_best, recovery_at_busy_ms=scenario.recovery_at_busy_ms,
        trace_sink=trace if trace_sink is not None else None, retain_trace=False)
    return metadata, result


def _execute_tasks(tasks, workers, trace_sink=None):
    if workers == 1:
        for task in tasks:
            yield _run_task(task, trace_sink)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        tasks = iter(tasks)
        for _ in range(2 * workers):
            task = next(tasks, None)
            if task is None:
                break
            pending.append(pool.submit(_run_task, task))
        while pending:
            result = pending.popleft().result()
            task = next(tasks, None)
            if task is not None:
                pending.append(pool.submit(_run_task, task))
            yield result


class CsvStream:
    def __init__(self, path):
        self.handle = (gzip.open(path, "wt", newline="", compresslevel=1) if path.suffix == ".gz"
                       else path.open("w", newline=""))
        self.writer = None

    def write(self, row):
        if self.writer is None:
            columns = [name for name in METADATA if name in row]
            columns.extend(sorted(row.keys() - set(columns)))
            self.writer = csv.DictWriter(self.handle, fieldnames=[*columns, "extra_json"])
            self.writer.writeheader()
        extra = {key: value for key, value in row.items() if key not in self.writer.fieldnames}
        formatted = {key: json.dumps(value, sort_keys=True, allow_nan=False)
                     if isinstance(value, (dict, list, tuple)) else value
                     for key, value in row.items() if key in self.writer.fieldnames}
        formatted["extra_json"] = json.dumps(extra, sort_keys=True, allow_nan=False) if extra else ""
        self.writer.writerow(formatted)
        self.handle.flush()

    def close(self):
        self.handle.close()


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def calibrate(config, seeds):
    """Check fixed-sample mean intervals under the Gaussian block likelihood."""
    from .timing import TimingEstimate

    rows = []
    for seed in seeds:
        rng = Random(f"mean-calibration:{seed}")
        estimate = TimingEstimate(prior_location_ms=20., block_size=config.block_size, mode=config.estimator,
                                  mean_estimator=config.mean_estimator, partial_blocks=config.partial_blocks)
        diagnostic_estimates = []
        partial_counts = set()
        mature_boundary = (1000 + config.block_size - 1) // config.block_size * config.block_size
        if config.partial_blocks:
            diagnostic_estimates = [(20., 20., estimate), *(
                (1., true_mean, TimingEstimate(prior_location_ms=1., block_size=config.block_size,
                     mode=config.estimator, mean_estimator=config.mean_estimator, partial_blocks=True))
                for true_mean in (20., 40.))]
            partial_counts = {count for remainder in range(1, config.block_size)
                              for count in (remainder, mature_boundary + remainder)}
        row = {"seed": seed, "estimator": config.estimator, "block_size": config.block_size,
               "true_mean_ms": 20., "raw_standard_deviation_ms": 2.}
        diagnostics = []
        for count in range(1, max(1000, max(partial_counts, default=0)) + 1):
            observation = rng.gauss(20., 2.)
            if observation <= 0:
                raise ValueError("Gaussian calibration produced a nonpositive duration")
            estimate.observe(observation)
            for prior, true_mean, diagnostic in diagnostic_estimates[1:]:
                diagnostic.observe(observation + true_mean - 20.)
            if count in (16, 1000):
                lower, upper = estimate.interval(.95)
                row[f"covered_{count}"] = lower <= 20. <= upper
                row[f"width_{count}"] = upper - lower
            if count == 1000:
                row["regime_resets"] = estimate.regime_changes
            if count in partial_counts:
                for prior, true_mean, diagnostic in diagnostic_estimates:
                    lower, upper = diagnostic.interval(.95)
                    diagnostics.append({"prior_location_ms": prior, "true_mean_ms": true_mean,
                                        "count": count, "partial_count": count % config.block_size,
                                        "history": "initial" if count < config.block_size else "mature",
                                        "covered": lower <= true_mean <= upper, "width_ms": upper - lower})
        if config.partial_blocks:
            row["partial_block_diagnostics"] = diagnostics
        rows.append(row)
    return rows


def nominate(rows, configurations, limit=3, *, reference_id="known"):
    """Pareto-filter tuning objectives before ranking at most three nominations."""
    if not 1 <= limit <= 3:
        raise ValueError("nomination limit must be one through three")
    by_key = {(row["scenario"], row["config_id"], row["seed"]): row for row in rows}
    definitions = scenarios()
    measured = []
    for configuration in configurations:
        if configuration.config.policy in BASELINES:
            continue
        ordinary = [r for r in rows if r["config_id"] == configuration.id and r["group"] == "core"]
        rates, tails = [], []
        for name in sorted({row["scenario"] for row in ordinary}):
            paired = [(row, by_key[(name, reference_id, row["seed"])]) for row in ordinary
                      if row["scenario"] == name and (name, reference_id, row["seed"]) in by_key]
            if not paired:
                continue
            denominator = fmean(baseline.get("busy_useful_work_per_ms", 0) for _, baseline in paired)
            if denominator > 0:
                rates.append(fmean(row["busy_useful_work_per_ms"] for row, _ in paired) / denominator)
            definition = definitions.get(name)
            for field, required in (("p99_ttft_ms", definition is None or definition.require_ttft),
                                    ("p99_decode_gap_ms", definition is None or definition.require_decode)):
                if not required:
                    continue
                if any(row.get(field) is None or not baseline.get(field) for row, baseline in paired):
                    tails.append(float("inf"))
                else:
                    tails.append(fmean(row[field] for row, _ in paired) /
                                 fmean(baseline[field] for _, baseline in paired))
        recovery = [r for r in rows if r["config_id"] == configuration.id and (
            r["group"] == "recovery" or definitions.get(r["scenario"]) is not None
            and definitions[r["scenario"]].recovery_best is not None)]
        success = fmean(r.get("recovery_ms") is not None and r["recovery_ms"] <= 60_000 for r in recovery) if recovery else 1.
        vector = (min(rates, default=0.), -max(tails, default=1.), success)
        if all(isfinite(value) for value in vector):
            measured.append((vector, configuration))
    frontier = [(vector, configuration) for vector, configuration in measured if not any(
        all(left >= right for left, right in zip(other, vector)) and other != vector
        for other, _ in measured)]
    ranked = []
    for (throughput, negative_tail, recovery), configuration in frontier:
        score = throughput - max(0., -negative_tail - 1.) - max(0., .95 - recovery)
        ranked.append((score, configuration.id, configuration))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(item[2] for item in ranked[:limit])


def rank_accepted_configurations(rows, gates, accepted):
    """Rank already accepted frozen configurations without fitting parameters."""
    result = []
    for identifier in accepted:
        core = [gate for gate in gates if gate["config_id"] == identifier and gate["group"] == "core"]
        rates = [gate["metrics"]["busy_useful_throughput_ratio"]["ratio"] for gate in core]
        tails = [gate["metrics"]["mean_seed_p99_decode_gap_ratio"]["ratio"] for gate in core
                 if "mean_seed_p99_decode_gap_ratio" in gate["metrics"]]
        expenses = [row["probe_ms"] for row in rows if row["config_id"] == identifier and row["group"] == "core"]
        if not rates or not expenses or any(not isfinite(value) or value <= 0 for value in rates):
            raise ValueError("accepted configuration lacks complete core selection measurements")
        result.append({"config_id": identifier,
                       "geomean_core_throughput_ratio": exp(fmean(log(rate) for rate in rates)),
                       "worst_core_mean_seed_p99_decode_gap_ratio": max(tails, default=1.),
                       "mean_core_probe_ms": fmean(expenses)})
    return sorted(result, key=lambda row: (-row["geomean_core_throughput_ratio"],
                                          row["worst_core_mean_seed_p99_decode_gap_ratio"],
                                          row["mean_core_probe_ms"], row["config_id"]))


def evaluate(rows, selected, configurations, *, replicates=2000, heldout=True,
             expected_seeds=None, comparison_ids=(), reference_id="known"):
    nominees = [c for c in configurations if c.config.policy not in BASELINES and c.id not in comparison_ids]
    result = []
    for scenario in selected:
        measured = [r for r in rows if r["scenario"] == scenario.name]
        for configuration in nominees:
            gate = {"scenario": scenario.name, "group": scenario.group, "config_id": configuration.id}
            candidate = [r for r in measured if r["config_id"] == configuration.id]
            if scenario.group == "core":
                gate.update(compare_workload(measured, configuration.id, configuration_count=len(nominees),
                                             replicates=replicates, heldout=heldout,
                                             require_ttft=scenario.require_ttft, require_decode=scenario.require_decode,
                                             reference_id=reference_id, expected_seeds=expected_seeds))
                gate["stationary_reset"] = stationary_reset_gate(candidate, heldout=heldout,
                                                                expected_seeds=expected_seeds)
                gate["passed"] &= gate["stationary_reset"]["passed"]
            elif scenario.group == "recovery":
                gate.update(recovery_gate(candidate, heldout=heldout,
                                           change_busy_ms=scenario.recovery_at_busy_ms,
                                           expected_seeds=expected_seeds))
            else:
                gate.update(passed=None, semantics="transient/stress measurements; no stationary performance gate")
                if scenario.recovery_best is not None:
                    gate["recovery"] = recovery_gate(candidate, heldout=heldout,
                                                      change_busy_ms=scenario.recovery_at_busy_ms,
                                                      expected_seeds=expected_seeds)
            result.append(gate)
    return result


def baseline_recovery(rows, selected, configurations, *, heldout=True,
                      expected_seeds=None, comparison_ids=()):
    references = [c for c in configurations if c.config.policy in BASELINES or c.id in comparison_ids]
    result = []
    for scenario in selected:
        if scenario.recovery_best is None:
            continue
        for configuration in references:
            measured = [row for row in rows if row["scenario"] == scenario.name
                        and row["config_id"] == configuration.id]
            result.append({"scenario": scenario.name, "group": scenario.group,
                           "config_id": configuration.id, **recovery_gate(
                               measured, heldout=heldout, expected_seeds=expected_seeds,
                               change_busy_ms=scenario.recovery_at_busy_ms)})
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preset", choices=("smoke", "screen", "validate"), default="smoke")
    result.add_argument("--scenarios", help="comma-separated scenario names or core/transient/stress/recovery/calibration/all")
    result.add_argument("--policies", default="all")
    result.add_argument("--configs", type=Path, help="JSON array of at most three configurations nominated on tuning seeds")
    result.add_argument("--comparison-configs", type=Path, help="JSON array of fixed comparison configurations, excluded from nomination and acceptance")
    result.add_argument("--reference", choices=("known", "ttl"), default="known",
                        help="primary reference for tuning selection and validation gates")
    result.add_argument("--seeds", type=seed_range)
    result.add_argument("--funding", choices=("funded", "unfunded", "both"), default="funded")
    result.add_argument("--epsilon", type=float)
    result.add_argument("--temperature", type=float)
    result.add_argument("--quantile", type=float)
    result.add_argument("--alpha", type=float)
    result.add_argument("--credit-cap-ms", dest="credit_cap_ms", type=float)
    result.add_argument("--refresh-every", dest="refresh_every", type=int)
    result.add_argument("--block-size", dest="block_size", type=int)
    result.add_argument("--probe-loss-ms", dest="probe_loss_ms", type=float)
    result.add_argument("--probe-extra-ms", dest="probe_extra_ms", type=float)
    result.add_argument("--refresh-busy-ms", dest="refresh_busy_ms", type=float)
    result.add_argument("--audit-every", dest="audit_every", type=int)
    result.add_argument("--probe-harm-ms", dest="probe_harm_ms", type=float,
                        help="targeted probe peer-delay budget in request-ms; omitted disables the bound")
    result.add_argument("--harm-alpha", dest="harm_alpha", type=float,
                        help="targeted peer-delay credit fraction; omitted disables peer-delay financing")
    result.add_argument("--initial-discovery", choices=("productive", "native"))
    result.add_argument("--mean-estimator", choices=("arithmetic", "ema"))
    result.add_argument("--partial-blocks", action="store_true", default=None,
                        help="condition timing uncertainty on fractional unfinished blocks")
    result.add_argument("--estimator", choices=("cumulative", "regime", "log-regime"))
    result.add_argument("--anneal", action="store_true", default=None)
    result.add_argument("--sweep", action="append", default=[], help="one-factor parameter=VALUE,VALUE; repeat for independent sweeps")
    result.add_argument("--screen-grid", action="store_true", help="policy-specific tuning grids, both estimators/funding variants, annealing and refresh-off controls")
    result.add_argument("--workers", type=int, default=1, help="independent CPU experiment processes; traces require one worker")
    result.add_argument("--warmup-ms", type=float)
    result.add_argument("--measurement-ms", type=float)
    result.add_argument("--max-steps", type=int, default=1_000_000)
    result.add_argument("--bootstrap-replicates", type=int, default=2000)
    result.add_argument("--output", type=Path)
    result.add_argument("--compress-requests", action="store_true", help="write lossless requests.csv.gz")
    result.add_argument("--trace", action="store_true", help="stream action events to trace.jsonl in --output")
    result.add_argument("--skip-calibration", action="store_true", help="diagnostic runs only; unavailable in validation")
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    from .learning_policy import PolicyConfig

    try:
        if args.trace and args.output is None:
            raise ValueError("--trace requires --output")
        if args.compress_requests and args.output is None:
            raise ValueError("--compress-requests requires --output")
        if args.workers < 1 or (args.trace and args.workers > 1):
            raise ValueError("workers must be positive; --trace requires --workers 1")
        if args.screen_grid and args.preset != "screen":
            raise ValueError("--screen-grid uses tuning seeds with --preset screen")
        if args.preset == "validate":
            evaluation_seeds(args.seeds)
            if args.warmup_ms not in (None, 30_000.) or args.measurement_ms not in (None, 120_000.):
                raise ValueError("validation requires 30000 occupied warmup ms and 120000 occupied measurement ms")
            if args.skip_calibration:
                raise ValueError("validation includes uncertainty calibration")
            if (args.policies != "all" or args.funding != "funded" or args.sweep
                    or any(getattr(args, name) is not None for name in (
                        "epsilon", "temperature", "quantile", "alpha", "credit_cap_ms",
                        "refresh_every", "block_size", "estimator", "anneal", "probe_loss_ms",
                        "probe_extra_ms", "refresh_busy_ms", "audit_every", "probe_harm_ms",
                        "harm_alpha", "initial_discovery", "mean_estimator", "partial_blocks"))):
                raise ValueError("validation policy parameters come exclusively from --configs")
        if args.bootstrap_replicates < 100 or args.max_steps < 1:
            raise ValueError("bootstrap needs at least 100 replicates and max-steps must be positive")
        seeds = args.seeds or (HELDOUT_SEEDS if args.preset == "validate" else
                               TRAINING_SEEDS if args.preset == "screen" else (0, 1))
        if args.preset != "validate" and not set(seeds) <= set(TRAINING_SEEDS):
            raise ValueError("screen and smoke must use tuning seeds 0:16")
        warmup = args.warmup_ms if args.warmup_ms is not None else (100. if args.preset == "smoke" else 30_000.)
        measurement = args.measurement_ms if args.measurement_ms is not None else (500. if args.preset == "smoke" else 120_000.)
        if not all(isfinite(n) and n >= 0 for n in (warmup, measurement)) or measurement == 0:
            raise ValueError("warmup must be nonnegative and measurement must be positive and finite")
        selection = args.scenarios or ("all" if args.preset == "validate" else
                                       SCREEN_SCENARIOS if args.preset == "screen" else SMOKE_SCENARIOS)
        selected = selected_scenarios(selection)
        configurations = build_configurations(args, PolicyConfig)
        comparisons = load_configurations(args.comparison_configs, PolicyConfig, comparison=True) if args.comparison_configs else ()
        comparison_ids = {configuration.id for configuration in comparisons}
        configurations = add_comparison_configurations(configurations, comparisons)
    except (TypeError, ValueError) as error:
        argument_parser.error(str(error))
    writers, trace_file = {}, None
    rows, calibrations = [], []
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        writers = {name: CsvStream(args.output / ("requests.csv.gz" if name == "requests"
                                                  and args.compress_requests else f"{name}.csv"))
                   for name in ("summary", "requests", "calibration")}
        _write_json(args.output / "configurations.json", [{"id": c.id, "config": asdict(c.config)} for c in configurations])
        if args.trace:
            trace_file = (args.output / "trace.jsonl").open("w")
    print(f"Synthetic timing experiment: {args.preset}; {len(selected)} scenarios, {len(configurations)} configurations, {len(seeds)} seeds.", flush=True)
    try:
        tasks = ((scenario, configuration, seed, warmup, measurement, args.max_steps)
                 for scenario in selected for configuration in configurations for seed in seeds)
        def trace(event):
            trace_file.write(json.dumps(event, allow_nan=False) + "\n")
        count = 0
        for metadata, result in _execute_tasks(tasks, args.workers, trace if trace_file is not None else None):
            row = metadata | result.summary
            rows.append({key: value for key, value in row.items() if key != "policy_state"})
            if writers:
                writers["summary"].write(row)
                for request in result.requests:
                    writers["requests"].write(metadata | request)
            count += 1
            if count % len(seeds) == 0:
                print(f"{metadata['scenario']} {metadata['config_id']}: {len(seeds)} seeds completed", flush=True)
        candidates = tuple(c for c in configurations if c.id not in comparison_ids)
        nominees = nominate(rows, candidates, reference_id=args.reference) if args.preset != "validate" else tuple(
            c for c in candidates if c.config.policy not in BASELINES)
        if args.output and args.preset == "screen":
            _write_json(args.output / "nominated.json", [{"id": c.id, "config": asdict(c.config)} for c in nominees])
        if not args.skip_calibration:
            for configuration in nominees:
                for measured in calibrate(configuration.config, seeds):
                    row = {"config_id": configuration.id, "policy": configuration.config.policy, **measured}
                    calibrations.append(row)
                    if writers:
                        writers["calibration"].write(row)
        evaluated = (configurations if args.preset == "validate" else
                     tuple(c for c in configurations if c.config.policy in BASELINES or c.id in comparison_ids) + nominees)
        options = {"replicates": args.bootstrap_replicates, "heldout": args.preset == "validate",
                   "expected_seeds": seeds if args.preset == "validate" else None,
                   "comparison_ids": comparison_ids}
        gates = evaluate(rows, selected, evaluated, reference_id=args.reference, **options)
        diagnostic_reference = "ttl" if args.reference == "known" else "known"
        reference_ids = tuple(dict.fromkeys((diagnostic_reference,
                                            *(c.id for c in comparisons if c.id != args.reference))))
        comparison_gates = [gate for reference in reference_ids
                            for gate in evaluate(rows, selected, evaluated, reference_id=reference, **options)
                            if gate["group"] == "core"]
        reference_recovery = baseline_recovery(rows, selected, configurations,
            heldout=args.preset == "validate", expected_seeds=options["expected_seeds"], comparison_ids=comparison_ids)
        calibration_results = [{"config_id": c.id, **calibration_gate(
            [r for r in calibrations if r["config_id"] == c.id], heldout=args.preset == "validate",
            expected_seeds=options["expected_seeds"])} for c in nominees]
        core_names = {name for name, scenario in scenarios().items() if scenario.group == "core"}
        complete = core_names <= {scenario.name for scenario in selected} and args.preset == "validate"
        complete_suite = set(scenarios()) <= {scenario.name for scenario in selected} and args.preset == "validate"
        accepted = [c.id for c in nominees if complete_suite
                    and all(g["passed"] for g in gates if g["config_id"] == c.id and g["passed"] is not None)
                    and all(g["passed"] for g in calibration_results if g["config_id"] == c.id)]
        accepted_ranking = rank_accepted_configurations(rows, gates, accepted)
        evaluation = {"complete_core_suite": complete, "complete_validation_suite": complete_suite,
                      "reference_id": args.reference,
                      "accepted_configurations": accepted,
                      "selected_configuration": accepted_ranking[0]["config_id"] if accepted_ranking else None,
                      "accepted_ranking": accepted_ranking,
                      "gates": gates, "calibration": calibration_results,
                      "comparison_gates": comparison_gates, "baseline_recovery": reference_recovery,
                      "comparison_configurations": sorted(comparison_ids),
                      "metrics": "throughput and p99 ratios compare means of per-seed metrics, with paired seed-cluster bounds",
                      "reset_gate": "0/64 stationary seeds empirically; no population false-switch certification"}
        if args.output:
            _write_json(args.output / "comparisons.json", evaluation)
        print(f"Accepted configurations: {', '.join(accepted) if accepted else 'none'}; complete heldout core suite: {complete}", flush=True)
        return 0
    finally:
        for writer in writers.values():
            writer.close()
        if trace_file is not None:
            trace_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
