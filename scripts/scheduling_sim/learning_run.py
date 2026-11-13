# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution costs hidden from graph selection and observed only on completion."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import exp, isfinite, log1p, sqrt
from random import Random

from .learning_policy import EXPLORATORY, LearningScheduler, PolicyConfig
from .model import Graph, Request, Value
from .simulator import Result, State
from .timing import GraphSpec, TimingObservation
from .utility import UtilityConfig


@dataclass(frozen=True)
class ExecutionConfig:
    cv: float = 0.0
    rho: float = 0.0
    outlier_probability: float = 0.0
    outlier_multiplier: float = 10.0
    poison_first: bool = False
    poison_graphs: tuple[str, ...] = ()
    drift_at_busy_ms: float | None = None
    drift_multipliers: tuple[tuple[str, float], ...] = ()
    drift_end_busy_ms: float | None = None
    missing_probability: float = 0.0
    zero_probability: float = 0.0
    context_coefficient: float = 0.0
    occupancy_coefficient: float = 0.0

    def __post_init__(self):
        if not isfinite(self.cv) or self.cv < 0 or not isfinite(self.rho) or not 0 <= self.rho < 1:
            raise ValueError("timing variation must be nonnegative and correlation in [0, 1)")
        probabilities = (self.outlier_probability, self.missing_probability, self.zero_probability)
        if any(not isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError("timing event probabilities must be in [0, 1]")
        if self.missing_probability + self.zero_probability > 1:
            raise ValueError("missing and zero timing probabilities cannot exceed one")
        if not isfinite(self.outlier_multiplier) or self.outlier_multiplier < 1:
            raise ValueError("outlier multiplier must be finite and at least one")
        if any(not isfinite(x) or x < 0 for x in (self.context_coefficient, self.occupancy_coefficient)):
            raise ValueError("context and occupancy coefficients must be nonnegative")
        if any(not isfinite(m) or m <= 0 for _, m in self.drift_multipliers):
            raise ValueError("drift multipliers must be positive and finite")
        for boundary in (self.drift_at_busy_ms, self.drift_end_busy_ms):
            if boundary is not None and (not isfinite(boundary) or boundary < 0):
                raise ValueError("drift boundaries must be finite and nonnegative")
        if self.drift_end_busy_ms is not None and (
                self.drift_at_busy_ms is None or self.drift_end_busy_ms <= self.drift_at_busy_ms):
            raise ValueError("temporary drift must end after its start")


class ExecutionModel:
    """Private physical means and independent per-graph stochastic processes."""

    def __init__(self, catalog: tuple[Graph, ...], config: ExecutionConfig = ExecutionConfig(), seed: int = 0):
        if not catalog or len({g.name for g in catalog}) != len(catalog):
            raise ValueError("execution catalog requires unique physical graph names")
        self.catalog = {g.name: g for g in catalog}
        self.config = config
        if set(dict(config.drift_multipliers)) - self.catalog.keys() or set(config.poison_graphs) - self.catalog.keys():
            raise ValueError("execution changes must name a prepared graph")
        self.seed = seed
        self.counts = {g.name: 0 for g in catalog}
        self.previous_noise: dict[str, float] = {}

    def nominal_means(self, views, busy_ms: float, *, context_tokens: int = 0) -> dict[str, float]:
        config = self.config
        drifting = (config.drift_at_busy_ms is not None and busy_ms >= config.drift_at_busy_ms
                    and (config.drift_end_busy_ms is None or busy_ms < config.drift_end_busy_ms))
        factors = dict(config.drift_multipliers) if drifting else {}
        decoders = sum(not v.remaining_prefill for v in views)
        prefills = sum(bool(v.remaining_prefill) for v in views)
        means = {}
        for name, graph in self.catalog.items():
            occupied = min(decoders, graph.decode_capacity) + int(bool(prefills and graph.prefill_capacity))
            capacity = graph.decode_capacity + int(bool(graph.prefill_capacity))
            scale = (1 + config.context_coefficient * context_tokens / 8192
                     + config.occupancy_coefficient * occupied / capacity)
            means[name] = graph.duration_ms * factors.get(name, 1.0) * scale
        return means

    def conditional_means(self, views, busy_ms: float, *, context_tokens: int = 0) -> dict[str, float]:
        means = self.nominal_means(views, busy_ms, context_tokens=context_tokens)
        config = self.config
        sigma = sqrt(log1p(config.cv ** 2))
        for name in means:
            if name in self.previous_noise:
                means[name] *= exp(sigma * config.rho * self.previous_noise[name]
                                   - .5 * sigma * sigma * config.rho * config.rho)
            if config.poison_first and not self.counts[name] and (
                    not config.poison_graphs or name in config.poison_graphs):
                means[name] *= 10
        return means

    def execute(self, graph_id: str, start_ms: float, mean_ms: float) -> TimingObservation:
        config = self.config
        count = self.counts[graph_id]
        self.counts[graph_id] += 1
        noise = Random(f"{self.seed}:timing:{graph_id}:{count}")
        z = noise.gauss(0, 1)
        if graph_id in self.previous_noise:
            z = config.rho * self.previous_noise[graph_id] + sqrt(1 - config.rho ** 2) * z
        self.previous_noise[graph_id] = z
        sigma = sqrt(log1p(config.cv ** 2))
        duration = mean_ms * exp(sigma * z - sigma * sigma / 2)
        multiplier = config.outlier_multiplier if noise.random() < config.outlier_probability else 1.0
        duration *= multiplier / (1 + config.outlier_probability * (config.outlier_multiplier - 1))
        if config.poison_first and count == 0 and (not config.poison_graphs or graph_id in config.poison_graphs):
            duration *= 10
        validity = Random(f"{self.seed}:observation:{graph_id}:{count}").random()
        measured = (None if validity < config.missing_probability else
                    0.0 if validity < config.missing_probability + config.zero_probability else duration)
        return TimingObservation(graph_id, start_ms, start_ms + duration, measured, 0.0)


def percentile(values: list[float], fraction: float = .99) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class Recovery:
    """Sustained best-arm exploitation in occupied-time windows after a change."""

    def __init__(self, best: str | None, change_ms: float, window_ms: float = 10_000):
        self.best = best
        self.change_ms = change_ms
        self.window_ms = window_ms
        self.intervals: list[tuple[float, float, bool]] = []
        self.recovered_at: float | None = None

    def observe(self, start: float, end: float, graph: str, category: str):
        if self.best is None or end <= self.change_ms:
            return
        if category == "ordinary":
            self.intervals.append((max(start, self.change_ms), end, graph == self.best))
        if end < self.change_ms + self.window_ms:
            return
        cutoff = end - self.window_ms
        retained, totals, goods = [], [], []
        for row in self.intervals:
            left, right, best = row
            if right > cutoff:
                retained.append(row)
                duration = right - max(left, cutoff)
                totals.append(duration)
                if best:
                    goods.append(duration)
        self.intervals = retained
        total = sum(totals)
        good = sum(goods)
        if total and good / total >= .95:
            if self.recovered_at is None:
                self.recovered_at = end - self.change_ms
        else:
            self.recovered_at = None


def run_learning(requests: list[Request], catalog: tuple[Graph, ...], policy_config: PolicyConfig,
                 *, seed: int = 0, value: Value = Value(64),
                 utility_config: UtilityConfig = UtilityConfig(16, 4, 100, .1, "service"),
                 execution_config: ExecutionConfig = ExecutionConfig(), warmup_ms: float = 30_000,
                 measurement_ms: float = 120_000, horizon_ms: float | None = None,
                 max_steps: int = 1_000_000, replenish: bool = False,
                 idle_gaps_ms: tuple[float, ...] = (), initial_observations: int = 0,
                 trace_sink: Callable[[dict], None] | None = None, retain_trace: bool = False,
                 recovery_best: str | None = None, recovery_at_busy_ms: float = 0,
                 drain_ms: float = 30_000) -> Result:
    if not requests or len({r.name for r in requests}) != len(requests):
        raise ValueError("learning experiments require uniquely named request templates")
    if any(not isfinite(x) or x < 0 for x in (warmup_ms, drain_ms, *idle_gaps_ms)) or measurement_ms <= 0:
        raise ValueError("simulation windows must be finite and nonnegative")
    if not isfinite(measurement_ms) or max_steps < 1 or initial_observations < 0:
        raise ValueError("simulation limits must be positive and finite")
    if horizon_ms is not None and (not isfinite(horizon_ms) or horizon_ms <= 0):
        raise ValueError("wall horizon must be positive and finite")
    if replenish and idle_gaps_ms:
        raise ValueError("continuous replenishment and drained idle cohorts are separate workloads")
    physical = ExecutionModel(catalog, execution_config, seed)
    policy = LearningScheduler(value, utility_config, tuple(GraphSpec.from_graph(g) for g in catalog),
                               policy_config, seed=seed)
    historical = ExecutionModel(catalog, replace(execution_config, poison_first=False,
                                                 drift_at_busy_ms=None, drift_end_busy_ms=None), seed=seed + 1)
    for graph in catalog:
        for _ in range(initial_observations):
            observation = historical.execute(graph.name, 0.0, graph.duration_ms)
            policy.seed_observation(replace(observation, start_ms=-observation.elapsed_ms, end_ms=0.0))
    templates = {r.name: r for r in requests}
    generations = {r.name: 0 for r in requests}
    states: dict[str, State] = {}
    origins: dict[str, str] = {}

    def spawn(template: Request, arrival: float):
        index = generations[template.name]
        generations[template.name] += 1
        name = f"{template.name}@{index}"
        request = replace(template, name=name, arrival_ms=arrival)
        states[name] = State(request, request.prompt_tokens, last_service_ms=arrival)
        origins[name] = template.name

    for request in requests:
        spawn(request, request.arrival_ms)
    now = busy = 0.0
    measurement_start = 0.0 if warmup_ms == 0 else None
    measurement_start_busy = 0.0
    measurement_end: float | None = None
    measurement_end_busy: float | None = None
    measurement_snapshot: dict | None = None
    measured_work = measured_busy = 0.0
    launches = graph_calls = idle_index = 0
    events = []
    completed: list[State] = []
    recovery = Recovery(recovery_best, recovery_at_busy_ms)
    draining = False
    stop_reason = "completed"
    phase_costs = {kind: 0.0 for kind in ("ordinary", "probe", "bootstrap")}
    phase_counts = {kind: 0 for kind in phase_costs}
    invalid_observations = 0
    last_probe_ms = 0.0
    first_output_calls: dict[str, int] = {}
    request_calls: dict[str, int] = {}

    while True:
        if launches >= max_steps:
            raise RuntimeError("learning simulation exceeded the graph launch limit")
        if horizon_ms is not None and now >= horizon_ms and not draining:
            measurement_end, measurement_end_busy = now, busy
            measurement_snapshot = policy.snapshot()
            draining = True
            stop_reason = "wall_horizon"
        if draining and now >= measurement_end + drain_ms:
            stop_reason += "_drain_limit"
            break
        if not states:
            if not draining and idle_index < len(idle_gaps_ms):
                arrival = now + idle_gaps_ms[idle_index]
                idle_index += 1
                for request in requests:
                    spawn(request, arrival + request.arrival_ms)
            else:
                break
        active = [s for s in states.values() if s.request.arrival_ms <= now]
        if not active:
            arrival = min(s.request.arrival_ms for s in states.values())
            if draining:
                break
            policy.idle()
            if horizon_ms is not None and arrival >= horizon_ms:
                now = horizon_ms
                continue
            now = arrival
            continue
        views = [s.view() for s in active]
        context = max(s.request.prompt_tokens - s.remaining_prefill + s.generated for s in active)
        means = physical.conditional_means(views, busy, context_tokens=context)
        nominal_means = physical.nominal_means(views, busy, context_tokens=context)
        if policy_config.policy == "known":
            policy.set_reference_means(means)
        decision = policy.choose(views, now)
        action = decision.action
        policy.commit(decision, launches)
        observation = physical.execute(action.graph.name, now, nominal_means[action.graph.name])
        duration = observation.elapsed_ms
        end = observation.end_ms
        committed = []
        finished = []
        for lane in action.lanes:
            state = states[lane.name]
            request = state.request
            request_calls[lane.name] = request_calls.get(lane.name, 0) + 1
            if lane.prefill:
                if lane.prefill > state.remaining_prefill:
                    raise AssertionError("selected prefill exceeds observable remaining work")
                state.remaining_prefill -= lane.prefill
                emitted = int(state.remaining_prefill == 0)
            else:
                emitted = 1
                remaining = min(request.output_tokens, request.max_new_tokens) - state.generated
                for position, probability in enumerate(request.acceptance[:lane.width - 1]):
                    rng = Random(f"{seed}:acceptance:{request.name}:{state.generated}:{position}")
                    if emitted >= remaining or rng.random() >= probability:
                        break
                    emitted += 1
            emitted = min(emitted, min(request.output_tokens, request.max_new_tokens) - state.generated)
            state.generated += emitted
            state.last_service_ms = end
            if emitted:
                if not state.deliveries:
                    first_output_calls[state.request.name] = request_calls[lane.name]
                state.deliveries.append((end, emitted))
            if state.generated == min(request.output_tokens, request.max_new_tokens):
                state.completion_ms = end
                finished.append(state)
            committed.append({"request": lane.name, "prefill": lane.prefill, "width": lane.width,
                              "expected_output": lane.expected_output, "emitted": emitted,
                              "weight": lane.weight})
        measured = measurement_start is not None and not draining
        useful_work = sum(row["weight"] * (row["prefill"] + value.decode_value * row["emitted"])
                          for row in committed)
        if measured:
            measured_work += useful_work
            measured_busy += duration
        if not draining:
            phase_costs[decision.category] += duration
            phase_counts[decision.category] += 1
            recovery.observe(busy, busy + duration, action.graph.name, decision.category)
        invalid_observations += int(not observation.valid)
        policy.complete(launches, observation, committed)
        if decision.category == "probe":
            last_probe_ms = duration
        if policy_config.funded and policy_config.policy in EXPLORATORY:
            allowance = policy_config.alpha / (1 - policy_config.alpha) * policy.ordinary_ms + last_probe_ms
            if policy.probe_ms > allowance + 1e-9 * max(1, busy, allowance):
                raise AssertionError("actual probe cost exceeds earned allowance plus one execution overrun")
        if trace_sink is not None or retain_trace:
            event = {"start_ms": now, "end_ms": end, "busy_start_ms": busy, "busy_end_ms": busy + duration,
                     "graph": action.graph.name, "duration_ms": duration,
                     "estimated_ms": decision.estimate_ms, "true_mean_ms": means[action.graph.name],
                     "category": decision.category, "reason": decision.reason,
                     "selection_score": decision.score, "measurement": measured, "drain": draining,
                     "valid_timing": observation.valid, "useful_work": useful_work, "lanes": committed,
                     "decision": decision.metadata}
            if trace_sink is not None:
                trace_sink(event)
            if retain_trace:
                events.append(event)
        now = end
        busy += duration
        launches += 1
        graph_calls += int(not draining)
        if measurement_start is None and busy >= warmup_ms:
            measurement_start, measurement_start_busy = now, busy
        elif measured and busy - measurement_start_busy >= measurement_ms:
            measurement_end, measurement_end_busy = now, busy
            measurement_snapshot = policy.snapshot()
            draining = True
            stop_reason = "occupied_window"
        for state in finished:
            completed.append(state)
            del states[state.request.name]
            if replenish and not draining:
                spawn(templates[origins[state.request.name]], now)

    if measurement_end is None:
        measurement_end, measurement_end_busy = now, busy
    start = measurement_start if measurement_start is not None else measurement_end
    rows, ttfts, gaps = [], [], []
    censored_ttft = censored_gap = unfinished = starved = 0
    for state in [*completed, *states.values()]:
        request = state.request
        if request.arrival_ms >= measurement_end:
            continue
        times = [t for t, _ in state.deliveries]
        cohort = start <= request.arrival_ms < measurement_end
        ttft = times[0] - request.arrival_ms if times else None
        if cohort:
            if ttft is None:
                censored_ttft += 1
            else:
                ttfts.append(ttft)
        relevant = state.completion_ms is None or state.completion_ms > start
        if relevant:
            unfinished += int(state.completion_ms is None)
            starved += int(state.completion_ms is None and not any(t > start for t in times))
        pairs = list(zip(times, times[1:]))
        if not request.prompt_tokens and times:
            pairs.insert(0, (request.arrival_ms, times[0]))
        relevant_gaps = [right - left for left, right in pairs
                         if right > start and left < measurement_end]
        gaps.extend(relevant_gaps)
        if state.completion_ms is None:
            left = times[-1] if times else request.arrival_ms
            if (times or not request.prompt_tokens) and left < measurement_end and now > start:
                censored_gap += 1
        rows.append({"name": request.name, "template": origins[request.name],
                     "arrival_ms": request.arrival_ms, "measurement_cohort": cohort,
                     "prefill_tokens": request.prompt_tokens - state.remaining_prefill,
                     "output_tokens": state.generated, "ttft_ms": ttft,
                     "completion_ms": state.completion_ms, "decode_gaps_ms": relevant_gaps,
                     "max_output_gap_ms": max(relevant_gaps, default=0.0),
                     "first_output_graph_calls": first_output_calls.get(request.name),
                     "unfinished": state.completion_ms is None})
    snapshot = measurement_snapshot if measurement_snapshot is not None else policy.snapshot()
    elapsed = measurement_end - start
    summary = {"elapsed_ms": measurement_end, "busy_ms": measurement_end_busy,
               "measurement_start_ms": start, "measurement_end_ms": measurement_end,
               "measurement_busy_ms": measured_busy, "measurement_elapsed_ms": elapsed,
               "measurement_complete": measured_busy >= measurement_ms,
               "useful_work": measured_work,
               "busy_useful_work_per_ms": measured_work / measured_busy if measured_busy else 0.0,
               "elapsed_useful_work_per_ms": measured_work / elapsed if elapsed else 0.0,
               "p99_ttft_ms": percentile(ttfts), "p99_decode_gap_ms": percentile(gaps),
               "ttft_samples": len(ttfts), "decode_gap_samples": len(gaps),
               "censored_ttft": censored_ttft, "censored_decode_gaps": censored_gap,
               "unfinished": unfinished, "starved": starved, "launches": graph_calls,
               "ordinary_ms": phase_costs["ordinary"], "probe_ms": phase_costs["probe"],
               "bootstrap_ms": phase_costs["bootstrap"], "probe_count": phase_counts["probe"],
               "bootstrap_count": phase_counts["bootstrap"], "invalid_observations": invalid_observations,
               "initial_observations_per_graph": initial_observations,
               "recovery_ms": recovery.recovered_at, "stop_reason": stop_reason,
               "regime_resets": snapshot.get("regime_resets", 0),
               "final_credit_ms": snapshot.get("credit_ms", 0), "policy_state": snapshot}
    return Result(summary, rows, events)
