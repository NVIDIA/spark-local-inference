# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observed-cost graph selection with explicit foreground exploration accounting."""

from dataclasses import dataclass, replace
from math import exp, isfinite, sqrt
from random import Random
from statistics import median

from .model import Action, Graph, Lane, Value, View, expected_output
from .scheduler import candidates
from .timing import GraphSpec, TimingEstimate, TimingObservation
from .utility import UtilityConfig, UtilityScheduler


POLICIES = ("known", "ttl", "jitter", "ema", "greedy", "epsilon", "softmax",
            "optimism", "thompson", "predictive", "targeted", "retained-refresh")
EXPLORATORY = frozenset(("epsilon", "softmax", "optimism", "thompson", "predictive", "targeted",
                         "retained-refresh"))
HARM_CAP_MS = 1.0


@dataclass(frozen=True)
class PolicyConfig:
    policy: str = "thompson"
    funded: bool = True
    epsilon: float = 0.05
    temperature: float = 0.1
    quantile: float = 0.05
    alpha: float = 0.02
    credit_cap_ms: float = 5.0
    refresh_every: int = 16
    block_size: int = 8
    estimator: str = "cumulative"
    anneal: bool = False
    ttl_ms: float = 30_000.0
    probe_loss_ms: float = 5.0
    probe_extra_ms: float = 0.0
    refresh_busy_ms: float = 5_000.0
    audit_every: int = 0
    probe_harm_ms: float | None = None
    harm_alpha: float | None = None
    initial_discovery: str = "productive"
    mean_estimator: str = "arithmetic"
    partial_blocks: bool = False

    def __post_init__(self):
        if self.policy not in POLICIES or self.estimator not in ("cumulative", "regime", "log-regime"):
            raise ValueError("unknown learning policy or estimator")
        if not isfinite(self.epsilon) or not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if not isfinite(self.quantile) or not 0 < self.quantile < 1:
            raise ValueError("optimistic quantile must be in (0, 1)")
        if not isfinite(self.alpha) or not 0 <= self.alpha < 1:
            raise ValueError("exploration fraction must be in [0, 1)")
        if any(not isfinite(x) or x <= 0 for x in (self.temperature, self.credit_cap_ms, self.ttl_ms)):
            raise ValueError("temperature, credit cap, and timing lifetime must be positive")
        if not isinstance(self.block_size, int) or self.block_size < 1:
            raise ValueError("timing blocks must contain a positive integer number of samples")
        if not isinstance(self.refresh_every, int) or self.refresh_every < 0:
            raise ValueError("refresh period must be a nonnegative integer")
        if any(not isfinite(x) or x < 0 for x in (
                self.probe_loss_ms, self.probe_extra_ms, self.refresh_busy_ms)):
            raise ValueError("probe limits and occupied refresh period must be finite and nonnegative")
        if type(self.audit_every) is not int or self.audit_every < 0 or self.audit_every == 1:
            raise ValueError("audit period must be zero or an integer of at least two")
        if self.probe_harm_ms is not None and (isinstance(self.probe_harm_ms, bool)
                or not isfinite(self.probe_harm_ms) or self.probe_harm_ms < 0):
            raise ValueError("peer service-delay limit must be finite and nonnegative")
        if self.harm_alpha is not None and (isinstance(self.harm_alpha, bool)
                or not isfinite(self.harm_alpha) or not 0 <= self.harm_alpha < 1):
            raise ValueError("peer service-delay fraction must be in [0, 1)")
        if self.harm_alpha is not None and self.probe_harm_ms is not None:
            raise ValueError("hard and funded peer service-delay controls are mutually exclusive")
        if self.initial_discovery not in ("productive", "native"):
            raise ValueError("initial discovery must be productive or native")
        if self.mean_estimator not in ("arithmetic", "ema"):
            raise ValueError("operational mean must be arithmetic or ema")
        if type(self.partial_blocks) is not bool:
            raise ValueError("partial timing blocks must be a boolean")
        if self.policy == "retained-refresh" and (not self.funded or self.estimator != "cumulative"):
            raise ValueError("retained-refresh requires funding and the cumulative timing estimator")
        if self.policy == "retained-refresh" and (self.probe_harm_ms is not None or self.harm_alpha is not None):
            raise ValueError("retained-refresh does not use peer service-delay controls")


@dataclass(frozen=True)
class Decision:
    action: Action
    category: str
    reason: str
    estimate_ms: float | None
    score: float
    metadata: dict


class LearningScheduler:
    def __init__(self, value: Value, utility_config: UtilityConfig, catalog: tuple[GraphSpec, ...],
                 config: PolicyConfig, seed: int = 0):
        if not catalog or len({g.name for g in catalog}) != len(catalog):
            raise ValueError("learning requires a nonempty catalog of unique physical graph names")
        self.value = value
        self.config = config
        self.catalog = tuple(sorted(catalog, key=lambda g: g.name))
        self.utility = UtilityScheduler(value, utility_config)
        self.estimates: dict[str, TimingEstimate] = {}
        self._ema: dict[str, float] = {}
        self._observed_at: dict[str, float] = {}
        self._expires: dict[str, float] = {}
        self._attempted: dict[str, float] = {}
        self._occupied_ms = 0.0
        self._observed_busy: dict[str, float] = {}
        self._attempted_busy: dict[str, float] = {}
        self._failed_confirmations: set[str] = set()
        self._contended_opportunities = 0
        self._references: dict[str, float] = {}
        self._rng = Random(f"{seed}:learning-policy")
        self._expiry_rng = Random(f"{seed}:timing-expiry")
        self._planned: Decision | None = None
        self._planned_reference: tuple[Action, int] | None = None
        self._plan_signature = None
        self._inflight: tuple[object, Decision] | None = None
        self._inflight_reference: tuple[Action, int] | None = None
        self._operation_ids: set[object] = set()
        self._now_ms = float("-inf")
        self.credit_ms = 0.0
        self.opportunities = 0
        self.probe_count = self.bootstrap_count = 0
        self.ordinary_ms = self.probe_ms = self.bootstrap_ms = 0.0
        self.ordinary_elapsed_ms = 0.0
        self.invalid_observations = 0
        self.harm_credit_ms = {"prefill": 0.0, "decode": 0.0}
        self.harm_ordinary_ms = {"prefill": 0.0, "decode": 0.0}
        self.harm_charged_ms = {"prefill": 0.0, "decode": 0.0}

    def set_reference_means(self, mapping: dict[str, float]):
        if self.config.policy != "known":
            raise ValueError("only the known-mean reference accepts hidden conditional means")
        if self._planned is not None or self._inflight is not None:
            raise ValueError("reference means cannot change during a decision or execution")
        if set(mapping) != {g.name for g in self.catalog} or any(
                not isfinite(x) or x <= 0 for x in mapping.values()):
            raise ValueError("reference costs must cover the catalog with positive finite means")
        self._references = dict(mapping)

    def _means(self, now_ms: float) -> dict[str, float | None]:
        if self.config.policy == "known":
            if not self._references:
                raise ValueError("known-mean policy requires conditional reference means")
            return dict(self._references)
        if self.config.policy in ("ttl", "jitter", "ema", "retained-refresh"):
            return {g.name: self._ema.get(g.name) if self.config.policy in ("ema", "retained-refresh")
                    or now_ms < self._expires.get(g.name, float("-inf")) else None
                    for g in self.catalog}
        return {g.name: self.estimates[g.name].mean_ms if g.name in self.estimates else None
                for g in self.catalog}

    def _graph(self, spec: GraphSpec, duration: float | None) -> Graph:
        # Unknown descriptors carry geometry only; their placeholder never enters a rate.
        return Graph(spec.name, duration if duration is not None else 1.0,
                     spec.prefill_capacity, spec.decode_capacity, spec.width)

    def _unknown_lane_value(self, lane: Lane) -> float:
        value = self.utility.lane_value(lane)
        if lane.expected_output > 0 and lane.name in self.utility.ages:
            age, reference, weight = self.utility.ages[lane.name]
            value += exp(-self.utility.log_scale) * self.utility.config.age_price * weight * reference * age
        return value

    def _actions(self, views: list[View], costs: dict[str, float | None],
                 *, filter_unknown_padding: bool = True) -> list[Action]:
        graphs = tuple(self._graph(g, costs[g.name]) for g in self.catalog)
        actions = candidates(views, graphs, self.value, lane_value=lambda lane, graph:
                             self._unknown_lane_value(lane) if costs[graph.name] is None
                             else self.utility.action_lane_value(lane, graph))
        if not filter_unknown_padding:
            return actions
        result = []
        for action in actions:
            graph = action.graph
            if costs[graph.name] is None and graph.prefill_capacity:
                prompt = next((lane for lane in action.lanes if lane.prefill), None)
                remaining = next((v.remaining_prefill for v in views if prompt and v.name == prompt.name), 0)
                if any(g.prefill_capacity and g.prefill_capacity < graph.prefill_capacity
                       and g.decode_capacity == graph.decode_capacity and g.width == graph.width
                       and (prompt is None or min(remaining, g.prefill_capacity) == prompt.prefill)
                       for g in self.catalog):
                    continue
            result.append(action)
        return result

    def _rank_key(self, action: Action, clocks: dict[str, float]):
        return (-self.utility.score(action), action.graph.duration_ms,
                min(clocks[l.name] for l in action.lanes),
                action.graph.prefill_capacity + action.graph.decode_capacity * action.graph.width,
                action.graph.name, tuple(l.name for l in action.lanes))

    def _bootstrap_key(self, action: Action, clocks: dict[str, float]):
        return (-sum(self._unknown_lane_value(lane) for lane in action.lanes),
                min(clocks[l.name] for l in action.lanes),
                action.graph.prefill_capacity + action.graph.decode_capacity * action.graph.width,
                action.graph.name, tuple(l.name for l in action.lanes))

    def _ensure_estimates(self, names: set[str], means: dict[str, float | None],
                          incumbent: Action | None):
        for graph in self.catalog:
            if graph.name not in names or graph.name in self.estimates:
                continue
            family = [means[g.name] for g in self.catalog if g.family == graph.family
                      and means[g.name] is not None]
            location = median(family) if family else (
                means[incumbent.graph.name] if incumbent is not None else 1.0)
            self.estimates[graph.name] = TimingEstimate(
                prior_location_ms=location, block_size=self.config.block_size, mode=self.config.estimator,
                mean_estimator=self.config.mean_estimator, partial_blocks=self.config.partial_blocks)

    def choose(self, views: list[View], now_ms: float) -> Decision:
        signature = (tuple(views), now_ms)
        if self._planned is not None:
            if signature != self._plan_signature:
                raise ValueError("an uncommitted decision must be committed before replanning")
            return self._planned
        if self._inflight is not None:
            raise ValueError("completion feedback is required before another decision")
        if not isfinite(now_ms) or now_ms < self._now_ms:
            raise ValueError("learning scheduler clock must be finite and monotonic")
        means = self._means(now_ms)
        native_discovery = (self.config.initial_discovery == "native"
                            or self.config.policy == "retained-refresh")
        graphs = tuple(self._graph(g, means[g.name]) for g in self.catalog)
        self.utility.prepare(views, graphs, now_ms,
                             known_graphs={name for name, mean in means.items() if mean is not None},
                             mask_unknown_prices=(self.config.policy in ("ttl", "jitter")
                                                  or native_discovery))
        actions = self._actions(views, means, filter_unknown_padding=self.config.policy in (
            "ttl", "jitter", "retained-refresh"))
        if not actions:
            raise ValueError("the prepared catalog cannot advance a live request")
        clocks = {v.name: v.last_service_ms for v in views}
        measured = [a for a in actions if means[a.graph.name] is not None]
        incumbent = min(measured, key=lambda a: self._rank_key(a, clocks)) if measured else None
        self._ensure_estimates({a.graph.name for a in actions}, means, incumbent)
        category, reason, opportunity = "ordinary", "mean", False
        sampled: dict[str, float] = {}
        targeted = {}
        unknown = [a for a in actions if means[a.graph.name] is None]
        native_unknown = []
        if native_discovery:
            native_unknown = [a for a in self._actions(views, means)
                              if means[a.graph.name] is None and not self.estimates[a.graph.name].count]
        soft_harm = self.config.policy == "targeted" and self.config.harm_alpha is not None
        phases = {"prefill" if lane.prefill else "decode" for lane in incumbent.lanes} if incumbent else set()
        unfunded_phases = sorted(phase for phase in phases if self.harm_credit_ms[phase] < HARM_CAP_MS)
        if incumbent is None:
            bootstrap = self._actions(views, means)
            selected = min(bootstrap, key=lambda a: self._bootstrap_key(a, clocks))
            category, reason = "bootstrap", "no_measured_productive_choice"
            if self.config.policy in ("ttl", "jitter") and any(
                    self.estimates[a.graph.name].count for a in unknown):
                category, reason = "probe", "expired_or_unobserved"
        elif self.config.policy in ("ttl", "jitter") and unknown:
            selected = min(unknown, key=lambda a: self._bootstrap_key(a, clocks))
            category, reason = "probe", "expired_or_unobserved"
        elif native_unknown:
            selected = min(native_unknown, key=lambda a: self._bootstrap_key(a, clocks))
            category, reason = "bootstrap", "initial_unknown_graph"
        elif self.config.policy == "retained-refresh":
            selected = incumbent
            stale = [action for action in measured if action.graph.name != incumbent.graph.name
                     and now_ms >= self._observed_at[action.graph.name] + self.config.ttl_ms]
            opportunity = bool(stale) and self.credit_ms >= self.config.credit_cap_ms
            if opportunity:
                selected = min(stale, key=lambda action: (
                    max(self._observed_at[action.graph.name],
                        self._attempted.get(action.graph.name, float("-inf"))),
                    self._rank_key(action, clocks)))
                category, reason = "probe", "stale_refresh"
        else:
            selected = incumbent
            alternatives = {a.graph.name for a in actions} - {incumbent.graph.name}
            opportunity = bool(alternatives) and self.config.policy in EXPLORATORY and (
                not self.config.funded or self.credit_ms >= self.config.credit_cap_ms) and (
                not soft_harm or not unfunded_phases)
            if opportunity:
                if self.config.policy == "targeted":
                    selected, reason, sampled, targeted = self._nominate_targeted(
                        views, means, actions, incumbent, clocks)
                else:
                    selected, reason, sampled = self._nominate(views, means, actions, incumbent, clocks)
                if selected.graph.name == incumbent.graph.name:
                    selected, reason = incumbent, "incumbent"
                else:
                    category = "probe"
        score = (sum(self._unknown_lane_value(lane) for lane in selected.lanes)
                 if means[selected.graph.name] is None and not sampled
                 else self.utility.score(selected))
        metadata = {"now_ms": now_ms, "opportunity": opportunity,
                    "opportunity_index": self.opportunities + int(opportunity),
                    "balance_before_ms": self.credit_ms, "mean_costs_ms": means,
                    "sampled_costs_ms": sampled, "log_scale": self.utility.log_scale,
                    "incumbent_graph": incumbent.graph.name if incumbent else None,
                    "incumbent_score": self.utility.score(incumbent) if incumbent else None,
                    "score_kind": "unknown_value" if means[selected.graph.name] is None and not sampled else "value_rate"}
        if self.config.policy == "targeted":
            metadata["targeted"] = {"occupied_ms": self._occupied_ms,
                                    "refresh_busy_ms": self.config.refresh_busy_ms, **targeted}
        if soft_harm:
            metadata["harm_budget"] = {
                "alpha": self.config.harm_alpha, "cap_request_ms": HARM_CAP_MS,
                "balances_before_ms": dict(self.harm_credit_ms),
                "required_phases": sorted(phases), "unfunded_phases": unfunded_phases,
                "reference": ({"graph": incumbent.graph.name,
                               "duration_ms": incumbent.graph.duration_ms,
                               "lanes": [vars(lane) for lane in incumbent.lanes],
                               "live_requests": len(views)} if incumbent else None)}
            self._planned_reference = (incumbent, len(views)) if incumbent else None
        self._planned = Decision(selected, category, reason, means[selected.graph.name], score, metadata)
        self._plan_signature = signature
        return self._planned

    def _nominate_targeted(self, views, means, actions, incumbent, clocks):
        names = sorted({action.graph.name for action in actions})
        nominal = {name: means[name] if means[name] is not None
                   else self.estimates[name].quantile(0.5) for name in names}
        optimistic = {name: self.estimates[name].quantile(self.config.quantile) for name in names}
        soft_harm = self.config.harm_alpha is not None
        guarded = self.config.probe_harm_ms is not None or soft_harm
        cohort_reference = incumbent if guarded else None
        ranked = self._best_by_graph(views, dict(means) | optimistic, clocks,
                                     cohort_reference=cohort_reference)
        audit_ranked = {}
        audit_fallback = {}
        safe_challengers = set(ranked)
        if guarded:
            def admissible(action):
                return self._probe_service_delay(action, incumbent, len(views), nominal, means)["admitted"]

            safe_ranked = self._best_by_graph(views, dict(means) | optimistic, clocks,
                                               admissible=admissible, cohort_reference=incumbent)
            safe_challengers = set(safe_ranked)
            ranked.update(safe_ranked)
            audit_fallback = self._best_by_graph(views, dict(means) | nominal, clocks,
                                                 cohort_reference=incumbent)
            audit_ranked = self._best_by_graph(views, dict(means) | nominal, clocks,
                                              admissible=admissible, cohort_reference=incumbent,
                                              harm_first=(lambda action: self._probe_service_delay(
                                                  action, incumbent, len(views), nominal, means)["total_request_ms"])
                                              if soft_harm else None)
        incumbent_score = self.utility.score(incumbent)
        gross_rate = sum(self.utility.action_lane_value(lane, incumbent.graph)
                         for lane in incumbent.lanes) / incumbent.graph.duration_ms
        details = {}
        for name in names:
            if name == incumbent.graph.name:
                continue
            estimate = self.estimates[name]
            candidate = ranked[name]
            mean_action = replace(candidate, graph=replace(candidate.graph, duration_ms=nominal[name]))
            mean_score = self.utility.score(mean_action)
            optimistic_score = self.utility.score(candidate)
            loss = max(0.0, incumbent_score - mean_score) * nominal[name] / gross_rate
            extra = max(0.0, nominal[name] - incumbent.graph.duration_ms)
            age = self._occupied_ms - self._observed_busy.get(name, 0.0)
            pending = bool(getattr(estimate, "pending_change", False))
            first = means[name] is None
            plausible = optimistic_score > incumbent_score
            admitted = plausible and (first or (
                loss <= self.config.probe_loss_ms and extra <= self.config.probe_extra_ms))
            if guarded:
                admitted &= name in safe_challengers
            details[name] = {"mean_estimate_ms": means[name], "admission_estimate_ms": nominal[name],
                             "optimistic_estimate_ms": optimistic[name], "mean_score": mean_score,
                             "optimistic_score": optimistic_score, "implied_loss_ms": loss,
                             "requests": [lane.name for lane in candidate.lanes],
                             "extra_blocking_ms": extra, "audit_age_ms": age,
                             "audit_due": bool(self.config.refresh_busy_ms and age >= self.config.refresh_busy_ms),
                             "pending_change": pending,
                             "confirmation_deferred": name in self._failed_confirmations,
                             "pending_count": getattr(estimate, "pending_count", 0),
                             "suspect_direction": getattr(estimate, "suspect_direction", 0),
                             "change_strength": getattr(estimate, "change_strength", 0.0),
                             "first_trial": first, "plausible": plausible, "admitted": admitted}
            if guarded:
                details[name]["service_delay"] = self._probe_service_delay(
                    candidate, incumbent, len(views), nominal, means)
                details[name]["audit_service_delay"] = self._probe_service_delay(
                    audit_ranked.get(name, audit_fallback[name]), incumbent, len(views), nominal, means)
                details[name]["deferred_reason"] = (
                    "peer_service_delay" if (pending or details[name]["audit_due"])
                    and name not in audit_ranked else None)
        pending = [name for name, row in details.items() if row["pending_change"]
                   and (not guarded or name in audit_ranked)]
        due = [name for name, row in details.items() if row["audit_due"]
               and (not guarded or name in audit_ranked)]
        priority = [name for name in pending if name not in self._failed_confirmations]
        eligible = priority or sorted(set(pending) | set(due))
        contenders = [ranked[name] for name, row in details.items() if row["admitted"]]
        selection = {"gross_value_rate": gross_rate, "candidates": details}
        if self.config.audit_every:
            audits = sorted(set(due) | (set(pending) & self._failed_confirmations))
            contenders = [action for action in contenders if action.graph.name not in pending]
            contended = bool(audits and contenders and not priority)
            audit_turn = contended and (self._contended_opportunities + 1) % self.config.audit_every == 0
            eligible = priority or (audits if audits and (not contenders or audit_turn) else [])
            selection["arbitration"] = {
                "period": self.config.audit_every, "contended": contended,
                "index": self._contended_opportunities + int(contended),
                "queue": ("confirmation" if priority else "audit" if eligible else
                          "challenger" if contenders else "incumbent"),
                "borrowed": bool(not priority and (bool(audits) != bool(contenders))),
                "audits": audits,
                "challengers": sorted(action.graph.name for action in contenders),
            }
        if eligible:
            name = min(eligible, key=lambda n: (self._attempted_busy.get(n, float("-inf")), n))
            reason = "confirmation" if name in pending else "occupied_audit"
            selected = (audit_ranked[name] if guarded else
                        self._best_by_graph(views, dict(means) | nominal, clocks)[name])
            mean_score = self.utility.score(selected)
            optimistic_score = self.utility.score(replace(
                selected, graph=replace(selected.graph, duration_ms=optimistic[name])))
            loss = max(0.0, incumbent_score - mean_score) * nominal[name] / gross_rate
            plausible = optimistic_score > incumbent_score
            details[name].update(mean_score=mean_score, optimistic_score=optimistic_score,
                                 implied_loss_ms=loss, plausible=plausible,
                                 admitted=plausible and (means[name] is None or (
                                     loss <= self.config.probe_loss_ms
                                     and details[name]["extra_blocking_ms"] <= self.config.probe_extra_ms)),
                                 requests=[lane.name for lane in selected.lanes])
            if guarded:
                details[name]["service_delay"] = self._probe_service_delay(
                    selected, incumbent, len(views), nominal, means)
            sampled = {name: nominal[name]} if means[name] is None else {}
        else:
            if not contenders:
                return incumbent, "incumbent", {}, selection
            selected = min(contenders, key=lambda action: self._rank_key(action, clocks))
            name = selected.graph.name
            reason = "first_trial" if means[name] is None else "utility_challenger"
            sampled = optimistic
        selected_details = {"gross_value_rate": gross_rate, "selected_graph": name, "candidates": details}
        if self.config.audit_every:
            selected_details["arbitration"] = selection["arbitration"]
        return selected, reason, sampled, selected_details

    def _probe_service_delay(self, action, incumbent, live_count, nominal, means, *,
                             elapsed_ms=None, committed=None):
        """Estimate positive service delay if a check precedes the incumbent."""
        duration = nominal[action.graph.name] if elapsed_ms is None else elapsed_ms
        lanes = {lane.name: lane for lane in action.lanes}
        actual = {row["request"]: row for row in committed} if committed is not None else None
        delays = {}
        phase_delays = {"prefill": 0.0, "decode": 0.0}
        for lane in incumbent.lanes:
            candidate = lanes.get(lane.name)
            if actual is None:
                supplied = candidate is not None and (
                    candidate.prefill >= lane.prefill if lane.prefill else candidate.expected_output > 0)
            else:
                row = actual.get(lane.name, {})
                supplied = row.get("prefill", 0) >= lane.prefill if lane.prefill else row.get("emitted", 0) > 0
            delays[lane.name] = max(0.0, duration - incumbent.graph.duration_ms) if supplied else duration
            phase_delays["prefill" if lane.prefill else "decode"] += delays[lane.name]
        single = live_count == 1 and self.config.harm_alpha is None
        total = 0.0 if single else sum(delays.values())
        if single:
            phase_delays = dict.fromkeys(phase_delays, 0.0)
        return {"request_delays_ms": delays, "total_request_ms": total,
                "phase_request_ms": phase_delays,
                "maximum_peer_delay_ms": 0.0 if single else max(delays.values(), default=0.0),
                "single_request_exemption": single,
                "duration_basis": ("actual_elapsed" if elapsed_ms is not None else
                                   "observed_mean" if means[action.graph.name] is not None else "prior_median"),
                "limit_request_ms": self.config.probe_harm_ms,
                "admitted": self.config.probe_harm_ms is None or single or total <= self.config.probe_harm_ms}

    def _nominate(self, views, means, actions, incumbent, clocks):
        index = self.opportunities + 1
        by_graph = {}
        for action in actions:
            key = self._rank_key(action, clocks) if means[action.graph.name] is not None else self._bootstrap_key(action, clocks)
            if action.graph.name not in by_graph or key < by_graph[action.graph.name][0]:
                by_graph[action.graph.name] = (key, action)
        alternatives = sorted(set(by_graph) - {incumbent.graph.name})
        if self.config.refresh_every and index % self.config.refresh_every == 0:
            name = min(alternatives, key=lambda n: (self._attempted.get(n, float("-inf")), n))
            return by_graph[name][1], "refresh", {}
        policy = self.config.policy
        if policy == "epsilon":
            if self._rng.random() < self.config.epsilon:
                return by_graph[self._rng.choice(sorted(by_graph))][1], "uniform", {}
            return incumbent, "incumbent", {}
        if policy == "softmax":
            # An unobserved arm has a positive prior location, not a measured normalizer.
            costs = {name: mean if mean is not None else self.estimates[name].quantile(0.5)
                     for name, mean in means.items() if name in by_graph}
            table = dict(means) | costs
            ranked = self._best_by_graph(views, table, clocks)
            low = min(self.utility.score(a) for a in ranked.values())
            high = max(self.utility.score(a) for a in ranked.values())
            if high == low:
                weights = [1.0] * len(ranked)
            else:
                factor = sqrt(1 + index / 100) if self.config.anneal else 1.0
                temperature = self.config.temperature / factor
                weights = [exp((self.utility.score(a) - high) / ((high - low) * temperature))
                           for a in ranked.values()]
            name = self._rng.choices(list(ranked), weights=weights, k=1)[0]
            return ranked[name], "softmax", costs
        draws = {}
        for name in sorted(by_graph):
            estimate = self.estimates[name]
            if policy == "optimism":
                draws[name] = estimate.quantile(self.config.quantile)
            elif policy == "predictive":
                draws[name] = estimate.sample_predictive(self._rng)
            else:
                draws[name] = estimate.sample_mean(self._rng)
        ranked = self._best_by_graph(views, dict(means) | draws, clocks)
        selected = min(ranked.values(), key=lambda a: self._rank_key(a, clocks))
        return selected, policy, draws

    def _cohort_actions(self, views, costs, incumbent):
        protected = {lane.name for lane in incumbent.lanes if not lane.prefill}
        prefills = [view for view in views if view.remaining_prefill]
        for spec in self.catalog:
            graph = self._graph(spec, costs[spec.name])
            rows = []
            if graph.decode_capacity:
                for view in views:
                    if not view.remaining_prefill:
                        width = min(graph.width, view.output_budget)
                        lane = Lane(view.name, 0, width, expected_output(view.acceptance, width), view.weight)
                        rows.append((view, lane))
                rows.sort(key=lambda row: (-self.utility.action_lane_value(row[1], graph),
                                          row[0].last_service_ms, row[0].name))
            same = [view for view, _ in rows if view.name in protected]
            other = [view for view, _ in rows if view.name not in protected]
            size = min(graph.decode_capacity, len(rows))
            # Every protected decode row reduces delay by the same fixed duration.
            for count in range(max(0, size - len(other)), min(size, len(same)) + 1):
                cohort = [*same[:count], *other[:size - count], *prefills]
                yield from candidates(cohort, (graph,), self.value,
                                      lane_value=self.utility.action_lane_value)

    def _best_by_graph(self, views, costs, clocks, *, admissible=None,
                       cohort_reference=None, harm_first=None):
        ranked = {}
        actions = (self._cohort_actions(views, costs, cohort_reference) if cohort_reference is not None
                   else self._actions(views, costs, filter_unknown_padding=False))
        def rank(action):
            key = self._rank_key(action, clocks)
            return (harm_first(action), key) if harm_first is not None else key
        for action in actions:
            if admissible is not None and not admissible(action):
                continue
            if action.graph.name not in ranked or rank(action) < rank(ranked[action.graph.name]):
                ranked[action.graph.name] = action
        return dict(sorted(ranked.items()))

    def commit(self, decision: Decision, operation_id):
        if self._inflight is not None or decision is not self._planned:
            raise ValueError("commit must name the scheduler's uncommitted decision")
        if operation_id in self._operation_ids:
            raise ValueError("operation identifiers cannot be reused")
        self._operation_ids.add(operation_id)
        self.opportunities += int(decision.metadata["opportunity"])
        self._attempted[decision.action.graph.name] = decision.metadata["now_ms"]
        if self.config.policy == "targeted":
            self._attempted_busy[decision.action.graph.name] = self._occupied_ms
            self._contended_opportunities += int(
                decision.metadata["targeted"].get("arbitration", {}).get("contended", False))
        self._inflight = (operation_id, decision)
        self._inflight_reference = self._planned_reference
        self._planned_reference = None
        self._planned = None
        self._plan_signature = None

    def _observe(self, observation: TimingObservation):
        if not observation.valid:
            self.invalid_observations += 1
            return
        name, total = observation.graph_id, observation.total_ms
        if name not in {g.name for g in self.catalog}:
            raise ValueError("timing observation names an unknown physical graph")
        if name not in self.estimates:
            self._ensure_estimates({name}, self._means(observation.end_ms)
                                   if self.config.policy != "known" else {g.name: None for g in self.catalog}, None)
        self.estimates[name].observe(total, gpu_ms=observation.gpu_ms, handoff_ms=observation.handoff_ms)
        fresh = observation.end_ms < self._expires.get(name, float("-inf"))
        self._ema[name] = ((7 * self._ema[name] + total) / 8
                           if name in self._ema and (self.config.policy == "ema" or fresh) else total)
        self._observed_at[name] = observation.end_ms
        lifetime = (self._expiry_rng.uniform(0.8, 1.2) * self.config.ttl_ms
                    if self.config.policy == "jitter" else self.config.ttl_ms)
        self._expires[name] = observation.end_ms + lifetime

    def seed_observation(self, observation: TimingObservation):
        if self._planned is not None or self._inflight is not None:
            raise ValueError("historical observations cannot change a planned or executing operation")
        self._observe(observation)
        self._attempted[observation.graph_id] = observation.start_ms

    def complete(self, operation_id, observation: TimingObservation, committed: list[dict]):
        if self._inflight is None or self._inflight[0] != operation_id:
            raise ValueError("completion must name the executing operation exactly once")
        decision = self._inflight[1]
        if observation.graph_id != decision.action.graph.name:
            raise ValueError("completion timing does not match the submitted physical graph")
        if not isfinite(observation.elapsed_ms) or observation.elapsed_ms <= 0:
            raise ValueError("completed execution must have positive finite elapsed time")
        if observation.start_ms != decision.metadata["now_ms"]:
            raise ValueError("completion interval must begin at dispatch")
        lanes = {lane.name: lane for lane in decision.action.lanes}
        if {row["request"] for row in committed} != set(lanes) or len(committed) != len(lanes):
            raise ValueError("completion must cover exactly the submitted cohort")
        for row in committed:
            lane = lanes[row["request"]]
            output_limit = int(lane.expected_output > 0) if lane.prefill else lane.width
            if not 0 <= row.get("prefill", 0) <= lane.prefill or not 0 <= row.get("emitted", 0) <= output_limit:
                raise ValueError("completion progress exceeds its submitted lane")
        self.utility.observe(observation.elapsed_ms, committed)
        self._observe(observation)
        duration = observation.elapsed_ms
        if self.config.policy == "targeted" and self.config.harm_alpha is not None:
            if decision.category == "probe":
                incumbent, live_count = self._inflight_reference
                charge = self._probe_service_delay(
                    decision.action, incumbent, live_count, {}, {},
                    elapsed_ms=duration, committed=committed)
                for phase, cost in charge["phase_request_ms"].items():
                    self.harm_credit_ms[phase] -= cost
                    self.harm_charged_ms[phase] += cost
                decision.metadata["harm_budget"]["actual_charge"] = charge
            elif decision.category == "ordinary":
                rate = self.config.harm_alpha / (1 - self.config.harm_alpha)
                for row in committed:
                    if not row.get("prefill", 0) and not row.get("emitted", 0):
                        continue
                    phase = "prefill" if lanes[row["request"]].prefill else "decode"
                    self.harm_ordinary_ms[phase] += duration
                    self.harm_credit_ms[phase] = min(HARM_CAP_MS, self.harm_credit_ms[phase] + rate * duration)
        if self.config.policy == "targeted":
            self._occupied_ms += duration
            if observation.valid:
                self._observed_busy[observation.graph_id] = self._occupied_ms
                self._failed_confirmations.discard(observation.graph_id)
            elif decision.reason == "confirmation":
                self._failed_confirmations.add(observation.graph_id)
        if decision.category == "probe":
            self.probe_count += 1
            self.probe_ms += duration
            self.credit_ms -= duration
        elif decision.category == "bootstrap":
            self.bootstrap_count += 1
            self.bootstrap_ms += duration
        else:
            self.ordinary_elapsed_ms += duration
            if observation.valid:
                self.ordinary_ms += duration
                self.credit_ms = min(self.config.credit_cap_ms,
                                     self.credit_ms + self.config.alpha / (1 - self.config.alpha) * duration)
        self._now_ms = observation.end_ms
        self._inflight = None
        self._inflight_reference = None

    def idle(self):
        if self._inflight is not None or self._planned is not None:
            raise ValueError("idle requires no planned or executing operation")
        self.credit_ms = min(0.0, self.credit_ms)
        if self.config.policy == "targeted" and self.config.harm_alpha is not None:
            self.harm_credit_ms = {phase: min(0.0, balance) for phase, balance in self.harm_credit_ms.items()}

    def snapshot(self) -> dict:
        result = {"policy": self.config.policy, "credit_ms": self.credit_ms,
                "balance_ms": self.credit_ms, "opportunities": self.opportunities,
                "ordinary_ms": self.ordinary_ms, "ordinary_elapsed_ms": self.ordinary_elapsed_ms,
                "probe_ms": self.probe_ms, "bootstrap_ms": self.bootstrap_ms,
                "probe_count": self.probe_count, "bootstrap_count": self.bootstrap_count,
                "invalid_observations": self.invalid_observations,
                "observation_count": sum(e.count for e in self.estimates.values()),
                "regime_resets": sum(e.regime_changes for e in self.estimates.values()),
                "costs": {name: estimate.snapshot() for name, estimate in sorted(self.estimates.items())}}
        if self.config.policy == "targeted":
            result["occupied_ms"] = self._occupied_ms
            result["observed_busy_ms"] = dict(sorted(self._observed_busy.items()))
            result["failed_confirmations"] = sorted(self._failed_confirmations)
            if self.config.audit_every:
                result["contended_opportunities"] = self._contended_opportunities
            if self.config.harm_alpha is not None:
                result["harm_budget"] = {"alpha": self.config.harm_alpha, "cap_request_ms": HARM_CAP_MS,
                                         "balances_ms": dict(self.harm_credit_ms),
                                         "ordinary_lane_request_ms": dict(self.harm_ordinary_ms),
                                         "charged_request_ms": dict(self.harm_charged_ms)}
        if self.config.initial_discovery == "native" or self.config.policy == "retained-refresh":
            result["initial_discovery_count"] = self.bootstrap_count
            result["initial_discovery_ms"] = self.bootstrap_ms
        return result
