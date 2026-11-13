# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online prices for useful-token rates and decode-delivery rates."""

from dataclasses import dataclass, replace
from math import exp, expm1, isfinite, log, log1p

from .model import Action, Graph, Lane, Value, View
from .scheduler import candidates


@dataclass(frozen=True)
class UtilityConfig:
    token_fairness: float
    delivery_fairness: float
    adaptation_ms: float
    age_price: float = 0.0
    rate_history: str = "exponential"

    def __post_init__(self):
        if any(not isfinite(v) or v < 0 for v in (self.token_fairness, self.delivery_fairness, self.age_price)):
            raise ValueError("fairness values must be finite and nonnegative")
        if not isfinite(self.adaptation_ms) or self.adaptation_ms <= 0:
            raise ValueError("price adaptation time must be positive and finite")
        if self.rate_history not in ("exponential", "cumulative", "service"):
            raise ValueError("rate history must be exponential, cumulative, or service")


@dataclass
class ServicePrice:
    """Fairness price using a rate estimate or the reciprocal entropy-dual price."""

    coefficient: float
    maximum_rate: float
    log_fraction: float
    elapsed_ms: float = 0.0

    @classmethod
    def initial(cls, coefficient: float, maximum_rate: float, active_count: int):
        if not isfinite(coefficient) or coefficient < 0:
            raise ValueError("utility coefficient must be finite and nonnegative")
        if not isfinite(maximum_rate) or maximum_rate <= 0 or active_count < 1:
            raise ValueError("service needs a positive finite maximum rate and active count")
        return cls(coefficient, maximum_rate, -log(active_count))

    @property
    def log_price(self) -> float:
        return log(self.coefficient) - self.log_fraction if self.coefficient else float("-inf")

    def value(self, service: float, log_scale: float) -> float:
        return exp(self.log_price - log(self.maximum_rate) - log_scale) * service

    def rebase(self, maximum_rate: float):
        """Preserve the marginal service price when its rate units change."""
        if not isfinite(maximum_rate) or maximum_rate <= 0:
            raise ValueError("service rate must be positive and finite")
        self.log_fraction += log(self.maximum_rate) - log(maximum_rate)
        self.maximum_rate = maximum_rate

    def observe(self, duration_ms: float, delivered: float, adaptation_ms: float,
                *, history="exponential"):
        if any(not isfinite(v) or v < 0 for v in (duration_ms, delivered)):
            raise ValueError("duration and delivered service must be finite and nonnegative")
        if not isfinite(adaptation_ms) or adaptation_ms <= 0:
            raise ValueError("price adaptation time must be positive and finite")
        if history not in ("exponential", "cumulative", "service"):
            raise ValueError("rate history must be exponential, cumulative, or service")
        if duration_ms == 0:
            if delivered:
                raise ValueError("positive service requires elapsed time")
            return
        if history == "service":
            previous = -self.log_fraction
            if delivered:
                normalized = delivered / self.maximum_rate
                decay = normalized / adaptation_ms
                previous -= decay
                incoming = log(-expm1(-decay)) + log(duration_ms) - log(normalized)
            else:
                incoming = log(duration_ms) - log(adaptation_ms)
            maximum = max(previous, incoming)
            self.log_fraction = -maximum - log(exp(previous - maximum) + exp(incoming - maximum))
            self.elapsed_ms += duration_ms
            return
        decay = (log1p(duration_ms / (adaptation_ms + self.elapsed_ms)) if history == "cumulative"
                 else duration_ms / adaptation_ms)
        self.elapsed_ms += duration_ms
        self.log_fraction -= decay
        if delivered:
            incoming = (log(-expm1(-decay)) + log(delivered)
                        - log(self.maximum_rate) - log(duration_ms))
            maximum = max(self.log_fraction, incoming)
            self.log_fraction = maximum + log(exp(self.log_fraction - maximum) + exp(incoming - maximum))


@dataclass
class RequestPrices:
    tokens: ServicePrice | None
    deliveries: ServicePrice | None
    weight: float


class UtilityScheduler:
    """Value-rate selection with separable proportional-fairness prices.

    The stationary utility is intrinsic useful work per time plus weighted log
    token and delivery rates. Prefill has no per-chunk delivery utility. Actual
    completions update prices; speculative expectations only score choices.
    """

    def __init__(self, value: Value, config: UtilityConfig):
        self.value = value
        self.config = config
        self.prices: dict[str, RequestPrices] = {}
        self.log_scale = 0.0
        self.ages: dict[str, tuple[float, float, float]] = {}
        self._references: dict[str, float] = {}
        self._token_normalizers: set[str] = set()
        self._delivery_normalizers: set[str] = set()
        self._mask_unknown_prices = False

    def prepare(self, views: list[View], catalog: tuple[Graph, ...], now_ms: float = 0.0,
                *, known_graphs: set[str] | None = None, mask_unknown_prices: bool = False):
        if not isfinite(now_ms) or any(now_ms < v.last_service_ms for v in views):
            raise ValueError("planning time must be finite and follow the last service")
        names = {v.name for v in views}
        self.prices = {name: prices for name, prices in self.prices.items() if name in names}
        self._references = {name: ref for name, ref in self._references.items() if name in names}
        self._token_normalizers.clear()
        self._delivery_normalizers.clear()
        self._mask_unknown_prices = mask_unknown_prices
        self.ages.clear()
        measured = catalog if known_graphs is None else tuple(g for g in catalog if g.name in known_graphs)
        decoding = sum(not view.remaining_prefill for view in views)
        reference_ms = min((decoding * graph.duration_ms / min(decoding, graph.decode_capacity)
                            for graph in measured if decoding and graph.decode_capacity), default=0.0)
        for view in views:
            actions = candidates([view], catalog, self.value)
            if not actions:
                raise ValueError("the prepared catalog cannot advance a live request")
            observed = [a for a in actions if known_graphs is None or a.graph.name in known_graphs]
            token_rate = max(((a.lanes[0].prefill + a.lanes[0].expected_output) / a.graph.duration_ms
                              for a in observed), default=0.0)
            delivery_rate = max((1 / a.graph.duration_ms for a in observed), default=0.0) if not view.remaining_prefill else 0
            if view.name not in self.prices:
                self.prices[view.name] = RequestPrices(None, None, view.weight)
            prices = self.prices[view.name]
            if token_rate:
                self._token_normalizers.add(view.name)
                if prices.tokens is None:
                    prices.tokens = ServicePrice.initial(
                        self.config.token_fairness * view.weight, token_rate, len(views))
                prices.tokens.rebase(token_rate)
            if delivery_rate:
                self._delivery_normalizers.add(view.name)
                if prices.deliveries is None:
                    prices.deliveries = ServicePrice.initial(
                        self.config.delivery_fairness * view.weight, delivery_rate, 1)
                    prices.deliveries.log_fraction = -log(reference_ms * delivery_rate)
                prices.deliveries.rebase(delivery_rate)
                self._references[view.name] = reference_ms
            reference = self._references.get(view.name)
            if not view.remaining_prefill and reference is not None and (
                    delivery_rate or not mask_unknown_prices):
                self.ages[view.name] = (now_ms - view.last_service_ms,
                                       reference, view.weight)
        self.log_scale = max([0.0, log(self.value.decode_value),
                             log(self.config.age_price) if self.config.age_price else float("-inf"),
                             *(price.log_price - log(price.maximum_rate)
                                    for row in self.prices.values()
                                    for price in (row.tokens, row.deliveries) if price is not None)])

    def lane_value(self, lane: Lane) -> float:
        prices = self.prices[lane.name]
        work = exp(-self.log_scale) * lane.useful_work(self.value)
        if prices.tokens is not None and (not self._mask_unknown_prices or lane.name in self._token_normalizers):
            work += prices.tokens.value(lane.prefill + lane.expected_output, self.log_scale)
        if prices.deliveries is not None and lane.expected_output > 0 and (
                not self._mask_unknown_prices or lane.name in self._delivery_normalizers):
            work += prices.deliveries.value(1, self.log_scale)
        return work

    def score(self, action: Action, *, preceding_ms: float = 0.0) -> float:
        """Value a delivery graph after any non-delivering prerequisite work."""
        if not isfinite(preceding_ms) or preceding_ms < 0:
            raise ValueError("prerequisite duration must be finite and nonnegative")
        duration = preceding_ms + action.graph.duration_ms
        graph = replace(action.graph, duration_ms=duration)
        reward = sum(self.action_lane_value(lane, graph) for lane in action.lanes)
        reward -= (exp(-self.log_scale) * self.config.age_price * duration * duration / 2
                   * sum(w for _, _, w in self.ages.values()))
        return reward / duration

    def action_lane_value(self, lane: Lane, graph: Graph) -> float:
        reward = self.lane_value(lane)
        if lane.name in self.ages and lane.expected_output > 0:
            age, reference, weight = self.ages[lane.name]
            reward += (exp(-self.log_scale) * self.config.age_price * weight * reference
                       * (age + graph.duration_ms))
        return reward

    def choose(self, views: list[View], catalog: tuple[Graph, ...], now_ms: float = 0.0) -> Action:
        self.prepare(views, catalog, now_ms)
        clocks = {v.name: v.last_service_ms for v in views}
        actions = candidates(views, catalog, self.value, lane_value=self.action_lane_value)
        return min(actions, key=lambda a: (
            -self.score(a), a.graph.duration_ms, min(clocks[lane.name] for lane in a.lanes),
            a.graph.prefill_capacity + a.graph.decode_capacity * a.graph.width,
            a.graph.name, tuple(lane.name for lane in a.lanes)))

    def observe(self, duration_ms: float, committed: list[dict]):
        delivered = {row["request"]: row for row in committed}
        for name, prices in self.prices.items():
            row = delivered.get(name, {})
            output = row.get("emitted", 0)
            if prices.tokens is not None and (not self._mask_unknown_prices or name in self._token_normalizers):
                prices.tokens.observe(duration_ms, row.get("prefill", 0) + output,
                                      self.config.adaptation_ms,
                                      history=self.config.rate_history)
            if prices.deliveries is not None and (not self._mask_unknown_prices or name in self._delivery_normalizers):
                prices.deliveries.observe(duration_ms, float(output > 0), self.config.adaptation_ms,
                                          history=self.config.rate_history)
            elif output and prices.deliveries is None:
                prices.deliveries = ServicePrice.initial(
                    self.config.delivery_fairness * prices.weight, 1 / duration_ms, 1)
