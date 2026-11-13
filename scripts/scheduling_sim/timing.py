# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable timing records and retained arithmetic-mean cost estimates."""

from dataclasses import dataclass, field
from functools import lru_cache
from math import exp, inf, isclose, isfinite, lgamma, log, log1p, sqrt
from random import Random

from .model import Graph


@dataclass(frozen=True)
class GraphSpec:
    name: str
    prefill_capacity: int = 0
    decode_capacity: int = 0
    width: int = 1

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a graph requires a name")
        if any(type(v) is not int for v in (self.prefill_capacity, self.decode_capacity, self.width)):
            raise ValueError("graph capacities and width must be integers")
        if min(self.prefill_capacity, self.decode_capacity) < 0 or self.width < 1:
            raise ValueError("invalid graph capacities")
        if not self.prefill_capacity and not self.decode_capacity:
            raise ValueError("a graph must have a prefill or decode lane")
        if not self.decode_capacity and self.width != 1:
            raise ValueError("prefill-only graphs cannot have speculative lanes")

    @classmethod
    def from_graph(cls, graph: Graph) -> "GraphSpec":
        return cls(graph.name, graph.prefill_capacity, graph.decode_capacity, graph.width)

    @property
    def family(self) -> str:
        if self.prefill_capacity and self.decode_capacity:
            return "mixed"
        return "prefill" if self.prefill_capacity else "decode"


@dataclass(frozen=True)
class TimingObservation:
    graph_id: str
    start_ms: float
    end_ms: float
    gpu_ms: float | None
    handoff_ms: float | None = 0.0

    def __post_init__(self):
        if not self.graph_id or not isinstance(self.graph_id, str):
            raise ValueError("a timing observation requires a graph id")
        if (not isfinite(self.start_ms) or not isfinite(self.end_ms)
                or self.end_ms <= self.start_ms or not isfinite(self.end_ms - self.start_ms)):
            raise ValueError("execution requires finite increasing timestamps")

    @property
    def elapsed_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def valid(self) -> bool:
        return (self.gpu_ms is not None and isfinite(self.gpu_ms) and self.gpu_ms > 0
                and self.handoff_ms is not None and isfinite(self.handoff_ms)
                and self.handoff_ms >= 0 and isfinite(self.gpu_ms + self.handoff_ms))

    @property
    def total_ms(self) -> float | None:
        return self.gpu_ms + self.handoff_ms if self.valid else None


def _beta_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) >= tiny else tiny)
    result = d
    for index in range(1, 513):
        twice = 2 * index
        coefficient = index * (b - index) * x / ((a + twice - 1) * (a + twice))
        d = 1.0 + coefficient * d
        c = 1.0 + coefficient / c
        d = 1.0 / (d if abs(d) >= tiny else tiny)
        c = c if abs(c) >= tiny else tiny
        result *= d * c
        coefficient = -(a + index) * (a + b + index) * x / ((a + twice) * (a + twice + 1))
        d = 1.0 + coefficient * d
        c = 1.0 + coefficient / c
        d = 1.0 / (d if abs(d) >= tiny else tiny)
        c = c if abs(c) >= tiny else tiny
        change = d * c
        result *= change
        if abs(change - 1.0) < 3e-14:
            return result
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return factor * _beta_fraction(a, b, x) / a
    return 1.0 - factor * _beta_fraction(b, a, 1 - x) / b


def _student_t_tail(x: float, degrees: float) -> float:
    if x == 0:
        return 0.5
    log_ratio = log(x) - log(degrees) / 2
    if log_ratio > 345:
        half = degrees / 2
        return exp(-log(2) - degrees * log_ratio - log(half)
                   - lgamma(half) - lgamma(0.5) + lgamma(half + 0.5))
    ratio = exp(log_ratio)
    square = ratio * ratio
    if square < 1:
        return 0.5 * (1 - _regularized_beta(square / (1 + square), 0.5, degrees / 2))
    return 0.5 * _regularized_beta(1 / (1 + square), degrees / 2, 0.5)


def student_t_cdf(value: float, degrees: float) -> float:
    """Student-t CDF evaluated through the regularized incomplete beta function."""
    if not isfinite(degrees) or degrees <= 0 or value != value:
        raise ValueError("Student-t requires positive finite degrees and a non-NaN value")
    if value == inf:
        return 1.0
    if value == -inf:
        return 0.0
    tail = _student_t_tail(abs(value), degrees)
    return tail if value < 0 else 1 - tail


def _student_t_upper_quantile(probability: float, degrees: float) -> float:
    if probability == 0:
        return inf
    low, high = 0.0, 1.0
    while _student_t_tail(high, degrees) > probability:
        high *= 2
        if not isfinite(high):
            return inf
    for _ in range(96):
        middle = low + (high - low) / 2
        if _student_t_tail(middle, degrees) > probability:
            low = middle
        else:
            high = middle
        if high - low <= 2e-13 * max(1.0, middle):
            break
    return low + (high - low) / 2


@lru_cache(maxsize=512)
def student_t_quantile(probability: float, degrees: float) -> float:
    """Invert Student-t tails without subtracting a small tail from one."""
    if not isfinite(probability) or not 0 <= probability <= 1 or not isfinite(degrees) or degrees <= 0:
        raise ValueError("Student-t requires a probability in [0, 1] and positive finite degrees")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -_student_t_upper_quantile(probability, degrees)
    return _student_t_upper_quantile(1 - probability, degrees)


@dataclass
class _Moments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def observe(self, value: float):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def merge(self, other: "_Moments"):
        if not other.count:
            return
        combined = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / combined
        self.mean += delta * other.count / combined
        self.count = combined

    def snapshot(self) -> dict:
        return {"count": self.count, "mean_ms": self.mean if self.count else None,
                "m2_ms2": self.m2, "variance_ms2": max(0.0, self.m2) / (self.count - 1)
                if self.count > 1 else None}


@dataclass(frozen=True)
class _Posterior:
    degrees: float
    location: float
    scale: float
    block_predictive_scale: float
    raw_predictive_scale: float

    def positive_quantile(self, probability: float) -> float:
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("a mean quantile requires a probability in [0, 1]")
        if probability == 0:
            return 0.0
        if probability == 1:
            return inf
        survival = 1 - student_t_cdf(-self.location / self.scale, self.degrees)
        tail = (1 - probability) * survival
        if tail <= 0.5:
            standard = _student_t_upper_quantile(tail, self.degrees)
        else:
            standard = -_student_t_upper_quantile(1 - tail, self.degrees)
        return max(0.0, self.location + self.scale * standard)

    def sample(self, rng: Random, *, predictive: bool = False) -> float:
        scale = self.raw_predictive_scale if predictive else self.scale
        while True:
            normal = rng.gauss(0, 1)
            chi_square = rng.gammavariate(self.degrees / 2, 2)
            value = self.location + scale * normal / sqrt(chi_square / self.degrees)
            if value > 0 and isfinite(value):
                return value

    def predictive_interval(self) -> tuple[float, float]:
        radius = student_t_quantile(0.995, self.degrees) * self.block_predictive_scale
        return self.location - radius, self.location + radius


@dataclass(frozen=True)
class _Block:
    total: _Moments
    gpu: _Moments
    handoff: _Moments


@dataclass
class _LogHistory:
    moments: _Moments = field(default_factory=_Moments)
    previous: float | None = None
    left: _Moments = field(default_factory=_Moments)
    right: _Moments = field(default_factory=_Moments)
    cross: float = 0.0

    def observe(self, total_ms: float):
        value = log(total_ms)
        if self.previous is not None:
            delta = self.previous - self.left.mean
            self.left.observe(self.previous)
            self.right.observe(value)
            self.cross += delta * (value - self.right.mean)
        self.moments.observe(value)
        self.previous = value

    @property
    def correlation(self) -> float:
        product = self.left.m2 * self.right.m2
        return self.cross / sqrt(product) if product > 0 else 0.0


def _correlation_factor(count: int, correlation: float) -> float:
    if correlation == 0:
        return 1.0
    if count <= 4:
        return 1 + 2 * sum((1 - lag / count) * correlation ** lag for lag in range(1, count))
    complement = 1 - correlation
    return ((1 + correlation) / complement
            - 2 * correlation * (-exp(count * log(correlation)) + 1)
            / (count * complement * complement))


@dataclass(frozen=True)
class _LogReference:
    mean: float
    variance: float
    raw_mean: float
    count: int
    correlation: float
    baseline_variance_factor: float
    degrees: int
    critical: float
    confirmation_level: float
    confirmation_critical: float

    def radius(self, count: int, *, confirmation: bool = False) -> float:
        critical = self.confirmation_critical if confirmation else self.critical
        return critical * sqrt(self.variance * (
            _correlation_factor(count, self.correlation) / count + self.baseline_variance_factor))


@dataclass(frozen=True)
class _RawTiming:
    total_ms: float
    gpu_ms: float | None
    handoff_ms: float | None


class TimingEstimate:
    """Retain raw timings and infer a positive mean from block means.

    The drift detector uses the symmetric NIG predictive working model. Selection
    draws condition its marginal mean distribution on positive duration.
    The optional EMA supplies the operational mean only. Posterior distributions
    and change detection retain the arithmetic statistics of the active regime.
    Optional partial blocks contribute their mean through a tempered Gaussian
    likelihood of weight count/block_size, replacing the preceding partial
    contribution. Complete blocks retain their unit likelihood weight.

    Log-regime confirmation uses raw log timings in groups of four. Its frozen
    baseline has at least 64 observations. The working lag-one correlation is
    max(0, r) + 2/sqrt(N - 3), capped at 0.99. For F(n) =
    1 + 2*sum((1-k/n)*rho**k), the log-mean difference variance is
    s²*(F(group)/group + F(N)/N). Student-t degrees use the largest power of two
    not exceeding N/F(N), minus one, with a floor of one. Two groups must exceed
    the central 99.99% interval on the same side with a 10% arithmetic shift.
    Their combined eight-sample mean must also exceed a lifetime confirmation
    threshold. Reference j spends 1e-4/(J*(J+1)), where J rounds j upward to a
    power of two. The reference counter survives regime changes. This nominal
    spending does not require independence between successive group checks.
    These correlation and predictive intervals are working models, not
    coverage guarantees under arbitrary dependence or non-lognormal timings.
    """

    def __init__(self, prior_location_ms: float = 1.0, block_size: int = 8, mode: str = "cumulative",
                 *, mean_estimator: str = "arithmetic", partial_blocks: bool = False):
        if (not isfinite(prior_location_ms) or prior_location_ms <= 0
                or not 0 < prior_location_ms * prior_location_ms < inf):
            raise ValueError("prior location must be positive and have a finite square")
        if type(block_size) is not int or block_size < 1:
            raise ValueError("block size must be a positive integer")
        if mode not in ("cumulative", "regime", "log-regime"):
            raise ValueError("timing mode must be cumulative, regime, or log-regime")
        if mean_estimator not in ("arithmetic", "ema"):
            raise ValueError("operational mean estimator must be arithmetic or ema")
        if type(partial_blocks) is not bool:
            raise ValueError("partial_blocks must be a bool")
        self._prior_location_ms = prior_location_ms
        self._block_size = block_size
        self._mode = mode
        self._mean_estimator = mean_estimator
        self._partial_blocks = partial_blocks
        self._ema_ms: float | None = None
        self._lifetime = _Moments()
        self._active = _Moments()
        self._lifetime_blocks = _Moments()
        self._active_blocks = _Moments()
        self._complete_active = _Moments()
        self._gpu = _Moments()
        self._handoff = _Moments()
        self._active_gpu = _Moments()
        self._active_handoff = _Moments()
        self._partial = _Moments()
        self._partial_gpu = _Moments()
        self._partial_handoff = _Moments()
        self._suspects: list[_Block] = []
        self._reference: tuple[float, float, float] | None = None
        self._direction = 0
        self._regime_changes = 0
        self._posterior_cache: _Posterior | None = None
        self._quantiles: dict[float, float] = {}
        self._log_history = _LogHistory()
        self._log_reference: _LogReference | None = None
        self._log_recent: list[_RawTiming] = []
        self._log_first: list[_RawTiming] = []
        self._log_direction = 0
        self._log_strength = 0.0
        self._log_looks = 0
        self._lifetime_partial = _Moments()

    @property
    def prior_location_ms(self) -> float:
        return self._prior_location_ms

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def mean_estimator(self) -> str:
        return self._mean_estimator

    @property
    def partial_blocks(self) -> bool:
        return self._partial_blocks

    @property
    def count(self) -> int:
        return self._lifetime.count

    @property
    def active_count(self) -> int:
        return self._active.count

    @property
    def block_count(self) -> int:
        return self._active_blocks.count

    @property
    def regime_changes(self) -> int:
        return self._regime_changes

    @property
    def mean_ms(self) -> float | None:
        return self.ema_ms if self.mean_estimator == "ema" else self.arithmetic_mean_ms

    @property
    def arithmetic_mean_ms(self) -> float | None:
        return self._active.mean if self._active.count else None

    @property
    def ema_ms(self) -> float | None:
        return self._ema_ms

    @property
    def pending_change(self) -> bool:
        return bool(self._log_direction) if self.mode == "log-regime" else bool(self._suspects)

    @property
    def suspect_direction(self) -> int:
        return self._log_direction if self.mode == "log-regime" else self._direction

    @property
    def suspect_blocks(self) -> int:
        return int(bool(self._log_first)) if self.mode == "log-regime" else len(self._suspects)

    @property
    def pending_count(self) -> int:
        if self.mode == "log-regime":
            return len(self._log_first) + len(self._log_recent) if self.pending_change else 0
        return sum(block.total.count for block in self._suspects)

    @property
    def change_strength(self) -> float:
        return self._log_strength if self.mode == "log-regime" else 0.0

    def _posterior(self) -> _Posterior:
        if self._posterior_cache is None:
            blocks = self._active_blocks
            kappa = 0.25 + blocks.count
            alpha = 2 + blocks.count / 2
            location = (0.25 * self.prior_location_ms + blocks.count * blocks.mean) / kappa
            beta = (self.prior_location_ms ** 2 + max(0.0, blocks.m2) / 2
                    + 0.25 * blocks.count / (2 * kappa) * (blocks.mean - self.prior_location_ms) ** 2)
            if self.partial_blocks and self._partial.count:
                weight = self._partial.count / self.block_size
                combined = kappa + weight
                beta += kappa * weight / (2 * combined) * (self._partial.mean - location) ** 2
                location = (kappa * location + weight * self._partial.mean) / combined
                kappa = combined
                alpha += weight / 2
            scale = sqrt(beta / alpha / kappa)
            self._posterior_cache = _Posterior(
                2 * alpha, location, scale, scale * sqrt(kappa + 1),
                scale * sqrt(self.block_size * kappa + 1),
            )
        return self._posterior_cache

    def quantile(self, probability: float) -> float:
        if probability not in self._quantiles:
            self._quantiles[probability] = self._posterior().positive_quantile(probability)
        return self._quantiles[probability]

    def interval(self, probability: float = 0.95) -> tuple[float, float]:
        """Return the posterior interval for the arithmetic mean of the active regime."""
        if not isfinite(probability) or not 0 < probability < 1:
            raise ValueError("an interval requires probability strictly between zero and one")
        tail = (1 - probability) / 2
        return self.quantile(tail), self.quantile(1 - tail)

    def sample_mean(self, rng: Random) -> float:
        """Sample the arithmetic mean of the active regime, independent of the operational estimator."""
        return self._posterior().sample(rng)

    def sample_predictive(self, rng: Random) -> float:
        """Draw a positive individual timing under the independent raw-noise model."""
        return self._posterior().sample(rng, predictive=True)

    def observe(self, total_ms: float | None, gpu_ms: float | None = None, handoff_ms: float | None = None) -> bool:
        if total_ms is None or not isfinite(total_ms) or total_ms <= 0:
            return False
        if gpu_ms is not None or handoff_ms is not None:
            if gpu_ms is None or not isfinite(gpu_ms) or gpu_ms <= 0:
                return False
            handoff = 0.0 if handoff_ms is None else handoff_ms
            if (not isfinite(handoff) or handoff < 0
                    or not isclose(total_ms, gpu_ms + handoff, rel_tol=1e-10, abs_tol=1e-12)):
                return False
        if self.partial_blocks:
            self._posterior_cache = None
            self._quantiles.clear()
        changed = (self._observe_log_change(_RawTiming(total_ms, gpu_ms, handoff_ms))
                   if self.mode == "log-regime" else None)
        self._lifetime.observe(total_ms)
        self._ema_ms = (total_ms if self._ema_ms is None
                        else self._ema_ms + (total_ms - self._ema_ms) / 8)
        if self.mode == "log-regime":
            self._lifetime_partial.observe(total_ms)
            if self._lifetime_partial.count == self.block_size:
                self._lifetime_blocks.observe(self._lifetime_partial.mean)
                self._lifetime_partial = _Moments()
        self._active.observe(total_ms)
        self._partial.observe(total_ms)
        for value, lifetime, active, partial in (
            (gpu_ms, self._gpu, self._active_gpu, self._partial_gpu),
            (handoff_ms, self._handoff, self._active_handoff, self._partial_handoff),
        ):
            if value is not None:
                lifetime.observe(value)
                active.observe(value)
                partial.observe(value)
        if self._partial.count == self.block_size:
            block = _Block(self._partial, self._partial_gpu, self._partial_handoff)
            self._partial, self._partial_gpu, self._partial_handoff = _Moments(), _Moments(), _Moments()
            self._close_block(block)
        if self.mode == "log-regime":
            self._log_history.observe(total_ms)
            if changed is not None:
                self._start_log_regime(changed)
        return True

    def _make_log_reference(self) -> _LogReference:
        history = self._log_history
        count = history.moments.count
        correlation = min(0.99, max(0.0, history.correlation) + 2 / sqrt(count - 3))
        baseline_factor = _correlation_factor(count, correlation) / count
        effective_count = max(1, int(1 / baseline_factor))
        degrees = max(1, (1 << (effective_count.bit_length() - 1)) - 1)
        self._log_looks += 1
        look_bucket = 1 << (self._log_looks - 1).bit_length()
        confirmation_level = 1e-4 / (look_bucket * (look_bucket + 1))
        return _LogReference(history.moments.mean, max(0.0, history.moments.m2) / (count - 1),
                             self._active.mean, count, correlation, baseline_factor, degrees,
                             student_t_quantile(0.99995, degrees), confirmation_level,
                             -student_t_quantile(confirmation_level / 2, degrees))

    def _clear_log_change(self):
        self._log_reference = None
        self._log_recent = []
        self._log_first = []
        self._log_direction = 0
        self._log_strength = 0.0

    def _observe_log_change(self, observation: _RawTiming) -> list[_RawTiming] | None:
        if self._log_reference is None:
            if self._log_history.moments.count < 64:
                return None
            self._log_reference = self._make_log_reference()
        reference = self._log_reference
        self._log_recent.append(observation)
        count = len(self._log_recent)
        difference = sum(log(row.total_ms) for row in self._log_recent) / count - reference.mean
        raw_difference = sum(row.total_ms for row in self._log_recent) / count - reference.raw_mean
        radius = reference.radius(count)
        direction = (1 if difference > radius and raw_difference >= 0.1 * reference.raw_mean else
                     -1 if difference < -radius and raw_difference <= -0.1 * reference.raw_mean else 0)
        if not self._log_first:
            self._log_direction = direction
        self._log_strength = abs(difference) / max(radius, 1e-12) if self.pending_change else 0.0
        if count < 4:
            return None
        if self._log_first:
            if direction == self._log_direction:
                changed = self._log_first + self._log_recent
                joint_difference = sum(log(row.total_ms) for row in changed) / len(changed) - reference.mean
                if joint_difference * direction > reference.radius(len(changed), confirmation=True):
                    self._clear_log_change()
                    return changed
            self._clear_log_change()
        elif direction:
            self._log_first, self._log_recent = self._log_recent, []
        else:
            self._clear_log_change()
        return None

    def _start_log_regime(self, observations: list[_RawTiming]):
        self._active, self._active_blocks, self._complete_active = _Moments(), _Moments(), _Moments()
        self._active_gpu, self._active_handoff = _Moments(), _Moments()
        self._partial, self._partial_gpu, self._partial_handoff = _Moments(), _Moments(), _Moments()
        self._log_history = _LogHistory()
        for row in observations:
            self._active.observe(row.total_ms)
            self._partial.observe(row.total_ms)
            self._log_history.observe(row.total_ms)
            for value, active, partial in ((row.gpu_ms, self._active_gpu, self._partial_gpu),
                                           (row.handoff_ms, self._active_handoff, self._partial_handoff)):
                if value is not None:
                    active.observe(value)
                    partial.observe(value)
            if self._partial.count == self.block_size:
                self._active_blocks.observe(self._partial.mean)
                self._complete_active.merge(self._partial)
                self._partial, self._partial_gpu, self._partial_handoff = _Moments(), _Moments(), _Moments()
        self._regime_changes += 1
        self._posterior_cache = None
        self._quantiles.clear()

    def _close_block(self, block: _Block):
        switched = False
        if self.mode == "regime" and self._active_blocks.count >= 8:
            if self._reference is None:
                low, high = self._posterior().predictive_interval()
                reference = (low, high, self._complete_active.mean)
            else:
                reference = self._reference
            low, high, mean = reference
            difference = block.total.mean - mean
            direction = (1 if block.total.mean > high else -1 if block.total.mean < low else 0)
            if direction and abs(difference) >= 0.1 * mean:
                if direction != self._direction:
                    self._suspects = []
                self._reference = reference
                self._direction = direction
                self._suspects.append(block)
                switched = len(self._suspects) == 3
            else:
                self._reference = None
                self._direction = 0
                self._suspects.clear()
        if self.mode != "log-regime":
            self._lifetime_blocks.observe(block.total.mean)
        self._active_blocks.observe(block.total.mean)
        self._complete_active.merge(block.total)
        if switched:
            self._active, self._active_blocks, self._complete_active = _Moments(), _Moments(), _Moments()
            self._active_gpu, self._active_handoff = _Moments(), _Moments()
            for suspect in self._suspects:
                self._active.merge(suspect.total)
                self._complete_active.merge(suspect.total)
                self._active_blocks.observe(suspect.total.mean)
                self._active_gpu.merge(suspect.gpu)
                self._active_handoff.merge(suspect.handoff)
            self._regime_changes += 1
            self._reference = None
            self._direction = 0
            self._suspects.clear()
        self._posterior_cache = None
        self._quantiles.clear()

    def snapshot(self) -> dict:
        posterior = self._posterior()
        snapshot = {"prior_location_ms": self.prior_location_ms, "mode": self.mode,
                "block_size": self.block_size, "count": self.count, "active_count": self.active_count,
                "block_count": self.block_count, "lifetime_block_count": self._lifetime_blocks.count,
                "mean_ms": self.mean_ms,
                "lifetime_mean_ms": self._lifetime.mean if self.count else None,
                "regime_changes": self.regime_changes, "partial_count": self._partial.count,
                "raw": self._lifetime.snapshot(), "active_raw": self._active.snapshot(),
                "gpu": self._gpu.snapshot(), "handoff": self._handoff.snapshot(),
                "active_gpu": self._active_gpu.snapshot(), "active_handoff": self._active_handoff.snapshot(),
                "posterior": {"degrees": posterior.degrees, "location_ms": posterior.location,
                              "scale_ms": posterior.scale,
                              "block_predictive_scale_ms": posterior.block_predictive_scale,
                              "raw_predictive_scale_ms": posterior.raw_predictive_scale}}
        if self.mean_estimator == "ema":
            snapshot.update(mean_estimator=self.mean_estimator,
                            arithmetic_mean_ms=self.arithmetic_mean_ms, ema_ms=self.ema_ms)
        if self.partial_blocks:
            snapshot["partial_blocks"] = True
        if self.mode == "log-regime":
            reference = self._log_reference
            snapshot["change_evidence"] = {
                "pending": self.pending_change, "direction": self.suspect_direction,
                "count": self.pending_count, "strength": self.change_strength,
                "reference_count": reference.count if reference is not None else None,
                "reference_correlation": reference.correlation if reference is not None else None,
                "reference_degrees": reference.degrees if reference is not None else None,
                "confirmation_level": reference.confirmation_level if reference is not None else None,
                "lifetime_looks": self._log_looks,
            }
        return snapshot
