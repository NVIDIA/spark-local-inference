# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
from math import atan, exp, inf, log1p, pi, sqrt
from random import Random
from statistics import mean, variance

import pytest

from scripts.scheduling_sim.model import Graph
from scripts.scheduling_sim.timing import (
    GraphSpec, TimingEstimate, TimingObservation, student_t_cdf, student_t_quantile,
    _correlation_factor,
)


@pytest.mark.parametrize("value", [-100, -3, -0.2, 0, 0.2, 3, 100])
def test_student_t_cdf_matches_closed_form_distributions(value):
    assert student_t_cdf(value, 1) == pytest.approx(0.5 + atan(value) / pi, abs=2e-13)
    assert student_t_cdf(value, 2) == pytest.approx(0.5 + value / (2 * sqrt(2 + value * value)), abs=2e-13)


def test_student_t_four_degrees_has_an_independent_exact_value():
    assert student_t_cdf(-1 / sqrt(2), 4) == pytest.approx(7 / 27, abs=2e-13)


def test_student_t_extreme_cauchy_tail_does_not_overflow_its_squared_value():
    assert student_t_cdf(-1e200, 1) == pytest.approx(1e-200 / pi, rel=1e-12, abs=0)
    assert student_t_quantile(1e-200, 1) == pytest.approx(-1e200 / pi, rel=1e-12)


@pytest.mark.parametrize("probability,degrees,expected", [
    (0.75, 1, 1.0),
    (0.975, 2, 4.302652729696142),
    (0.975, 4, 2.7764451051977987),
    (0.975, 10, 2.2281388519649385),
    (0.995, 30, 2.7499956535670305),
])
def test_student_t_quantiles_match_reference_values(probability, degrees, expected):
    assert student_t_quantile(probability, degrees) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("degrees", [1, 2, 4, 4.125, 4.75, 12, 129, 1000, 100_000])
@pytest.mark.parametrize("probability", [1e-10, 0.005, 0.1, 0.5, 0.9, 0.995, 1 - 1e-10])
def test_student_t_quantile_inverts_both_tails(probability, degrees):
    quantile = student_t_quantile(probability, degrees)
    assert student_t_cdf(quantile, degrees) == pytest.approx(probability, abs=2e-10)
    assert student_t_quantile(1 - probability, degrees) == pytest.approx(-quantile, rel=2e-7, abs=1e-10)


def test_student_t_endpoints_and_invalid_arguments():
    assert student_t_quantile(0, 4) == -inf
    assert student_t_quantile(1, 4) == inf
    assert student_t_cdf(-inf, 4) == 0
    assert student_t_cdf(inf, 4) == 1
    for invalid in (-1, 2, float("nan")):
        with pytest.raises(ValueError):
            student_t_quantile(invalid, 4)
    for invalid in (0, -1, inf, float("nan")):
        with pytest.raises(ValueError):
            student_t_cdf(0, invalid)


def test_graph_spec_copies_geometry_without_execution_truth():
    graph = Graph("prefill", 66, prefill_capacity=4096)
    spec = GraphSpec.from_graph(graph)
    assert spec == GraphSpec("prefill", prefill_capacity=4096)
    assert "duration_ms" not in spec.__dataclass_fields__
    assert spec.family == "prefill"
    assert GraphSpec("decode", decode_capacity=4, width=8).family == "decode"
    assert GraphSpec("mixed", 512, 4, 8).family == "mixed"
    with pytest.raises(FrozenInstanceError):
        spec.width = 2


@pytest.mark.parametrize("gpu,handoff", [(None, 0), (0, 0), (-1, 0), (float("nan"), 0),
                                        (1, None), (1, -1), (1, inf)])
def test_missing_timing_preserves_execution_elapsed_time(gpu, handoff):
    observation = TimingObservation("graph", 20, 50, gpu, handoff)
    assert observation.elapsed_ms == 30
    assert not observation.valid
    assert observation.total_ms is None


def test_timing_observation_separates_wall_and_cost_samples():
    observation = TimingObservation("graph", 20, 50, 10, 2)
    assert observation.valid
    assert observation.total_ms == 12
    assert observation.elapsed_ms == 30
    assert TimingObservation("historical", -30, -1, 10).elapsed_ms == 29


def test_partial_blocks_update_raw_mean_without_fake_posterior_observations():
    estimate = TimingEstimate(prior_location_ms=2)
    prior = estimate.interval()
    for _ in range(7):
        assert estimate.observe(10)
    assert estimate.mean_ms == 10
    assert estimate.count == 7
    assert estimate.block_count == 0
    assert estimate.interval() == prior
    estimate.observe(10)
    assert estimate.block_count == 1
    assert estimate.interval() != prior


def test_prior_and_block_configuration_are_frozen_after_initialization():
    estimate = TimingEstimate(prior_location_ms=7, block_size=32, mode="regime")
    for name, value in (("prior_location_ms", 1), ("block_size", 8), ("mode", "cumulative"),
                        ("mean_estimator", "ema"), ("partial_blocks", True)):
        with pytest.raises(AttributeError):
            setattr(estimate, name, value)
    assert estimate.prior_location_ms == 7
    assert estimate.block_size == 32
    assert estimate.mode == "regime"
    assert estimate.mean_estimator == "arithmetic"


def test_nig_posterior_uses_block_means_and_observed_dispatch_mean():
    estimate = TimingEstimate(prior_location_ms=2)
    for value in [2] * 8 + [4] * 8:
        estimate.observe(value)
    snapshot = estimate.snapshot()
    assert estimate.mean_ms == 3
    assert snapshot["raw"]["variance_ms2"] == pytest.approx(16 / 15)
    assert snapshot["posterior"]["degrees"] == 6
    assert snapshot["posterior"]["location_ms"] == pytest.approx(26 / 9)
    assert snapshot["posterior"]["scale_ms"] == pytest.approx(sqrt(184 / 243))


def test_truncated_prior_sampling_has_independent_probability_and_mean():
    estimate = TimingEstimate()
    rng = Random(927)
    draws = [estimate.sample_mean(rng) for _ in range(30_000)]
    assert min(draws) > 0
    assert sum(value <= 2 for value in draws) / len(draws) == pytest.approx(13 / 20, abs=0.01)
    assert mean(draws) == pytest.approx(1.8, abs=0.06)
    assert estimate.quantile(0.65) == pytest.approx(2, abs=1e-11)


@pytest.mark.parametrize("block_size", [8, 32])
def test_a_thousand_samples_keep_counts_and_narrow_mean_uncertainty(block_size):
    estimate = TimingEstimate(prior_location_ms=10, block_size=block_size)
    for _ in range(16):
        estimate.observe(10)
    short = estimate.interval()
    for _ in range(984):
        estimate.observe(10)
    long = estimate.interval()
    assert long[0] < 10 < long[1]
    assert long[1] - long[0] < (short[1] - short[0]) / 4
    assert estimate.count == estimate.active_count == 1000
    assert estimate.block_count == 1000 // block_size
    before = estimate.snapshot()
    estimate.quantile(0.01)
    estimate.sample_mean(Random(0))
    assert estimate.snapshot() == before
    observation = TimingObservation("graph", 3_600_000, 3_600_010, 10)
    estimate.observe(observation.total_ms, gpu_ms=observation.gpu_ms, handoff_ms=observation.handoff_ms)
    assert estimate.count == 1001


def test_posterior_mean_draws_do_not_sample_persistent_execution_jitter():
    estimate = TimingEstimate(prior_location_ms=10)
    rng = Random(100)
    for _ in range(1000):
        estimate.observe(10 + rng.uniform(-2, 2))
    means = [estimate.sample_mean(rng) for _ in range(4000)]
    predictions = [estimate.sample_predictive(rng) for _ in range(4000)]
    assert variance(predictions) > 30 * variance(means)


@pytest.mark.parametrize("block_size", [8, 32])
def test_predictive_draws_recover_individual_execution_variability(block_size):
    estimate = TimingEstimate(prior_location_ms=1, block_size=block_size)
    rng = Random(901)
    timings = [10 + rng.choice((-4, 4)) for _ in range(32_768)]
    for timing in timings:
        estimate.observe(timing)
    block_means = [mean(timings[start:start + block_size])
                   for start in range(0, len(timings), block_size)]
    predictions = [estimate.sample_predictive(rng) for _ in range(20_000)]
    # Positive truncation removes a small lower tail near zero.
    assert variance(predictions) == pytest.approx(variance(timings), rel=0.2)
    assert variance(predictions) > block_size * variance(block_means) / 2
    posterior = estimate.snapshot()["posterior"]
    assert posterior["raw_predictive_scale_ms"] > 2 * posterior["block_predictive_scale_ms"]


@pytest.mark.parametrize("mean_estimator", ["arithmetic", "ema"])
def test_invalid_samples_are_transactional_and_component_counts_are_observations(mean_estimator):
    estimate = TimingEstimate(mean_estimator=mean_estimator)
    assert estimate.observe(12, gpu_ms=10, handoff_ms=2)
    assert estimate.observe(10, gpu_ms=10)
    before = estimate.snapshot()
    for total, gpu, handoff in [(None, None, None), (0, None, None), (-1, None, None), (inf, None, None),
                                (float("nan"), None, None), (1, 0, 1), (10, 9, 2),
                                (10, None, 0), (10, 11, -1)]:
        assert not estimate.observe(total, gpu_ms=gpu, handoff_ms=handoff)
        assert estimate.snapshot() == before
    assert before["gpu"]["count"] == 2
    assert before["gpu"]["mean_ms"] == 10
    assert before["handoff"]["count"] == 1
    assert before["handoff"]["mean_ms"] == 2


@pytest.mark.parametrize("block_size", [8, 32])
@pytest.mark.parametrize("before,after", [(40, 1), (10, 100)])
def test_confirmed_regime_retains_lifetime_data_and_seeds_only_triggering_blocks(block_size, before, after):
    estimate = TimingEstimate(prior_location_ms=before, block_size=block_size, mode="regime")
    for _ in range(32 * block_size):
        estimate.observe(before, gpu_ms=before * 0.8, handoff_ms=before * 0.2)
    for _ in range(2 * block_size):
        estimate.observe(after, gpu_ms=after * 0.8, handoff_ms=after * 0.2)
    assert estimate.regime_changes == 0
    for _ in range(block_size):
        estimate.observe(after, gpu_ms=after * 0.8, handoff_ms=after * 0.2)
    assert estimate.regime_changes == 1
    assert estimate.count == 35 * block_size
    assert estimate.active_count == 3 * block_size
    assert estimate.block_count == 3
    assert estimate.mean_ms == after
    snapshot = estimate.snapshot()
    assert snapshot["lifetime_mean_ms"] == pytest.approx((32 * before + 3 * after) / 35)
    assert snapshot["gpu"]["count"] == estimate.count
    assert snapshot["active_gpu"]["count"] == estimate.active_count
    assert snapshot["active_handoff"]["mean_ms"] == pytest.approx(after * 0.2)


def test_cumulative_mode_retains_a_mature_changed_distribution_without_resets():
    estimate = TimingEstimate(prior_location_ms=40)
    for value in [40] * 1000 + [1] * 24:
        estimate.observe(value)
    assert estimate.count == estimate.active_count == 1024
    assert estimate.regime_changes == 0
    assert estimate.mean_ms == pytest.approx(40024 / 1024)


def test_regime_mode_adapts_after_a_thousand_samples_without_losing_their_moments():
    estimate = TimingEstimate(prior_location_ms=40, mode="regime")
    for value in [40] * 1000 + [1] * 24:
        estimate.observe(value)
    assert estimate.count == 1024
    assert estimate.active_count == 24
    assert estimate.mean_ms == 1
    assert estimate.regime_changes == 1
    assert estimate.snapshot()["lifetime_mean_ms"] == pytest.approx(40024 / 1024)


def test_frozen_reference_does_not_absorb_the_first_two_suspect_blocks():
    estimate = TimingEstimate(prior_location_ms=10, mode="regime")
    for value in [10] * 64 + [100] * 24:
        estimate.observe(value)
    assert estimate.regime_changes == 1
    assert estimate.active_count == 24
    assert estimate.mean_ms == 100


def test_drift_requires_completed_blocks_and_at_least_eight_reference_blocks():
    estimate = TimingEstimate(prior_location_ms=10, mode="regime")
    for value in [10] * 7 * 8 + [100] * 3 * 8:
        estimate.observe(value)
    assert estimate.regime_changes == 0
    assert estimate.count == 80


def test_a_broad_early_predictive_interval_can_hide_a_downshift():
    estimate = TimingEstimate(prior_location_ms=40, mode="regime")
    for _ in range(8 * 8):
        estimate.observe(40)
    posterior = estimate.snapshot()["posterior"]
    lower = (posterior["location_ms"] - student_t_quantile(0.995, posterior["degrees"])
             * posterior["block_predictive_scale_ms"])
    assert lower < 1
    for _ in range(3 * 8):
        estimate.observe(1)
    assert estimate.regime_changes == 0


def test_an_isolated_outlier_is_not_a_regime_change():
    estimate = TimingEstimate(prior_location_ms=10, mode="regime")
    for value in [10] * 1000 + [10_000] + [10] * 1000:
        estimate.observe(value)
    assert estimate.regime_changes == 0
    assert estimate.count == estimate.active_count == 2001


@pytest.mark.parametrize("kwargs", [{"prior_location_ms": 0}, {"prior_location_ms": inf},
                                    {"block_size": 0}, {"block_size": 1.5}, {"mode": "ttl"},
                                    {"mean_estimator": "ttl"}, {"mean_estimator": None},
                                    {"mean_estimator": True}])
def test_invalid_estimator_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TimingEstimate(**kwargs)


@pytest.mark.parametrize("correlation", [0., .1, .8, .95, .99])
@pytest.mark.parametrize("count", [1, 2, 4, 8, 64, 1000, 40_000])
def test_finite_correlation_factor_matches_direct_covariance_sum(count, correlation):
    expected = 1 + 2 * sum((1 - lag / count) * correlation ** lag for lag in range(1, count))
    assert _correlation_factor(count, correlation) == pytest.approx(expected, rel=2e-12)


@pytest.mark.parametrize("block_size", [8, 32])
def test_log_detector_retains_mature_arithmetic_mean_and_posterior(block_size):
    estimates = [TimingEstimate(prior_location_ms=10, block_size=block_size, mode=mode)
                 for mode in ("cumulative", "regime", "log-regime")]
    for value in [9., 11.] * 500:
        for estimate in estimates:
            estimate.observe(value, gpu_ms=value * .8, handoff_ms=value * .2)
    reference = estimates[0].snapshot()
    for estimate in estimates[1:]:
        snapshot = estimate.snapshot()
        assert estimate.count == estimate.active_count == 1000
        assert estimate.mean_ms == 10
        assert snapshot["posterior"] == reference["posterior"]
        assert snapshot["raw"] == reference["raw"]
        assert snapshot["gpu"] == reference["gpu"]
        assert snapshot["handoff"] == reference["handoff"]
        assert estimate.interval() == estimates[0].interval()
        assert estimate.sample_mean(Random(21)) == estimates[0].sample_mean(Random(21))
        assert not estimate.pending_change
        assert estimate.regime_changes == 0


@pytest.mark.parametrize("block_size", [8, 32])
@pytest.mark.parametrize("before,after", [(20., 5.), (5., 20.)])
def test_log_detector_flags_first_changed_raw_sample_and_confirms_after_eight(block_size, before, after):
    estimate = TimingEstimate(prior_location_ms=before, block_size=block_size, mode="log-regime")
    for i in range(1000):
        estimate.observe(before * (1.01 if i % 2 else .99))
    before_snapshot = estimate.snapshot()
    estimate.observe(after, gpu_ms=after * .8, handoff_ms=after * .2)
    assert estimate.pending_change
    assert estimate.pending_count == 1
    assert estimate.suspect_direction == (1 if after > before else -1)
    assert estimate.change_strength > 1
    reference = estimate.snapshot()["change_evidence"]
    assert reference["reference_count"] == 1000
    for _ in range(6):
        estimate.observe(after, gpu_ms=after * .8, handoff_ms=after * .2)
        assert estimate.pending_change
        assert estimate.regime_changes == 0
        assert estimate.snapshot()["change_evidence"]["reference_count"] == 1000
    assert estimate.pending_count == 7
    assert estimate.suspect_blocks == 1
    estimate.observe(after, gpu_ms=after * .8, handoff_ms=after * .2)
    assert estimate.regime_changes == 1
    assert estimate.count == 1008
    assert estimate.active_count == 8
    assert estimate.mean_ms == after
    assert estimate.block_count == 8 // block_size
    assert estimate.snapshot()["partial_count"] == 8 % block_size
    assert estimate.snapshot()["raw"]["mean_ms"] == pytest.approx((1000 * before + 8 * after) / 1008)
    assert estimate.snapshot()["active_gpu"]["count"] == 8
    assert estimate.snapshot()["active_handoff"]["mean_ms"] == pytest.approx(after * .2)
    assert not estimate.pending_change
    assert estimate.pending_count == 0
    assert before_snapshot["count"] == 1000


def test_log_detector_requires_mature_history_and_two_consistent_groups():
    estimate = TimingEstimate(prior_location_ms=10, mode="log-regime")
    for value in [10.] * 63 + [1000.]:
        estimate.observe(value)
    assert not estimate.pending_change
    assert estimate.regime_changes == 0
    estimate = TimingEstimate(prior_location_ms=10, mode="log-regime")
    for value in [10.] * 1000 + [40.] * 4:
        estimate.observe(value)
    assert estimate.pending_change
    assert estimate.suspect_blocks == 1
    for _ in range(4):
        estimate.observe(2.5)
    assert estimate.regime_changes == 0
    assert not estimate.pending_change
    assert estimate.active_count == estimate.count == 1008


def test_log_detector_ignores_a_single_extreme_and_preserves_invalid_sample_state():
    estimate = TimingEstimate(mode="log-regime")
    for value in [3.] * 1000 + [3000.] + [3.] * 3:
        estimate.observe(value)
    assert estimate.pending_change
    before = estimate.snapshot()
    assert not estimate.observe(None)
    assert not estimate.observe(0)
    assert not estimate.observe(3, gpu_ms=0)
    assert estimate.snapshot() == before
    for _ in range(4):
        estimate.observe(3.)
    assert not estimate.pending_change
    assert estimate.regime_changes == 0
    assert estimate.count == 1008


@pytest.mark.parametrize("trend,expected_rho,expected_degrees", [
    (False, 2 / sqrt(61), 31), (True, .99, 1),
])
def test_log_reference_correlation_cap_and_effective_degree_floor(trend, expected_rho, expected_degrees):
    estimate = TimingEstimate(mode="log-regime")
    for i in range(64):
        estimate.observe(exp(i / 100) if trend else 1.)
    estimate.observe(2.)
    evidence = estimate.snapshot()["change_evidence"]
    assert evidence["reference_correlation"] == pytest.approx(expected_rho)
    assert evidence["reference_degrees"] == expected_degrees


def test_log_confirmation_preserves_independent_lifetime_block_partition():
    estimate = TimingEstimate(mode="log-regime")
    for value in [10.] * 68 + [2.5] * 8 + [2.5] * 8:
        estimate.observe(value)
    snapshot = estimate.snapshot()
    assert estimate.regime_changes == 1
    assert snapshot["count"] == 84
    assert snapshot["active_count"] == 16
    assert snapshot["lifetime_block_count"] == 10
    assert snapshot["block_count"] == 2


def test_log_confirmation_allowance_survives_regime_changes():
    estimate = TimingEstimate(mode="log-regime")
    for value in [20.] * 1000:
        estimate.observe(value)
    estimate.observe(5.)
    first = estimate.snapshot()["change_evidence"]
    first_look = first["lifetime_looks"]
    assert first_look > 1
    assert first["confirmation_level"] <= 1e-4 / (first_look * (first_look + 1))
    for value in [5.] * 7:
        estimate.observe(value)
    assert estimate.regime_changes == 1
    assert estimate.snapshot()["change_evidence"]["lifetime_looks"] == first_look
    for value in [5.] * 56:
        estimate.observe(value)
    estimate.observe(20.)
    second = estimate.snapshot()["change_evidence"]
    assert second["lifetime_looks"] == first_look + 1
    assert second["confirmation_level"] <= first["confirmation_level"]
    for value in [20.] * 7:
        estimate.observe(value)
    assert estimate.regime_changes == 2
    assert estimate.count == 1072
    assert estimate.active_count == 8


@pytest.mark.parametrize("rho", [0., .8, .95])
@pytest.mark.parametrize("mean_estimator", ["arithmetic", "ema"])
def test_log_confirmation_does_not_reset_a_long_stationary_correlated_trace(rho, mean_estimator):
    rng = Random("log-regime-development:15")
    sigma = sqrt(log1p(.5 ** 2))
    innovation_scale = sqrt(1 - rho * rho)
    z = rng.gauss(0, 1)
    estimate = TimingEstimate(mode="log-regime", mean_estimator=mean_estimator)
    for i in range(40_000):
        if i:
            z = rho * z + innovation_scale * rng.gauss(0, 1)
        estimate.observe(3 * exp(sigma * z - sigma * sigma / 2))
    assert estimate.regime_changes == 0
    assert estimate.count == estimate.active_count == 40_000


@pytest.mark.parametrize("before,after", [(20., 5.), (5., 20.)])
@pytest.mark.parametrize("seed", range(16))
def test_log_confirmation_detects_mature_fourfold_change_in_eight_noisy_samples(seed, before, after):
    rng = Random(f"log-regime-shift-development:{seed}")
    sigma = sqrt(log1p(.05 ** 2))
    estimate = TimingEstimate(prior_location_ms=before, mode="log-regime")
    for _ in range(1000):
        estimate.observe(before * exp(sigma * rng.gauss(0, 1) - sigma * sigma / 2))
    assert not estimate.regime_changes
    estimate.observe(after * exp(sigma * rng.gauss(0, 1) - sigma * sigma / 2))
    assert estimate.pending_change
    for _ in range(7):
        estimate.observe(after * exp(sigma * rng.gauss(0, 1) - sigma * sigma / 2))
    assert estimate.regime_changes == 1
    assert estimate.active_count == 8
    assert estimate.count == 1008


def test_operational_ema_uses_observed_total_with_one_eighth_gain():
    estimate = TimingEstimate(mean_estimator="ema")
    assert estimate.mean_ms is estimate.arithmetic_mean_ms is estimate.ema_ms is None
    assert estimate.observe(32, gpu_ms=24, handoff_ms=8)
    assert estimate.mean_ms == estimate.ema_ms == 32
    assert estimate.observe(16, gpu_ms=12, handoff_ms=4)
    assert estimate.mean_ms == estimate.ema_ms == 30
    assert estimate.arithmetic_mean_ms == 24
    assert estimate.observe(48, gpu_ms=36, handoff_ms=12)
    assert estimate.mean_ms == estimate.ema_ms == 32.25
    assert estimate.arithmetic_mean_ms == 32
    assert estimate.count == estimate.active_count == 3
    assert estimate.block_count == 0
    snapshot = estimate.snapshot()
    assert snapshot["mean_estimator"] == "ema"
    assert snapshot["mean_ms"] == snapshot["ema_ms"] == 32.25
    assert snapshot["arithmetic_mean_ms"] == snapshot["active_raw"]["mean_ms"] == 32


@pytest.mark.parametrize("mode", ["cumulative", "regime", "log-regime"])
def test_arithmetic_defaults_keep_the_snapshot_shape(mode):
    default = TimingEstimate(mode=mode)
    explicit = TimingEstimate(mode=mode, mean_estimator="arithmetic")
    for value in [9., 11.] * 32:
        default.observe(value)
        explicit.observe(value)
    assert default.mean_ms == explicit.mean_ms == 10
    assert default.snapshot() == explicit.snapshot()
    assert not {"mean_estimator", "arithmetic_mean_ms", "ema_ms"} & default.snapshot().keys()


@pytest.mark.parametrize("block_size", [8, 32])
def test_operational_ema_retains_a_thousand_arithmetic_samples_and_their_posterior(block_size):
    arithmetic = TimingEstimate(prior_location_ms=10, block_size=block_size, mode="log-regime")
    ema = TimingEstimate(prior_location_ms=10, block_size=block_size, mode="log-regime",
                         mean_estimator="ema")
    for value in [9., 11.] * 500:
        for estimate in (arithmetic, ema):
            estimate.observe(value, gpu_ms=value * .8, handoff_ms=value * .2)
    assert arithmetic.count == ema.count == ema.active_count == 1000
    assert ema.arithmetic_mean_ms == arithmetic.mean_ms == 10
    assert ema.mean_ms != ema.arithmetic_mean_ms
    assert ema.interval() == arithmetic.interval()
    assert ema.sample_mean(Random(21)) == arithmetic.sample_mean(Random(21))
    assert ema.sample_predictive(Random(22)) == arithmetic.sample_predictive(Random(22))
    snapshot = ema.snapshot()
    for key in ("raw", "active_raw", "gpu", "handoff", "active_gpu", "active_handoff", "posterior"):
        assert snapshot[key] == arithmetic.snapshot()[key]
    before = ema.snapshot()
    resumed = TimingObservation("graph", 3_600_000, 3_600_100, 8, 2)
    assert ema.interval() == arithmetic.interval()
    assert ema.snapshot() == before
    ema.observe(resumed.total_ms, gpu_ms=resumed.gpu_ms, handoff_ms=resumed.handoff_ms)
    assert ema.count == 1001
    assert ema.mean_ms == pytest.approx(before["ema_ms"] + (10 - before["ema_ms"]) / 8)
    assert ema.arithmetic_mean_ms == 10


@pytest.mark.parametrize("mode", ["cumulative", "regime", "log-regime"])
@pytest.mark.parametrize("block_size", [8, 32])
def test_operational_ema_keeps_each_arithmetic_detector_decision_identical(mode, block_size):
    arithmetic = TimingEstimate(prior_location_ms=10, block_size=block_size, mode=mode)
    ema = TimingEstimate(prior_location_ms=10, block_size=block_size, mode=mode, mean_estimator="ema")
    values = [10.] * 1000 + [40.] * 96 + [10.] * 96 + [4000.] + [10.] * 32
    for value in values:
        for estimate in (arithmetic, ema):
            estimate.observe(value, gpu_ms=value * .8, handoff_ms=value * .2)
        assert ema.arithmetic_mean_ms == arithmetic.mean_ms
        assert (ema.regime_changes, ema.pending_change, ema.pending_count, ema.suspect_direction,
                ema.active_count, ema.count) == (
                    arithmetic.regime_changes, arithmetic.pending_change, arithmetic.pending_count,
                    arithmetic.suspect_direction, arithmetic.active_count, arithmetic.count)
    snapshot = ema.snapshot()
    for key in ("mean_estimator", "arithmetic_mean_ms", "ema_ms"):
        snapshot.pop(key)
    snapshot["mean_ms"] = ema.arithmetic_mean_ms
    assert snapshot == arithmetic.snapshot()
    assert ema.interval() == arithmetic.interval()
    if mode == "cumulative":
        assert ema.regime_changes == 0
    else:
        assert ema.regime_changes >= 1


@pytest.mark.parametrize("mode,changed_samples", [("regime", 24), ("log-regime", 8)])
def test_operational_ema_survives_a_confirmed_regime_change(mode, changed_samples):
    estimate = TimingEstimate(prior_location_ms=20, mode=mode, mean_estimator="ema")
    for value in [20.] * 1000 + [5.] * changed_samples:
        estimate.observe(value)
    assert estimate.regime_changes == 1
    assert estimate.count == 1000 + changed_samples
    assert estimate.active_count == changed_samples
    assert estimate.arithmetic_mean_ms == 5
    assert estimate.mean_ms == pytest.approx(5 + 15 * (7 / 8) ** changed_samples)
    assert estimate.mean_ms > estimate.arithmetic_mean_ms


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", []])
def test_partial_block_configuration_requires_a_bool(invalid):
    with pytest.raises(ValueError, match="partial_blocks"):
        TimingEstimate(partial_blocks=invalid)


def test_complete_block_conditioning_remains_the_default_control():
    default = TimingEstimate()
    explicit = TimingEstimate(partial_blocks=False)
    assert not default.partial_blocks
    for value in [9., 11.] * 500 + [40.]:
        for estimate in (default, explicit):
            estimate.observe(value)
        assert default.snapshot() == explicit.snapshot()
    assert "partial_blocks" not in default.snapshot()


def test_partial_block_conditioning_moves_a_mismatched_prior_before_eight_observations():
    estimate = TimingEstimate(prior_location_ms=1, partial_blocks=True)
    previous = estimate.interval()
    for count in range(1, 7):
        estimate.observe(40)
        assert estimate.interval() != previous
        previous = estimate.interval()
        assert estimate.count == estimate.active_count == count
        assert estimate.block_count == 0
        assert estimate.snapshot()["posterior"]["degrees"] == 4 + count / 8
    posterior = estimate.snapshot()["posterior"]
    assert posterior["location_ms"] == 121 / 4
    assert posterior["scale_ms"] == pytest.approx(sqrt(4595 / 76))
    assert estimate.quantile(.05) == pytest.approx(15.109491270121088)
    assert estimate.snapshot()["partial_blocks"] is True
    assert estimate.mean_ms == estimate.arithmetic_mean_ms == 40


def test_partial_replacement_does_not_accumulate_overlapping_means():
    estimate = TimingEstimate(prior_location_ms=2, block_size=4, partial_blocks=True)
    for value in (2., 4., 6.):
        estimate.observe(value)
        estimate.interval()
    posterior = estimate.snapshot()["posterior"]
    # One partial mean of 4 has weight 3/4, rather than three overlapping means.
    assert posterior["degrees"] == 19 / 4
    assert posterior["location_ms"] == 7 / 2
    assert posterior["scale_ms"] == pytest.approx(sqrt(35 / 19))
    assert estimate.sample_mean(Random(1)) != TimingEstimate(prior_location_ms=2).sample_mean(Random(1))


@pytest.mark.parametrize("block_size", [8, 32])
def test_partial_promotion_matches_complete_block_likelihood_without_duplicate_evidence(block_size):
    complete = TimingEstimate(prior_location_ms=2, block_size=block_size)
    partial = TimingEstimate(prior_location_ms=2, block_size=block_size, partial_blocks=True)
    for index, value in enumerate(([40.] + [2.] * (block_size - 1) + [3., 5.] * block_size), 1):
        for estimate in (complete, partial):
            estimate.observe(value, gpu_ms=value * .75, handoff_ms=value * .25)
            assert all(0 < bound < inf for bound in estimate.interval())
        if index % block_size == 0:
            assert partial.snapshot()["posterior"] == complete.snapshot()["posterior"]
            assert partial.interval() == complete.interval()
            assert partial.sample_mean(Random(7)) == complete.sample_mean(Random(7))
            assert partial.sample_predictive(Random(8)) == complete.sample_predictive(Random(8))
            assert partial.count == complete.count == index
            assert partial.block_count == complete.block_count == index // block_size


@pytest.mark.parametrize("block_size", [8, 32])
def test_partial_conditioning_retains_a_thousand_raw_observations(block_size):
    estimate = TimingEstimate(prior_location_ms=10, block_size=block_size, partial_blocks=True)
    for count in range(1, 1001):
        estimate.observe(10, gpu_ms=8, handoff_ms=2)
        if count == 16:
            short = estimate.interval()
    before = estimate.snapshot()
    mature = estimate.interval()
    assert before["posterior"]["degrees"] == 4 + 1000 / block_size
    estimate.observe(40, gpu_ms=32, handoff_ms=8)
    after = estimate.snapshot()
    assert estimate.count == estimate.active_count == 1001
    assert estimate.block_count == 1001 // block_size
    assert estimate.arithmetic_mean_ms == pytest.approx(10040 / 1001)
    assert after["raw"]["count"] == after["gpu"]["count"] == after["handoff"]["count"] == 1001
    assert after["posterior"]["degrees"] == 4 + 1001 / block_size
    assert after["posterior"] != before["posterior"]
    assert mature[1] - mature[0] < (short[1] - short[0]) / 4


@pytest.mark.parametrize("invalid,components", [
    (None, {}), (0, {}), (-1, {}), (float("nan"), {}), (inf, {}),
    (10, {"gpu_ms": 0}), (10, {"gpu_ms": 9, "handoff_ms": 2}),
])
def test_invalid_timing_preserves_partial_evidence_and_cached_quantiles(invalid, components):
    estimate = TimingEstimate(partial_blocks=True)
    estimate.observe(40)
    before = estimate.snapshot()
    quantile = estimate.quantile(.05)
    posterior = estimate._posterior_cache
    cached_quantiles = dict(estimate._quantiles)
    assert not estimate.observe(invalid, **components)
    assert estimate.snapshot() == before
    assert estimate._posterior_cache is posterior
    assert estimate._quantiles == cached_quantiles
    assert estimate.quantile(.05) == quantile


@pytest.mark.parametrize("mode", ["cumulative", "regime", "log-regime"])
@pytest.mark.parametrize("block_size", [8, 32])
def test_partial_caches_do_not_contaminate_completed_block_detector_references(mode, block_size):
    complete = TimingEstimate(prior_location_ms=10, block_size=block_size, mode=mode)
    partial = TimingEstimate(prior_location_ms=10, block_size=block_size, mode=mode, partial_blocks=True)
    values = [10.] * 1000 + [40.] * 96 + [10.] * 96 + [4000.] + [10.] * 32
    for value in values:
        for estimate in (complete, partial):
            estimate.observe(value, gpu_ms=value * .8, handoff_ms=value * .2)
            estimate.quantile(.05)
            estimate.interval()
        assert partial._reference == complete._reference
        assert partial._log_reference == complete._log_reference
        assert (partial.regime_changes, partial.pending_change, partial.pending_count,
                partial.suspect_direction, partial.active_count, partial.count) == (
                    complete.regime_changes, complete.pending_change, complete.pending_count,
                    complete.suspect_direction, complete.active_count, complete.count)
    for key in ("raw", "active_raw", "gpu", "handoff", "active_gpu", "active_handoff"):
        assert partial.snapshot()[key] == complete.snapshot()[key]
    if mode != "cumulative":
        assert partial.regime_changes >= 1
