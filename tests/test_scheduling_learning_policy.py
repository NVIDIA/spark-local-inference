# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from itertools import combinations
from math import exp, log
from random import Random

import pytest

from scripts.scheduling_sim.learning_policy import LearningScheduler, PolicyConfig
from scripts.scheduling_sim.model import Action, Graph, Lane, Value, View, expected_output
from scripts.scheduling_sim.timing import GraphSpec, TimingEstimate, TimingObservation
from scripts.scheduling_sim.utility import UtilityConfig, UtilityScheduler


PREFILLS = (GraphSpec("small", prefill_capacity=64), GraphSpec("large", prefill_capacity=128))
UTILITY = UtilityConfig(16, 4, 100, 0.1, "service")


def scheduler(policy="greedy", catalog=PREFILLS, **kwargs):
    return LearningScheduler(Value(64), UTILITY, catalog, PolicyConfig(policy=policy, **kwargs), seed=31)


def view(name="p", remaining=1024):
    return View(name, remaining, 4096, (1.0,) * 7)


def seed(s, name, duration, end=0.0, count=1):
    for _ in range(count):
        s.seed_observation(TimingObservation(name, end - duration, end, duration, 0))


def finish(s, decision, duration, operation=1, valid=True):
    s.commit(decision, operation)
    start = decision.metadata["now_ms"]
    observation = TimingObservation(decision.action.graph.name, start, start + duration,
                                    duration if valid else None, 0)
    committed = [{"request": lane.name, "prefill": lane.prefill,
                  "emitted": int(lane.expected_output > 0) if lane.prefill else 1}
                 for lane in decision.action.lanes]
    s.complete(operation, observation, committed)
    return start + duration


@pytest.mark.parametrize("policy", ["ttl", "jitter", "ema", "greedy", "epsilon", "softmax",
                                    "optimism", "thompson", "predictive", "retained-refresh"])
def test_cold_policy_bootstraps_useful_work_without_fairness_or_credit(policy):
    s = scheduler(policy)
    decision = s.choose([view()], 0)
    assert decision.category == "bootstrap"
    assert decision.action.graph.name == "large"
    assert decision.estimate_ms is None
    assert decision.metadata["score_kind"] == "unknown_value"
    assert s.utility.prices["p"].tokens is None
    finish(s, decision, 10)
    assert s.snapshot()["bootstrap_ms"] == 10
    assert s.snapshot()["observation_count"] == 1
    assert s.credit_ms == 0


def test_cold_padding_uses_smallest_shape_for_equal_useful_work():
    s = scheduler(catalog=(*PREFILLS, GraphSpec("padded", prefill_capacity=512)))
    decision = s.choose([view(remaining=17)], 0)
    assert decision.action.graph.name == "small"
    assert decision.action.lanes[0].prefill == 17


def test_no_measured_phase_can_bootstrap_after_another_phase():
    s = scheduler(catalog=(*PREFILLS, GraphSpec("decode", decode_capacity=1)))
    seed(s, "large", 10)
    decision = s.choose([view(remaining=0)], 0)
    assert decision.category == "bootstrap"
    assert decision.action.graph.name == "decode"


def test_unseen_prompt_competing_with_measured_decode_is_not_free_bootstrap():
    s = scheduler("epsilon", catalog=(*PREFILLS, GraphSpec("decode", decode_capacity=1)), epsilon=1)
    seed(s, "decode", 10)
    decision = s.choose([view(), view("d", 0)], 0)
    assert decision.category == "ordinary"
    assert decision.action.graph.name == "decode"


@pytest.mark.parametrize("policy", ["ema", "greedy"])
def test_retained_greedy_keeps_cost_after_idle_without_exploring(policy):
    s = scheduler(policy)
    seed(s, "large", 10)
    decision = s.choose([view()], 60_000)
    assert decision.category == "ordinary"
    assert decision.action.graph.name == "large"
    assert decision.estimate_ms == pytest.approx(10)


def test_ttl_strict_boundary_replaces_expired_ema():
    s = scheduler("ttl")
    seed(s, "large", 8)
    seed(s, "large", 16, end=1)
    assert s._means(1)["large"] == pytest.approx(9)
    assert s._means(30_000)["large"] == pytest.approx(9)
    assert s._means(30_001)["large"] is None
    decision = s.choose([view()], 30_001)
    assert decision.category == "probe"
    finish(s, decision, 2)
    assert s._means(30_003)["large"] == 2


def test_unobserved_ttl_shape_outranks_the_measured_winner():
    s = scheduler("ttl")
    seed(s, "large", 1)
    decision = s.choose([view()], 1)
    assert decision.action.graph.name == "small"
    assert decision.category == "probe"
    assert decision.estimate_ms is None


def test_jitter_is_seeded_and_inside_the_declared_interval():
    first, second = scheduler("jitter"), scheduler("jitter")
    for s in (first, second):
        seed(s, "large", 10)
    assert first._expires == second._expires
    assert 24_000 <= first._expires["large"] <= 36_000


@pytest.mark.parametrize("policy", ["epsilon", "softmax", "optimism", "thompson", "predictive"])
def test_funded_policies_need_full_credit_and_bootstrap_does_not_earn_it(policy):
    s = scheduler(policy, epsilon=1)
    seed(s, "large", 10)
    seed(s, "small", 1000)
    s.credit_ms = 4.999
    decision = s.choose([view()], 0)
    assert decision.action.graph.name == "large"
    assert not decision.metadata["opportunity"]
    finish(s, decision, 1)
    assert s.credit_ms == s.config.credit_cap_ms
    next_decision = s.choose([view()], 1)
    assert next_decision.metadata["opportunity"]


def test_full_probe_cost_creates_debt_without_earning_probe_credit():
    s = scheduler("epsilon", epsilon=1, refresh_every=0)
    seed(s, "large", 10)
    seed(s, "small", 1000)
    now = finish(s, s.choose([view()], 0), 245)
    assert s.credit_ms == pytest.approx(5)
    decision = s.choose([view()], now)
    assert decision.category == "probe"
    assert decision.action.graph.name == "small"
    now = finish(s, decision, 100, operation=2)
    assert s.credit_ms == pytest.approx(-95)
    assert s.probe_ms == 100
    assert s.ordinary_ms == 245
    assert s.choose([view()], now).category == "ordinary"


def test_completed_probe_time_obeys_the_foreground_allowance_with_one_overdraw():
    s = scheduler("epsilon", epsilon=1, refresh_every=0)
    seed(s, "large", 10)
    seed(s, "small", 1000)
    rng = Random(129)
    now, largest_probe = 0.0, 0.0
    for operation in range(1, 301):
        decision = s.choose([view()], now)
        if decision.category == "probe":
            duration = rng.uniform(100, 10_000)
            largest_probe = max(largest_probe, duration)
        else:
            duration = rng.uniform(1, 500)
        now = finish(s, decision, duration, operation)
        allowance = s.ordinary_ms * s.config.alpha / (1 - s.config.alpha)
        assert s.probe_ms <= allowance + largest_probe + 1e-9
        assert s.credit_ms >= -largest_probe - 1e-9
    assert s.probe_count > 0


def test_idle_discards_saved_credit_and_retains_negative_balance_and_counts():
    s = scheduler()
    seed(s, "large", 10, count=10)
    s.credit_ms = 5
    s.idle()
    assert s.credit_ms == 0
    s.credit_ms = -30
    s.idle()
    assert s.credit_ms == -30
    assert s.estimates["large"].count == 10


def test_invalid_ordinary_observation_earns_nothing_and_keeps_the_estimate():
    s = scheduler()
    seed(s, "large", 10)
    decision = s.choose([view()], 0)
    before = replace(s.utility.prices["p"].tokens)
    expected = replace(before)
    expected.observe(245, decision.action.lanes[0].prefill, UTILITY.adaptation_ms, history="service")
    finish(s, decision, 245, valid=False)
    assert s.credit_ms == 0
    assert s.estimates["large"].count == 1
    assert s.snapshot()["invalid_observations"] == 1
    after = s.utility.prices["p"].tokens
    assert after.log_fraction == pytest.approx(expected.log_fraction)
    assert after.log_fraction != before.log_fraction
    assert after.elapsed_ms == 245


@pytest.mark.parametrize("policy", ["greedy", "ttl"])
def test_invalid_gpu_sample_still_accounts_for_published_decode_service(policy):
    s = scheduler(policy, catalog=(GraphSpec("decode", decode_capacity=1),))
    seed(s, "decode", 10)
    decision = s.choose([view(remaining=0)], 0)
    expected_token = replace(s.utility.prices["p"].tokens)
    expected_delivery = replace(s.utility.prices["p"].deliveries)
    for price in (expected_token, expected_delivery):
        price.observe(100, 1, UTILITY.adaptation_ms, history="service")
    finish(s, decision, 100, valid=False)
    prices = s.utility.prices["p"]
    assert prices.tokens.log_fraction == pytest.approx(expected_token.log_fraction)
    assert prices.deliveries.log_fraction == pytest.approx(expected_delivery.log_fraction)
    assert prices.tokens.elapsed_ms == prices.deliveries.elapsed_ms == 100
    assert s.credit_ms == 0
    assert s.estimates["decode"].count == 1


def test_invalid_probe_still_pays_elapsed_cost():
    s = scheduler("epsilon", epsilon=1, funded=False, refresh_every=0)
    seed(s, "large", 10)
    seed(s, "small", 1000)
    decision = s.choose([view()], 0)
    assert decision.category == "probe"
    finish(s, decision, 100, valid=False)
    assert s.credit_ms == -100
    assert s.estimates["small"].count == 1


def test_retry_is_identical_and_opportunity_changes_only_at_commit():
    s = scheduler("thompson", funded=False, refresh_every=0)
    seed(s, "large", 10)
    seed(s, "small", 1000)
    first = s.choose([view()], 0)
    rng_state = s._rng.getstate()
    assert s.choose([view()], 0) is first
    assert s._rng.getstate() == rng_state
    assert s.opportunities == 0
    s.commit(first, 1)
    assert s.opportunities == 1
    with pytest.raises(ValueError, match="completion feedback"):
        s.choose([view()], 0)
    with pytest.raises(ValueError, match="uncommitted"):
        s.commit(first, 2)


def test_incumbent_nominations_still_advance_refresh_opportunities():
    s = scheduler("epsilon", epsilon=0, funded=False, refresh_every=2)
    seed(s, "large", 10)
    seed(s, "small", 1000)
    first = s.choose([view()], 40_000)
    assert first.category == "ordinary"
    now = finish(s, first, 10)
    second = s.choose([view()], now)
    assert second.reason == "refresh"
    assert second.category == "probe"
    assert second.action.graph.name == "small"


@pytest.mark.parametrize("nominee", ["large", "small"])
def test_epsilon_samples_all_physical_keys_and_does_not_anneal(monkeypatch, nominee):
    s = scheduler("epsilon", epsilon=0.05, funded=False, refresh_every=0, anneal=True)
    seed(s, "large", 10)
    seed(s, "small", 1000)
    s.opportunities = 99
    populations = []
    monkeypatch.setattr(s._rng, "random", lambda: 0.04)
    def choose(population):
        populations.append(tuple(population))
        return nominee
    monkeypatch.setattr(s._rng, "choice", choose)
    decision = s.choose([view()], 0)
    assert populations == [("large", "small")]
    assert decision.action.graph.name == nominee
    assert decision.category == ("ordinary" if nominee == "large" else "probe")


def test_uniform_epsilon_has_equal_support_including_the_incumbent():
    s = scheduler("epsilon", epsilon=1, funded=False, refresh_every=0)
    seed(s, "large", 10)
    seed(s, "small", 1000)
    counts = {"large": 0, "small": 0}
    now = 0.0
    for operation in range(1000):
        decision = s.choose([view()], now)
        name = decision.action.graph.name
        counts[name] += 1
        assert decision.category == ("ordinary" if name == "large" else "probe")
        now = finish(s, decision, 10 if name == "large" else 1000, operation)
    assert 420 <= counts["large"] <= 580
    assert sum(counts.values()) == 1000


@pytest.mark.parametrize("anneal, expected_temperature", [(False, 0.1), (True, 0.1 / (2 ** 0.5))])
def test_softmax_anneals_on_the_declared_hundred_opportunity_scale(monkeypatch, anneal, expected_temperature):
    s = scheduler("softmax", temperature=0.1, funded=False, refresh_every=0, anneal=anneal)
    seed(s, "large", 10)
    seed(s, "small", 1000)
    s.opportunities = 99
    weights_by_name = {}
    def choose(population, weights, k):
        weights_by_name.update(zip(population, weights))
        return ["large"]
    monkeypatch.setattr(s._rng, "choices", choose)
    decision = s.choose([view()], 0)
    assert decision.metadata["opportunity_index"] == 100
    assert weights_by_name["large"] == 1
    assert weights_by_name["small"] == pytest.approx(exp(-1 / expected_temperature))


def test_funded_refresh_samples_a_mature_observed_arm_without_a_ttl_wait():
    s = scheduler("epsilon", epsilon=0, refresh_every=16)
    seed(s, "large", 10, count=1000)
    seed(s, "small", 1000, count=1000)
    now = finish(s, s.choose([view()], 0), 245)
    assert s.credit_ms == s.config.credit_cap_ms
    for operation in range(2, 17):
        decision = s.choose([view()], now)
        assert decision.category == "ordinary"
        assert decision.metadata["opportunity"]
        now = finish(s, decision, 10, operation=operation)
    decision = s.choose([view()], now)
    assert now < s.config.ttl_ms
    assert decision.metadata["opportunity_index"] == 16
    assert decision.reason == "refresh"
    assert decision.category == "probe"
    assert decision.action.graph.name == "small"
    assert decision.estimate_ms == 1000


def test_all_cohorts_reuse_one_sample_and_freeze_observed_fairness(monkeypatch):
    s = scheduler("thompson", catalog=(GraphSpec("one", decode_capacity=1),
                                      GraphSpec("two", decode_capacity=2)), funded=False, refresh_every=0)
    seed(s, "one", 10)
    seed(s, "two", 15)
    calls = {}
    for name, estimate in s.estimates.items():
        def sample(rng, name=name):
            calls[name] = calls.get(name, 0) + 1
            return 0.01 if name == "one" else 1000
        monkeypatch.setattr(estimate, "sample_mean", sample)
    views = [view("d0", 0), view("d1", 0)]
    decision = s.choose(views, 0)
    assert calls == {"one": 1, "two": 1}
    assert s.utility.prices["d0"].tokens.maximum_rate == pytest.approx(0.1)
    assert s.utility.ages["d0"][1] == pytest.approx(15)
    assert decision.metadata["sampled_costs_ms"] == {"one": 0.01, "two": 1000}


def test_same_graph_nomination_executes_the_full_mean_incumbent(monkeypatch):
    s = scheduler("thompson", funded=False, refresh_every=0)
    seed(s, "large", 1)
    seed(s, "small", 1000)
    for name, estimate in s.estimates.items():
        monkeypatch.setattr(estimate, "sample_mean", lambda rng, name=name: 1 if name == "large" else 1000)
    decision = s.choose([view()], 0)
    assert decision.category == "ordinary"
    assert decision.reason == "incumbent"
    assert decision.action.graph.duration_ms == 1
    assert decision.action.graph.name == decision.metadata["incumbent_graph"]


def test_funded_discovery_can_measure_a_faster_unknown_padded_graph(monkeypatch):
    catalog = (GraphSpec("fitting", prefill_capacity=64), GraphSpec("padded", prefill_capacity=4096))
    s = scheduler("thompson", catalog=catalog, refresh_every=0)
    seed(s, "fitting", 5)
    first = s.choose([view(remaining=64)], 0)
    now = finish(s, first, 245)
    for name, estimate in s.estimates.items():
        monkeypatch.setattr(estimate, "sample_mean", lambda rng, name=name: 1 if name == "padded" else 5)
    decision = s.choose([view(remaining=64)], now)
    assert decision.category == "probe"
    assert decision.action.graph.name == "padded"
    assert decision.action.lanes[0].prefill == 64
    finish(s, decision, 1, operation=2)
    assert s.estimates["padded"].mean_ms == 1


def test_one_completion_and_operation_identifier_are_accepted_exactly_once():
    s = scheduler()
    seed(s, "large", 10)
    decision = s.choose([view()], 0)
    finish(s, decision, 10)
    with pytest.raises(ValueError, match="exactly once"):
        s.complete(1, TimingObservation("large", 0, 10, 10), [])
    next_decision = s.choose([view()], 10)
    with pytest.raises(ValueError, match="reused"):
        s.commit(next_decision, 1)


def test_failed_completion_validation_does_not_mutate_accounting():
    s = scheduler()
    seed(s, "large", 10)
    decision = s.choose([view()], 0)
    s.commit(decision, 1)
    before = s.snapshot()
    with pytest.raises(ValueError, match="cohort"):
        s.complete(1, TimingObservation("large", 0, 10, 10), [])
    assert s.snapshot() == before


def test_known_mean_policy_matches_existing_utility_without_counting_truth_as_observations():
    graphs = (Graph("small", 4, prefill_capacity=64), Graph("large", 5, prefill_capacity=128))
    s = scheduler("known")
    s.set_reference_means({g.name: g.duration_ms for g in graphs})
    expected = UtilityScheduler(Value(64), UTILITY).choose([view()], graphs)
    actual = s.choose([view()], 0)
    assert actual.action == expected
    assert s.snapshot()["observation_count"] == 0
    with pytest.raises(ValueError, match="only the known"):
        scheduler().set_reference_means({"large": 1, "small": 2})


def test_mature_history_survives_idle_without_confidence_decay():
    s = scheduler()
    seed(s, "large", 10, count=1000)
    before = s.estimates["large"].snapshot()
    s.idle()
    s.choose([view()], 1_000_000)
    assert s.estimates["large"].snapshot() == before


def test_unknown_normalizer_preserves_physical_prices_but_native_mask_can_hide_them():
    policy = UtilityScheduler(Value(64), UTILITY)
    graphs = (Graph("prefill", 10, prefill_capacity=128),)
    views = [view()]
    policy.prepare(views, graphs)
    price = policy.prices["p"].tokens
    original = price.log_price - log(price.maximum_rate)
    policy.prepare(views, graphs, 1, known_graphs=set())
    action = policy.choose(views, graphs, 1)
    assert policy.prices["p"].tokens is price
    assert price.log_price - log(price.maximum_rate) == original
    policy.prepare(views, graphs, 1, known_graphs=set(), mask_unknown_prices=True)
    assert policy.lane_value(action.lanes[0]) == pytest.approx(
        exp(-policy.log_scale) * action.lanes[0].useful_work(Value(64)))


@pytest.mark.parametrize("changes", [{"policy": "invalid"}, {"alpha": 1}, {"alpha": -1},
                                     {"epsilon": 2}, {"temperature": 0}, {"quantile": 0},
                                     {"block_size": 0}, {"refresh_every": -1}, {"ttl_ms": 0}])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(PolicyConfig(), **changes)


def targeted_decode(monkeypatch, narrow_ms=16, **changes):
    graphs = (GraphSpec("wide", decode_capacity=1, width=8),
              GraphSpec("narrow", decode_capacity=1, width=3))
    s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100), graphs,
                          PolicyConfig(policy="targeted", refresh_busy_ms=0, **changes))
    seed(s, "wide", 30, count=64)
    seed(s, "narrow", narrow_ms)
    monkeypatch.setattr(s.estimates["narrow"], "quantile", lambda probability: 10)
    s.credit_ms = 5
    return s


def test_targeted_probe_corrects_a_plausible_cheap_arm_without_changing_ordinary_selection(monkeypatch):
    s = targeted_decode(monkeypatch)
    s.credit_ms = 4.999
    ordinary = s.choose([view("d", 0)], 0)
    assert ordinary.action.graph.name == "wide"
    assert ordinary.category == "ordinary"
    finish(s, ordinary, 30)
    decision = s.choose([view("d", 0)], 30)
    assert decision.action.graph.name == "narrow"
    assert decision.category == "probe"
    assert decision.reason == "utility_challenger"
    row = decision.metadata["targeted"]["candidates"]["narrow"]
    assert row["mean_score"] < decision.metadata["incumbent_score"] < row["optimistic_score"]
    assert row["implied_loss_ms"] == pytest.approx(4.75)
    assert row["extra_blocking_ms"] == 0
    assert decision.metadata["mean_costs_ms"]["narrow"] == 16


@pytest.mark.parametrize("limit,expected", [(5, "wide"), (10, "narrow")])
def test_targeted_loss_admission_uses_incumbent_value_milliseconds(monkeypatch, limit, expected):
    s = targeted_decode(monkeypatch, narrow_ms=18, probe_loss_ms=limit)
    decision = s.choose([view("d", 0)], 0)
    row = decision.metadata["targeted"]["candidates"]["narrow"]
    assert row["implied_loss_ms"] == pytest.approx(6.75)
    assert row["plausible"]
    assert decision.action.graph.name == expected


@pytest.mark.parametrize("extra,expected", [(0, "wide"), (5, "narrow")])
def test_targeted_extra_blocking_limit_is_separate_from_utility_loss(monkeypatch, extra, expected):
    s = targeted_decode(monkeypatch, narrow_ms=35, probe_loss_ms=100, probe_extra_ms=extra)
    decision = s.choose([view("d", 0)], 0)
    assert decision.metadata["targeted"]["candidates"]["narrow"]["extra_blocking_ms"] == 5
    assert decision.action.graph.name == expected


def test_targeted_unknown_trial_does_not_treat_prior_median_as_a_known_delay(monkeypatch):
    graphs = (GraphSpec("wide", decode_capacity=1, width=8),
              GraphSpec("narrow", decode_capacity=1, width=3))
    s = LearningScheduler(Value(64), UTILITY, graphs,
                          PolicyConfig(policy="targeted", refresh_busy_ms=0))
    seed(s, "wide", 30, count=64)
    estimate = TimingEstimate(prior_location_ms=1000)
    s.estimates["narrow"] = estimate
    monkeypatch.setattr(estimate, "quantile", lambda probability: 10 if probability < .5 else 1000)
    s.credit_ms = 5
    decision = s.choose([view("d", 0)], 0)
    assert decision.action.graph.name == "narrow"
    assert decision.reason == "first_trial"
    assert decision.estimate_ms is None
    row = decision.metadata["targeted"]["candidates"]["narrow"]
    assert row["first_trial"] and row["admitted"]
    assert row["admission_estimate_ms"] == 1000
    assert row["extra_blocking_ms"] == 970
    assert s.utility.prices["d"].tokens.maximum_rate == pytest.approx(8 / 30)
    finish(s, decision, 100)
    assert s.credit_ms == -95
    assert s.probe_ms == 100
    assert s.ordinary_ms == 0


def test_targeted_unknown_trial_still_requires_plausible_utility_improvement(monkeypatch):
    s = targeted_decode(monkeypatch)
    s.estimates["narrow"] = TimingEstimate(prior_location_ms=1000)
    decision = s.choose([view("d", 0)], 0)
    assert decision.action.graph.name == "wide"
    assert not decision.metadata["targeted"]["candidates"]["narrow"]["plausible"]


def test_targeted_audit_uses_occupied_time_and_finances_one_observation():
    s = scheduler("targeted", refresh_busy_ms=200)
    seed(s, "large", 10, count=1000)
    seed(s, "small", 1000, count=1000)
    first = s.choose([view()], 1_000_000)
    assert first.category == "ordinary"
    assert s.snapshot()["occupied_ms"] == 0
    now = finish(s, first, 245)
    decision = s.choose([view()], now)
    assert decision.reason == "occupied_audit"
    assert decision.action.graph.name == "small"
    row = decision.metadata["targeted"]["candidates"]["small"]
    assert row["audit_age_ms"] == 245
    assert row["audit_due"] and not row["admitted"]
    assert s.choose([view()], now) is decision
    assert s.opportunities == 0
    count = s.estimates["small"].count
    now = finish(s, decision, 10, operation=2)
    assert s.opportunities == 1
    assert s.estimates["small"].count == count + 1
    assert s.credit_ms == -5
    assert s.snapshot()["observed_busy_ms"]["small"] == 255
    s.idle()
    s.credit_ms = 5
    next_decision = s.choose([view()], now + 1_000_000)
    assert next_decision.category == "ordinary"
    assert not next_decision.metadata["targeted"]["candidates"]["small"]["audit_due"]


def test_targeted_invalid_audits_rotate_without_refreshing_observation_age():
    graphs = (GraphSpec("a", prefill_capacity=64), GraphSpec("b", prefill_capacity=128),
              GraphSpec("good", prefill_capacity=4096))
    s = scheduler("targeted", catalog=graphs, refresh_busy_ms=200)
    for name, duration in (("a", 1000), ("b", 1000), ("good", 2)):
        seed(s, name, duration, count=1000)
    now = finish(s, s.choose([view()], 0), 245)
    decision = s.choose([view()], now)
    assert decision.reason == "occupied_audit"
    assert decision.action.graph.name == "a"
    now = finish(s, decision, 1, operation=2, valid=False)
    s.credit_ms = 5
    other = s.choose([view()], now)
    assert other.reason == "occupied_audit"
    assert other.action.graph.name == "b"
    assert other.metadata["targeted"]["candidates"]["a"]["audit_age_ms"] == 246


def test_targeted_pending_change_gets_funded_confirmation_without_waiting_for_audit(monkeypatch):
    s = targeted_decode(monkeypatch, narrow_ms=1000)
    estimate = s.estimates["narrow"]
    monkeypatch.setattr(TimingEstimate, "pending_change", property(lambda item: item is estimate))
    decision = s.choose([view("d", 0)], 0)
    assert decision.reason == "confirmation"
    assert decision.action.graph.name == "narrow"
    assert decision.action.graph.duration_ms == 1000
    row = decision.metadata["targeted"]["candidates"]["narrow"]
    assert row["pending_change"] and not row["admitted"]
    finish(s, decision, 10)
    assert s.credit_ms == -5
    blocked = s.choose([view("d", 0)], 10)
    assert blocked.category == "ordinary"


@pytest.mark.parametrize("field,value", [("probe_loss_ms", -1), ("probe_extra_ms", float("inf")),
                                        ("refresh_busy_ms", float("nan"))])
def test_targeted_configuration_limits_are_validated(field, value):
    with pytest.raises(ValueError):
        replace(PolicyConfig(), **{field: value})


@pytest.mark.parametrize("policy", ["known", "ttl", "jitter", "ema", "greedy", "epsilon",
                                    "softmax", "optimism", "thompson", "predictive"])
def test_targeted_controls_do_not_change_existing_policy_decisions_or_accounting(policy):
    left = scheduler(policy)
    right = scheduler(policy, probe_loss_ms=0, probe_extra_ms=1000, refresh_busy_ms=1,
                      audit_every=4, probe_harm_ms=0)
    if policy == "known":
        for s in (left, right):
            s.set_reference_means({"small": 10, "large": 20})
    now = 0
    for operation in range(1, 21):
        a, b = (s.choose([view()], now) for s in (left, right))
        assert a == b
        duration = 10 if a.action.graph.name == "small" else 20
        for s, decision in ((left, a), (right, b)):
            finish(s, decision, duration, operation, valid=operation % 7 != 0)
        now += duration
        assert left.snapshot() == right.snapshot()
        if operation == 10:
            left.idle()
            right.idle()
            now += 60_000


def test_targeted_quantile_is_shared_across_cohorts_and_retries(monkeypatch):
    s = targeted_decode(monkeypatch)
    calls = []
    monkeypatch.setattr(s.estimates["narrow"], "quantile", lambda probability: calls.append(probability) or 10)
    decision = s.choose([view("a", 0), view("b", 0)], 0)
    assert calls == [s.config.quantile]
    assert decision.metadata["sampled_costs_ms"]["narrow"] == 10
    assert s.choose([view("a", 0), view("b", 0)], 0) is decision
    assert calls == [s.config.quantile]


@pytest.mark.parametrize("probe_harm_ms", [None, 0])
def test_targeted_one_sample_audit_can_fund_confirmation_of_a_mature_cost_change(probe_harm_ms):
    s = scheduler("targeted", estimator="log-regime", refresh_busy_ms=200,
                  probe_harm_ms=probe_harm_ms)
    seed(s, "large", 10, count=1000)
    seed(s, "small", 1000, count=1000)
    now = 0
    probes = []
    for operation in range(1, 151):
        decision = s.choose([view()], now)
        if decision.category == "probe":
            assert decision.action.graph.name == "small"
            probes.append(decision.reason)
        duration = 1 if decision.action.graph.name == "small" else 10
        now = finish(s, decision, duration, operation)
        if s.estimates["small"].regime_changes:
            break
    assert probes[0] == "occupied_audit"
    assert probes[1:] == ["confirmation"] * 7
    assert s.estimates["small"].count == 1008
    assert s.estimates["small"].mean_ms == 1
    assert not s.estimates["small"].pending_change
    decision = s.choose([view()], now)
    assert decision.action.graph.name == "small"
    assert decision.category == "ordinary"


def test_targeted_audit_uses_the_mean_best_cohort_and_traces_that_cohort(monkeypatch):
    graphs = (GraphSpec("incumbent", decode_capacity=2),
              GraphSpec("audit", decode_capacity=1, width=8))
    s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100, 1), graphs,
                          PolicyConfig(policy="targeted"))
    seed(s, "incumbent", 5, count=1000)
    seed(s, "audit", 100, count=1000)
    monkeypatch.setattr(s.estimates["audit"], "quantile", lambda probability: 1)
    s._occupied_ms = 5000
    s.credit_ms = 5
    views = [View("yield", 0, 4096, (1.0,) * 7),
             View("weight", 0, 4096, (0.0,) * 7, weight=2)]
    decision = s.choose(views, 0)
    assert decision.reason == "occupied_audit"
    assert [lane.name for lane in decision.action.lanes] == ["weight"]
    optimistic = s._best_by_graph(views, {"incumbent": 5, "audit": 1},
                                  {view.name: view.last_service_ms for view in views})
    assert [lane.name for lane in optimistic["audit"].lanes] == ["yield"]
    row = decision.metadata["targeted"]["candidates"]["audit"]
    assert row["requests"] == ["weight"]
    assert row["mean_score"] == s.utility.score(decision.action)
    at_optimistic_cost = replace(decision.action, graph=replace(decision.action.graph, duration_ms=1))
    assert row["optimistic_score"] == s.utility.score(at_optimistic_cost)


def test_targeted_invalid_confirmation_rotates_due_audits_without_erasing_change_evidence():
    graphs = (GraphSpec("a-due", prefill_capacity=128),
              GraphSpec("z-pending", prefill_capacity=64),
              GraphSpec("ordinary", prefill_capacity=4096))
    s = scheduler("targeted", catalog=graphs, estimator="log-regime", refresh_busy_ms=200)
    for name, duration in (("a-due", 1000), ("z-pending", 20), ("ordinary", 10)):
        seed(s, name, duration, count=1000)
    seed(s, "z-pending", 5)
    pending = s.estimates["z-pending"]
    assert pending.pending_change and pending.pending_count == 1
    now = finish(s, s.choose([view()], 0), 245)
    decision = s.choose([view()], now)
    assert decision.reason == "confirmation"
    assert decision.action.graph.name == "z-pending"
    now = finish(s, decision, 1, operation=2, valid=False)
    assert pending.count == 1001
    assert pending.pending_change and pending.pending_count == 1
    assert s.snapshot()["failed_confirmations"] == ["z-pending"]
    for operation in range(3, 13):
        decision = s.choose([view()], now)
        if decision.category == "probe":
            break
        now = finish(s, decision, 10, operation)
    assert decision.reason == "occupied_audit"
    assert decision.action.graph.name == "a-due"
    assert decision.metadata["targeted"]["candidates"]["z-pending"]["confirmation_deferred"]
    now = finish(s, decision, 1000, operation=operation)
    s.credit_ms = 5
    confirmation = s.choose([view()], now)
    assert confirmation.reason == "confirmation"
    assert confirmation.action.graph.name == "z-pending"
    finish(s, confirmation, 5, operation=operation + 1)
    assert pending.pending_change and pending.pending_count == 2
    assert not s.snapshot()["failed_confirmations"]


def arbitration_scheduler(monkeypatch, audit_every=4):
    graphs = (GraphSpec("wide", decode_capacity=1, width=8),
              GraphSpec("narrow", decode_capacity=1, width=3),
              GraphSpec("audit", decode_capacity=1))
    s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100), graphs,
                          PolicyConfig(policy="targeted", audit_every=audit_every))
    for name, duration in (("wide", 30), ("narrow", 16), ("audit", 1000)):
        seed(s, name, duration, count=1000)
    monkeypatch.setattr(s.estimates["narrow"], "quantile", lambda probability: 10)
    s._occupied_ms = 5000
    s._observed_busy["narrow"] = 5000
    return s


def test_audit_arbitration_commits_three_challengers_then_one_audit_without_free_probe_time(monkeypatch):
    s = arbitration_scheduler(monkeypatch)
    now, operation = 0, 0
    queues = []
    for index in range(8):
        while True:
            decision = s.choose([view("d", 0)], now)
            if decision.category == "probe":
                break
            assert s._contended_opportunities == index
            operation += 1
            now = finish(s, decision, 30, operation)
            assert s._contended_opportunities == index
        arbitration = decision.metadata["targeted"]["arbitration"]
        assert arbitration["contended"]
        assert arbitration["index"] == index + 1
        assert not arbitration["borrowed"]
        queues.append(arbitration["queue"])
        assert s.choose([view("d", 0)], now) is decision
        with pytest.raises(ValueError, match="uncommitted decision"):
            s.commit(replace(decision), operation + 1)
        assert s._contended_opportunities == index
        operation += 1
        duration = 10 if arbitration["queue"] == "audit" else 16
        now = finish(s, decision, duration, operation, valid=arbitration["queue"] != "audit")
        assert s._contended_opportunities == index + 1
        assert s.credit_ms == 5 - duration
        assert s.probe_ms <= s.ordinary_ms * s.config.alpha / (1 - s.config.alpha) + 16
    assert queues == ["challenger", "challenger", "challenger", "audit"] * 2
    assert s.snapshot()["contended_opportunities"] == 8


@pytest.mark.parametrize("queue", ["audit", "challenger"])
def test_audit_arbitration_borrows_empty_queue_without_advancing_phase(monkeypatch, queue):
    s = arbitration_scheduler(monkeypatch)
    s.credit_ms = 5
    if queue == "audit":
        monkeypatch.setattr(s.estimates["narrow"], "quantile", lambda probability: 1000)
    else:
        s._observed_busy["audit"] = s._occupied_ms
    decision = s.choose([view("d", 0)], 0)
    arbitration = decision.metadata["targeted"]["arbitration"]
    assert arbitration["queue"] == queue
    assert arbitration["borrowed"] and not arbitration["contended"]
    now = finish(s, decision, 10)
    assert s._contended_opportunities == 0
    monkeypatch.setattr(s.estimates["narrow"], "quantile", lambda probability: 10)
    s._observed_busy["audit"] = 0
    s.credit_ms = 5
    next_decision = s.choose([view("d", 0)], now)
    arbitration = next_decision.metadata["targeted"]["arbitration"]
    assert arbitration["contended"] and arbitration["index"] == 1
    assert arbitration["queue"] == "challenger"


def test_audit_arbitration_does_not_delay_healthy_confirmation_or_advance_phase(monkeypatch):
    s = arbitration_scheduler(monkeypatch)
    estimate = s.estimates["audit"]
    monkeypatch.setattr(TimingEstimate, "pending_change", property(lambda item: item is estimate))
    s.credit_ms = 5
    decision = s.choose([view("d", 0)], 0)
    assert decision.reason == "confirmation"
    assert decision.action.graph.name == "audit"
    arbitration = decision.metadata["targeted"]["arbitration"]
    assert arbitration["queue"] == "confirmation"
    assert not arbitration["contended"]
    finish(s, decision, 10)
    assert s._contended_opportunities == 0


def test_failed_confirmation_cannot_also_claim_a_challenger_slot(monkeypatch):
    s = arbitration_scheduler(monkeypatch)
    estimate = s.estimates["narrow"]
    monkeypatch.setattr(TimingEstimate, "pending_change", property(lambda item: item is estimate))
    s._failed_confirmations.add("narrow")
    s.credit_ms = 5
    decision = s.choose([view("d", 0)], 0)
    assert decision.metadata["targeted"]["candidates"]["narrow"]["admitted"]
    arbitration = decision.metadata["targeted"]["arbitration"]
    assert "narrow" in arbitration["audits"]
    assert "narrow" not in arbitration["challengers"]
    assert arbitration["queue"] == "audit"
    assert not arbitration["contended"]


def test_audit_arbitration_phase_survives_idle_and_request_generation_changes(monkeypatch):
    s = arbitration_scheduler(monkeypatch)
    s.credit_ms = 5
    first = s.choose([view("first", 0)], 0)
    now = finish(s, first, 16)
    assert s._contended_opportunities == 1
    s.idle()
    assert s.credit_ms == -11
    s.credit_ms = 5
    second = s.choose([view("second", 0)], now + 1_000_000)
    arbitration = second.metadata["targeted"]["arbitration"]
    assert arbitration["contended"] and arbitration["index"] == 2
    assert arbitration["queue"] == "challenger"


def test_default_audit_priority_ignores_arbitration_phase_and_preserves_trace_shape(monkeypatch):
    s = arbitration_scheduler(monkeypatch, audit_every=0)
    s.credit_ms = 5
    s._contended_opportunities = 3
    decision = s.choose([view("d", 0)], 0)
    assert decision.reason == "occupied_audit"
    assert decision.action.graph.name == "audit"
    assert list(decision.metadata["targeted"]) == [
        "occupied_ms", "refresh_busy_ms", "gross_value_rate", "selected_graph", "candidates"]
    assert "arbitration" not in decision.metadata["targeted"]
    assert "contended_opportunities" not in s.snapshot()
    finish(s, decision, 10)
    assert s._contended_opportunities == 3


@pytest.mark.parametrize("period", [1, -1, True, 4.0])
def test_invalid_audit_arbitration_period_is_rejected(period):
    with pytest.raises(ValueError, match="audit period"):
        PolicyConfig(audit_every=period)


def peer_probe_scheduler(*, probe_harm_ms=5, safe=True, refresh_busy_ms=5000):
    graphs = (GraphSpec("incumbent", decode_capacity=4, width=8),
              GraphSpec("a-narrow", decode_capacity=1, width=8))
    if safe:
        graphs += (GraphSpec("safe", decode_capacity=4),)
    s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100), graphs,
                          PolicyConfig(policy="targeted", probe_loss_ms=100,
                                       probe_extra_ms=100, refresh_busy_ms=refresh_busy_ms,
                                       probe_harm_ms=probe_harm_ms))
    durations = [("incumbent", 35), ("a-narrow", 20)]
    if safe:
        durations.append(("safe", 5))
    for name, duration in durations:
        seed(s, name, duration, count=1000)
    s._occupied_ms = 5000
    s.credit_ms = 5
    return s


def test_peer_service_delay_defers_narrow_audit_without_starving_a_full_cohort():
    s = peer_probe_scheduler()
    views = [view(f"d{i}", 0) for i in range(4)]
    attempts, observed = dict(s._attempted_busy), dict(s._observed_busy)
    decision = s.choose(views, 0)
    assert decision.reason == "occupied_audit"
    assert decision.action.graph.name == "safe"
    assert {lane.name for lane in decision.action.lanes} == {v.name for v in views}
    rows = decision.metadata["targeted"]["candidates"]
    harm = rows["a-narrow"]["audit_service_delay"]
    assert sorted(harm["request_delays_ms"].values()) == [0, 20, 20, 20]
    assert harm["total_request_ms"] == 60
    assert harm["maximum_peer_delay_ms"] == 20
    assert not harm["admitted"]
    assert rows["a-narrow"]["audit_due"]
    assert rows["a-narrow"]["deferred_reason"] == "peer_service_delay"
    assert rows["safe"]["service_delay"]["total_request_ms"] == 0
    assert s.choose(views, 0) is decision
    assert s._attempted_busy == attempts and s._observed_busy == observed
    assert s.opportunities == s._contended_opportunities == 0
    assert s._occupied_ms == 5000
    finish(s, decision, 20)
    assert s._attempted_busy == {"safe": 5000}
    assert s.credit_ms == -15
    assert s.probe_ms == 20
    assert s.ordinary_ms == 0


def test_peer_service_delay_filters_optimistic_challengers_and_healthy_confirmations(monkeypatch):
    views = [view(f"d{i}", 0) for i in range(4)]
    s = peer_probe_scheduler(refresh_busy_ms=0)
    for name in ("a-narrow", "safe"):
        monkeypatch.setattr(s.estimates[name], "quantile", lambda probability: 1)
    pending = s.estimates["a-narrow"]
    monkeypatch.setattr(TimingEstimate, "pending_change", property(lambda item: item is pending))
    decision = s.choose(views, 0)
    assert decision.reason == "utility_challenger"
    assert decision.action.graph.name == "safe"
    row = decision.metadata["targeted"]["candidates"]["a-narrow"]
    assert row["pending_change"] and row["plausible"]
    assert not row["admitted"]
    assert row["deferred_reason"] == "peer_service_delay"
    assert row["service_delay"]["total_request_ms"] == 60


def test_peer_service_delay_considers_cohorts_before_collapsing_a_physical_graph():
    graphs = (GraphSpec("incumbent", prefill_capacity=64, decode_capacity=1),
              GraphSpec("check", prefill_capacity=128, decode_capacity=1))
    views = [view("decode", 0), replace(view("short", 64), weight=2),
             replace(view("long", 128), weight=1.5)]
    decisions = []
    for limit in (None, 5):
        s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100), graphs,
                              PolicyConfig(policy="targeted", probe_harm_ms=limit))
        seed(s, "incumbent", 10, count=1000)
        seed(s, "check", 12, count=1000)
        s._occupied_ms = 5000
        s.credit_ms = 5
        decision = s.choose(views, 0)
        assert decision.action.graph.name == "check"
        assert decision.reason == "occupied_audit"
        decisions.append(decision)
        if limit is not None:
            row = decision.metadata["targeted"]["candidates"]["check"]
            assert row["requests"] == [lane.name for lane in decision.action.lanes]
            assert row["service_delay"]["request_delays_ms"] == {"decode": 2, "short": 2}
            assert row["service_delay"]["total_request_ms"] == 4
            assert row["mean_score"] == s.utility.score(decision.action)
    assert {lane.name for lane in decisions[0].action.lanes} == {"decode", "long"}
    assert {lane.name for lane in decisions[1].action.lanes} == {"decode", "short"}


@pytest.mark.parametrize("count,expected", [(1, "a-narrow"), (4, "incumbent")])
def test_unknown_peer_service_delay_uses_a_disclosed_prior_and_preserves_lone_discovery(monkeypatch, count, expected):
    s = peer_probe_scheduler(safe=False)
    s.estimates["a-narrow"] = TimingEstimate(prior_location_ms=1000)
    monkeypatch.setattr(s.estimates["a-narrow"], "quantile", lambda probability: 1 if probability < .5 else 1000)
    decision = s.choose([view(f"d{i}", 0) for i in range(count)], 0)
    assert decision.action.graph.name == expected
    row = decision.metadata["targeted"]["candidates"]["a-narrow"]
    harm = row["audit_service_delay"]
    assert harm["duration_basis"] == "prior_median"
    assert harm["single_request_exemption"] == (count == 1)
    assert harm["total_request_ms"] == (0 if count == 1 else 3965)
    assert row["first_trial"]


@pytest.mark.parametrize("limit", [-1, float("inf"), float("nan"), True])
def test_invalid_peer_service_delay_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="peer service-delay"):
        PolicyConfig(probe_harm_ms=limit)


def test_peer_service_delay_searches_safe_decode_subsets_before_value_ranking(monkeypatch):
    graphs = (GraphSpec("incumbent", decode_capacity=1, width=8),
              GraphSpec("check", decode_capacity=1))
    s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100), graphs,
                          PolicyConfig(policy="targeted", probe_harm_ms=5, refresh_busy_ms=0))
    seed(s, "incumbent", 10, count=1000)
    seed(s, "check", 6, count=1000)
    monkeypatch.setattr(s.estimates["check"], "quantile", lambda probability: 1)
    s.credit_ms = 5
    decision = s.choose([view("A", 0), View("B", 0, 4096, (0.0,) * 7, weight=2)], 0)
    assert decision.reason == "utility_challenger"
    assert decision.action.graph.name == "check"
    assert [lane.name for lane in decision.action.lanes] == ["A"]
    row = decision.metadata["targeted"]["candidates"]["check"]
    assert row["implied_loss_ms"] == pytest.approx(4.75)
    assert row["service_delay"]["total_request_ms"] == 0


@pytest.mark.parametrize("limit", [0, 6, 12, 18])
def test_grouped_decode_cohorts_match_exhaustive_safe_subset_selection(limit):
    graphs = (GraphSpec("incumbent", decode_capacity=3, width=8),
              GraphSpec("check", decode_capacity=2))
    views = [View(f"d{i}", 0, 4096, (i / 5,) * 7, weight=(7 - i) / 3,
                  last_service_ms=i) for i in range(6)]
    s = LearningScheduler(Value(64), UtilityConfig(3, 2, 100), graphs,
                          PolicyConfig(policy="targeted", probe_harm_ms=limit))
    costs = {"incumbent": 10, "check": 6}
    catalog = tuple(s._graph(g, costs[g.name]) for g in graphs)
    s.utility.prepare(views, catalog, 10)
    incumbent = Action(catalog[0], tuple(Lane(v.name, 0, 8, expected_output(v.acceptance, 8), v.weight)
                                        for v in views[:3]))
    clocks = {v.name: v.last_service_ms for v in views}
    def admissible(action):
        return s._probe_service_delay(action, incumbent, len(views), costs, costs)["admitted"]
    actual = s._best_by_graph(views, costs, clocks, admissible=admissible,
                              cohort_reference=incumbent)
    exhaustive = [Action(catalog[1], tuple(Lane(v.name, 0, 1, 1, v.weight) for v in group))
                  for group in combinations(views, 2)]
    feasible = [action for action in exhaustive if admissible(action)]
    if feasible:
        expected = min(feasible, key=lambda action: s._rank_key(action, clocks))
        assert {lane.name for lane in actual["check"].lanes} == {lane.name for lane in expected.lanes}
        assert s.utility.score(actual["check"]) == pytest.approx(s.utility.score(expected))
    else:
        assert "check" not in actual


def soft_mixed_scheduler(**changes):
    graphs = (GraphSpec("incumbent", prefill_capacity=128, decode_capacity=2, width=8),
              GraphSpec("check", prefill_capacity=128, decode_capacity=2))
    s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100), graphs,
                          PolicyConfig(policy="targeted", harm_alpha=.002, **changes))
    seed(s, "incumbent", 10, count=1000)
    seed(s, "check", 5, count=1000)
    return s


def test_harm_credit_uses_productive_actual_lane_phases_even_when_gpu_timing_is_missing():
    s = soft_mixed_scheduler()
    views = [view("prompt"), view("d0", 0), view("d1", 0)]
    decision = s.choose(views, 0)
    count = s.estimates["incumbent"].count
    s.commit(decision, 1)
    rows = [{"request": "prompt", "prefill": 64, "emitted": 0},
            {"request": "d0", "prefill": 0, "emitted": 1},
            {"request": "d1", "prefill": 0, "emitted": 0}]
    s.complete(1, TimingObservation("incumbent", 0, 2000, None, 0), rows)
    assert s.harm_ordinary_ms == {"prefill": 2000, "decode": 2000}
    assert s.harm_credit_ms == {"prefill": 1, "decode": 1}
    assert s.harm_charged_ms == {"prefill": 0, "decode": 0}
    assert s.estimates["incumbent"].count == count
    assert s.ordinary_ms == s.credit_ms == 0
    assert s.choose(views, 2000).category == "ordinary"


@pytest.mark.parametrize("valid", [False, True])
def test_harm_budget_requires_both_phases_and_settles_partial_progress_once(valid):
    s = soft_mixed_scheduler()
    views = [view("prompt"), view("d0", 0), view("d1", 0)]
    s._occupied_ms = 5000
    s.credit_ms = 5
    s.harm_credit_ms["decode"] = 1
    ordinary = s.choose(views, 0)
    assert ordinary.category == "ordinary"
    assert ordinary.metadata["harm_budget"]["unfunded_phases"] == ["prefill"]
    assert not ordinary.metadata["opportunity"]
    now = finish(s, ordinary, 10)
    s.harm_credit_ms["prefill"] = 1
    decision = s.choose(views, now)
    assert decision.reason == "occupied_audit"
    assert decision.action.graph.name == "check"
    assert decision.metadata["targeted"]["candidates"]["check"]["service_delay"]["total_request_ms"] == 0
    reference = s._planned_reference
    assert s.choose(views, now) is decision
    with pytest.raises(ValueError, match="uncommitted decision"):
        s.commit(replace(decision), 2)
    assert s._planned_reference == reference
    assert s.harm_credit_ms == {"prefill": 1, "decode": 1}
    s.commit(decision, 2)
    rows = [{"request": "prompt", "prefill": 64, "emitted": 0},
            {"request": "d0", "prefill": 0, "emitted": 1},
            {"request": "d1", "prefill": 0, "emitted": 0}]
    observation = TimingObservation("check", now, now + 25, 25 if valid else None, 0)
    s.complete(2, observation, rows)
    charge = decision.metadata["harm_budget"]["actual_charge"]
    assert charge["request_delays_ms"] == {"prompt": 25, "d0": 15, "d1": 25}
    assert charge["phase_request_ms"] == {"prefill": 25, "decode": 40}
    assert charge["duration_basis"] == "actual_elapsed"
    assert s.harm_charged_ms == {"prefill": 25, "decode": 40}
    assert s.harm_credit_ms == {"prefill": -24, "decode": -39}
    assert s.credit_ms == -20 and s.probe_ms == 25
    with pytest.raises(ValueError, match="exactly once"):
        s.complete(2, observation, rows)
    assert s.harm_charged_ms == {"prefill": 25, "decode": 40}
    s.idle()
    assert s.harm_credit_ms == {"prefill": -24, "decode": -39}


def test_soft_audit_minimizes_harm_before_value_while_challenger_keeps_value(monkeypatch):
    graphs = (GraphSpec("incumbent", decode_capacity=1, width=8),
              GraphSpec("check", decode_capacity=1))
    views = [view("A", 0), View("B", 0, 4096, (0.0,) * 7, weight=2)]
    for refresh, expected in ((5000, "A"), (0, "B")):
        s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100), graphs,
                              PolicyConfig(policy="targeted", harm_alpha=.002, refresh_busy_ms=refresh))
        seed(s, "incumbent", 10, count=1000)
        seed(s, "check", 6, count=1000)
        monkeypatch.setattr(s.estimates["check"], "quantile", lambda probability: 1)
        s.credit_ms = 5
        s.harm_credit_ms["decode"] = 1
        s._occupied_ms = 5000
        decision = s.choose(views, 0)
        assert decision.action.graph.name == "check"
        assert [lane.name for lane in decision.action.lanes] == [expected]


def test_soft_harm_budget_recovers_a_mature_overestimate_under_continuous_contention():
    graphs = (GraphSpec("incumbent", decode_capacity=1), GraphSpec("check", decode_capacity=1))
    s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100), graphs,
                          PolicyConfig(policy="targeted", estimator="log-regime", harm_alpha=.002,
                                       refresh_busy_ms=200))
    seed(s, "incumbent", 10, count=1000)
    seed(s, "check", 1000, count=1000)
    views = [view(f"d{i}", 0) for i in range(4)]
    now, probes = 0, []
    for operation in range(1, 301):
        decision = s.choose(views, now)
        if decision.category == "probe":
            probes.append(decision.reason)
        now = finish(s, decision, 1 if decision.action.graph.name == "check" else 10, operation)
        if s.estimates["check"].regime_changes:
            break
    assert probes == ["occupied_audit", *(["confirmation"] * 7)]
    assert s.estimates["check"].count == 1008
    assert s.estimates["check"].mean_ms == 1
    assert s.harm_charged_ms == {"prefill": 0, "decode": 0}
    assert s.choose(views, now).action.graph.name == "check"


@pytest.mark.parametrize("duration,charge,balance", [(25, 15, -14), (5, 0, 1)])
def test_soft_budget_charges_lone_prefill_delay_without_exempting_rediscovery(duration, charge, balance):
    graphs = (GraphSpec("incumbent", prefill_capacity=64), GraphSpec("check", prefill_capacity=128))
    s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100), graphs,
                          PolicyConfig(policy="targeted", harm_alpha=.002))
    seed(s, "incumbent", 10, count=1000)
    seed(s, "check", 1000, count=1000)
    s._occupied_ms = 5000
    s.credit_ms = 5
    s.harm_credit_ms["prefill"] = 1
    decision = s.choose([view()], 0)
    assert decision.reason == "occupied_audit"
    assert decision.action.graph.name == "check"
    finish(s, decision, duration)
    actual = decision.metadata["harm_budget"]["actual_charge"]
    assert not actual["single_request_exemption"]
    assert actual["phase_request_ms"] == {"prefill": charge, "decode": 0}
    assert s.harm_charged_ms["prefill"] == charge
    assert s.harm_credit_ms["prefill"] == balance


def test_harm_budget_bound_holds_with_actual_overruns_invalid_timings_and_idle():
    graphs = (GraphSpec("incumbent", decode_capacity=2, width=8), GraphSpec("check", decode_capacity=1))
    s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100), graphs,
                          PolicyConfig(policy="targeted", harm_alpha=.005, refresh_busy_ms=20))
    seed(s, "incumbent", 10, count=1000)
    seed(s, "check", 1000, count=1000)
    now, last_charge, probe_count = 0, 0, 0
    for operation in range(1, 2501):
        decision = s.choose([view("A", 0), view("B", 0)], now)
        duration = (20, 50, 2)[probe_count % 3] if decision.category == "probe" else 10
        now = finish(s, decision, duration, operation, valid=operation % 11 != 0)
        if decision.category == "probe":
            probe_count += 1
            last_charge = decision.metadata["harm_budget"]["actual_charge"]["phase_request_ms"]["decode"]
        earned = .005 / .995 * s.harm_ordinary_ms["decode"]
        assert s.harm_charged_ms["decode"] <= earned + last_charge + 1e-8
        if operation % 137 == 0:
            before = dict(s.harm_credit_ms)
            s.idle()
            assert s.harm_credit_ms == {phase: min(0, balance) for phase, balance in before.items()}
            assert s.harm_charged_ms["decode"] <= earned + last_charge + 1e-8
    assert probe_count >= 3


def test_harm_bound_retains_each_phases_latest_exposing_probe_across_phase_changes():
    graphs = (GraphSpec("decode-good", decode_capacity=1, width=8),
              GraphSpec("decode-check", decode_capacity=1),
              GraphSpec("prefill-good", prefill_capacity=64),
              GraphSpec("prefill-check", prefill_capacity=128))
    s = LearningScheduler(Value(64), UtilityConfig(0, 0, 100), graphs,
                          PolicyConfig(policy="targeted", harm_alpha=.002, refresh_busy_ms=200))
    for graph in graphs:
        seed(s, graph.name, 1000 if "check" in graph.name else 10, count=1000)
    last_exposing = {"prefill": 0, "decode": 0}
    now = 0
    for operation, phase, duration, expected in (
            (1, "decode", 500, "ordinary"), (2, "decode", 30, "probe"),
            (3, "prefill", 1500, "ordinary"), (4, "prefill", 40, "probe")):
        decision = s.choose([view(phase, 0 if phase == "decode" else 1024)], now)
        assert decision.category == expected
        assert decision.metadata["harm_budget"]["required_phases"] == [phase]
        now = finish(s, decision, duration, operation)
        if expected == "probe":
            charge = decision.metadata["harm_budget"]["actual_charge"]["phase_request_ms"]
            last_exposing[phase] = charge[phase]
        for exposed in last_exposing:
            earned = .002 / .998 * s.harm_ordinary_ms[exposed]
            assert s.harm_charged_ms[exposed] <= earned + last_exposing[exposed] + 1e-9
        if operation == 2:
            decode_debt = s.harm_credit_ms["decode"]
            assert decode_debt < 0
        if operation > 2:
            assert s.harm_credit_ms["decode"] == decode_debt
    assert charge["decode"] == 0
    assert s.harm_charged_ms["decode"] > .002 / .998 * s.harm_ordinary_ms["decode"] + charge["decode"]


def test_native_initial_discovery_uses_bootstrap_accounting_without_earning_credit():
    s = scheduler("targeted", initial_discovery="native", harm_alpha=.002)
    now = 0
    for operation, expected, duration in ((1, "large", 20), (2, "small", 5)):
        decision = s.choose([view()], now)
        assert decision.category == "bootstrap"
        assert decision.action.graph.name == expected
        now = finish(s, decision, duration, operation)
        assert s.credit_ms == s.probe_ms == s.ordinary_ms == 0
        assert s.harm_credit_ms == {"prefill": 0, "decode": 0}
        assert s.harm_charged_ms == {"prefill": 0, "decode": 0}
    snapshot = s.snapshot()
    assert snapshot["initial_discovery_count"] == snapshot["bootstrap_count"] == 2
    assert snapshot["initial_discovery_ms"] == snapshot["bootstrap_ms"] == 25
    assert s.choose([view()], now).category == "ordinary"


def test_native_initial_discovery_masks_unmeasured_decode_prices_but_retains_history():
    graphs = (GraphSpec("prefill", prefill_capacity=64), GraphSpec("decode", decode_capacity=1))
    for discovery in ("productive", "native"):
        s = scheduler("targeted", catalog=graphs, initial_discovery=discovery)
        seed(s, "prefill", 8)
        first = s.choose([view("request", 64)], 0)
        now = finish(s, first, 8)
        price = s.utility.prices["request"].tokens
        decision = s.choose([view("request", 0)], now)
        assert decision.category == "bootstrap" and decision.action.graph.name == "decode"
        assert s.utility.prices["request"].tokens is price
        assert "request" not in s.utility._token_normalizers
        intrinsic = exp(-s.utility.log_scale) * decision.action.lanes[0].useful_work(s.value)
        value = s.utility.lane_value(decision.action.lanes[0])
        if discovery == "native":
            assert value == pytest.approx(intrinsic)
        else:
            assert value > intrinsic


def test_operational_ema_forwarding_retains_arithmetic_posterior_and_unfunded_history():
    arithmetic, ema = (scheduler("targeted", mean_estimator=mode) for mode in ("arithmetic", "ema"))
    for s in (arithmetic, ema):
        seed(s, "large", 20)
        seed(s, "large", 4)
    assert arithmetic._means(0)["large"] == 12
    assert ema._means(0)["large"] == 18
    assert ema.estimates["large"].arithmetic_mean_ms == 12
    assert arithmetic.estimates["large"].interval() == ema.estimates["large"].interval()
    assert ema.credit_ms == ema.ordinary_ms == ema.bootstrap_ms == 0


@pytest.mark.parametrize("changes", [{"harm_alpha": -1}, {"harm_alpha": 1},
    {"harm_alpha": float("nan")}, {"harm_alpha": True},
    {"harm_alpha": .002, "probe_harm_ms": 0}, {"initial_discovery": "unknown"},
    {"mean_estimator": "unknown"}])
def test_soft_harm_and_estimator_configuration_is_explicit(changes):
    with pytest.raises(ValueError):
        PolicyConfig(**changes)


def test_retained_refresh_uses_native_ema_and_keeps_expired_costs_available():
    s = scheduler("retained-refresh")
    seed(s, "large", 20)
    seed(s, "large", 4, end=1)
    seed(s, "small", 1000)
    assert s.estimates["large"].mean_ms == 12
    assert s._means(1)["large"] == s._means(1_000_000)["large"] == 18
    s.credit_ms = 4.999
    decision = s.choose([view()], 1_000_000)
    assert decision.category == "ordinary"
    assert decision.action.graph.name == "large"
    assert decision.estimate_ms == 18
    assert not decision.metadata["opportunity"]


def test_retained_refresh_preserves_native_fresh_and_expired_ema_updates():
    s = scheduler("retained-refresh")
    seed(s, "large", 8)
    seed(s, "large", 16, end=1)
    assert s._means(30_001)["large"] == 9
    seed(s, "large", 1, end=30_000)
    assert s._means(30_000)["large"] == 8
    seed(s, "large", 2, end=60_000)
    assert s._means(60_000)["large"] == 2
    assert s._means(3_600_000)["large"] == 2
    assert s.credit_ms == 0


def test_retained_refresh_intrinsic_native_discovery_matches_ttl_startup_actions():
    ttl, retained = scheduler("ttl"), scheduler("retained-refresh")
    now = 0
    for operation, expected, duration in ((1, "large", 20), (2, "small", 5), (3, "small", 5)):
        first, second = ttl.choose([view()], now), retained.choose([view()], now)
        assert first.action == second.action
        assert second.action.graph.name == expected
        assert first.score == second.score
        assert ttl.utility._token_normalizers == retained.utility._token_normalizers
        if operation <= 2:
            assert second.category == "bootstrap"
        else:
            assert second.category == "ordinary"
        finish(ttl, first, duration, operation)
        now = finish(retained, second, duration, operation)
    snapshot = retained.snapshot()
    assert snapshot["initial_discovery_count"] == 2
    assert snapshot["initial_discovery_ms"] == 25
    assert retained.probe_ms == 0


def test_retained_refresh_native_filter_skips_unknown_padding_with_no_extra_work():
    graphs = (GraphSpec("fitting", prefill_capacity=64), GraphSpec("padded", prefill_capacity=4096))
    s = scheduler("retained-refresh", catalog=graphs)
    decision = s.choose([view(remaining=64)], 0)
    assert decision.category == "bootstrap"
    assert decision.action.graph.name == "fitting"
    now = finish(s, decision, 10)
    decision = s.choose([view(remaining=64)], now)
    assert decision.category == "ordinary"
    assert decision.action.graph.name == "fitting"
    assert s._means(now)["padded"] is None
    assert s.credit_ms == 0


def test_retained_refresh_masks_unknown_phase_prices_and_preserves_service_history():
    graphs = (GraphSpec("prefill", prefill_capacity=64), GraphSpec("decode", decode_capacity=1))
    s = scheduler("retained-refresh", catalog=graphs)
    seed(s, "prefill", 8)
    now = finish(s, s.choose([view("request", 64)], 0), 8)
    price = s.utility.prices["request"].tokens
    decision = s.choose([view("request", 0)], now)
    assert decision.category == "bootstrap"
    assert decision.action.graph.name == "decode"
    assert s.utility.prices["request"].tokens is price
    assert "request" not in s.utility._token_normalizers
    intrinsic = exp(-s.utility.log_scale) * decision.action.lanes[0].useful_work(s.value)
    assert s.utility.lane_value(decision.action.lanes[0]) == pytest.approx(intrinsic)


@pytest.mark.parametrize("acceptance,requests", [(0., 1), (1., 1), (0., 2), (1., 2)])
def test_retained_refresh_preserves_ttl_ordinary_verification_and_cohort_choices(acceptance, requests):
    graphs = (GraphSpec("one", decode_capacity=1), GraphSpec("two", decode_capacity=2),
              GraphSpec("mid", decode_capacity=1, width=3), GraphSpec("wide", decode_capacity=1, width=8))
    ttl, retained = (scheduler(policy, catalog=graphs) for policy in ("ttl", "retained-refresh"))
    for s in (ttl, retained):
        for name, duration in (("one", 10), ("two", 16), ("mid", 18), ("wide", 30)):
            seed(s, name, duration)
    views = [View(f"d{i}", 0, 4096, (acceptance,) * 7) for i in range(requests)]
    first, second = ttl.choose(views, 0), retained.choose(views, 0)
    assert first.action == second.action
    assert first.score == second.score
    assert first.category == second.category == "ordinary"
    if requests == 1:
        assert second.action.graph.width == (8 if acceptance else 1)


def test_retained_refresh_invalid_ordinary_timing_keeps_cost_and_earns_no_credit():
    s = scheduler("retained-refresh")
    seed(s, "large", 10)
    seed(s, "small", 1000)
    decision = s.choose([view()], 0)
    assert decision.category == "ordinary"
    finish(s, decision, 245, valid=False)
    assert s.credit_ms == s.ordinary_ms == 0
    assert s.ordinary_elapsed_ms == 245
    assert s._means(245)["large"] == 10
    assert s._observed_at["large"] == 0
    assert s.utility.prices["p"].tokens.elapsed_ms == 245


@pytest.mark.parametrize("alpha,ordinary_steps", [(.002, 250), (.005, 100)])
def test_retained_refresh_idle_preserves_costs_but_requires_busy_credit(alpha, ordinary_steps):
    s = scheduler("retained-refresh", alpha=alpha)
    seed(s, "large", 10)
    seed(s, "small", 1000)
    counts = {name: estimate.count for name, estimate in s.estimates.items()}
    s.credit_ms = 5
    s.idle()
    assert s.credit_ms == 0
    assert {name: estimate.count for name, estimate in s.estimates.items()} == counts
    now = 60_000
    for operation in range(1, ordinary_steps + 1):
        decision = s.choose([view()], now)
        assert decision.category == "ordinary"
        assert decision.action.graph.name == "large"
        assert not decision.metadata["opportunity"]
        now = finish(s, decision, 10, operation)
    assert s.credit_ms == 5
    decision = s.choose([view()], now)
    assert decision.category == "probe" and decision.reason == "stale_refresh"
    assert decision.action.graph.name == "small"
    assert decision.estimate_ms == 1000
    finish(s, decision, 100, operation="invalid", valid=False)
    assert s.credit_ms == -95
    s.idle()
    assert s.credit_ms == -95
    assert s._means(now + 100)["small"] == 1000


def test_retained_refresh_checks_only_measured_stale_alternatives_at_the_strict_boundary():
    s = scheduler("retained-refresh")
    seed(s, "large", 10)
    seed(s, "small", 1000)
    s.credit_ms = 5
    first = s.choose([view()], 29_999)
    assert first.category == "ordinary"
    assert not first.metadata["opportunity"]
    now = finish(s, first, 1)
    second = s.choose([view()], now)
    assert now == 30_000
    assert second.category == "probe"
    assert second.action.graph.name == "small"
    now = finish(s, second, 2, operation=2)
    assert s._means(now)["small"] == 2
    assert s.credit_ms == 3
    third = s.choose([view()], now)
    assert third.category == "ordinary"
    assert not third.metadata["opportunity"]


def test_retained_refresh_invalid_attempt_rotates_and_is_charged_exactly_once():
    graphs = tuple(GraphSpec(name, decode_capacity=1) for name in ("incumbent", "a", "b"))
    s = scheduler("retained-refresh", catalog=graphs, alpha=.5)
    for name, duration in (("incumbent", 10), ("a", 20), ("b", 30)):
        seed(s, name, duration)
    s.credit_ms = 5
    decision = s.choose([view(remaining=0)], 30_000)
    assert decision.action.graph.name == "a"
    assert s.choose([view(remaining=0)], 30_000) is decision
    assert s.opportunities == 0 and s.credit_ms == 5
    s.commit(decision, "check-a")
    assert s.opportunities == 1 and s.credit_ms == 5
    before = s.snapshot()
    with pytest.raises(ValueError, match="cohort"):
        s.complete("check-a", TimingObservation("a", 30_000, 30_100, None), [])
    assert s.snapshot() == before
    s.complete("check-a", TimingObservation("a", 30_000, 30_100, None),
               [{"request": "p", "prefill": 0, "emitted": 1}])
    assert s.credit_ms == -95
    assert s._observed_at["a"] == 0
    assert s._attempted["a"] == 30_000
    assert s.estimates["a"].count == 1
    with pytest.raises(ValueError, match="exactly once"):
        s.complete("check-a", TimingObservation("a", 30_000, 30_100, None), [])
    assert s.credit_ms == -95 and s.probe_count == 1 and s.probe_ms == 100
    now = 30_100
    for operation in range(10):
        ordinary = s.choose([view(remaining=0)], now)
        assert ordinary.category == "ordinary"
        now = finish(s, ordinary, 10, operation)
    assert s.credit_ms == 5
    retry = s.choose([view(remaining=0)], now)
    assert retry.category == "probe"
    assert retry.action.graph.name == "b"


def test_retained_refresh_oldest_priority_uses_valid_completion_and_attempt_then_utility():
    graphs = tuple(GraphSpec(name, decode_capacity=1) for name in ("incumbent", "a-slow", "z-fast"))
    s = scheduler("retained-refresh", catalog=graphs)
    for name, duration in (("incumbent", 1), ("a-slow", 30), ("z-fast", 20)):
        seed(s, name, duration)
    s.credit_ms = 5
    decision = s.choose([view(remaining=0)], 30_000)
    assert decision.action.graph.name == "z-fast"
    now = finish(s, decision, 20)
    s.credit_ms = 5
    decision = s.choose([view(remaining=0)], now)
    assert decision.action.graph.name == "a-slow"


def test_retained_refresh_never_uses_uncertainty_or_phase_funding(monkeypatch):
    s = scheduler("retained-refresh")
    seed(s, "large", 10)
    seed(s, "small", 1000)
    def unexpected(*args, **kwargs):
        pytest.fail("retained refresh must not query the uncertainty model")
    for estimate in s.estimates.values():
        for name in ("quantile", "sample_mean", "sample_predictive"):
            monkeypatch.setattr(estimate, name, unexpected)
    s.credit_ms = 5
    decision = s.choose([view()], 30_000)
    assert decision.category == "probe"
    assert "harm_budget" not in decision.metadata
    finish(s, decision, 100)
    assert s.harm_credit_ms == s.harm_charged_ms == {"prefill": 0, "decode": 0}


@pytest.mark.parametrize("changes", [{"funded": False}, {"estimator": "regime"},
                                     {"estimator": "log-regime"}, {"harm_alpha": .002},
                                     {"probe_harm_ms": 5}])
def test_retained_refresh_requires_its_funded_cumulative_contract(changes):
    with pytest.raises(ValueError, match="retained-refresh"):
        PolicyConfig(policy="retained-refresh", **changes)


@pytest.mark.parametrize("alpha", [.002, .005])
def test_retained_refresh_allowance_covers_overdraw_invalid_timings_and_idle(alpha):
    s = scheduler("retained-refresh", alpha=alpha, ttl_ms=200)
    seed(s, "large", 10)
    seed(s, "small", 1000)
    rng = Random(129)
    now = 0
    last_probe = 0
    for operation in range(1, 4001):
        decision = s.choose([view()], now)
        if decision.category == "probe":
            assert s.credit_ms == s.config.credit_cap_ms
            duration = (20, 200, 1000)[s.probe_count % 3]
        else:
            duration = rng.uniform(10, 500)
        now = finish(s, decision, duration, operation, valid=operation % 11 != 0)
        if decision.category == "probe":
            last_probe = duration
        allowance = alpha / (1 - alpha) * s.ordinary_ms
        assert s.probe_ms <= allowance + last_probe + 1e-8
        assert s.credit_ms >= -last_probe - 1e-8
        if operation % 137 == 0:
            balance = s.credit_ms
            s.idle()
            assert s.credit_ms == min(0, balance)
            now += 10_000
    assert s.probe_count >= 3
    assert s.invalid_observations > 0
