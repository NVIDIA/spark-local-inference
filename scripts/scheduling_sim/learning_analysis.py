# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired experiment comparisons and explicit scheduler acceptance criteria."""

from dataclasses import dataclass
from math import ceil, isfinite
from random import Random
from statistics import fmean, median


HELDOUT_SEEDS = tuple(range(10_000, 10_064))
TRAINING_SEEDS = tuple(range(16))


def evaluation_seeds(seeds=None):
    """Validate an evaluation sample independent of the tuning seeds."""
    values = HELDOUT_SEEDS if seeds is None else tuple(seeds)
    if (len(values) != 64 or len(set(values)) != 64
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in values)):
        raise ValueError("evaluation requires exactly 64 distinct nonnegative integer seeds")
    if set(values) & set(TRAINING_SEEDS):
        raise ValueError("evaluation seeds must not overlap tuning seeds 0:16")
    return values


def _expected_seeds(heldout, expected_seeds, observed):
    if expected_seeds is not None and not heldout:
        raise ValueError("explicit evaluation seeds require heldout mode")
    return set(evaluation_seeds(expected_seeds)) if heldout else set(observed)


@dataclass(frozen=True)
class RatioInterval:
    ratio: float
    lower: float
    upper: float
    seeds: int
    confidence: float


def quantile(values, probability):
    """Nearest-rank quantile, including exact endpoints."""
    rows = sorted(values)
    if not rows or not 0 <= probability <= 1:
        raise ValueError("quantile requires samples and a probability in [0, 1]")
    return rows[max(0, ceil(probability * len(rows)) - 1)]


def paired_ratio(candidate, reference, *, confidence=.95, replicates=2000, seed=0):
    """Bootstrap the ratio of means by resampling paired seed clusters."""
    if candidate.keys() != reference.keys() or not candidate:
        raise ValueError("paired comparison requires the same nonempty seed set")
    if not .5 < confidence < 1 or replicates < 100:
        raise ValueError("bootstrap needs confidence in (.5, 1) and at least 100 replicates")
    pairs = [(float(candidate[s]), float(reference[s])) for s in sorted(candidate)]
    if any(not isfinite(c) or not isfinite(r) or c < 0 or r <= 0 for c, r in pairs):
        raise ValueError("paired metrics require finite nonnegative values and positive references")
    rng = Random(seed)
    count = len(pairs)
    samples = []
    for _ in range(replicates):
        selected = [pairs[rng.randrange(count)] for _ in pairs]
        samples.append(sum(c for c, _ in selected) / sum(r for _, r in selected))
    return RatioInterval(fmean(c for c, _ in pairs) / fmean(r for _, r in pairs),
                         quantile(samples, 1 - confidence), quantile(samples, confidence),
                         count, confidence)


def compare_workload(rows, candidate_id, *, configuration_count=1, replicates=2000,
                     heldout=True, require_ttft=True, require_decode=True,
                     reference_id="known", expected_seeds=None):
    """Apply gates to one configuration and one declared core workload."""
    if not 1 <= configuration_count <= 3:
        raise ValueError("validation requires one through three frozen configurations")
    by_config = {}
    for row in rows:
        if row["config_id"] not in (candidate_id, reference_id):
            continue
        group = by_config.setdefault(row["config_id"], {})
        if row["seed"] in group:
            raise ValueError("duplicate seed/configuration measurement")
        group[row["seed"]] = row
    candidate, reference = by_config.get(candidate_id, {}), by_config.get(reference_id, {})
    expected = _expected_seeds(heldout, expected_seeds, reference)
    checks = {"paired_seed_set": bool(expected) and set(candidate) == set(reference) == expected}
    result = {"config_id": candidate_id, "reference_id": reference_id, "checks": checks, "metrics": {},
              "interval_semantics": "ratio of mean per-seed metrics; paired seed-cluster bootstrap"}
    if not checks["paired_seed_set"]:
        result["passed"] = False
        return result
    checks["measurement_complete"] = all(candidate[s].get("measurement_complete") is True
                                              and reference[s].get("measurement_complete") is True
                                              for s in expected)
    confidence = 1 - .05 / configuration_count
    specifications = [("busy_useful_work_per_ms", "busy_useful_throughput_ratio", .98, "lower")]
    if require_ttft:
        specifications.append(("p99_ttft_ms", "mean_seed_p99_ttft_ratio", 1.05, "upper"))
    if require_decode:
        specifications.append(("p99_decode_gap_ms", "mean_seed_p99_decode_gap_ratio", 1.05, "upper"))
    for field, label, threshold, direction in specifications:
        try:
            interval = paired_ratio({s: candidate[s][field] for s in expected},
                                    {s: reference[s][field] for s in expected},
                                    confidence=confidence, replicates=replicates,
                                    seed=(f"{candidate_id}:{label}" if reference_id == "known"
                                          else f"{candidate_id}:{reference_id}:{label}"))
        except (KeyError, TypeError, ValueError) as error:
            result["metrics"][label] = {"error": str(error)}
            checks[label] = False
            continue
        result["metrics"][label] = vars(interval) | {"threshold": threshold, "bound": direction}
        checks[label] = interval.lower >= threshold if direction == "lower" else interval.upper <= threshold
    for field in ("unfinished", "starved"):
        checks["no_extra_" + field] = all(
            field in candidate[s] and field in reference[s]
            and candidate[s][field] <= reference[s][field] for s in expected)
    for field, required in (("censored_ttft", require_ttft), ("censored_decode_gaps", require_decode)):
        if required:
            checks["no_" + field] = all(candidate[s].get(field) == reference[s].get(field) == 0
                                            for s in expected)
    for field, required in (("ttft_samples", require_ttft), ("decode_gap_samples", require_decode)):
        if required:
            checks["sufficient_" + field] = all(candidate[s].get(field, 0) >= 100
                                                   and reference[s].get(field, 0) >= 100 for s in expected)
    result["passed"] = all(checks.values())
    return result


def recovery_gate(rows, *, heldout=True, change_busy_ms=0., expected_seeds=None):
    if not isfinite(change_busy_ms) or change_busy_ms < 0:
        raise ValueError("recovery change time must be finite and nonnegative")
    by_seed = {row["seed"]: row for row in rows}
    expected = _expected_seeds(heldout, expected_seeds, by_seed)
    paired = bool(expected) and set(by_seed) == expected and len(by_seed) == len(rows)
    observed = all(row.get("measurement_complete") is True
                   and isinstance(row.get("busy_ms"), (int, float))
                   and isfinite(row["busy_ms"]) and row["busy_ms"] >= change_busy_ms + 60_000
                   for row in by_seed.values())
    recovered = sum(isinstance(row.get("recovery_ms"), (int, float))
                    and not isinstance(row["recovery_ms"], bool)
                    and isfinite(row["recovery_ms"]) and 0 <= row["recovery_ms"] <= 60_000
                    for row in by_seed.values())
    required = ceil(.95 * len(expected))
    return {"passed": paired and observed and recovered >= required, "seeds": len(by_seed),
            "complete_recovery_exposure": observed,
            "recovered_within_60000_busy_ms": recovered, "required": required,
            "definition": "95% of ordinary execution time uses the true best graph in sustained 10-second occupied windows"}


def stationary_reset_gate(rows, *, heldout=True, expected_seeds=None):
    by_seed = {row["seed"]: row for row in rows}
    expected = _expected_seeds(heldout, expected_seeds, by_seed)
    paired = bool(expected) and set(by_seed) == expected and len(by_seed) == len(rows)
    available = all("regime_resets" in row for row in by_seed.values())
    changed = sum(row.get("regime_resets", 0) > 0 for row in by_seed.values())
    return {"passed": paired and available and changed / len(expected) <= .01,
            "seeds": len(by_seed), "seeds_with_reset": changed,
            "semantics": "empirical seed frequency; not a confidence-certified population bound"}


def calibration_gate(rows, *, heldout=True, expected_seeds=None):
    by_seed = {row["seed"]: row for row in rows}
    expected = _expected_seeds(heldout, expected_seeds, by_seed)
    checks = {"seed_set": bool(expected) and set(by_seed) == expected and len(rows) == len(by_seed)}
    metrics = {}
    if not by_seed:
        return {"passed": False, "checks": checks, "metrics": metrics}
    for count in (16, 1000):
        coverage = fmean(float(row[f"covered_{count}"]) for row in by_seed.values())
        metrics[f"coverage_{count}"] = coverage
        checks[f"coverage_{count}"] = coverage >= .90
    width16 = median(row["width_16"] for row in by_seed.values())
    width1000 = median(row["width_1000"] for row in by_seed.values())
    metrics.update(median_width_16=width16, median_width_1000=width1000)
    checks["mature_width"] = width1000 <= width16 / 4
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics,
            "semantics": "95% intervals for the mean at fixed sample counts under the matched Gaussian block likelihood"}
