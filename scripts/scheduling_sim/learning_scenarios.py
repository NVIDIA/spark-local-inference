# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic timing-learning workloads with fixed experiment classifications."""

from dataclasses import dataclass, field

from .model import Graph, Request
from .utility import UtilityConfig
from .workloads import catalog


@dataclass(frozen=True)
class Scenario:
    name: str
    group: str
    requests: tuple[Request, ...]
    graphs: tuple[Graph, ...]
    execution: dict = field(default_factory=dict)
    replenish: bool = True
    idle_gaps_ms: tuple[float, ...] = ()
    initial_observations: int = 0
    warmup_ms: float | None = None
    measurement_ms: float | None = None
    horizon_ms: float | None = None
    recovery_best: str | None = None
    recovery_at_busy_ms: float = 0.0
    utility_config: UtilityConfig = UtilityConfig(16, 4, 100, .1, "service")
    require_ttft: bool = True
    require_decode: bool = True


def _decode(name, *, acceptance=.8, output=128):
    return Request(name, 0, output, acceptance=(acceptance,) * 7)


def scenarios():
    graphs = catalog()
    full = (Request("prompt", 8192, 128),)
    families = {
        "prefill-8192": full,
        "tail-64": (Request("prompt", 64, 1),),
        "tail-576": (Request("prompt", 576, 1),),
        "decode": (_decode("decode"),),
        **{f"prefill-{count}-decoders": (Request("prompt", 8192, 128),
               *(_decode(f"decode-{index}") for index in range(count))) for count in (1, 4, 16)},
    }
    result = {}

    def add(scenario):
        if scenario.name in result:
            raise ValueError("experiment scenario names must be unique")
        result[scenario.name] = scenario

    for family, requests in families.items():
        for cv in (0., .05, .2, .5):
            add(Scenario(f"{family}-cv{cv:g}", "core", requests, graphs, {"cv": cv},
                         require_decode=not family.startswith("tail-")))
    add(Scenario("prefill-4-low-acceptance-decoders-cv0.2", "core",
                 (Request("prompt", 8192, 128),
                  *(_decode(f"decode-{index}", acceptance=0.) for index in range(4))),
                 graphs, {"cv": .2}))
    add(Scenario("prefill-4-heterogeneous-decoders-mixed-cv0.2", "core",
                 (Request("prompt", 8192, 128),
                  *(_decode(f"decode-{index}", acceptance=acceptance)
                    for index, acceptance in enumerate((0., .3, .8, 1.)))),
                 catalog(mixed=True), {"cv": .2}))
    for count in (2, 16):
        add(Scenario(f"{count}-prefills", "core",
                     tuple(Request(f"prompt-{index}", 8192, 128) for index in range(count)), graphs))
    knee = (Graph("prefill-64", 3., prefill_capacity=64),
            Graph("prefill-512", 1., prefill_capacity=512),
            Graph("prefill-4096", 16., prefill_capacity=4096),
            Graph("decode-b1-w1", 15., decode_capacity=1),
            Graph("decode-b1-w8", 30., decode_capacity=1, width=8))
    add(Scenario("catalog-prefill-efficiency-knee", "core", full, knee))
    crossover = tuple(Graph(f"decode-b{batch}-w{width}", duration, decode_capacity=batch, width=width)
                      for batch, timings in ((1, ((1, 5.), (3, 7.), (8, 20.))),
                                             (4, ((1, 7.), (3, 18.), (8, 35.))))
                      for width, duration in timings)
    for count in (1, 4):
        add(Scenario(f"catalog-width-crossover-{count}", "core",
                     tuple(_decode(f"decode-{index}", acceptance=1.) for index in range(count)), crossover))
    for tokens in (64, 576, 8192):
        add(Scenario(f"cold-{tokens}", "transient", (Request("prompt", tokens, 128),), graphs,
                     replenish=False, warmup_ms=0))
    for seconds in (29, 31, 120, 3600):
        add(Scenario(f"idle-{seconds}s", "transient", (Request("prompt", 8192, 1),), graphs,
                     replenish=False, idle_gaps_ms=(seconds * 1000., 0., 0., 0.), warmup_ms=0,
                     require_decode=False))
    add(Scenario("idle-short-burst", "transient", (Request("prompt", 64, 1),), graphs,
                 replenish=False, idle_gaps_ms=(3_600_000., 0., 0., 0., 0., 0., 0.),
                 warmup_ms=0, require_decode=False))
    add(Scenario("mature-1000", "calibration", full, graphs, initial_observations=1000))
    add(Scenario("poisoned-first-10x", "stress", full, graphs,
                 {"poison_first": True, "poison_graphs": ("prefill-4096", "decode-b1-w8")}))
    competing = families["prefill-4-decoders"]
    for rho in (.8, .95):
        add(Scenario(f"correlated-rho{rho:g}", "stress", competing, graphs, {"cv": .2, "rho": rho}))
    add(Scenario("rare-10x-outliers", "stress", competing, graphs,
                 {"cv": .05, "outlier_probability": .001, "outlier_multiplier": 10.}))
    add(Scenario("temporary-global-slowdown", "stress", competing, graphs,
                 {"cv": .05, "drift_at_busy_ms": 30_000., "drift_end_busy_ms": 60_000.,
                  "drift_multipliers": tuple((g.name, 3.) for g in graphs)}))
    add(Scenario("context-mismatch", "stress", full, graphs,
                 {"cv": .05, "context_coefficient": .5}))
    add(Scenario("occupancy-mismatch", "stress", competing, graphs,
                 {"cv": .05, "occupancy_coefficient": .5}))
    expensive = (Graph("decode-one", 10., decode_capacity=1),
                 Graph("decode-four", 12., decode_capacity=4),
                 Graph("decode-sixteen-expensive", 5000., decode_capacity=16))
    add(Scenario("expensive-unknown", "stress", tuple(_decode(f"d{i}", output=256) for i in range(16)),
                 expensive, warmup_ms=0))
    for field in ("missing_probability", "zero_probability"):
        add(Scenario(field.removesuffix("_probability") + "-timings", "stress", competing, graphs,
                     {"cv": .05, field: .1}))
    for count in (2, 4):
        choices = tuple(Graph(f"choice-{i}", 10. * (i + 1), decode_capacity=1) for i in range(count))
        for kind, multipliers in (
                ("slow-fast", (("choice-1", .25),)),
                ("fast-slow", (("choice-0", 3.),))):
            add(Scenario(f"recovery-{count}-{kind}", "recovery", (_decode("decode", output=4096),), choices,
                         {"cv": .05, "drift_at_busy_ms": 30_000., "drift_multipliers": multipliers},
                         recovery_best="choice-1", recovery_at_busy_ms=30_000.,
                         utility_config=UtilityConfig(0, 0, 100, 0, "service")))
        add(Scenario(f"recovery-{count}-poison", "recovery", (_decode("decode", output=4096),), choices,
                     {"cv": .05, "poison_first": True, "poison_graphs": ("choice-0",)},
                     recovery_best="choice-0", recovery_at_busy_ms=0., warmup_ms=0,
                     utility_config=UtilityConfig(0, 0, 100, 0, "service")))
        add(Scenario(f"mature-1000-recovery-{count}-slow-fast", "stress",
                     (_decode("decode", output=4096),), choices,
                     {"cv": .05, "drift_at_busy_ms": 30_000., "drift_multipliers": (("choice-1", .25),)},
                     initial_observations=1000, recovery_best="choice-1", recovery_at_busy_ms=30_000.,
                     utility_config=UtilityConfig(0, 0, 100, 0, "service")))
    return result


def selected_scenarios(selection="all"):
    available = scenarios()
    names = selection.split(",") if isinstance(selection, str) else selection
    selected = {}
    for name in names:
        matches = (available if name == "all" else
                   {n: s for n, s in available.items() if s.group == name})
        if not matches:
            if name not in available:
                raise ValueError(f"unknown learning scenario or group: {name}")
            matches = {name: available[name]}
        selected.update(matches)
    return tuple(selected.values())


SMOKE_SCENARIOS = "prefill-8192-cv0,idle-31s,recovery-2-slow-fast"
SCREEN_SCENARIOS = ("prefill-8192-cv0,prefill-4-decoders-cv0.2,tail-64-cv0.2,"
                    "catalog-prefill-efficiency-knee,recovery-2-slow-fast,poisoned-first-10x,"
                    "mature-1000-recovery-2-slow-fast")
