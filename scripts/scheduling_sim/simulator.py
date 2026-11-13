# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nonpreemptive execution, hidden EOS, and sampled acceptance outcomes."""

from dataclasses import dataclass, field
from math import isfinite
from random import Random

from .model import Graph, Request, Value, View
from .scheduler import choose
from .utility import UtilityScheduler


@dataclass
class State:
    request: Request
    remaining_prefill: int
    generated: int = 0
    last_service_ms: float = 0.0
    deliveries: list[tuple[float, int]] = field(default_factory=list)
    completion_ms: float | None = None

    def view(self) -> View:
        r = self.request
        return View(r.name, self.remaining_prefill, r.max_new_tokens - self.generated,
                    r.acceptance, r.weight, self.last_service_ms)


@dataclass
class Result:
    summary: dict
    requests: list[dict]
    trace: list[dict]


def run(requests: list[Request], catalog: tuple[Graph, ...], value: Value,
        *, seed: int = 0, horizon_ms: float = 10_000, max_steps: int = 100_000,
        scheduler: UtilityScheduler | None = None) -> Result:
    if len({r.name for r in requests}) != len(requests):
        raise ValueError("request names must be unique")
    if not isfinite(horizon_ms) or horizon_ms <= 0 or max_steps < 1:
        raise ValueError("simulation limits must be positive and finite")
    states = {r.name: State(r, r.prompt_tokens, last_service_ms=r.arrival_ms) for r in requests}
    rngs = {r.name: Random(f"{seed}:{r.name}") for r in requests}
    now = busy = 0.0
    events = []
    while now < horizon_ms:
        unfinished = [s for s in states.values() if s.completion_ms is None]
        if not unfinished:
            break
        active = [s for s in unfinished if s.request.arrival_ms <= now]
        if not active:
            arrival = min(s.request.arrival_ms for s in unfinished)
            now = min(arrival, horizon_ms)
            continue
        if len(events) >= max_steps:
            raise RuntimeError("simulation exceeded the graph launch limit")
        views = [s.view() for s in active]
        action = scheduler.choose(views, catalog, now) if scheduler is not None else choose(views, catalog, value)
        selection_score = scheduler.score(action) if scheduler is not None else action.score(value)
        end = now + action.graph.duration_ms
        committed = []
        for lane in action.lanes:
            state = states[lane.name]
            r = state.request
            emitted = 0
            if lane.prefill:
                state.remaining_prefill -= lane.prefill
                emitted = int(state.remaining_prefill == 0)
            else:
                emitted = 1
                remaining = min(r.output_tokens, r.max_new_tokens) - state.generated
                for probability in r.acceptance[:lane.width - 1]:
                    if emitted >= remaining or rngs[r.name].random() >= probability:
                        break
                    emitted += 1
            emitted = min(emitted, min(r.output_tokens, r.max_new_tokens) - state.generated)
            state.generated += emitted
            state.last_service_ms = end
            if emitted:
                state.deliveries.append((end, emitted))
            if state.generated == min(r.output_tokens, r.max_new_tokens):
                state.completion_ms = end
            committed.append({"request": lane.name, "prefill": lane.prefill,
                              "width": lane.width, "expected_output": lane.expected_output,
                              "emitted": emitted, "weight": lane.weight})
        events.append({"start_ms": now, "end_ms": end, "graph": action.graph.name,
                       "duration_ms": action.graph.duration_ms,
                       "prefill_capacity": action.graph.prefill_capacity,
                       "decode_capacity": action.graph.decode_capacity,
                       "verification_width": action.graph.width,
                       "expected_work": action.expected_work(value), "score": action.score(value),
                       "selection_score": selection_score,
                       "selection_log_scale": scheduler.log_scale if scheduler is not None else 0.0,
                       "lanes": committed})
        if scheduler is not None:
            scheduler.observe(action.graph.duration_ms, committed)
        busy += action.graph.duration_ms
        now = end
    rows = []
    for state in states.values():
        r = state.request
        times = [t for t, _ in state.deliveries]
        gaps = [b - a for a, b in zip(times, times[1:])]
        if times and r.prompt_tokens == 0:
            gaps.insert(0, times[0] - r.arrival_ms)
        if times and state.completion_ms is None:
            gaps.append(now - times[-1])
        elif not times and r.prompt_tokens == 0:
            gaps.append(max(0.0, now - r.arrival_ms))
        rows.append({"name": r.name, "prefill_tokens": r.prompt_tokens - state.remaining_prefill,
                     "output_tokens": state.generated,
                     "ttft_ms": times[0] - r.arrival_ms if times else None,
                     "completion_ms": state.completion_ms,
                     "max_output_gap_ms": max(gaps, default=0.0),
                     "decode_age_area_ms2": sum(gap * gap / 2 for gap in gaps),
                     "unfinished": state.completion_ms is None})
    outputs = sum(r["output_tokens"] for r in rows)
    prefill = sum(r["prefill_tokens"] for r in rows)
    weighted_work = sum((r["prefill_tokens"] + value.decode_value * r["output_tokens"]) * spec.weight
                        for r, spec in zip(rows, requests))
    summary = {"elapsed_ms": now, "busy_ms": busy, "launches": len(events),
               "prefill_tokens": prefill, "output_tokens": outputs,
               "prefill_tok_s": prefill * 1000 / now if now else 0.0,
               "output_tok_s": outputs * 1000 / now if now else 0.0,
               "useful_work_per_ms": weighted_work / now if now else 0.0,
               "max_output_gap_ms": max((r["max_output_gap_ms"] for r in rows), default=0.0),
               "decode_age_area_ms2": sum(r["decode_age_area_ms2"] for r in rows),
               "unfinished": sum(r["unfinished"] for r in rows)}
    return Result(summary, rows, events)
