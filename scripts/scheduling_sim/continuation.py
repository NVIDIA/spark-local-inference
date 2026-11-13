# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serial-policy continuation values composed from independent request moments."""

from dataclasses import dataclass, replace
from math import isfinite

from .oracle import Decision, Metrics, Objective, Oracle, Progress, State


@dataclass(frozen=True)
class Isolated:
    metrics: Metrics = Metrics()
    duration_second_ms2: float = 0.0
    first_output_ms: float = 0.0


@dataclass(frozen=True)
class Features:
    isolated: Metrics = Metrics()
    duration_second_ms2: float = 0.0
    live: float = 0.0
    prefill: float = 0.0
    decode: float = 0.0
    decode_age_ms: float = 0.0
    decode_first_ms: float = 0.0
    age_until_first_ms2: float = 0.0

    def plus(self, other: "Features", probability: float) -> "Features":
        return Features(self.isolated.plus(other.isolated, probability),
                        *(getattr(self, name) + probability * getattr(other, name)
                          for name in self.__dataclass_fields__ if name != "isolated"))


class SerialReference:
    """Finish each request in input order using isolated useful-work-rate choices.

    Per-request tables contain only remaining prompt and attained output states.
    Joint acceptance branches are unnecessary when combining their moments.
    """

    def __init__(self, oracle: Oracle, objective: Objective, *, time_value: float):
        if not isfinite(time_value) or time_value <= 0:
            raise ValueError("GPU time value must be positive and finite")
        self.oracle = oracle
        self.objective = objective
        self.time_value = time_value
        self.single = tuple(Oracle((replace(job, age_ms=0, weight=1),), oracle.graphs,
                                   max_states=oracle.max_states) for job in oracle.jobs)
        self.tables: list[dict[tuple[int, int], Isolated]] = [dict() for _ in oracle.jobs]
        self.steps: list[dict[tuple[int, int], tuple]] = [dict() for _ in oracle.jobs]

    def isolated(self, index: int, progress: Progress) -> Isolated:
        if progress.done:
            return Isolated()
        table, steps = self.tables[index], self.steps[index]
        root = progress.prefill, progress.generated
        stack = [root]
        model = self.single[index]
        while stack:
            key = stack[-1]
            if key in table:
                stack.pop()
                continue
            if key not in steps:
                state = (Progress(*key, 0.0),)
                action = model.immediate(self.objective)(state)
                metrics, branches = model.transition(state, action)
                steps[key] = action, metrics, branches
            action, immediate, branches = steps[key]
            missing = [(p[0].prefill, p[0].generated) for _, p in branches
                       if not p[0].done and (p[0].prefill, p[0].generated) not in table]
            if missing:
                stack.extend(missing)
                continue
            metrics = immediate
            second = immediate.elapsed_ms**2
            first = immediate.elapsed_ms
            for probability, successor in branches:
                p = successor[0]
                child = Isolated() if p.done else table[p.prefill, p.generated]
                metrics = metrics.plus(child.metrics, probability)
                second += probability * (2 * immediate.elapsed_ms * child.metrics.elapsed_ms
                                         + child.duration_second_ms2)
                if p.generated == key[1]:
                    first += probability * child.first_output_ms
            table[key] = Isolated(metrics, second, first)
            stack.pop()
        return table[root]

    def _features(self, index: int, progress: Progress) -> Features:
        if progress.done:
            return Features()
        solo = self.isolated(index, progress)
        decode = float(progress.prefill == 0)
        return Features(solo.metrics, solo.duration_second_ms2, 1.0, 1.0 - decode,
                        decode, decode * progress.age_ms, decode * solo.first_output_ms,
                        decode * progress.age_ms * solo.first_output_ms)

    def _combine(self, features: tuple[Features, ...]) -> Metrics:
        mean = second = 0.0
        total = Metrics()
        for job, f in zip(self.oracle.jobs, features):
            m, w = f.isolated, job.weight
            total = total.plus(Metrics(
                m.elapsed_ms, w * m.prefill_tokens, w * m.output_tokens,
                w * (m.prefill_wait_ms + f.prefill * mean),
                w * (m.unfinished_wait_ms + f.live * mean),
                w * (m.decode_age_ms2 + f.age_until_first_ms2
                     + (f.decode_age_ms + f.decode_first_ms) * mean + 0.5 * f.decode * second)))
            second += f.duration_second_ms2 + 2 * mean * m.elapsed_ms
            mean += m.elapsed_ms
        return total

    def metrics(self, state: State) -> Metrics:
        return self._combine(tuple(self._features(i, p) for i, p in enumerate(state)))

    def value(self, state: State) -> float:
        metrics = self.metrics(state)
        return self.objective.net_work(metrics) - self.time_value * metrics.elapsed_ms

    def choose_serial(self, state: State) -> Decision:
        index = next(i for i, p in enumerate(state) if not p.done)
        p = state[index]
        self.isolated(index, p)
        action = self.steps[index][p.prefill, p.generated][0]
        return Decision(action.graph, index if action.prefill is not None else None,
                        (index,) if action.decodes else ())

    def expected_after(self, state: State, action: Decision) -> Metrics:
        features = []
        duration = action.graph.duration_ms
        for i, p in enumerate(state):
            if p.done:
                features.append(Features())
                continue
            tokens = min(p.prefill, action.graph.prefill_capacity) if action.prefill == i else 0
            emits = i in action.decodes or (tokens > 0 and tokens == p.prefill)
            if emits:
                width = action.graph.width if i in action.decodes else 1
                feature = Features()
                for probability, emitted, done in self.oracle._output(i, p, width):
                    after = Progress(p.prefill - tokens, p.generated + emitted, 0.0, done)
                    feature = feature.plus(self._features(i, after), probability)
                features.append(feature)
            else:
                after = Progress(p.prefill - tokens, p.generated, p.age_ms + duration if not p.prefill else 0)
                features.append(self._features(i, after))
        return self._combine(tuple(features))

    def immediate_metrics(self, state: State, action: Decision) -> Metrics:
        duration = action.graph.duration_ms
        prefill = output = first_wait = unfinished = age = 0.0
        for i, (job, p) in enumerate(zip(self.oracle.jobs, state)):
            if p.done:
                continue
            w = job.weight
            unfinished += w * duration
            if p.prefill:
                first_wait += w * duration
            else:
                age += w * (p.age_ms * duration + duration * duration / 2)
            tokens = min(p.prefill, action.graph.prefill_capacity) if action.prefill == i else 0
            prefill += w * tokens
            if i in action.decodes or (tokens > 0 and tokens == p.prefill):
                width = action.graph.width if i in action.decodes else 1
                output += w * sum(chance * count for chance, count, _ in self.oracle._output(i, p, width))
        return Metrics(duration, prefill, output, first_wait, unfinished, age)

    def score(self, state: State, action: Decision) -> float:
        immediate = self.immediate_metrics(state, action)
        after = self.expected_after(state, action)
        following = self.objective.net_work(after) - self.time_value * after.elapsed_ms
        return (self.objective.net_work(immediate) + following - self.value(state)) / immediate.elapsed_ms

    def choose(self, state: State) -> Decision:
        return max(self.oracle.choices(state), key=lambda action: self.score(state, action))

    @property
    def table_entries(self) -> int:
        return sum(len(table) for table in self.tables)
