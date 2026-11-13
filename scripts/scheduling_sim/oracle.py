# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact completion-episode planning with hidden EOS and observed acceptance.

Useful work, GPU time, and waiting have explicit prices. A separate ratio
solver tests normalization by episode duration. All requests are present at
the root; every policy must finish them. Graphs have no duration ceiling.
"""

from dataclasses import dataclass
from functools import cache
from itertools import combinations, product
from math import isfinite
from typing import Callable

from .model import Graph, expected_output


@dataclass(frozen=True)
class Job:
    name: str
    prefill: int
    length_mass: tuple[tuple[int, float], ...]
    acceptance: tuple[float, ...] = (1.0,) * 7
    generated: int = 0
    age_ms: float = 0.0
    weight: float = 1.0

    def __post_init__(self):
        if not self.name or any(not isinstance(n, int) or n < 0 for n in (self.prefill, self.generated)):
            raise ValueError("invalid job name or remaining work")
        if self.prefill and (self.generated or self.age_ms):
            raise ValueError("a prefill job cannot already have output or stream age")
        if not isfinite(self.age_ms) or self.age_ms < 0 or not isfinite(self.weight) or self.weight <= 0:
            raise ValueError("invalid job age or weight")
        if not self.length_mass or len({n for n, _ in self.length_mass}) != len(self.length_mass):
            raise ValueError("output lengths require a nonempty, unique discrete prior")
        if any(not isinstance(n, int) or n < 1 or not isfinite(p) or p <= 0 for n, p in self.length_mass):
            raise ValueError("output lengths and their probability masses must be positive")
        if not any(n > self.generated for n, _ in self.length_mass):
            raise ValueError("the prior assigns no probability to a still-live job")
        expected_output(self.acceptance, 1)


@dataclass(frozen=True)
class Progress:
    prefill: int
    generated: int
    age_ms: float
    done: bool = False


State = tuple[Progress, ...]


@dataclass(frozen=True)
class Decision:
    graph: Graph
    prefill: int | None = None
    decodes: tuple[int, ...] = ()


@dataclass(frozen=True)
class Metrics:
    elapsed_ms: float = 0.0
    prefill_tokens: float = 0.0
    output_tokens: float = 0.0
    prefill_wait_ms: float = 0.0
    unfinished_wait_ms: float = 0.0
    decode_age_ms2: float = 0.0

    def plus(self, other: "Metrics", probability: float = 1.0) -> "Metrics":
        return Metrics(*(getattr(self, field) + probability * getattr(other, field)
                         for field in self.__dataclass_fields__))


@dataclass(frozen=True)
class Objective:
    decode_value: float
    prefill_wait_price: float = 0.0
    unfinished_wait_price: float = 0.0
    decode_age_price: float = 0.0

    def __post_init__(self):
        if not isfinite(self.decode_value) or self.decode_value <= 0:
            raise ValueError("decode value must be positive and finite")
        if any(not isfinite(p) or p < 0 for p in (
                self.prefill_wait_price, self.unfinished_wait_price, self.decode_age_price)):
            raise ValueError("waiting prices must be finite and nonnegative")

    def useful_work(self, metrics: Metrics) -> float:
        return metrics.prefill_tokens + self.decode_value * metrics.output_tokens

    def net_work(self, metrics: Metrics) -> float:
        return (self.useful_work(metrics) - self.prefill_wait_price * metrics.prefill_wait_ms
                - self.unfinished_wait_price * metrics.unfinished_wait_ms
                - self.decode_age_price * metrics.decode_age_ms2)

    def rate(self, metrics: Metrics) -> float:
        return self.net_work(metrics) / metrics.elapsed_ms


@dataclass(frozen=True)
class Result:
    first: Decision | None
    metrics: Metrics
    rate: float
    states: int
    iterations: int = 0
    residual: float = 0.0


class Oracle:
    def __init__(self, jobs: tuple[Job, ...], graphs: tuple[Graph, ...], *, max_states: int = 100_000):
        if not jobs or len({j.name for j in jobs}) != len(jobs):
            raise ValueError("oracle jobs require nonempty unique names")
        if not graphs or len({g.name for g in graphs}) != len(graphs):
            raise ValueError("oracle graphs require nonempty unique names")
        if max_states < 1:
            raise ValueError("the oracle state limit must be positive")
        self.jobs = jobs
        self.graphs = graphs
        self.root = tuple(Progress(j.prefill, j.generated, j.age_ms) for j in jobs)
        self.max_states = max_states
        self._seen: set[State] = set()
        self._transitions: dict[tuple[State, Decision], tuple[Metrics, tuple[tuple[float, State], ...]]] = {}

    def choices(self, state: State) -> tuple[Decision, ...]:
        self._seen.add(state)
        if len(self._seen) > self.max_states:
            raise RuntimeError(f"oracle exceeded {self.max_states} states")
        prefills = [i for i, p in enumerate(state) if not p.done and p.prefill]
        decodes = [i for i, p in enumerate(state) if not p.done and not p.prefill]
        if not prefills and not decodes:
            return ()
        choices = []
        for graph in self.graphs:
            for count in range(min(len(decodes), graph.decode_capacity) + 1):
                for cohort in combinations(decodes, count):
                    for prefill in (None, *prefills) if graph.prefill_capacity else (None,):
                        if cohort or prefill is not None:
                            choices.append(Decision(graph, prefill, cohort))
        if not choices:
            raise ValueError("the graph catalog cannot finish the oracle workload")
        return tuple(choices)

    def _output(self, index: int, progress: Progress, width: int) -> tuple[tuple[float, int, bool], ...]:
        job = self.jobs[index]
        lengths = [(n, mass) for n, mass in job.length_mass if n > progress.generated]
        total = sum(mass for _, mass in lengths)
        width = min(width, max(n for n, _ in lengths) - progress.generated)
        expected_output(job.acceptance, width)
        accepted = []
        survival = 1.0
        for emitted, probability in enumerate(job.acceptance[:width - 1], 1):
            accepted.append((emitted, survival * (1 - probability)))
            survival *= probability
        accepted.append((width, survival))
        outcomes: dict[tuple[int, bool], float] = {}
        for (length, mass), (count, chance) in product(lengths, accepted):
            if not chance:
                continue
            remaining = length - progress.generated
            key = (min(remaining, count), remaining <= count)
            outcomes[key] = outcomes.get(key, 0.0) + mass / total * chance
        return tuple((p, emitted, done) for (emitted, done), p in sorted(outcomes.items()))

    def transition(self, state: State, action: Decision) -> tuple[Metrics, tuple[tuple[float, State], ...]]:
        key = state, action
        if key in self._transitions:
            return self._transitions[key]
        duration = action.graph.duration_ms
        updated = tuple(Progress(p.prefill, p.generated, p.age_ms + duration if not p.prefill and not p.done else 0,
                                 p.done) for p in state)
        branches = [(1.0, updated)]
        prefill_work = output_work = 0.0
        for index, before in enumerate(state):
            tokens = min(before.prefill, action.graph.prefill_capacity) if action.prefill == index else 0
            if tokens:
                prefill_work += self.jobs[index].weight * tokens
            emits = index in action.decodes or (tokens > 0 and tokens == before.prefill)
            if emits:
                width = action.graph.width if index in action.decodes else 1
                outcomes = self._output(index, before, width)
                output_work += self.jobs[index].weight * sum(p * count for p, count, _ in outcomes)
                successors = []
                for (branch_p, branch), (chance, count, done) in product(branches, outcomes):
                    after = list(branch)
                    after[index] = Progress(before.prefill - tokens, before.generated + count, 0.0, done)
                    successors.append((branch_p * chance, tuple(after)))
                branches = successors
            elif tokens:
                branches = [(chance, tuple(Progress(before.prefill - tokens, before.generated, 0.0)
                                          if i == index else p for i, p in enumerate(branch)))
                            for chance, branch in branches]
        metrics = Metrics(duration, prefill_work, output_work,
                          duration * sum(j.weight for j, p in zip(self.jobs, state) if p.prefill and not p.done),
                          duration * sum(j.weight for j, p in zip(self.jobs, state) if not p.done),
                          sum(j.weight * (p.age_ms * duration + duration * duration / 2)
                              for j, p in zip(self.jobs, state) if not p.prefill and not p.done))
        result = metrics, tuple(branches)
        self._transitions[key] = result
        return result

    def _total(self, state: State, action: Decision, continuation: Callable[[State], Metrics]) -> Metrics:
        total, branches = self.transition(state, action)
        for probability, successor in branches:
            total = total.plus(continuation(successor), probability)
        return total

    def evaluate(self, policy: Callable[[State], Decision], objective: Objective) -> Result:
        @cache
        def visit(state):
            choices = self.choices(state)
            if not choices:
                return Metrics()
            action = policy(state)
            if action not in choices:
                raise ValueError("policy returned an unavailable action")
            return self._total(state, action, visit)

        metrics = visit(self.root)
        return Result(policy(self.root), metrics, objective.rate(metrics), len(self._seen))

    def immediate(self, objective: Objective, *, waiting: bool = False) -> Callable[[State], Decision]:
        def choose(state):
            def score(action):
                metrics, _ = self.transition(state, action)
                reward = objective.net_work(metrics) if waiting else objective.useful_work(metrics)
                return reward / metrics.elapsed_ms
            return max(self.choices(state), key=score)
        return choose

    def rollout(self, objective: Objective, *, time_value: float,
                baseline: Callable[[State], Decision] | None = None) -> Callable[[State], Decision]:
        """Price future states by a supplied policy's expected completion value.

        One fixed positive time price defines the continuation potential. Each
        action maximizes expected reward plus potential change per unit time.
        With exact baseline values this cannot reduce expected episode surplus.
        The default baseline chooses immediate useful work per unit time.
        """
        if not isfinite(time_value) or time_value <= 0:
            raise ValueError("GPU time value must be positive and finite")
        baseline = baseline or self.immediate(objective)

        @cache
        def continuation(state):
            if not self.choices(state):
                return Metrics()
            return self._total(state, baseline(state), continuation)

        def potential(state):
            metrics = continuation(state)
            return objective.net_work(metrics) - time_value * metrics.elapsed_ms

        @cache
        def choose(state):
            before = potential(state)

            def score(action):
                metrics, branches = self.transition(state, action)
                after = sum(chance * potential(successor) for chance, successor in branches)
                return (objective.net_work(metrics) + after - before) / metrics.elapsed_ms

            return max(self.choices(state), key=score)

        return choose

    def _solve_at_price(self, objective: Objective, time_value: float) -> tuple[Decision, Metrics]:
        policy = {}

        @cache
        def visit(state):
            choices = self.choices(state)
            if not choices:
                return Metrics()
            totals = [(action, self._total(state, action, visit)) for action in choices]
            action, metrics = max(totals, key=lambda pair: objective.net_work(pair[1]) - time_value * pair[1].elapsed_ms)
            policy[state] = action
            return metrics

        metrics = visit(self.root)
        return policy[self.root], metrics

    def solve_priced(self, objective: Objective, *, time_value: float) -> Result:
        """Maximize expected useful work minus waiting costs and priced GPU time."""
        if not isfinite(time_value) or time_value <= 0:
            raise ValueError("GPU time value must be positive and finite")
        first, metrics = self._solve_at_price(objective, time_value)
        return Result(first, metrics, objective.rate(metrics), len(self._seen), 1)

    def solve(self, objective: Objective, *, tolerance: float = 1e-10, max_iterations: int = 100) -> Result:
        if not isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
            raise ValueError("invalid oracle solver limits")
        rate = 0.0
        for iteration in range(1, max_iterations + 1):
            first, metrics = self._solve_at_price(objective, rate)
            residual = objective.net_work(metrics) - rate * metrics.elapsed_ms
            if abs(residual) <= tolerance * max(1.0, abs(objective.net_work(metrics))):
                return Result(first, metrics, objective.rate(metrics), len(self._seen), iteration, residual)
            rate = objective.rate(metrics)
        raise RuntimeError(f"oracle ratio optimization did not converge: residual {residual:g}")
