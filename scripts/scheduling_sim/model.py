# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit graph costs and useful work. Times are milliseconds."""

from dataclasses import dataclass
from math import isfinite


def expected_output(probabilities: tuple[float, ...], width: int) -> float:
    if width < 1 or width > len(probabilities) + 1:
        raise ValueError("verification width exceeds the acceptance profile")
    if any(not isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("conditional acceptance probabilities must be in [0, 1]")
    survival = 1.0
    output = 1.0
    for probability in probabilities[:width - 1]:
        survival *= probability
        output += survival
    return output


@dataclass(frozen=True)
class Graph:
    name: str
    duration_ms: float
    prefill_capacity: int = 0
    decode_capacity: int = 0
    width: int = 1

    def __post_init__(self):
        if not self.name or not isfinite(self.duration_ms) or self.duration_ms <= 0:
            raise ValueError("graphs require a name and a positive finite duration")
        if min(self.prefill_capacity, self.decode_capacity) < 0 or self.width < 1:
            raise ValueError("invalid graph capacities")
        if not self.prefill_capacity and not self.decode_capacity:
            raise ValueError("a graph must have a prefill or decode lane")
        if not self.decode_capacity and self.width != 1:
            raise ValueError("prefill-only graphs cannot have speculative lanes")


@dataclass(frozen=True)
class View:
    """Observable state. The actual EOS position is absent."""

    name: str
    remaining_prefill: int
    output_budget: int
    acceptance: tuple[float, ...]
    weight: float = 1.0
    last_service_ms: float = 0.0

    def __post_init__(self):
        if self.remaining_prefill < 0 or self.output_budget < 1:
            raise ValueError("invalid remaining request work")
        if not isfinite(self.weight) or self.weight <= 0:
            raise ValueError("request weight must be positive and finite")
        expected_output(self.acceptance, 1)


@dataclass(frozen=True)
class Value:
    """One prefill token is one unit; one output token is decode_value units."""

    decode_value: float

    def __post_init__(self):
        if not isfinite(self.decode_value) or self.decode_value <= 0:
            raise ValueError("decode token value must be positive and finite")


@dataclass(frozen=True)
class Lane:
    name: str
    prefill: int
    width: int
    expected_output: float
    weight: float

    def useful_work(self, value: Value) -> float:
        return self.weight * (self.prefill + value.decode_value * self.expected_output)


@dataclass(frozen=True)
class Action:
    graph: Graph
    lanes: tuple[Lane, ...]

    def expected_work(self, value: Value) -> float:
        return sum(lane.useful_work(value) for lane in self.lanes)

    def score(self, value: Value) -> float:
        return self.expected_work(value) / self.graph.duration_ms


@dataclass(frozen=True)
class Request:
    name: str
    prompt_tokens: int
    output_tokens: int
    arrival_ms: float = 0.0
    acceptance: tuple[float, ...] = (1.0,) * 7
    max_new_tokens: int = 4096
    weight: float = 1.0

    def __post_init__(self):
        if not self.name or self.prompt_tokens < 0 or self.output_tokens < 1:
            raise ValueError("invalid request dimensions")
        if not isfinite(self.arrival_ms) or self.arrival_ms < 0:
            raise ValueError("arrival must be finite and nonnegative")
        View(self.name, self.prompt_tokens, self.max_new_tokens, self.acceptance, self.weight)
