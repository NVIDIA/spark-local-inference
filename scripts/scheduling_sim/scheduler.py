# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Maximum expected useful work per millisecond over prepared graph choices."""

from collections.abc import Callable

from .model import Action, Graph, Lane, Value, View, expected_output


def candidates(views: list[View], catalog: tuple[Graph, ...], value: Value,
               *, lane_value: Callable[[Lane, Graph], float] | None = None) -> list[Action]:
    if len({v.name for v in views}) != len(views):
        raise ValueError("request names must be unique")
    prefills = [v for v in views if v.remaining_prefill]
    decodes = [v for v in views if not v.remaining_prefill]
    actions = []
    reward = lane_value if lane_value is not None else lambda lane, graph: lane.useful_work(value)
    for graph in catalog:
        rows = []
        if graph.decode_capacity:
            for view in decodes:
                width = min(graph.width, view.output_budget)
                emitted = expected_output(view.acceptance, width)
                rows.append((view, Lane(view.name, 0, width, emitted, view.weight)))
            # Shape cost is fixed: its optimal subset contains the largest
            # weighted contributions. Empty physical rows earn no work.
            rows.sort(key=lambda row: (-reward(row[1], graph), row[0].last_service_ms, row[0].name))
            rows = [lane for _, lane in rows[:graph.decode_capacity]]
        for view in [None, *prefills] if graph.prefill_capacity else [None]:
            lanes = list(rows)
            if view is not None:
                tokens = min(view.remaining_prefill, graph.prefill_capacity)
                emitted = float(tokens == view.remaining_prefill)
                lanes.append(Lane(view.name, tokens, 0, emitted, view.weight))
            if lanes:
                actions.append(Action(graph, tuple(lanes)))
    return actions


def rank(views: list[View], catalog: tuple[Graph, ...], value: Value) -> list[Action]:
    clocks = {view.name: view.last_service_ms for view in views}
    return sorted(candidates(views, catalog, value), key=lambda a: (
        -a.score(value), a.graph.duration_ms,
        min(clocks[lane.name] for lane in a.lanes),
        a.graph.prefill_capacity + a.graph.decode_capacity * a.graph.width,
        a.graph.name, tuple(lane.name for lane in a.lanes),
    ))


def choose(views: list[View], catalog: tuple[Graph, ...], value: Value) -> Action:
    ranked = rank(views, catalog, value)
    if not ranked:
        raise ValueError("the prepared catalog cannot advance a live request")
    return ranked[0]
