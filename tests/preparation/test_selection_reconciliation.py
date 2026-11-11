# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Rank-specific preparation catalogs keep common operator storage compatible."""

from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from b12x._lib.scratch import ScratchBufferSpec
from b12x.preparation import DetectedDevice, MemoryRequirements, Plan, PreparationSession, PreparedCall

from ignition.components.native_operator import NativeConfig, native_contract
from ignition.preparation.collectives import CollectivePreparation, _selected_plan_catalog, dist


@dataclass(frozen=True)
class Query:
    rows: int


def request(name, *, split=3, rows=1, dependencies=(), pinned=False):
    contract = native_contract(component_id="test.rank_scratch", query_type=Query,
        tactics=lambda query: (split, 5 - split))
    plan = Plan(contract=contract, query=Query(rows), shared=not dependencies,
        override=NativeConfig(split) if pinned else None,
        _compile_jobs=lambda config, device: (),
        _memory_requirements=lambda config, device: MemoryRequirements(scratch=(
            ScratchBufferSpec(name="partial", shape=(config.tactic * 263168,),
                              dtype=torch.uint8, device=torch.device("cpu")),)),
        _materialize=lambda selection, device: SimpleNamespace(split=selection.config.tactic))
    return plan.request(name=name, dependencies=dependencies,
                        prepare_call=lambda state: PreparedCall(run=lambda: state.split))


def exchange(monkeypatch, requests):
    owner = CollectivePreparation.__new__(CollectivePreparation)
    owner.group, owner.ranks = object(), tuple(range(len(requests)))
    catalogs = []
    for rank in requests:
        catalog = {}
        for _, key, payload, config in _selected_plan_catalog(rank):
            catalog.setdefault(key, {}).setdefault(payload, config)
        catalogs.append(catalog)
    monkeypatch.setattr(dist, "all_gather_object",
                        lambda target, local, **kwargs: target.__setitem__(slice(None), catalogs))
    return owner


@pytest.mark.parametrize("repeat_preparation", [False, True])
def test_common_plans_reconcile_before_scratch_binding(monkeypatch, repeat_preparation):
    ranks = [(request("attention"), request("rank_zero_only", rows=2)),
             (request("attention", split=2),)]
    with ExitStack() as scope:
        sessions = [scope.enter_context(PreparationSession(device=DetectedDevice(None, None),
                    compile_workers=0, prime=False)) for _ in ranks]
        for session, requests in zip(sessions, ranks, strict=True):
            session.prepare(requests, autotune=False)
            if repeat_preparation:
                session.prepare(requests, autotune=False)
        original_handles = [requests[0].plan.handle for requests in ranks]
        private_state = ranks[0][1].plan.prepared
        assert [requests[0].plan.scratch_specs()[0].nbytes for requests in ranks] == [789504, 526336]

        owner = exchange(monkeypatch, ranks)
        for session, requests in zip(sessions, ranks, strict=True):
            configs, changed = owner.reconcile_selections(requests)
            assert changed
            session.prepare(requests, autotune=False, selected_configs=configs)

        assert [requests[0].plan.scratch_specs()[0].nbytes for requests in ranks] == [789504, 789504]
        assert [requests[0].plan.handle for requests in ranks] == original_handles
        assert ranks[0][1].plan.prepared is private_state
        assert ranks[1][0].plan.prepared.state.split == 3


def test_matching_selections_need_no_rematerialization(monkeypatch):
    ranks = [(request("attention"), request("rank_zero_only", rows=2)), (request("attention"),)]
    with ExitStack() as scope:
        for requests in ranks:
            session = scope.enter_context(PreparationSession(device=DetectedDevice(None, None),
                compile_workers=0, prime=False))
            session.prepare(requests, autotune=False)
        owner = exchange(monkeypatch, ranks)
        assert all(owner.reconcile_selections(requests) == ({}, False) for requests in ranks)


def test_duplicate_declarations_do_not_hide_local_disagreement(monkeypatch):
    ranks = [(request("a"), request("b", split=2)), (request("attention"),)]
    with ExitStack() as scope:
        for requests in ranks:
            for item in requests:
                session = scope.enter_context(PreparationSession(device=DetectedDevice(None, None),
                    compile_workers=0, prime=False))
                session.prepare((item,), autotune=False)
        owner = exchange(monkeypatch, ranks)
        configs, changed = owner.reconcile_selections(ranks[0])
        assert changed and configs == {ranks[0][1].plan: NativeConfig(3)}


@pytest.mark.parametrize("distinct", ["override", "dependency"])
def test_distinct_declarations_keep_their_selected_configurations(monkeypatch, distinct):
    ranks = [(request("input", rows=2), request("attention", dependencies=("input",), pinned=distinct == "override")),
             (request("input", rows=3 if distinct == "dependency" else 2),
              request("attention", split=2, dependencies=("input",), pinned=distinct == "override"))]
    with ExitStack() as scope:
        for requests in ranks:
            session = scope.enter_context(PreparationSession(device=DetectedDevice(None, None),
                compile_workers=0, prime=False))
            session.prepare(requests, autotune=False)
        owner = exchange(monkeypatch, ranks)
        assert all(owner.reconcile_selections(requests) == ({}, False) for requests in ranks)
