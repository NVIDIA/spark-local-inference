# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharded tuning relays the contributions required by b12x preparation jobs."""

from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from ignition.preparation.collectives import _tuning_results
from ignition.preparation.tuning_replicas import authorization


@pytest.mark.parametrize("replicas", [False, True])
@pytest.mark.parametrize("values", [(1, 2), (1, 2, 3, 4, 5, 6)])
def test_finalist_and_ordinary_jobs_consolidate_across_four_shards(tmp_path, monkeypatch, replicas, values):
    from b12x.preparation import (
        DetectedDevice, FinalistPolicy, Knob, MemoryRequirements, Plan,
        PreparationSession, PreparedCall, TuningContract, _measurement,
    )
    from b12x.preparation._cache import SelectionCache

    def prepare(calls, **kwargs):
        yield
        return SimpleNamespace(calls=calls, close=lambda: None, completed_rounds=0,
            planned_rounds=0, active_count=len(calls), latest_round_us=())

    def measure(race, **kwargs):
        yield
        return _measurement.RaceMeasurements(tuple(float(call.output) for call in race.calls), 0)

    monkeypatch.setattr(_measurement, "prepare_race_steps", prepare)
    monkeypatch.setattr(_measurement, "measure_race_steps", measure)
    contract = TuningContract(
        component_id="test.finalist_consolidation", query_schema_version=1, config_schema_version=1,
        query_fields=frozenset({"rows"}), config_fields=frozenset({"width"}),
        encode_query=lambda query: {"rows": query}, encode_config=lambda config: {"width": config},
        decode_config=lambda payload: payload["width"], validate_query=lambda query, device: None,
        validate_config=lambda query, config, device: None, default_config=lambda query, device: 1,
        knobs=(Knob(name="width", values=values),))

    def request(name, rows, *, finalists=False):
        plan = Plan(contract=contract, query=rows,
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(),
            _materialize=lambda selection, device: selection.config)
        return plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state),
            benchmark_call=lambda state: PreparedCall(run=lambda: state, produce=lambda: None),
            finalist_policy=FinalistPolicy(family_fields=("width",)) if finalists else None)

    with ExitStack() as stack:
        jobs = []
        for rank in range(4):
            engine = stack.enter_context(PreparationSession(device=DetectedDevice(None, None),
                cache_dir=tmp_path / str(rank), compile_workers=0))
            engine._cache = SelectionCache(tmp_path / str(rank),
                {"schema_version": 23, "tuning_cache_version": 1})
            engine.configure_tuning_shard(rank, (0, 1, 2, 3))
            jobs.append(engine.begin((request("families", 1, finalists=True), request("ordinary", 2))))
        states = [job.advance() for job in jobs]
        snapshots = tuple(state.ready_cache for state in states)
        assert all(snapshot is not None for snapshot in snapshots)
        for rank, job in enumerate(jobs):
            states[rank] = job.advance(cache=snapshots)
            for _ in range(100):
                if states[rank].ready_tuning:
                    break
                states[rank] = job.advance()
            assert len(states[rank].ready_tuning) == 2
        shards = [tuple(state.ready_tuning) for state in states]
        if replicas:
            # Each replica has two serving ranks; only corresponding ranks exchange candidates.
            peers = []
            for shard in shards:
                peers.extend(((False, (), None, None, (), None), (False, (), None, None, shard, None)))
            tuning = authorization(peers, replica=0, rank=1, tensor_parallel=2, ranks=(0, 1))["tuning"]
        else:
            tuning = _tuning_results(item for shard in shards for item in shard)
        for job in jobs:
            state = job.advance(tuning=tuning)
            for _ in range(100):
                if state.done:
                    break
                state = job.advance()
            assert state.done
            result = job.result()
            assert result.selections["families"].config == 1
            assert result.selections["ordinary"].config == 1
            assert {item.selection.config for item in result.finalists["families"]} == set(values)
            assert result.coverage["families"]["measured_count"] == len(values)
        assert len(tuning) == 5
        finalists = [item for item in tuning if item.finalist_policy is not None]
        assert sum(item.assignment is None for item in finalists) == max(0, 4 - len(values))
