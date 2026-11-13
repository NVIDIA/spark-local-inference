# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Local candidate replicas preserve independent tensor-parallel groups."""

import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from ignition.preparation.tuning_replicas import ReplicaPreparation, TuningReplica, authorization


def state(*, cache=None, races=(), collective=(), done=False, collective_tuning=None):
    return done, collective, None, cache, races, collective_tuning


def race(key, latency, candidate):
    return SimpleNamespace(key=key, latency_us=latency, candidate_index=candidate,
                           assignment={"candidate": candidate})


def test_caches_and_winners_are_exchanged_between_corresponding_tp_ranks():
    values = [state(cache=value) for value in ("replica0.rank0", "replica0.rank1",
                                              "replica1.rank0", "replica1.rank1")]
    assert authorization(values, replica=0, rank=1, tensor_parallel=2, ranks=(0, 1)) == {
        "cache": ("replica0.rank1", "replica1.rank1")}
    winners = [race("rank0", 5, 0), race("rank1", 2, 0), race("rank0", 1, 1), race("rank1", 4, 1)]
    values = [state(races=(value,)) for value in winners]
    assert authorization(values, replica=0, rank=0, tensor_parallel=2, ranks=(0, 1))["tuning"] is winners[2]
    assert authorization(values, replica=1, rank=1, tensor_parallel=2, ranks=(0, 1))["tuning"] is winners[1]


def test_collectives_only_wait_for_their_own_replica():
    ready = state(collective=(("sum", (0, 1)),))
    values = [ready, ready, state(), state()]
    assert authorization(values, replica=0, rank=0, tensor_parallel=2, ranks=(0, 1)) == {"collective_key": "sum"}
    assert authorization(values, replica=1, rank=0, tensor_parallel=2, ranks=(0, 1)) == {"collective_key": None}
    with pytest.raises(ValueError, match="serving TP group"):
        authorization(values, replica=0, rank=0, tensor_parallel=2, ranks=(0, 2))


def test_collective_samples_stay_within_their_tp_replica():
    contributions = [SimpleNamespace(ranks=(0, 1), latency=value) for value in (2., 3., 7., 5.)]
    values = [state(collective_tuning=value) for value in contributions]
    for replica in range(2):
        for rank in range(2):
            assert authorization(values, replica=replica, rank=rank, tensor_parallel=2, ranks=(0, 1)) == {
                "collective_tuning": tuple(contributions[replica * 2:(replica + 1) * 2])}
    values[3] = state()
    assert authorization(values, replica=0, rank=0, tensor_parallel=2, ranks=(0, 1)) == {
        "collective_tuning": tuple(contributions[:2])}
    assert authorization(values, replica=1, rank=0, tensor_parallel=2, ranks=(0, 1)) == {"collective_key": None}


def test_mismatched_tuning_races_fail_before_selecting_another_ranks_kernel():
    values = [state(races=(race("one", 1, 0),)), state(races=(race("two", 2, 1),))]
    with pytest.raises(RuntimeError, match="different candidate races"):
        authorization(values, replica=0, rank=0, tensor_parallel=1, ranks=(0,))


def _transport(directory, member):
    from datetime import timedelta
    import torch.distributed as dist

    replica, rank = divmod(member, 2)
    dist.init_process_group("gloo", init_method=(Path(directory) / f"tp-{replica}").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    role = TuningReplica(str(Path(directory) / "replicas"), replica, 2, 2)
    from ignition.preparation import host_memory

    memory_info = host_memory.device_memory_info
    host_memory.device_memory_info = lambda device: (0, (96 << 30) - member * (1 << 20))
    try:
        assert host_memory.virtual_device_memory(None, dist.group.WORLD, tuning_replica=role) == (
            (96 << 30) - 3 * (1 << 20))
    finally:
        host_memory.device_memory_info = memory_info
    coordinator = ReplicaPreparation(role, rank, ranks=(0, 1))
    assert coordinator.gather((replica, rank)) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert dist.get_rank() == rank and dist.get_world_size() == 2
    members = [None, None]
    dist.all_gather_object(members, (replica, rank))
    assert members == [(replica, 0), (replica, 1)]
    closed = []
    job = SimpleNamespace(
        advance=lambda **allowed: SimpleNamespace(done=False, ready_collectives=(),
                                                   ready_cache=None, ready_tuning=(), ready_collective_tuning=None),
        close=lambda: closed.append(True))

    def begin(requests, **options):
        if member == 3:
            raise ValueError('candidate compilation failed')
        return job

    with pytest.raises(RuntimeError, match='candidate compilation failed'):
        coordinator.prepare(SimpleNamespace(begin=begin), (), progress=None, autotune=True)
    assert closed == ([] if member == 3 else [True])
    del coordinator
    dist.destroy_process_group()


def test_replica_control_does_not_replace_default_tp_groups(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_transport, args=(str(tmp_path), member)) for member in range(4)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=45)
        assert [process.exitcode for process in processes] == [0] * 4
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()


def _finish_rank(snapshot, config, **options):
    import torch.distributed as dist
    from ignition.preparation.tuning_replicas import TuningComplete

    assert dist.get_world_size() == len(config.devices)
    dist.barrier()
    raise TuningComplete()


def _finish_replica():
    import torch.distributed as dist
    from ignition.preparation.rank_group import prepare_group
    from ignition.preparation.tuning_replicas import TuningComplete

    with pytest.raises(TuningComplete):
        prepare_group(None, SimpleNamespace(devices=(0, 1)), _finish_rank)
    assert not dist.is_initialized()
    assert not multiprocessing.active_children()


def test_preparation_only_tp_group_drains_and_joins_its_ranks():
    process = multiprocessing.get_context('spawn').Process(target=_finish_replica)
    process.start()
    try:
        process.join(timeout=45)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
        process.join()
