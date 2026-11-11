# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate compiler concurrency beside mapped model storage."""

import pytest
from types import SimpleNamespace

from ignition.preparation import host_memory
from ignition.preparation.host_memory import worker_limit


def _cgroup_files(tmp_path, *, group="/tenant/ranks/job", mount_root="/tenant", limits=None):
    mount = tmp_path / "cgroup mount"
    mount.mkdir()
    for name, (maximum, current) in (limits or {}).items():
        directory = mount / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "memory.max").write_text(str(maximum))
        (directory / "memory.current").write_text(str(current))
    cgroups, mounts = tmp_path / "cgroup", tmp_path / "mountinfo"
    cgroups.write_text(f"0::{group}\n")
    escaped = str(mount).replace(" ", r"\040")
    mounts.write_text(f"42 1 0:7 {mount_root} {escaped} rw - cgroup2 cgroup rw\n")
    return cgroups, mounts


def test_cgroup_memory_uses_tightest_remaining_ancestor_allowance(tmp_path):
    files = _cgroup_files(tmp_path, limits={"": (1000, 700), "ranks": (700, 500),
                                          "ranks/job": ("max", 100)})
    assert host_memory._cgroup_available_bytes(*files) == 200


@pytest.mark.parametrize("maximum,current,expected", [(1000, 500, 500), (1000, 1100, 0), ("max", 100, None)])
def test_cgroup_memory_handles_finite_unlimited_and_exceeded_limits(tmp_path, maximum, current, expected):
    files = _cgroup_files(tmp_path, limits={"ranks/job": (maximum, current)})
    assert host_memory._cgroup_available_bytes(*files) == expected


def test_cgroup_mount_for_unrelated_subtree_does_not_limit_process(tmp_path):
    files = _cgroup_files(tmp_path, mount_root="/other", limits={"": (1000, 900)})
    assert host_memory._cgroup_available_bytes(*files) is None


def test_cgroup_v1_does_not_supply_a_unified_memory_limit(tmp_path):
    cgroups, mounts = _cgroup_files(tmp_path)
    cgroups.write_text("4:memory:/tenant/ranks/job\n")
    assert host_memory._cgroup_available_bytes(cgroups, mounts) is None


@pytest.mark.parametrize("remaining,expected", [(None, 8192), (16384, 8192), (4096, 4096), (0, 0)])
def test_host_available_memory_is_bounded_by_cgroup_allowance(monkeypatch, remaining, expected):
    from pathlib import Path

    read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs:
                        "MemAvailable: 8 kB\n" if str(path) == "/proc/meminfo" else read_text(path, *args, **kwargs))
    monkeypatch.setattr(host_memory, "_cgroup_available_bytes", lambda: remaining)
    assert host_memory.available_bytes() == expected


def test_loading_reclaims_promoted_cycles_when_host_headroom_is_low(monkeypatch):
    import gc
    import weakref

    class Owner:
        pass

    enabled = gc.isenabled()
    gc.disable()
    try:
        owner = Owner()
        owner.cycle = owner
        observer = weakref.ref(owner)
        gc.collect(0)
        del owner
        monkeypatch.setattr(host_memory, "available_bytes", lambda: 2 * host_memory._HEADROOM_BYTES)
        host_memory.release_unused(generation=0)
        assert observer() is not None
        monkeypatch.setattr(host_memory, "available_bytes", lambda: host_memory._HEADROOM_BYTES - 1)
        host_memory.release_unused(generation=0)
        assert observer() is None
    finally:
        if enabled:
            gc.enable()


def test_cpu_worker_budget_uses_affinity_and_is_shared_by_local_ranks(monkeypatch):
    monkeypatch.setattr(host_memory, "os", SimpleNamespace(sched_getaffinity=lambda _: set(range(48))))
    assert host_memory.cpu_workers() == 48
    assert host_memory.cpu_workers(pools=4) == 12
    assert host_memory.cpu_workers(24, pools=4) == 6
    assert host_memory.cpu_workers(80, pools=4) == 20
    with pytest.raises(ValueError, match="one process"):
        host_memory.cpu_workers(2, pools=4)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_cpu_worker_budgets_are_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        host_memory.cpu_workers(count)


def test_cpu_worker_budget_without_affinity_uses_cpu_count(monkeypatch):
    monkeypatch.setattr(host_memory, "os", SimpleNamespace(cpu_count=lambda: 24))
    assert host_memory.cpu_workers(pools=2) == 12


@pytest.mark.parametrize("pools", [1, 2, 4])
@pytest.mark.parametrize("gib", [16, 24, 64, 128])
def test_compiler_pools_leave_host_headroom(pools, gib):
    workers = worker_limit(8, pools=pools, available=gib << 30)
    assert 1 <= workers <= 8
    assert pools * workers * (2 << 30) + (8 << 30) <= gib << 30


def test_insufficient_host_ram_fails_before_workers_start():
    with pytest.raises(MemoryError, match="available host memory"):
        worker_limit(8, pools=4, available=15 << 30)


def test_unified_memory_budgets_host_and_device_storage_together():
    assert worker_limit(8, available=24 << 30) == 8
    assert worker_limit(8, available=24 << 30, device_bytes=2 << 30) == 4
    assert worker_limit(8, pools=2, available=24 << 30, device_bytes=2 << 30) == 2
    with pytest.raises(MemoryError, match="available host memory"):
        worker_limit(8, available=11 << 30, device_bytes=2 << 30)


@pytest.mark.parametrize("platform", ["posix", "nt"])
@pytest.mark.parametrize("available,expected", [(12 << 30, 12 << 30), (40 << 30, 32 << 30)])
def test_integrated_device_uses_available_ram(monkeypatch, available, expected, platform):
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 << 30, 32 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: available)
    monkeypatch.setattr(host_memory, "os", SimpleNamespace(name=platform))
    assert host_memory.device_memory_info("cuda:0") == (
        4 << 30 if platform == "nt" else expected, 32 << 30)


def test_windows_device_budget_includes_dedicated_memory_beyond_host_ram(monkeypatch):
    import torch

    monkeypatch.setattr(host_memory, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (105 << 30, 110 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 48 << 30)
    assert host_memory.device_memory_info("cuda:0") == (105 << 30, 110 << 30)
    workers, race = host_memory.tuning_limits("cuda:0", 8, headroom_bytes=8 << 30)
    assert workers * (2 << 30) + race + (1 << 30) + (8 << 30) <= 48 << 30


def test_discrete_device_budget_is_independent_of_host_ram(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (24 << 30, 32 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: pytest.fail("discrete VRAM must not use host capacity"))
    assert host_memory.device_memory_info("cuda:0") == (24 << 30, 32 << 30)


@pytest.mark.parametrize("pools", [1, 16])
def test_candidate_scratch_leaves_device_headroom_growth_and_pending_storage(monkeypatch, pools):
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (5 << 30, 96 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    assert host_memory.candidate_scratch_limit(0, pools=pools, headroom_bytes=1 << 30,
        pending_persistent_bytes=512 << 20) == 2560 << 20


def test_candidate_scratch_shares_integrated_memory_between_local_ranks(monkeypatch):
    import torch

    monkeypatch.setattr(host_memory, "device_memory_info", lambda device: (40 << 30, 121 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 40 << 30)
    assert host_memory.candidate_scratch_limit(0, pools=4, headroom_bytes=1 << 30,
        pending_persistent_bytes=512 << 20) == 6656 << 20


def test_candidate_scratch_never_spends_reserved_memory(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1 << 30, 96 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    assert host_memory.candidate_scratch_limit(0, headroom_bytes=1 << 30) == 0


@pytest.mark.parametrize("pools", [1, 2, 4])
@pytest.mark.parametrize("race_budget", [None, 1 << 20, 64 << 30])
def test_discrete_tuning_keeps_device_headroom_without_dividing_vram_across_ranks(
    monkeypatch, pools, race_budget
):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 << 30, 96 << 30))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 72 << 30)
    workers, race = host_memory.tuning_limits("cuda:0", 16, pools=pools,
        headroom_bytes=1 << 30, race_budget=race_budget)
    assert workers == min(16, 32 // pools)
    assert 0 < race <= 512 << 20
    assert 4 * race + (2 << 30) <= 4 << 30
    if race_budget is not None:
        assert race <= race_budget


def test_discrete_small_workloads_keep_a_positive_race_budget_without_workers(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (768 << 20, 24 << 30))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: pytest.fail("no compiler pool needs host RAM"))
    assert host_memory.tuning_limits("cuda:0", 0, headroom_bytes=1 << 30) == (0, 1)


@pytest.mark.parametrize("pools", [1, 2, 4])
@pytest.mark.parametrize("headroom", [0, 16 << 30])
@pytest.mark.parametrize("race_budget", [None, 1 << 20, 64 << 30])
def test_tuning_races_and_compilers_share_one_ram_budget(monkeypatch, pools, headroom, race_budget):
    import torch

    available = 40 << 30
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (available, 121 << 30))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: available)
    workers, race = host_memory.tuning_limits("cuda:0", 8, pools=pools,
        headroom_bytes=headroom, race_budget=race_budget)
    assert workers > 0 and race > 0
    assert pools * (workers * (2 << 30) + race + (1 << 30)) + max(8 << 30, headroom) <= available
    if race_budget is not None:
        assert race <= race_budget
    if race_budget == 1 << 20:
        assert race == race_budget
