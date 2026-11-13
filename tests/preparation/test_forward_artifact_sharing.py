# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ranks compile each forward variant once, and ranks on separate hosts receive the others by path."""

import multiprocessing
import os
from pathlib import Path

import pytest


def _rank(rank, world, store, home, owned, outcome):
    import torch.distributed as dist

    from ignition.preparation.aot import _share_artifacts

    # Each rank's working directory stands in for its host's disk; artifact paths are identical.
    os.chdir(home)
    try:
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
        _share_artifacts(owned, dist.group.WORLD, tuple(range(world)))
        outcome.put((rank, {path: Path(path).read_bytes() for path in ("forwards/a.bin", "forwards/b.bin")}))
        dist.destroy_process_group()
    except Exception as error:  # noqa: BLE001 - The parent reports every rank's failure.
        outcome.put((rank, repr(error)))


def test_each_variant_compiles_on_the_least_loaded_rank_that_binds_it():
    from ignition.preparation.aot import _variant_compilers

    shared = [{name: "shared" for name in "abcdef"}] * 4
    assert list(_variant_compilers(shared).values()) == [0, 1, 2, 3, 0, 1]
    # Rank zero samples; its peers consume, so neither variant compiles on a rank that cannot bind it.
    sampled = [{"decode": "sample", "resume": "sample"}] + [{"decode": "consume", "resume": "consume"}] * 3
    assert _variant_compilers(sampled) == {("decode", "sample"): 0, ("decode", "consume"): 1,
                                           ("resume", "sample"): 0, ("resume", "consume"): 2}


def test_every_rank_receives_what_it_lacks_from_the_rank_that_compiled_it(tmp_path):
    context = multiprocessing.get_context("fork")
    homes = [tmp_path / f"host{rank}" for rank in range(3)]
    for home in homes:
        (home / "forwards").mkdir(parents=True)
    (homes[0] / "forwards/a.bin").write_bytes(b"compiled a")
    (homes[1] / "forwards/b.bin").write_bytes(b"compiled b")
    # Rank two compiled nothing, already holds a, and holds a stale b.
    (homes[2] / "forwards/a.bin").write_bytes(b"compiled a")
    (homes[2] / "forwards/b.bin").write_bytes(b"stale b")
    owned = [["forwards/a.bin"], ["forwards/b.bin"], []]
    outcome = context.Queue()
    processes = [context.Process(target=_rank, args=(rank, 3, tmp_path / "store", homes[rank], owned, outcome))
                 for rank in range(3)]
    for process in processes:
        process.start()
    results = dict(outcome.get(timeout=60) for _ in processes)
    for process in processes:
        process.join(timeout=60)
    expected = {"forwards/a.bin": b"compiled a", "forwards/b.bin": b"compiled b"}
    assert results == {rank: expected for rank in range(3)}


def test_ranks_lend_missing_compilations_until_lending_finishes_no_sooner():
    from ignition.preparation.aot import _balance_compilations

    # Rank zero's sampling variants leave it most of a TP16 catalog; lending levels every rank at three waves.
    assigned = _balance_compilations([13] + [2] * 15, [1] * 16)
    assert max(map(len, assigned)) == 3 and len(assigned[0]) == 3
    assert sorted(job for jobs in assigned for job in jobs) == sorted(
        [(0, index) for index in range(13)] + [(rank, index) for rank in range(1, 16) for index in range(2)])
    # Ranks keep their own forwards, and a rank with two workers finishes two per wave.
    assigned = _balance_compilations([9, 0, 0], [2, 1, 1])
    assert [len(jobs) for jobs in assigned] == [5, 2, 2]
    assert all(owner == 0 for jobs in assigned for owner, _ in jobs)
    assert _balance_compilations([1, 1], [1, 1]) == [[(0, 0)], [(1, 0)]]
    # A rank whose GPU has no room for a compiler lends every forward it owns.
    assigned = _balance_compilations([13, 2, 2, 2], [1, 1, 0, 1])
    assert assigned[2] == [] and sorted(job for jobs in assigned for job in jobs) == sorted(
        [(0, index) for index in range(13)] + [(rank, index) for rank in (1, 2, 3) for index in range(2)])
    assert max(map(len, assigned)) == 7


class _Forward:
    def __init__(self, variant):
        self.compilation_variant = variant


class _CompilerMetadata:
    pass


class _Artifact:
    def __init__(self, execute):
        from inspect import signature
        from types import SimpleNamespace

        from ignition.preparation.aot import StorageBackend
        from torch._dynamo.guards import GuardManagerWrapper

        backend = StorageBackend.__new__(StorageBackend)
        backend.artifact = SimpleNamespace(inner_fn=SimpleNamespace(compiled_fn=execute))
        backend._serialized_artifact = b"portable backend"
        backend._tuning_frozen = False
        guards = SimpleNamespace(check=lambda arguments: isinstance(arguments["forward"], _Forward),
                                 check_verbose=lambda arguments: "forward type differs")
        self._artifacts = SimpleNamespace(signature=signature(lambda forward: None),
            runtime_env=SimpleNamespace(closure=()), guard_manager=GuardManagerWrapper(root=guards),
            compiled_fn=backend, guards_state=b"portable guards", compiler_graph=_CompilerMetadata())
        self.fn = execute

    def __call__(self, owner):
        self.fn(owner)

    def save_compiled_function(self, path):
        assert Path(path).exists()


def _replace_gpu_preparation():
    from types import SimpleNamespace
    import torch
    from ignition.preparation import compile_pool, inductor_tuning

    class Selections(inductor_tuning.InductorSelections):
        def register(self, bound):
            state = bound.artifact._artifacts
            self.forwards[id(bound)] = state.compiled_fn, state.compiler_graph, {}

        def freeze(self, bound):
            self.forwards[id(bound)][0]._tuning_frozen = True

    inductor_tuning.InductorSelections = Selections
    compile_pool.compiler_resources = lambda *args, **kwargs: compile_pool.CompilerResources(
        1, 128 << 30, 16, 32 << 30, 8 << 30)
    torch.cuda.current_stream = lambda device: SimpleNamespace(synchronize=lambda: None)


class _Pool:
    """Compiles by writing the artifact path with the compiling rank, as a worker would."""

    workers = 1

    def __init__(self, device, **kwargs):
        self.workers = kwargs.get("workers", 1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, forward, path):
        from ignition.preparation.forward_metadata import dumps

        return self.submit_payload(dumps(forward), path)

    def submit_payload(self, payload, path):
        from concurrent.futures import Future
        import torch.distributed as dist

        from ignition.preparation.forward_metadata import loads

        forward = loads(payload)
        Path(path).write_text(f"{forward.compilation_variant} compiled by rank {dist.get_rank()}")
        future = Future()
        future.set_result(None)
        return future


def _lending_rank(rank, world, store, root, outcome, lifecycle=None, fail_shutdown=False, scenario=None):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch.distributed as dist

    from ignition.preparation import aot, compile_pool

    try:
        class Pool(_Pool):
            def __exit__(self, *args):
                if lifecycle is not None:
                    started, closed, finished = lifecycle
                    if rank == 0:
                        assert started.wait(20), "borrowed compilation did not start"
                        assert not finished.is_set(), "borrowed compilation finished before owner shutdown"
                        closed.set()
                if fail_shutdown and rank == 0:
                    raise RuntimeError("compiler shutdown failed on rank 0")

            def submit_payload(self, payload, path):
                from ignition.preparation.forward_metadata import loads

                if scenario == "two_hosts":
                    assert hosts[loads(payload).rank] == hosts[rank], "forward metadata crossed hosts"
                if scenario == "worker_failure" and rank == 1:
                    from concurrent.futures import Future

                    future = Future()
                    future.set_exception(RuntimeError("forward worker failed on rank 1"))
                    return future
                if lifecycle is not None and rank == 1 and loads(payload).compilation_variant == "sample":
                    started, closed, finished = lifecycle
                    started.set()
                    assert closed.wait(20), "idle owner retained its compiler while borrowed work was running"
                    result = super().submit_payload(payload, path)
                    finished.set()
                    return result
                return super().submit_payload(payload, path)

        compile_pool.CompilePool = Pool
        _replace_gpu_preparation()
        hosts = ("a", "b", "a", "b") if scenario == "two_hosts" else ("host",) * world
        if scenario == "two_hosts":
            os.chdir(root / hosts[rank])
            root = Path("forwards")
        if scenario in ("two_hosts", "zero_owners", "no_capacity", "worker_failure"):
            capacity = (rank in (2, 3) if scenario == "two_hosts" else rank == 1)
            if scenario == "no_capacity":
                capacity = False
            compile_pool.compiler_resources = lambda *args, **kwargs: compile_pool.CompilerResources(
                int(capacity), 40 << 30, 16, 32 << 30, 8 << 30)
        aot.ForwardCache._load = lambda self, forward, path, required=False, before_run=None: aot.BoundForward(
            _Artifact(lambda owner: None), forward, True)
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
        cache = aot.ForwardCache.__new__(aot.ForwardCache)
        cache.root, cache.device, cache.headroom_bytes, cache.options = root, "cpu", 0, aot.MappingProxyType({})
        cache.compile_workers = None
        cache.lock_gpu_clocks = False
        variant = "sample" if rank == 0 else "consume"
        functions = {name: _Forward(variant) for name in "abcd"}
        for forward in functions.values():
            forward.rank = rank
        coordinator = SimpleNamespace(group=dist.group.WORLD, ranks=tuple(range(world)), hosts=hosts,
            spans_hosts=len(set(hosts)) > 1,
            local_ranks=world, barrier=lambda key, ranks: dist.monitored_barrier(group=dist.group.WORLD))
        selected = cache._prepare_ranks(functions, {name: name for name in functions}, coordinator, progress=None,
                                        materialize=lambda: None, warmup=nullcontext, before_run=None)
        if scenario == "two_hosts":
            outcome.put((rank, (sorted(selected), sorted(path.read_text() for path in root.glob("*.bin")))))
        else:
            outcome.put((rank, sorted(selected)))
    except Exception:  # noqa: BLE001 - The parent reports every rank's failure.
        import traceback

        outcome.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_one_host_lends_a_ranks_missing_compilations_to_idle_ranks(tmp_path):
    context = multiprocessing.get_context("fork")
    root = tmp_path / "forwards"
    root.mkdir()
    outcome = context.Queue()
    processes = [context.Process(target=_lending_rank, args=(rank, 3, tmp_path / "store", root, outcome))
                 for rank in range(3)]
    for process in processes:
        process.start()
    results = dict(outcome.get(timeout=120) for _ in processes)
    for process in processes:
        process.join(timeout=60)
    assert results == {rank: list("abcd") for rank in range(3)}
    # Rank zero owns four sampling variants and its peers two consuming variants each: rank zero lends one.
    compiled = sorted(path.read_text() for path in root.glob("*.bin"))
    assert compiled.count("sample compiled by rank 0") == 3
    assert compiled.count("sample compiled by rank 1") == 1
    assert len(compiled) == 8 and not list((root / "lent").iterdir())


def test_idle_compiler_closes_before_a_peer_finishes_borrowed_work(tmp_path):
    context = multiprocessing.get_context("fork")
    root = tmp_path / "forwards"
    root.mkdir()
    outcome = context.Queue()
    lifecycle = tuple(context.Event() for _ in range(3))
    processes = [context.Process(target=_lending_rank,
        args=(rank, 3, tmp_path / "store", root, outcome, lifecycle)) for rank in range(3)]
    try:
        for process in processes:
            process.start()
        results = dict(outcome.get(timeout=90) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert results == {rank: list("abcd") for rank in range(3)}
        assert all(event.is_set() for event in lifecycle)
        assert len(list(root.glob("*.bin"))) == 8
        assert not list((root / "lent").iterdir())
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)


def test_compiler_shutdown_failure_is_reported_to_every_rank(tmp_path):
    context = multiprocessing.get_context("fork")
    root = tmp_path / "forwards"
    root.mkdir()
    outcome = context.Queue()
    processes = [context.Process(target=_lending_rank,
        args=(rank, 3, tmp_path / "store", root, outcome, None, True)) for rank in range(3)]
    try:
        for process in processes:
            process.start()
        results = dict(outcome.get(timeout=90) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert set(results) == set(range(3))
        assert all("compiler shutdown failed on rank 0" in result for result in results.values())
        assert len(list(root.glob("*.bin"))) == 8
        assert not list((root / "lent").iterdir())
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)


@pytest.mark.parametrize("scenario", ["zero_owners", "two_hosts", "no_capacity", "worker_failure"])
def test_host_admission_lends_to_local_workers_and_reports_failures_collectively(tmp_path, scenario):
    context = multiprocessing.get_context("fork")
    world = 4 if scenario == "two_hosts" else 3
    root = tmp_path / "forwards"
    root.mkdir()
    if scenario == "two_hosts":
        for host in ("a", "b"):
            (root / host / "forwards").mkdir(parents=True)
    outcome = context.Queue()
    processes = [context.Process(target=_lending_rank,
        args=(rank, world, tmp_path / "store", root, outcome, None, False, scenario)) for rank in range(world)]
    try:
        for process in processes:
            process.start()
        results = dict(outcome.get(timeout=120) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        if scenario == "two_hosts":
            for selected, compiled in results.values():
                assert selected == list("abcd")
                assert len(compiled) == 8
                assert compiled.count("sample compiled by rank 2") == 4
                assert all("rank 2" in item or "rank 3" in item for item in compiled)
            assert results[0] == results[1] == results[2] == results[3]
        elif scenario == "zero_owners":
            assert results == {rank: list("abcd") for rank in range(world)}
            assert all("rank 1" in path.read_text() for path in root.glob("*.bin"))
        else:
            reason = "no capacity for a forward compiler" if scenario == "no_capacity" else "forward worker failed on rank 1"
            assert all(reason in result for result in results.values())
        assert not list(root.rglob("*.forward"))
        assert not list(root.rglob("*.tmp"))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)


def _cached_rank(rank, world, store, root, deferred, outcome):
    import gc
    import weakref
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch.distributed as dist

    from ignition.preparation import aot, compile_pool

    try:
        compile_pool.CompilePool = _Pool
        _replace_gpu_preparation()
        restored, portable, warmed = [], [], []

        def load(self, forward, path, required=False, before_run=None):
            if required:
                gc.collect()
                assert all(reference() is None for reference in portable), "previous portable artifact is retained"
            artifact = _Artifact(warmed.append)
            bound = aot.BoundForward(artifact, forward, True)
            if not required:
                restored.append(weakref.ref(bound))
            else:
                portable.extend((weakref.ref(artifact), weakref.ref(artifact._artifacts.compiler_graph)))
            return bound

        def materialize():
            gc.collect()
            assert restored and all(reference() is None for reference in restored)

        aot.ForwardCache._load = load
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
        cache = aot.ForwardCache.__new__(aot.ForwardCache)
        cache.root, cache.device, cache.headroom_bytes, cache.options = root, "cpu", 0, aot.MappingProxyType({})
        cache.compile_workers = None
        cache.lock_gpu_clocks = False
        functions = {name: _Forward("shared") for name in "abcd"}
        signatures = {name: name for name in functions}
        for name, forward in functions.items():
            cache._path(forward, name=f"{name}.shared", signature=name).write_text("cached")
        dist.barrier()
        coordinator = SimpleNamespace(group=dist.group.WORLD, ranks=tuple(range(world)), spans_hosts=False,
            hosts=("host",) * world,
            local_ranks=world, barrier=lambda key, ranks: dist.monitored_barrier(group=dist.group.WORLD))
        selected = cache._prepare_ranks(functions, signatures, coordinator, progress=None,
            materialize=materialize if deferred else None, warmup=nullcontext, before_run=None)
        assert set(warmed) == set(functions.values())
        assert all(warmed.count(forward) == 2 for forward in functions.values())
        assert all(bound.cache_hit for bound in selected.values())
        gc.collect()
        assert all(reference() is None for reference in portable)
        for bound in selected.values():
            assert bound.prepare_capture() == 0
            bound()
        assert all(warmed.count(forward) == 3 for forward in functions.values())
        bound = selected["a"]
        owner, bound.owner = bound.owner, object()
        with pytest.raises(aot.BindingMismatch, match="compiled guards"):
            bound()
        bound.owner = owner
        assert all(warmed.count(forward) == 3 for forward in functions.values())
        outcome.put((rank, sorted(selected)))
        dist.destroy_process_group()
    except Exception:
        import traceback

        outcome.put((rank, traceback.format_exc()))


@pytest.mark.parametrize("deferred", [False, True])
def test_cached_catalog_releases_restoration_owners_before_rebinding(tmp_path, deferred):
    context = multiprocessing.get_context("fork")
    root = tmp_path / "forwards"
    root.mkdir()
    outcome = context.Queue()
    processes = [context.Process(target=_cached_rank,
        args=(rank, 3, tmp_path / "store", root, deferred, outcome)) for rank in range(3)]
    for process in processes:
        process.start()
    results = dict(outcome.get(timeout=120) for _ in processes)
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    assert results == {rank: list("abcd") for rank in range(3)}
