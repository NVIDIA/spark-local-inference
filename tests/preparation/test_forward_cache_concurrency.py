# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Concurrent model launches retain their compiled variant until binding finishes."""

from concurrent.futures import ThreadPoolExecutor, wait
from threading import Event
from types import SimpleNamespace

from ignition.preparation.aot import ForwardCache


def test_catalog_publication_waits_for_all_bindings(tmp_path, monkeypatch):
    published, release, attempted, second_entered = (Event() for _ in range(4))
    first, second = object.__new__(ForwardCache), object.__new__(ForwardCache)
    first.root = second.root = tmp_path
    artifact = tmp_path / 'forward.bin'

    def prepare(cache, functions, signatures, **kwargs):
        value = functions['pages']
        if value == 64:
            second_entered.set()
        artifact.write_text(str(value))
        if value == 32:
            published.set()
            assert release.wait(5)
        return int(artifact.read_text())

    def competing_launch():
        attempted.set()
        return second.prepare_many({'pages': 64}, {})

    monkeypatch.setattr(ForwardCache, '_prepare_many', prepare)
    with ThreadPoolExecutor(2) as pool:
        left = pool.submit(first.prepare_many, {'pages': 32}, {})
        try:
            assert published.wait(5)
            right = pool.submit(competing_launch)
            assert attempted.wait(5)
            assert not second_entered.wait(.1)
        finally:
            release.set()
        assert left.result(timeout=5) == 32
        assert right.result(timeout=5) == 64


def test_a_hosts_first_rank_holds_its_cache_lock_for_every_rank_there(tmp_path, monkeypatch):
    import torch.distributed as dist

    cache = object.__new__(ForwardCache)
    cache.root = tmp_path
    barriers = []
    monkeypatch.setattr(dist, 'get_rank', lambda: 1)
    monkeypatch.setattr(dist, 'barrier', lambda group: barriers.append(group))
    monkeypatch.setattr(ForwardCache, '_prepare_many', lambda *args, **kwargs: 'bound')
    # Rank one shares rank zero's host and cache, so rank zero's lock covers it.
    local = SimpleNamespace(ranks=(0, 1), hosts=('a', 'a'), group='ranks')
    with cache._publication_lock():
        assert cache.prepare_many({}, {}, coordinator=local) == 'bound'
    # Ranks hold every host's lock before compiling and until every rank has bound.
    assert barriers == ['ranks', 'ranks']
    # Alone on its host, rank one holds that host's lock itself, after host a's turn to lock.
    barriers.clear()
    remote = SimpleNamespace(ranks=(0, 1), hosts=('a', 'b'), group='ranks')
    with ThreadPoolExecutor(1) as pool:
        with cache._publication_lock():
            binding = pool.submit(cache.prepare_many, {}, {}, coordinator=remote)
            wait([binding], timeout=.1)
            assert not binding.done() and barriers == ['ranks']
        assert binding.result(timeout=5) == 'bound' and barriers == ['ranks'] * 3
