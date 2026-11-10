# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU allocation-routing and exception ownership checks for Qwen preparation."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.core_storage import CoreStorage
from ignition.recipes.qwen38 import bootstrap


class Routing:
    """Record allocator hooks while retaining the real storage context managers."""

    def __init__(self):
        self.events, self.active, self.counter = [], set(), 0

    def pool(self, *args):
        self.counter += 1
        return SimpleNamespace(id=self.counter, allocator=lambda: object())

    def begin(self, device, pool):
        self.events.append(("begin", pool))
        self.active.add(pool)

    def end(self, device, pool):
        self.events.append(("end", pool))
        self.active.remove(pool)

    def release(self, device, pool):
        self.events.append(("release", pool))

    def storage(self):
        storage = CoreStorage.__new__(CoreStorage)
        storage.device = 0
        storage._allocator = self.pool()
        storage._preparation_pools = set()
        storage.numeric_storage = contextlib.nullcontext()
        storage._scope = torch.cuda.use_mem_pool(self.pool(), device=0)
        storage.seal_low_window = lambda: self.events.append(("seal", 0))
        storage.host_allocate = lambda *args: None
        storage.peer_storage = None
        return storage

    def assert_scopes_closed_once(self):
        begins = [pool for event, pool in self.events if event == "begin"]
        ends = [pool for event, pool in self.events if event == "end"]
        releases = [pool for event, pool in self.events if event == "release"]
        assert len(begins) == 2
        assert sorted(ends) == sorted(releases) == sorted(begins)
        assert not self.active
        # Allocation routing returns to the parent before its storage scope exits.
        assert self.events.index(("end", begins[1])) < self.events.index(("end", begins[0]))


@pytest.fixture(autouse=True)
def routing(monkeypatch):
    hooks = Routing()
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU test initialized CUDA"))
    monkeypatch.setattr(torch.cuda, "MemPool", hooks.pool)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: hooks.events.append(("synchronize", 0)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: hooks.events.append(("empty-cache", 0)))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    for target, replacement in (("_cuda_beginAllocateCurrentThreadToPool", hooks.begin),
            ("_cuda_endAllocateToPool", hooks.end), ("_cuda_releasePool", hooks.release)):
        monkeypatch.setattr(torch.cuda.memory, target, replacement)
    manifest = SimpleNamespace(metadata_sha256={"config.json": "config"}, headers_sha256="headers")
    monkeypatch.setattr(bootstrap, "inspect_qwen38_checkpoint", lambda snapshot: manifest)
    monkeypatch.setattr(bootstrap, "phase", lambda *args, **kwargs: contextlib.nullcontext())
    return hooks


def assert_failure_closes_routing(routing, failure):
    storage = routing.storage()
    try:
        with pytest.raises(type(failure)) as raised:
            with storage:
                bootstrap.prepare_model(Path("/unused"), bootstrap.PreparationConfig(), core_storage=storage)
        assert raised.value is failure
        # Keep the traceback alive: garbage collection must not restore routing.
        assert failure.__traceback__ is not None
        routing.assert_scopes_closed_once()
    finally:
        failure.__traceback__ = None


def test_weight_load_failure_closes_routing_with_exception_retained(monkeypatch, routing):
    failure = RuntimeError("injected weight load failure")

    def load(*args, **kwargs):
        raise failure

    monkeypatch.setattr(bootstrap.ModelWeights, "load", load)
    assert_failure_closes_routing(routing, failure)


def test_construction_failure_closes_routing_after_owner_cleanup(monkeypatch, routing):
    failure = RuntimeError("injected model-state construction failure")
    owners = []
    session = SimpleNamespace(__enter__=lambda: None, close=lambda: owners.append("session"))
    table = SimpleNamespace(close=lambda: owners.append("table"))
    monkeypatch.setattr(bootstrap.ModelWeights, "load", lambda *args, **kwargs: object())
    monkeypatch.setattr(bootstrap.PLETable, "load", lambda *args, **kwargs: table)
    monkeypatch.setattr(bootstrap, "preparation_session", lambda *args, **kwargs: session)

    @contextlib.contextmanager
    def phase(progress, label, **kwargs):
        if label == "Allocate model state":
            raise failure
        yield

    monkeypatch.setattr(bootstrap, "phase", phase)
    assert_failure_closes_routing(routing, failure)
    assert owners == ["session", "table"]


def test_success_preserves_early_close_arguments_and_no_grad(monkeypatch, routing):
    result, config = object(), object()
    snapshot, progress, group, storage = (object() for _ in range(4))
    events = []

    def prepare(actual_snapshot, actual_config, **kwargs):
        assert actual_snapshot is snapshot and actual_config is config
        assert not torch.is_grad_enabled()
        transient = kwargs.pop("transient")
        assert kwargs == dict(progress=progress, compile_decoder=False, rank=2,
            group=group, core_storage=storage)
        transient.callback(events.append, "close-at-tuning")
        with transient:
            pass
        events.append("compile-start")
        return result

    monkeypatch.setattr(bootstrap, "_prepare_model", prepare)
    assert bootstrap.prepare_model(snapshot, config, progress=progress, compile_decoder=False,
        rank=2, group=group, core_storage=storage) is result
    assert events == ["close-at-tuning", "compile-start"]
    assert routing.events == []


def test_cleanup_failure_adds_note_without_masking_primary_exception(monkeypatch):
    failure = ValueError("original preparation failure")

    def close():
        raise RuntimeError("allocation cleanup failure")

    def prepare(*args, transient, **kwargs):
        transient.callback(close)
        raise failure

    monkeypatch.setattr(bootstrap, "_prepare_model", prepare)
    with pytest.raises(ValueError) as raised:
        bootstrap.prepare_model(Path("/unused"))
    assert raised.value is failure
    assert any("allocation cleanup failure" in note for note in failure.__notes__)


def test_cleanup_failure_propagates_when_preparation_succeeded(monkeypatch):
    failure = RuntimeError("allocation cleanup failure")

    def close():
        raise failure

    def prepare(*args, transient, **kwargs):
        transient.callback(close)
        return object()

    monkeypatch.setattr(bootstrap, "_prepare_model", prepare)
    with pytest.raises(RuntimeError) as raised:
        bootstrap.prepare_model(Path("/unused"))
    assert raised.value is failure
