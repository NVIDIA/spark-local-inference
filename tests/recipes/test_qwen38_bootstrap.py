# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU ownership tests for prepared-model handoff and confirmed-drain cleanup."""

from __future__ import annotations

import gc
import sys
import weakref
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from ignition.preparation.execution import GraphSignature
from ignition.preparation.model import PreparedModel


class Resource:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def close(self):
        self.events.append(self.name)

    def reset(self):
        self.events.append(self.name)


def model_fixture(monkeypatch, *, synchronize=None, server=None):
    events = []
    weights = Resource("weights", events)
    weights_ref = weakref.ref(weights)
    model = PreparedModel(
        device=0,
        graphs={"decode": Resource("graph", events)},
        buffers={"frame_index": object(), "submission_sequence": object(),
                 "completion_records": SimpleNamespace(shape=(2, 4, 32))},
        signatures={"decode": GraphSignature("decode", 1, 1)},
        resources=(weights,),
        server_config={},
        memory_report={},
        session=Resource("session", events),
        close_resources=(Resource("table", events),),
    )
    monkeypatch.setattr(model, "manifest", lambda: {"test_manifest": True})
    monkeypatch.setattr("torch.cuda.synchronize", synchronize or (lambda device: events.append("synchronize")))
    monkeypatch.setitem(sys.modules, "flare_heliopause", SimpleNamespace(Server=server))
    return model, events, weights_ref


def thread_call(call, *, entered=None):
    errors = []

    def run():
        if entered is not None:
            entered.set()
        try:
            call()
        except BaseException as error:  # noqa: BLE001 - Propagate worker failures to the test thread.
            errors.append(error)

    thread = Thread(target=run, daemon=True)
    thread.start()
    return thread, errors


def join(thread):
    thread.join(timeout=3)
    assert not thread.is_alive(), "prepared-model handoff deadlocked"


def test_close_excludes_concurrent_native_start(monkeypatch):
    closing, release, attempted, constructed = Event(), Event(), Event(), Event()

    def synchronize(device):
        closing.set()
        assert release.wait(3)

    def server(*args):
        constructed.set()
        raise AssertionError("native server must not start while or after the owner closes")

    model, events, weights = model_fixture(monkeypatch, synchronize=synchronize, server=server)

    def start():
        assert closing.wait(3)
        attempted.set()
        model.start_server()

    def release_close():
        assert attempted.wait(3)
        try:
            assert not constructed.wait(0.05)
            assert not model.closed and not events and weights() is not None
        finally:
            release.set()

    starter, start_errors = thread_call(start)
    observer, observer_errors = thread_call(release_close)
    model.close()
    join(observer)
    join(starter)
    assert not observer_errors
    assert len(start_errors) == 1 and "closed" in str(start_errors[0])
    assert not constructed.is_set()
    assert events == ["graph", "session", "table"]
    assert model.closed and not model.resources and not model.buffers and not model.graphs
    gc.collect()
    assert weights() is None


def test_native_start_excludes_close_until_owner_handoff_finishes(monkeypatch):
    constructing, release, attempted, drained = Event(), Event(), Event(), Event()

    class Runtime:
        drained = False

        def close(self):
            self.drained = True
            drained.set()

    def server(*args):
        constructing.set()
        assert release.wait(3)
        return Runtime()

    model, events, _ = model_fixture(monkeypatch, server=server)
    starter, start_errors = thread_call(model.start_server)
    assert constructing.wait(3)

    def release_startup():
        assert attempted.wait(3)
        try:
            assert not drained.wait(0.05)
            assert events == ["synchronize"]
        finally:
            release.set()

    observer, observer_errors = thread_call(release_startup)
    attempted.set()
    model.close()
    join(starter)
    join(observer)
    assert not start_errors and not observer_errors
    assert drained.is_set() and model.closed
    assert events == ["synchronize", "synchronize", "graph", "session", "table"]
    model.close()
    assert events.count("graph") == 1


def test_wrong_thread_close_preserves_runtime_and_all_owners(monkeypatch):
    class Runtime:
        drained = False

        def close(self):
            raise AssertionError("wrong-thread cleanup must not stop native serving")

    model, events, weights = model_fixture(monkeypatch)
    runtime = Runtime()
    model.runtime, model._consumed = runtime, True
    closer, errors = thread_call(model.close)
    join(closer)
    assert len(errors) == 1 and "preparation thread" in str(errors[0])
    assert model.runtime is runtime and not model.closed
    assert model.graphs and model.buffers and model.resources and weights() is not None
    assert not events


@pytest.mark.parametrize("raises", [False, True])
def test_unconfirmed_native_drain_retains_every_prepared_resource(monkeypatch, raises):
    class Runtime:
        drained = False

        def close(self):
            if raises:
                raise RuntimeError("CUDA stream drain failed")

    model, events, weights = model_fixture(monkeypatch)
    runtime = Runtime()
    model.runtime, model._consumed = runtime, True
    with pytest.raises(RuntimeError, match="drain"):
        model.close()
    gc.collect()
    assert not model.closed and model.runtime is runtime
    assert model.graphs and model.buffers and model.resources
    assert weights() is not None and not events


def test_failed_startup_retention_blocks_cleanup_and_second_handoff(monkeypatch):
    error = RuntimeError("native startup failed without confirmed drain")
    error.resources_retained = True

    def server(*args):
        raise error

    model, events, weights = model_fixture(monkeypatch, server=server)
    with pytest.raises(RuntimeError) as raised:
        model.start_server()
    assert raised.value is error and model.consumed
    with pytest.raises(RuntimeError, match="remain retained"):
        model.close()
    with pytest.raises(RuntimeError, match="consumed or closed"):
        model.start_server()
    assert not model.closed and weights() is not None
    assert model.graphs and model.buffers and model.resources
    assert events == ["synchronize"]


def test_confirmed_drain_releases_resources_even_when_runtime_reports_error(monkeypatch):
    failure = RuntimeError("model worker reported a device completion error")

    class Runtime:
        drained = True

        def close(self):
            raise failure

    model, events, weights = model_fixture(monkeypatch)
    model.runtime, model._consumed = Runtime(), True
    with pytest.raises(RuntimeError) as raised:
        model.close()
    assert raised.value is failure
    assert model.closed and model.runtime is None
    assert not model.resources and not model.buffers and not model.graphs
    assert events == ["synchronize", "graph", "session", "table"]
    gc.collect()
    assert weights() is None
    model.close()
    assert events.count("graph") == 1


def test_context_manager_preserves_primary_failure_when_cleanup_is_undrained(monkeypatch):
    model, events, weights = model_fixture(monkeypatch)
    model._startup_undrained = True
    primary = ValueError("preparation consumer failed")
    with pytest.raises(ValueError) as raised, model:
        raise primary
    assert raised.value is primary
    assert any("cleanup failed" in note and "remain retained" in note for note in primary.__notes__)
    assert not events and weights() is not None and not model.closed


def test_drained_owner_can_retry_incomplete_resource_cleanup(monkeypatch):
    class Runtime:
        drained = True

        def close(self):
            pass

    class Session(Resource):
        calls = 0

        def close(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("preparation resource cleanup failed once")
            super().close()

    model, events, weights = model_fixture(monkeypatch)
    model.session = Session("session", events)
    model.runtime, model._consumed = Runtime(), True
    with pytest.raises(RuntimeError, match="failed once"):
        model.close()
    assert not model.closed and model.resources and weights() is not None
    assert "table" not in events
    model.close()
    assert model.closed and model.runtime is None
    assert events.count("table") == 1
    assert not model.resources and not model.graphs and not model.buffers
    gc.collect()
    assert weights() is None


@pytest.mark.parametrize("fails", [False, True])
def test_peer_resources_release_only_after_native_rank_drains(monkeypatch, fails):
    def serve_rank(owner, socket, *, rank, world):
        owner._consumed = True
        assert (socket, rank, world) == ("peer socket", 1, 2)
        if fails:
            raise RuntimeError("peer drain failed")

    monkeypatch.setattr("ignition.preparation.execution.PreparedExecution.serve_rank", serve_rank)
    model, events, weights = model_fixture(monkeypatch)
    if fails:
        with pytest.raises(RuntimeError, match="peer drain failed"):
            model.serve_rank("peer socket", rank=1, world=2)
        with pytest.raises(RuntimeError, match="must retain"):
            model.close()
        assert not events and not model.closed and weights() is not None
    else:
        model.serve_rank("peer socket", rank=1, world=2)
        model.close()
        assert model.closed and events == ["synchronize", "graph", "session", "table"]
        gc.collect()
        assert weights() is None


def test_snapshot_budget_covers_the_selected_speculative_layout():
    from ignition.recipes.qwen38.bootstrap import PreparationConfig
    from ignition.recipes.qwen38.state import HybridStateLayout

    ordinary = HybridStateLayout.build().row_bytes
    speculative = HybridStateLayout.build(speculative_tokens=7).row_bytes
    assert speculative > ordinary
    PreparationConfig(snapshot_budget_bytes=ordinary)
    with pytest.raises(ValueError, match="at least one state row"):
        PreparationConfig(mtp=True, snapshot_budget_bytes=ordinary)
    config = PreparationConfig(mtp=True,
                               snapshot_budget_bytes=speculative)
    assert config.snapshot_budget_bytes == speculative


def test_moe_decode_a16_requires_separate_graph_roles_and_aligned_experts():
    from ignition.recipes.qwen38.bootstrap import PreparationConfig

    assert PreparationConfig(force_moe_decode_a16=True).force_moe_decode_a16
    assert PreparationConfig(devices=(0, 1), force_moe_decode_a16=True).force_moe_decode_a16
    with pytest.raises(ValueError, match="separate prefill and decode graphs"):
        PreparationConfig(force_moe_decode_a16=True, mixed_prefill=True)
    with pytest.raises(ValueError, match="intermediate width divisible by 64"):
        PreparationConfig(devices=(0, 1, 2, 3), force_moe_decode_a16=True)
