# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from ignition.preparation.graphs import PreparedGraphBundle


def bundle(**overrides):
    arguments = {"device": 0, "graphs": {"decode": object()}, "buffers": {}}
    arguments.update(overrides)
    return PreparedGraphBundle(**arguments)


def test_import_does_not_load_torch_or_native_extension():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import ignition.preparation.graphs; "
        "assert 'torch' not in sys.modules; assert 'flare_heliopause' not in sys.modules",
    ], check=True)


@pytest.mark.parametrize("arguments", [
    {"device": -1}, {"device": True}, {"device": "cuda:0"},
    {"graphs": {}}, {"graphs": {" ": object()}},
    {"buffers": {"": object()}}, {"buffers": {1: object()}},
])
def test_invalid_identity_rejected_before_cuda_import(arguments):
    with pytest.raises(ValueError):
        bundle(**arguments)


def test_registration_metadata_is_copied_and_read_only():
    graphs = {"decode": object(), "prefill": object()}
    buffers = {"state": object()}
    resources = [object()]
    prepared = bundle(graphs=graphs, buffers=buffers, resources=resources)
    graphs.clear()
    buffers.clear()
    resources.clear()
    assert tuple(prepared.graphs) == ("decode", "prefill")
    assert tuple(prepared.buffers) == ("state",)
    assert len(prepared.resources) == 1
    assert dict(prepared.graph_indices) == {"decode": 0, "prefill": 1}
    with pytest.raises(TypeError):
        prepared.graph_indices["decode"] = 3


def test_raw_replay_requires_qualification_binding_before_cuda(monkeypatch):
    prepared = bundle()
    monkeypatch.setitem(sys.modules, "flare_heliopause", SimpleNamespace())

    def forbidden():
        pytest.fail("qualification feature rejection must precede CUDA validation")

    monkeypatch.setattr(prepared, "_validate_resources", forbidden)
    with pytest.raises(RuntimeError, match="native qualification Cargo feature"):
        prepared.start_worker()
    assert not prepared.consumed


def test_failed_registration_allows_retry_but_success_is_one_shot(monkeypatch):
    prepared = bundle()
    calls = 0
    receiver = object()

    def register(capacity):
        nonlocal calls
        calls += 1
        assert capacity == 3
        if calls == 1:
            raise RuntimeError("worker initialization failed before ready")
        return receiver

    monkeypatch.setattr(prepared, "_register", register)
    with pytest.raises(RuntimeError, match="initialization failed"):
        prepared.start_worker(queue_capacity=3)
    assert not prepared.consumed
    assert prepared.start_worker(queue_capacity=3) is receiver
    assert prepared.consumed
    with pytest.raises(RuntimeError, match="already been consumed"):
        prepared.start_worker(queue_capacity=3)
    assert calls == 2


def test_concurrent_registration_only_transfers_once(monkeypatch):
    prepared = bundle()
    entered = Event()
    release = Event()
    receiver = object()

    def register(_capacity):
        entered.set()
        assert release.wait(5)
        return receiver

    monkeypatch.setattr(prepared, "_register", register)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(prepared.start_worker)
        assert entered.wait(5)
        second = executor.submit(prepared.start_worker)
        release.set()
        assert first.result(timeout=5) is receiver
        with pytest.raises(RuntimeError, match="already been consumed"):
            second.result(timeout=5)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_never_consumes_bundle(capacity):
    prepared = bundle()
    with pytest.raises(ValueError, match="queue_capacity"):
        prepared.start_worker(queue_capacity=capacity)
    assert not prepared.consumed
