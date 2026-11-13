# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from ignition.direct.executor import StepRunner


def test_eager_execution_requires_the_recipe_to_retain_its_forwards():
    with pytest.raises(RuntimeError, match="retains no prepared forwards"):
        StepRunner(SimpleNamespace(forwards=None), eager=True)


class Runner:
    """A rank's step runner that reports a fixed graph time."""

    def __init__(self, nanos):
        self.nanos, self.steps = nanos, []

    def run(self, step):
        self.steps.append(step["sequence"])
        return self.nanos

    def close(self):
        pass


def test_a_step_takes_its_slowest_ranks_graph_time(monkeypatch):
    import socket
    import threading

    from ignition.direct import executor

    peers = {f"rank{rank}": Runner(nanos) for rank, nanos in ((1, 7_000_000), (2, 9_500_000))}
    monkeypatch.setattr(executor, "StepRunner", lambda model, eager: peers[threading.current_thread().name])
    roots, threads = [], []
    for name in peers:
        root, peer = socket.socketpair()
        roots += [root, peer]
        threads.append(threading.Thread(target=executor.serve_peer, args=(None, peer), kwargs={"eager": False},
                                        name=name))
        threads[-1].start()
    steps = object.__new__(executor.RankSteps)
    steps.runner, steps.peers, steps.stopped = Runner(8_000_000), tuple(roots[::2]), False
    assert steps.run({"graph": "speculative16x6", "uploads": [], "frame": 1, "sequence": 1}) == 9_500_000
    peers["rank2"].nanos = 1
    assert steps.run({"graph": "speculative16x6", "uploads": [], "frame": 0, "sequence": 2}) == 8_000_000
    steps.stop()
    for thread in threads:
        thread.join(timeout=10)
    for connection in roots:
        connection.close()
    assert [peer.steps for peer in peers.values()] == [[1, 2], [1, 2]]


def test_combined_round_reads_terminal_confidence_without_a_separate_proposal():
    import torch
    from ignition.direct.executor import RankSteps
    steps = object.__new__(RankSteps)
    values = torch.tensor([[[0., 0.], [0., 0.]], [[.25, .5], [.75, 1.]]])
    steps.runner = SimpleNamespace(confidence_graphs=frozenset({'combined'}),
                                   model=SimpleNamespace(buffers={"proposal_confidence": values}))
    assert steps.confidence('combined', 1) == [[.25, .5], [.75, 1.]]
    assert steps.confidence('width-one', 1) is None


def startup_model(events, *, status_value=0, coordinator=None):
    import torch

    status = torch.tensor([0], dtype=torch.int32)
    def replay():
        assert status.item() == -1
        events.append("replay")
        status.fill_(status_value)
    tuner = SimpleNamespace(graph=SimpleNamespace(replay=replay, reset=lambda: events.append("reset")),
                            status_buffer="status", outputs=("selector",))
    manifest = {"graphs": [{"name": "decode", "tuned_bindings": [{"buffer": "selector", "tuning_graph": "tune"}]}],
                "buffers": [dict(name=name, dtype="int32", nbytes=4, shape=[1], strides=[1])
                            for name in ("selector", "status")],
                "tuning_graphs": [{"name": "unused"},
                                  {"name": "tune", "outputs": ["selector"], "status_buffer": "status"}]}
    return SimpleNamespace(manifest=lambda: manifest, session=SimpleNamespace(_coordinator=coordinator),
                           tuning_graphs={"tune": tuner}, buffers={"status": status},
                           device=0, signatures={}, resources=(), forwards={}, graphs={"tune": tuner.graph})


def test_direct_startup_completes_before_eager_graph_release(monkeypatch):
    from contextlib import nullcontext
    import torch

    events = []
    model = startup_model(events)
    stream = SimpleNamespace(query=lambda: events.append("drain") or True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: None)
    StepRunner(model, eager=True)
    assert events == ["replay", "drain", "reset", "empty_cache"]


@pytest.mark.parametrize("statuses", [(0, 0), (0, 7), (7, 7)])
def test_startup_waits_for_every_rank_and_rejects_any_failed_status(monkeypatch, statuses):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    from threading import Barrier
    import torch
    from ignition.preparation.startup_tuning import replay_startup_tuning

    barrier = Barrier(2, timeout=5)
    values = [None, None]
    finished = []
    class Coordinator:
        def __init__(self, rank):
            self.rank = rank
        def same_on_every_rank(self, value):
            values[self.rank] = value
            barrier.wait()
            agreed = values[0] == values[1]
            barrier.wait()
            return agreed
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    def run(rank):
        model = startup_model([], status_value=statuses[rank], coordinator=Coordinator(rank))
        try:
            replay_startup_tuning(model, SimpleNamespace(query=lambda: True))
        except RuntimeError:
            return "failed"
        finished.append(rank)
        return "ready"
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, range(2)))
    expected = "ready" if statuses == (0, 0) else "failed"
    assert results == [expected, expected]
    assert len(finished) == (2 if expected == "ready" else 0)


def test_startup_rank_catalog_mismatch_rejects_before_launch(monkeypatch):
    from ignition.preparation.startup_tuning import replay_startup_tuning
    events = []
    model = startup_model(events, coordinator=SimpleNamespace(same_on_every_rank=lambda value: False))
    with pytest.raises(RuntimeError, match="order or binding schema differs"):
        replay_startup_tuning(model, None)
    assert events == []


def test_startup_unknown_drain_requires_process_teardown(monkeypatch):
    from contextlib import nullcontext
    import os
    import torch
    from ignition.preparation.startup_tuning import replay_startup_tuning

    class Terminated(BaseException):
        pass
    def abort():
        raise Terminated
    def query():
        raise RuntimeError("CUDA completion unavailable")
    monkeypatch.setattr(os, "abort", abort)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    with pytest.raises(Terminated):
        replay_startup_tuning(startup_model([]), SimpleNamespace(query=query))


@pytest.mark.parametrize('difference', ['outputs', 'status', 'serving-binding'])
def test_startup_agrees_full_binding_schema_before_launch(difference):
    from ignition.preparation.startup_tuning import replay_startup_tuning

    remote_producer = ('tune', ('selector',), 'status')
    remote_consumer = ('decode', (('selector', 'tune'),))
    if difference == 'outputs':
        remote_producer = ('tune', ('other-selector',), 'status')
    elif difference == 'status':
        remote_producer = ('tune', ('selector',), 'other-status')
    else:
        remote_consumer = ('decode', (('other-selector', 'tune'),))
    remote = ((remote_producer,), (remote_consumer,),
              tuple((name, 'int32', 4, (1,), (1,)) for name in ('selector', 'status')))
    events = []
    model = startup_model(events, coordinator=SimpleNamespace(same_on_every_rank=lambda value: value == remote))
    with pytest.raises(RuntimeError, match='binding schema differs'):
        replay_startup_tuning(model, None)
    assert events == []


@pytest.mark.parametrize("buffer", ("selector", "status"))
@pytest.mark.parametrize("field,value", (("dtype", "float32"), ("nbytes", 8),
                                        ("shape", [2]), ("strides", [2])))
def test_startup_rank_storage_mismatch_rejects_before_launch(monkeypatch, buffer, field, value):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import torch
    from ignition.preparation.startup_tuning import replay_startup_tuning

    barrier = Barrier(2, timeout=5)
    signatures = [None, None]
    events = [[], []]

    class Coordinator:
        def __init__(self, rank):
            self.rank = rank

        def same_on_every_rank(self, signature):
            signatures[self.rank] = signature
            barrier.wait()
            return signatures[0] == signatures[1]

    def unexpected_stream(*args):
        raise AssertionError("rank storage disagreement must reject before CUDA access")

    monkeypatch.setattr(torch.cuda, "stream", unexpected_stream)

    def run(rank):
        model = startup_model(events[rank], coordinator=Coordinator(rank))
        if rank == 1:
            descriptor = next(item for item in model.manifest()["buffers"] if item["name"] == buffer)
            descriptor[field] = value
        with pytest.raises(RuntimeError, match="binding schema differs"):
            replay_startup_tuning(model, None)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(run, range(2)))
    assert events == [[], []]
