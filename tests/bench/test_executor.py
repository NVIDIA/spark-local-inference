# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Rank timing reduction, snapshot ordering, and benchmark peer failure handling."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
import socket
import threading
from types import SimpleNamespace

import pytest

from ignition.bench import executor


class Rank:
    def __init__(self, nanos, value):
        self.nanos, self.value = nanos, value
        self.saved = None
        self.commands = []
        self.failure = None
        self.restore_gate = None
        self.restoring = threading.Event()
        self.benchmark_state = {"value": value}
        self.runner = SimpleNamespace(device="cpu", stream=None, records=lambda frame: frame)

    def execute(self, command):
        operation = command["operation"]
        self.commands.append(operation)
        if operation == "snapshot":
            self.saved = self.value
        elif operation == "restore":
            self.restoring.set()
            if self.restore_gate is not None:
                assert self.restore_gate.wait(timeout=5)
            self.value = self.saved
        elif operation == "run":
            if self.failure is not None:
                raise RuntimeError(self.failure)
            self.value += 1
            return {"graph_ns": self.nanos}
        elif operation in ("release", "close"):
            self.saved = None
        else:
            raise AssertionError(operation)
        return {}


@contextmanager
def rank_executor(monkeypatch, ranks):
    monkeypatch.setattr(executor, "_RankBenchmark", lambda model: model)
    connections, threads, errors = [], [], []

    def serve(rank, connection):
        try:
            executor.serve_peer(rank, connection)
        except BaseException as error:
            errors.append(error)

    for rank in ranks[1:]:
        root, peer = socket.socketpair()
        connections.extend((root, peer))
        thread = threading.Thread(target=serve, args=(rank, peer), daemon=True)
        thread.start()
        threads.append(thread)
    prepared = SimpleNamespace(model=ranks[0], start_benchmark=lambda: connections[::2])
    steps = executor.BenchmarkExecutor(prepared)
    try:
        yield steps
    finally:
        steps.close()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
        for connection in connections:
            connection.close()
        assert not errors


def test_every_rank_restores_before_replay_and_slowest_rank_sets_latency(monkeypatch):
    ranks = [Rank(7_000_000, 3), Rank(9_500_000, 5), Rank(8_000_000, 7)]
    with rank_executor(monkeypatch, ranks) as steps:
        steps.snapshot()
        for sequence in range(3):
            steps.restore()
            assert [rank.value for rank in ranks] == [3, 5, 7]
            assert steps.run(dict(graph="round", sequence=sequence)) == 9_500_000
            assert [rank.value for rank in ranks] == [4, 6, 8]
        ranks[1].nanos = 1
        assert steps.run(dict(graph="round", sequence=3)) == 8_000_000
        steps.restore()
        steps.release_snapshot()
        assert all(rank.saved is None for rank in ranks)
    assert all(rank.commands[-1] == "close" for rank in ranks)


def test_restore_waits_for_peer_completion_before_returning(monkeypatch):
    ranks = [Rank(1, 3), Rank(2, 5)]
    gate = ranks[1].restore_gate = threading.Event()
    with rank_executor(monkeypatch, ranks) as steps:
        steps.snapshot()
        ranks[1].value = 99
        with ThreadPoolExecutor(1) as pool:
            restore = pool.submit(steps.restore)
            try:
                assert ranks[1].restoring.wait(timeout=5)
                assert not restore.done()
                assert ranks[1].value == 99
            finally:
                gate.set()
            restore.result(timeout=5)
        assert ranks[1].value == 5


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_rank_failure_identifies_rank_and_consumes_all_replies(monkeypatch, rank):
    ranks = [Rank(1, 3), Rank(2, 5), Rank(3, 7)]
    ranks[rank].failure = "captured benchmark replay changed allocated CUDA memory"
    with rank_executor(monkeypatch, ranks) as steps:
        with pytest.raises(RuntimeError, match=rf"\({rank}, .*changed allocated CUDA memory"):
            steps.run(dict(graph="round", sequence=1))
        ranks[rank].failure = None
        assert steps.run(dict(graph="round", sequence=2)) == 3


def test_local_rank_rejects_replay_allocations(monkeypatch):
    import torch

    rank = object.__new__(executor._RankBenchmark)
    rank.runner = SimpleNamespace(device="cuda:0", run=lambda step: 1)
    sizes = iter((1024, 2048))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: next(sizes))
    with pytest.raises(RuntimeError, match="changed allocated CUDA memory"):
        rank.execute(dict(operation="run", step={}))


def test_two_device_graphs_restore_each_ranks_mutable_state():
    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("set FLARE_RUN_CUDA_TESTS=1 on two assigned CUDA devices")
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("two assigned CUDA devices are required")
    from ignition.preparation.execution import GraphSignature

    models, values = [], []
    for device in range(2):
        with torch.cuda.device(device):
            value = torch.full((16,), device + 3, dtype=torch.int64, device=device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                value.add_(device + 1)
            stream.synchronize()
            value.fill_(device + 3)
            torch.cuda.synchronize(device)
            buffers = dict(frame_index=torch.zeros(1, dtype=torch.int32, device=device),
                           submission_sequence=torch.zeros(1, dtype=torch.int64, device=device),
                           completion_records=torch.zeros((2, 1, 24), dtype=torch.int64, device=device))
            models.append(SimpleNamespace(device=device, resources=(), session=SimpleNamespace(),
                manifest=lambda: {"graphs": [{"name": "round"}], "buffers": []},
                signatures={"round": GraphSignature("speculative", 1, 8)}, graphs={"round": graph},
                benchmark_state={"value": value}, buffers=buffers))
            values.append(value)
    root, peer = socket.socketpair()
    errors = []

    def serve():
        try:
            executor.serve_peer(models[1], peer)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    prepared = SimpleNamespace(model=models[0], start_benchmark=lambda: (root,))
    steps = executor.BenchmarkExecutor(prepared)
    try:
        steps.snapshot()
        for sequence in range(3):
            steps.restore()
            assert [value.cpu().tolist() for value in values] == [[3] * 16, [4] * 16]
            elapsed = steps.run(dict(graph="round", uploads=(), frame=sequence % 2, sequence=sequence + 1))
            assert elapsed > 0
            assert [value.cpu().tolist() for value in values] == [[4] * 16, [6] * 16]
        steps.restore()
        steps.release_snapshot()
    finally:
        steps.close()
        thread.join(timeout=10)
        root.close()
        peer.close()
        assert not thread.is_alive()
        assert not errors
        for model in models:
            model.graphs["round"].reset()
