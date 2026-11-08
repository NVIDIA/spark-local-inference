# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Coordinate restored graph measurements across every preparation rank."""

from ignition.preparation.host_ranks import receive_json, receive_object, send_json, send_object


class StateSnapshot:
    """Same-device copies of explicitly declared persistent tensors, outside replay timing."""

    def __init__(self, tensors, stream):
        import torch

        self.entries = {}
        seen = set()
        with torch.cuda.stream(stream), torch.no_grad():
            for name, tensor in tensors.items():
                if tensor.device.type != "cuda":
                    raise ValueError(f"benchmark state {name} must be a CUDA tensor")
                key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
                if key in seen:
                    continue
                seen.add(key)
                saved = torch.empty_like(tensor)
                saved.copy_(tensor, non_blocking=True)
                self.entries[name] = (tensor, saved)
        stream.synchronize()

    def restore(self, stream):
        import torch

        with torch.cuda.stream(stream), torch.no_grad():
            for tensor, saved in self.entries.values():
                tensor.copy_(saved, non_blocking=True)


class _RankBenchmark:
    def __init__(self, model):
        from ignition.direct.executor import StepRunner

        if not getattr(model, "benchmark_state", None):
            raise ValueError("model preparation did not declare complete benchmark state")
        self.runner = StepRunner(model, eager=False)
        self.snapshot = None

    def execute(self, command):
        import torch

        operation = command["operation"]
        runner = self.runner
        if operation == "run":
            allocated = torch.cuda.memory_allocated(runner.device)
            elapsed = runner.run(command["step"])
            if torch.cuda.memory_allocated(runner.device) != allocated:
                raise RuntimeError("captured benchmark replay changed allocated CUDA memory")
            return {"graph_ns": elapsed}
        if operation == "snapshot":
            if self.snapshot is not None:
                raise RuntimeError("benchmark rank already owns a state snapshot")
            self.snapshot = StateSnapshot(runner.model.benchmark_state, runner.stream)
        elif operation == "restore":
            if self.snapshot is None:
                raise RuntimeError("benchmark rank has no state snapshot to restore")
            self.snapshot.restore(runner.stream)
            runner.stream.synchronize()
        elif operation == "release":
            runner.stream.synchronize()
            self.snapshot = None
        elif operation == "close":
            runner.close()
            self.snapshot = None
        else:
            raise ValueError(f"unknown benchmark rank operation {operation!r}")
        return {}


class BenchmarkExecutor:
    """Run real prompt setup and restored measurements on every prepared rank."""

    def __init__(self, prepared):
        self.model = getattr(prepared, "model", prepared)
        if not getattr(self.model, "benchmark_state", None):
            raise ValueError("model preparation did not declare complete benchmark state")
        self.peers = tuple(prepared.start_benchmark()) if prepared is not self.model else ()
        try:
            self.local = _RankBenchmark(self.model)
        except BaseException:
            if self.peers:
                prepared.supervisor.abort("benchmark rank startup failed")
            raise
        self.device, self.stream = self.local.runner.device, self.local.runner.stream
        self.closed = False

    def _exchange(self, command):
        for peer in self.peers:
            send_object(peer, command)
        results, failures = [], []
        try:
            results.append(self.local.execute(command))
        except Exception as error:
            failures.append((0, f"{type(error).__name__}: {error}"))
        for rank, peer in enumerate(self.peers, 1):
            try:
                reply = receive_json(peer)
                if "error" in reply:
                    failures.append((rank, reply["error"]))
                else:
                    results.append(reply)
            except ConnectionError as error:
                failures.append((rank, str(error)))
        if failures:
            raise RuntimeError(f"benchmark rank operation {command['operation']!r} failed: {failures}")
        return results

    def run(self, step):
        return max(result["graph_ns"] for result in self._exchange(dict(operation="run", step=step)))

    def records(self, frame):
        return self.local.runner.records(frame)

    def snapshot(self):
        self._exchange(dict(operation="snapshot"))

    def restore(self):
        """Drain every rank's restoration before any rank starts a timed graph."""
        self._exchange(dict(operation="restore"))

    def release_snapshot(self):
        self._exchange(dict(operation="release"))

    def close(self):
        if not self.closed:
            self.closed = True
            self._exchange(dict(operation="close"))


def serve_peer(model, connection):
    """Execute benchmark commands until rank zero confirms its final drain."""
    rank = _RankBenchmark(model)
    while True:
        command = receive_object(connection)
        try:
            reply = rank.execute(command)
        except Exception as error:
            reply = {"error": f"{type(error).__name__}: {error}"}
        send_json(connection, reply)
        if command["operation"] == "close":
            return
