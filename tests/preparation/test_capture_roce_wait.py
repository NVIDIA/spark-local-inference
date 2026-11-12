# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiled warmup across hosts waits longer for a peer than the graphs capture does."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

from ignition.preparation import capture
from ignition.preparation.collectives import WARMUP_SPIN_LIMIT


def test_compiled_warmup_widens_the_roce_wait_and_capture_keeps_the_serving_wait(monkeypatch):
    waits = []
    stream = SimpleNamespace(wait_stream=lambda value: None, synchronize=lambda: None)
    runtime = SimpleNamespace(spin_limit=20_000_000)

    @contextmanager
    def capture_scope(*, stream):
        waits.append(("capture", runtime.spin_limit))
        yield

    runtime.capture = capture_scope

    class Forward:
        cache_hit = False
        capture_allocation_sizes = ()

        def prepare_capture(self):
            return 0

        def retire_after_capture(self):
            pass

        def __call__(self):
            pass

    forward = Forward()

    class Cache:
        def __init__(self, **kwargs):
            pass

        def prepare_many(self, functions, signatures, **kwargs):
            waits.append(("warm", runtime.spin_limit))
            return functions

    def all_gather_object(output, value, group):
        output[:] = [value] * len(output)

    coordinator = SimpleNamespace(spans_hosts=True, ranks=(0, 1), group=None, barrier=lambda name, ranks: None)
    monkeypatch.setattr(capture, "ForwardCache", Cache)
    monkeypatch.setattr(capture, "capture_graph", lambda **options: "graph")
    monkeypatch.setattr(capture.torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(capture.torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(capture.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(capture.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(capture.torch.cuda, "graph", lambda *args, **kwargs: nullcontext())
    graphs, _ = capture.capture_forwards({"decode": forward}, {"decode": SimpleNamespace(kind="decode")},
        session=SimpleNamespace(namespace=SimpleNamespace(to_dict=lambda: {}), capture=nullcontext),
        device="cuda:0", stream=stream, pool=None, collective=runtime, coordinator=coordinator)
    assert graphs == {"decode": "graph"}
    assert waits == [("warm", WARMUP_SPIN_LIMIT), ("capture", 20_000_000)]
    assert runtime.spin_limit == 20_000_000
