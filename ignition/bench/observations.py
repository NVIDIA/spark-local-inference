# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Untimed captured forwards with explicit metadata and acceptance destinations."""

from functools import partial
from pathlib import Path


class ObservationForward:
    def __init__(self, forward, *, rows, device, draft=False, acceptance_width=None):
        import torch

        self.forward = forward
        layout = {}
        for role in (("target", "draft") if draft else ("target",)):
            layout.update({f"{role}.{name}": (shape, dtype) for name, shape, dtype in (
                ("live_counts", (2,), torch.int32), ("starts", (rows,), torch.int64),
                ("lengths", (rows,), torch.int32), ("query_start_loc", (rows + 1,), torch.int32))})
        if acceptance_width is not None:
            from ignition.components.speculation import SpeculativeAcceptance

            acceptance = SpeculativeAcceptance.storage_layout(rows, acceptance_width)
            layout.update({f"acceptance.{name}": acceptance[name] for name in
                           ("offered_drafts", "accepted_drafts", "survival", "valid")})
            layout.update({f"commit.{name}": (shape, dtype) for name, shape, dtype in (
                ("accepted_drafts", (rows,), torch.int32), ("emitted_count", (rows,), torch.int32),
                ("emitted_ids", (rows, acceptance_width), torch.int64))})
        self.outputs = {name: torch.empty(shape, dtype=dtype, device=device)
                        for name, (shape, dtype) in layout.items()}

    def __call__(self):
        observations = self.forward.observe()
        for name, output in self.outputs.items():
            output.copy_(observations[name])


class CapturedObservers:
    """Own private observation graphs and their kernels until replay has drained."""

    def __init__(self, device, forwards, graphs, compiled, names):
        self.device = device
        self.forwards, self.graphs, self.compiled = forwards, graphs, compiled
        self.observers = {name: partial(self.replay, private) for name, private in names.items()}
        self.closed = False

    def replay(self, name):
        if self.closed:
            raise RuntimeError("benchmark observation graphs are closed")
        self.graphs[name].replay()
        return self.forwards[name].outputs

    def close(self):
        import torch

        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.compiled.clear()
        for forward in self.forwards.values():
            forward.outputs.clear()
        self.forwards.clear()
        self.observers.clear()
        self.closed = True


def capture_observers(forwards, signatures, *, device, cache_root=None, **kwargs):
    """Capture private metadata graphs alongside an already captured serving catalog."""
    from ignition.preparation.capture import capture_forwards

    names = {name: f"benchmark-observation-{name}" for name in forwards}
    wrappers = {}
    for name, forward in forwards.items():
        signature = signatures[name]
        if signature.kind not in ("prefill", "speculative"):
            raise ValueError("benchmark observations require prefill or speculative graphs")
        wrappers[names[name]] = ObservationForward(forward, rows=signature.batch_capacity, device=device,
            draft=signature.kind == "speculative" and forward.draft is not None)
    private_signatures = {names[name]: signatures[name] for name in forwards}
    root = None if cache_root is None else Path(cache_root) / "benchmark-observations"
    graphs, compiled = capture_forwards(wrappers, private_signatures, device=device, cache_root=root,
        report_catalog=False, label="Capture benchmark observations", **kwargs)
    return CapturedObservers(device, wrappers, graphs, compiled, names)
