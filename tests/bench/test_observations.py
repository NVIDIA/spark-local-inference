"""Private benchmark graph capture, replay dispatch, and resource lifetimes."""

from types import SimpleNamespace

import pytest
import torch

from ignition.bench.observations import CapturedObservers, capture_observers
from ignition.preparation.execution import GraphSignature


def test_capture_observers_keeps_private_graphs_out_of_serving_catalog(monkeypatch):
    captured = {}
    replays = []

    def eager_observe():
        raise AssertionError("observation replay must not execute an eager forward")

    forwards = {"prefill128": SimpleNamespace(observe=eager_observe),
                "round2x8": SimpleNamespace(observe=eager_observe, draft=object())}
    signatures = {"prefill128": GraphSignature("prefill", 1, 128),
                  "round2x8": GraphSignature("speculative", 2, 8)}

    def capture(functions, private_signatures, **kwargs):
        captured.update(functions=functions, signatures=private_signatures, kwargs=kwargs)
        graphs = {}
        for name, forward in functions.items():
            def replay(name=name, forward=forward):
                replays.append(name)
                forward.outputs["target.starts"].fill_(512)
            graphs[name] = SimpleNamespace(replay=replay)
        return graphs, {name: object() for name in functions}

    monkeypatch.setattr("ignition.preparation.capture.capture_forwards", capture)
    owner = capture_observers(forwards, signatures, device="cpu", cache_root="/tmp/kernels",
                              session=object(), compile_decoder=True)
    assert set(forwards) == {"prefill128", "round2x8"}
    assert set(owner.observers) == set(forwards)
    assert set(captured["functions"]) == {"benchmark-observation-prefill128", "benchmark-observation-round2x8"}
    assert str(captured["kwargs"]["cache_root"]) == "/tmp/kernels/benchmark-observations"
    assert captured["kwargs"]["report_catalog"] is False
    prefill = owner.observers["prefill128"]()
    round_values = owner.observers["round2x8"]()
    assert prefill["target.starts"].tolist() == [512]
    assert round_values["target.starts"].tolist() == [512, 512]
    assert round_values["draft.starts"].dtype == torch.int64
    assert not any(name.startswith(("acceptance.", "commit.")) for name in round_values)
    assert replays == ["benchmark-observation-prefill128", "benchmark-observation-round2x8"]


def test_observer_close_drains_and_resets_before_releasing_kernel_and_output_owners(monkeypatch):
    events = []

    class Owners(dict):
        def clear(self):
            events.append("kernels")
            super().clear()

    class Outputs(dict):
        def clear(self):
            events.append("outputs")
            super().clear()

    outputs = Outputs(value=torch.ones(1))
    compiled = Owners(private=object())

    def reset():
        assert compiled and outputs
        events.append("reset")

    graph = SimpleNamespace(reset=reset)
    owner = CapturedObservers("cuda:0", {"private": SimpleNamespace(outputs=outputs)},
                              {"private": graph}, compiled, {"round": "private"})
    replay = owner.observers["round"]
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("drain"))
    owner.close()
    owner.close()
    assert events == ["drain", "reset", "kernels", "outputs"]
    assert owner.closed and not owner.graphs and not owner.compiled and not owner.forwards
    with pytest.raises(RuntimeError, match="closed"):
        replay()


@pytest.mark.parametrize("failure", ["drain", "reset"])
def test_observer_close_retains_resources_when_drain_or_reset_fails(monkeypatch, failure):
    outputs = {"value": torch.ones(1)}
    compiled = {"private": object()}

    def fail():
        raise RuntimeError("cannot close graph")

    graph = SimpleNamespace(reset=fail if failure == "reset" else lambda: None)
    owner = CapturedObservers("cuda:0", {"private": SimpleNamespace(outputs=outputs)},
                              {"private": graph}, compiled, {"round": "private"})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: fail() if failure == "drain" else None)
    with pytest.raises(RuntimeError, match="cannot close graph"):
        owner.close()
    assert not owner.closed
    assert owner.graphs["private"] is graph and owner.compiled and owner.forwards and outputs and owner.observers
