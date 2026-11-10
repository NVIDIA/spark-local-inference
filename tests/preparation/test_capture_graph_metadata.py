# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export graph metadata follows instantiation, reset, and collection."""

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
import gc
import itertools
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch.cuda.graphs import CUDAGraph, _RetainedCallbacks

from ignition.preparation import capture
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.runtime_package import _graphs, capture_graph, instantiate_graph


class Graph:
    register_post_instantiate_hook = CUDAGraph.register_post_instantiate_hook
    register_destroy_callback = CUDAGraph.register_destroy_callback

    def __init__(self, keep_graph=False):
        self.keep_graph = keep_graph
        self._post_instantiate_hooks = OrderedDict()
        self._arm()
        self.template = self.executable = 0
        self.instantiations = 0
        self._executable = next(self._handles)

    def _arm(self):
        self._retained = _RetainedCallbacks()
        self._finalizer = weakref.finalize(self, self._retained.fire)

    def capture_end(self):
        self.template = 11
        if not self.keep_graph:
            self.instantiate()
            self.template = 0

    def instantiate(self):
        assert self.template == 11
        self.executable = self._executable
        self.instantiations += 1
        for hook in self._post_instantiate_hooks.values():
            hook(self)

    def raw_cuda_graph(self):
        if not self.template:
            raise RuntimeError("graph template was released")
        return self.template

    def raw_cuda_graph_exec(self):
        if not self.executable:
            raise RuntimeError("graph is not instantiated")
        return self.executable

    def reset(self):
        self._retained.fire()
        self._finalizer.detach()
        self._arm()
        self.template = self.executable = 0


class Recorder:
    def __init__(self):
        self.records = {}
        self.events = []
        self.fail = False

    def record_graph(self, template, executable):
        assert template == 11 and executable >= 12
        self.records[executable] = template
        self.events.append("record")
        if self.fail:
            raise RuntimeError("native graph copy failed")

    def release_graph(self, executable):
        self.records.pop(executable, None)
        self.events.append("release")


@pytest.fixture
def recorder(monkeypatch):
    native = Recorder()
    monkeypatch.setattr(Graph, "_handles", itertools.count(12), raising=False)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(ModuleImages, "current", classmethod(lambda cls: SimpleNamespace(native=native)))
    return native


def test_recording_releases_template_and_reset_invalidates_reused_executable(recorder):
    graph = capture_graph(retain_graphs=True)
    assert not graph.keep_graph
    graph.capture_end()
    instantiate_graph(graph)
    assert graph.instantiations == 1
    assert _graphs({"decode": graph}) == [dict(name="decode", graph=0, executable=12)]
    with pytest.raises(RuntimeError, match="template was released"):
        graph.raw_cuda_graph()
    graph.reset()
    assert not recorder.records
    assert graph._flare_graph_metadata.executable == 0
    graph.capture_end()
    assert recorder.records == {12: 11}
    assert recorder.events == ["record", "release", "record"]
    reference = weakref.ref(graph)
    del graph
    gc.collect()
    assert reference() is None
    assert not recorder.records
    assert recorder.events[-1] == "release"


def test_recording_failure_releases_partial_metadata_and_propagates(recorder):
    graph = capture_graph(retain_graphs=True)
    recorder.fail = True
    with pytest.raises(RuntimeError, match="native graph copy failed"):
        graph.capture_end()
    assert not recorder.records
    assert graph._flare_graph_metadata.executable == 0
    graph.reset()
    recorder.fail = False
    graph.capture_end()
    assert recorder.records == {12: 11}


def test_recording_preserves_live_template_and_unretained_capture_paths(monkeypatch, recorder):
    unretained = capture_graph()
    unretained.capture_end()
    assert not recorder.records
    monkeypatch.setattr(ModuleImages, "current", classmethod(lambda cls: None))
    retained = capture_graph(retain_graphs=True)
    retained.capture_end()
    instantiate_graph(retained)
    assert retained.keep_graph
    assert _graphs({"decode": retained}) == [dict(name="decode", graph=11, executable=13)]


def test_shared_capture_records_after_all_warmups_without_reinstantiation(monkeypatch, recorder):
    calls, state = [], {"capturing": False}
    stream = SimpleNamespace(wait_stream=lambda other: None, synchronize=lambda: None)

    @contextmanager
    def graph_scope(graph, **kwargs):
        assert calls[:2] == [("first", False), ("second", False)]
        state["capturing"] = True
        try:
            yield
        finally:
            state["capturing"] = False
        graph.capture_end()

    for name, value in dict(current_stream=lambda device: stream, stream=lambda stream: nullcontext(),
                            synchronize=lambda device: None, graph=graph_scope,
                            mem_get_info=lambda device: (1024, 2048), memory_allocated=lambda device: 0,
                            memory_reserved=lambda device: 0).items():
        monkeypatch.setattr(torch.cuda, name, value)
    functions = {name: lambda name=name: calls.append((name, state["capturing"]))
                 for name in ("first", "second")}
    graphs, compiled = capture.capture_forwards(functions,
        {name: SimpleNamespace(kind="decode") for name in functions}, session=SimpleNamespace(capture=nullcontext),
        device="cpu", stream=stream, pool=None, compile_decoder=False, retain_graphs=True)
    assert not compiled
    assert all(graph.instantiations == 1 for graph in graphs.values())
    assert calls == [("first", False), ("second", False), ("first", True), ("second", True)]
    assert recorder.events == ["record", "record"]
