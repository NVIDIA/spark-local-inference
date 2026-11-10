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
from ignition.preparation.runtime_package import (
    RecordedExportGraph, _graphs, capture_graph, instantiate_graph, record_export_graph,
)


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
        self.templates = {}
        self.sequence = itertools.count(1)

    def record_graph(self, template, executable):
        assert template == 11 and executable >= 12
        self.records[executable] = template
        self.events.append("record")
        if self.fail:
            raise RuntimeError("native graph copy failed")

    def release_graph(self, executable):
        self.records.pop(executable, None)
        self.events.append("release")

    def record_template(self, template, flags):
        assert template == 11 and flags == 1
        if self.fail:
            raise RuntimeError("native graph copy failed")
        recorded = next(self.sequence)
        self.templates[recorded] = template
        self.events.append("record-template")
        return recorded

    def release_template(self, recorded):
        self.templates.pop(recorded, None)
        self.events.append("release-template")


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


def test_export_metadata_outlives_template_and_retains_external_owners(monkeypatch, recorder):
    import flare_heliopause

    monkeypatch.setattr(flare_heliopause, "current_context", lambda: (100, 0))
    storage = SimpleNamespace(pool=SimpleNamespace(id=(0, 1)), graph_pool=SimpleNamespace(id=(0, 2)))
    graph = Graph(keep_graph=True)
    graph.pools = lambda: [(0, 2)]
    graph.capture_end()
    kernel_owner = SimpleNamespace()
    owner = record_export_graph(graph, storage, owners=(kernel_owner,))
    assert graph.template == graph.executable == graph.instantiations == 0
    assert owner.pools == (storage.pool, storage.graph_pool)
    assert owner.owners == (kernel_owner,)
    assert _graphs({"decode": owner}, recorder=recorder) == [dict(name="decode", recorded=1)]
    with pytest.raises(ValueError, match="another native module recorder"):
        _graphs({"decode": owner}, recorder=Recorder())
    assert recorder.templates == {1: 11}
    owner.reset()
    owner.reset()
    assert owner.pools == owner.owners == ()
    assert not recorder.templates
    assert recorder.events == ["record-template", "release-template"]
    with pytest.raises(ValueError, match="released"):
        _graphs({"decode": owner}, recorder=recorder)


def test_export_rejects_unowned_pool_without_destroying_template(monkeypatch, recorder):
    storage = SimpleNamespace(pool=SimpleNamespace(id=(0, 1)), graph_pool=SimpleNamespace(id=(0, 2)))
    graph = Graph(keep_graph=True)
    graph.pools = lambda: [(0, 3)]
    graph.capture_end()
    with pytest.raises(ValueError, match="without an external owner"):
        record_export_graph(graph, storage)
    assert graph.template == 11
    assert not recorder.templates


def test_export_releases_metadata_on_collection(recorder):
    owner = RecordedExportGraph(recorder, 1, 100, ())
    recorder.templates[1] = 11
    reference = weakref.ref(owner)
    del owner
    gc.collect()
    assert reference() is None
    assert not recorder.templates


def test_export_resources_accept_recorded_graphs_and_executable_tuners(monkeypatch, recorder):
    import flare_heliopause
    from ignition.preparation.graphs import PreparedGraphBundle

    monkeypatch.setattr(flare_heliopause, "current_context", lambda: (100, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda ordinal: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda ordinal: (12, 0))
    recorded = RecordedExportGraph(recorder, 1, 100, ())
    tuner = Graph(keep_graph=True)
    tuner.capture_end()
    tuner.instantiate()
    bundle = PreparedGraphBundle(device=0, graphs={"decode": recorded, "tuner": tuner}, buffers={})
    assert bundle._validate_resources(export_graphs=True) == (100, [1, tuner.executable], [])
    with pytest.raises(TypeError, match="CUDAGraph"):
        bundle._validate_resources()
    recorded.context = 101
    with pytest.raises(ValueError, match="another CUDA context"):
        bundle._validate_resources(export_graphs=True)
    recorded.context = 100
    recorded.reset()
    with pytest.raises(ValueError, match="released"):
        bundle._validate_resources(export_graphs=True)


def test_export_preserves_collective_consumers_after_replacing_cuda_graph(monkeypatch, recorder):
    import flare_heliopause
    from ignition.preparation.startup_tuning import record_consumer

    registry = SimpleNamespace(captures={})
    session = SimpleNamespace(capture=nullcontext, _startup_collectives=registry)
    storage = SimpleNamespace(transient=nullcontext, pool=SimpleNamespace(id=(0, 1)),
                              graph_pool=SimpleNamespace(id=(0, 2)))
    stream = SimpleNamespace(wait_stream=lambda other: None, synchronize=lambda: None)
    collective = object()
    def forward():
        record_consumer(collective)
    @contextmanager
    def graph_scope(graph, **kwargs):
        yield
        graph.capture_end()
    monkeypatch.setattr(flare_heliopause, "current_context", lambda: (100, 0))
    monkeypatch.setattr(Graph, "pools", lambda self: [(0, 2)], raising=False)
    for name, value in dict(current_stream=lambda device: stream, stream=lambda stream: nullcontext(),
                            synchronize=lambda device: None, graph=graph_scope).items():
        monkeypatch.setattr(torch.cuda, name, value)
    graphs, _ = capture.capture_forwards({"decode": forward}, {"decode": SimpleNamespace(kind="decode")},
        session=session, device="cpu", stream=stream, pool=storage.graph_pool.id, compile_decoder=False,
        retain_graphs=True, warmup_storage=storage)
    graph = graphs["decode"]
    assert isinstance(graph, RecordedExportGraph)
    assert registry.captures == {id(graph): {id(collective)}}
