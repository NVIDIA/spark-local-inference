# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent startup outputs remain separate from serving inputs and capacities."""

from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.execution import GraphSignature, PreparedExecution, TunedBinding, TuningGraph
from ignition.preparation.startup_tuning import capture_consumers, record_consumer


def execution(**changes):
    values = dict(device=0, max_active=4, graphs={"decode": object()},
        signatures={"decode": GraphSignature("decode", 1, 1, ("tokens",))},
        buffers={"frame_index": torch.zeros(1, dtype=torch.int32),
                 "submission_sequence": torch.zeros(1, dtype=torch.int64),
                 "completion_records": torch.zeros((2, 4, 32), dtype=torch.int64),
                 "tokens": torch.zeros(1, dtype=torch.int64),
                 "selector": torch.ones(1, dtype=torch.int32),
                 "status": torch.ones(1, dtype=torch.int32)},
        tuning_graphs={"choose": TuningGraph(object(), ("selector",), "status")},
        tuned_bindings={"decode": (TunedBinding("selector", "choose"),)})
    return PreparedExecution(**(values | changes))


def test_manifest_has_separate_tuners_without_host_uploads(monkeypatch):
    bundle = execution()
    spans = [(4096 + index * 4096, tensor.numel() * tensor.element_size())
             for index, tensor in enumerate(bundle.buffers.values())]
    monkeypatch.setattr(bundle, "_validate_resources", lambda: (9, [11, 12], spans))
    manifest = bundle.manifest()
    assert manifest["abi_version"] == 6
    assert set(bundle.signatures) == set(bundle.estimated_durations_ns) == {"decode"}
    assert set(bundle.graphs) == {"decode", "choose"}
    assert manifest["tuning_graphs"] == [dict(name="choose", handle=12,
                                              outputs=["selector"], status_buffer="status")]
    assert manifest["graphs"][0]["tuned_bindings"] == [dict(buffer="selector", tuning_graph="choose")]
    assert {binding["buffer_name"] for binding in manifest["graphs"][0]["input_bindings"]} == {
        "frame_index", "submission_sequence", "tokens"}


def test_invalid_startup_dependencies_and_uploads_are_rejected():
    with pytest.raises(ValueError, match="output of its tuning graph"):
        execution(tuned_bindings={"decode": (TunedBinding("tokens", "choose"),)})
    with pytest.raises(ValueError, match="host uploads"):
        execution(signatures={"decode": GraphSignature("decode", 1, 1, ("selector",))})
    with pytest.raises(ValueError, match="int32"):
        execution(tuning_graphs={"choose": TuningGraph(object(), ("selector",), "tokens")})
    with pytest.raises(ValueError, match="distinct"):
        execution(tuning_graphs={"decode": TuningGraph(object(), ("selector",), "status")})


def test_capture_dependencies_follow_invoked_consumers_and_graph_identity():
    registry = SimpleNamespace(captures={})
    session = SimpleNamespace(_startup_collectives=registry)
    first, second, unused = object(), object(), object()
    graph, other_graph = object(), object()
    record_consumer(unused)
    with capture_consumers(session, graph):
        record_consumer(first)
        record_consumer(first)
    with capture_consumers(session, other_graph):
        record_consumer(second)
    assert registry.captures == {id(graph): {id(first)}, id(other_graph): {id(second)}}


def test_registry_reuses_producer_until_its_consumers_change(monkeypatch):
    from b12x.comm import pcie
    from ignition.preparation.startup_tuning import CollectiveStartupTuning

    prepared = []
    def prepare(specs, *, cache_policy):
        assert cache_policy == "cold"
        producer = SimpleNamespace(specs=tuple(specs))
        prepared.append(producer)
        return producer
    def spec(plan, *, profiles):
        assert profiles == (("full", 1),)
        return plan
    monkeypatch.setattr(pcie, "joint_startup_spec", spec, raising=False)
    monkeypatch.setattr(pcie, "prepare_joint_startup_tuner", prepare, raising=False)
    registry = CollectiveStartupTuning(object())
    state = SimpleNamespace(runtime=object())
    first = SimpleNamespace(key=("latency", 256), selector=torch.zeros(1), state=state)
    registry.owners.append(SimpleNamespace(plan=first, runtime_profiles=(("full", 1),), runtime_cache_policy="cold"))
    registry.prepare()
    registry.prepare()
    assert len(prepared) == 1
    second = SimpleNamespace(key=first.key, selector=torch.zeros(1), state=state)
    registry.owners.append(SimpleNamespace(plan=second, runtime_profiles=(("full", 1),), runtime_cache_policy="cold"))
    registry.prepare()
    assert len(prepared) == 2
    assert registry.groups[(first.key, id(state.runtime))].specs == (first, second)
    registry.prepare()
    assert len(prepared) == 2

    separate = SimpleNamespace(key=first.key, selector=torch.zeros(1),
                               state=SimpleNamespace(runtime=object()))
    registry.owners.append(SimpleNamespace(plan=separate, runtime_profiles=(("full", 1),), runtime_cache_policy="cold"))
    registry.prepare()
    assert len(prepared) == 3
    assert len(registry.groups) == 2
    assert registry.groups[(separate.key, id(separate.state.runtime))].specs == (separate,)


def test_standalone_registration_prepares_each_declared_consumer(monkeypatch):
    from ignition.preparation.startup_tuning import CollectiveStartupTuning, register_collective

    snapshots = []
    monkeypatch.setattr(CollectiveStartupTuning, "prepare", lambda self: snapshots.append(tuple(self.owners)))
    session = SimpleNamespace()
    first, second = object(), object()
    register_collective(first, session)
    register_collective(second, session)
    assert snapshots == [(first,), (first, second)]


def test_catalog_retains_exportable_graph_and_selector_storage(monkeypatch):
    from contextlib import contextmanager, nullcontext
    from ignition.preparation.runtime_package import _graphs
    from ignition.preparation.startup_tuning import CollectiveStartupTuning

    events = []
    class Graph:
        def __init__(self, *, keep_graph):
            self.keep_graph = keep_graph
            self.instantiated = False
        def instantiate(self):
            self.instantiated = True
            events.append("instantiate")
        def raw_cuda_graph(self):
            assert self.keep_graph
            return 11
        def raw_cuda_graph_exec(self):
            assert self.instantiated
            return 12
    class Stream:
        def __init__(self, *, device):
            assert device == 2
        def wait_stream(self, stream):
            assert stream == "current-2"
            events.append("wait")
        def synchronize(self):
            events.append("drain")
    @contextmanager
    def device(ordinal):
        assert ordinal == 2
        events.append("device-enter")
        yield
        events.append("device-exit")
    @contextmanager
    def capture(graph, *, stream, pool):
        assert isinstance(stream, Stream)
        assert pool == (1, 7)
        events.append("capture-enter")
        yield
        events.append("capture-exit")
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda ordinal: f"current-{ordinal}")
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda ordinal: None)
    selector = torch.tensor([7], dtype=torch.uint32)
    owner = SimpleNamespace(plan=SimpleNamespace(prepared=SimpleNamespace(state=SimpleNamespace(selector=selector))))
    producer = SimpleNamespace(specs=(SimpleNamespace(selector=selector),),
        status=torch.ones(1, dtype=torch.int32), run=lambda: events.append("run"))
    registry = CollectiveStartupTuning(SimpleNamespace(device=SimpleNamespace(ordinal=2), capture=nullcontext,
        benchmark_storage=SimpleNamespace(graph_pool=SimpleNamespace(id=(1, 7)))))
    serving = object()
    registry.owners = [owner]
    registry.groups = {"key": producer}
    registry.owner_groups = {id(owner): "key"}
    registry.captures[id(serving)] = {id(owner)}
    buffers, tuners, bindings, resources = registry.catalog({"decode": serving})
    output = buffers[bindings["decode"][0].buffer]
    assert output.dtype == torch.int32 and output.data_ptr() == selector.data_ptr()
    output.fill_(5)
    assert selector.item() == 5
    assert resources == (producer,)
    assert _graphs({name: value.graph for name, value in tuners.items()}) == [
        dict(name="collective-tuning-0", graph=11, executable=12)]
    assert events == ["device-enter", "wait", "capture-enter", "run", "capture-exit",
                      "instantiate", "drain", "device-exit"]


def test_storage_compiled_collective_records_actual_consumer(monkeypatch):
    from ignition.components.collectives import _OWNERS, _all_reduce
    from ignition.preparation.aot import StorageBackend

    class Owner:
        pass
    owner = Owner()
    owner.plan = object()
    owner.runtime = SimpleNamespace(all_reduce=lambda source, *, out, plan: out.copy_(source))
    handle = 987654321
    monkeypatch.setitem(_OWNERS, handle, owner)
    source, output = torch.arange(8, dtype=torch.float32), torch.zeros(8)
    def forward(source, output):
        _all_reduce(source, output, handle)
        return output
    compiled = StorageBackend(forward, (source, output))
    registry = SimpleNamespace(captures={})
    session = SimpleNamespace(_startup_collectives=registry)
    graph = object()
    with capture_consumers(session, graph):
        compiled(source, output)
    torch.testing.assert_close(output, source)
    assert registry.captures[id(graph)] == {id(owner)}


def test_manifest_emits_explicit_empty_startup_catalog(monkeypatch):
    bundle = execution(tuning_graphs={}, tuned_bindings={})
    spans = [(4096 + index * 4096, tensor.numel() * tensor.element_size())
             for index, tensor in enumerate(bundle.buffers.values())]
    monkeypatch.setattr(bundle, "_validate_resources", lambda: (9, [11], spans))
    manifest = bundle.manifest()
    assert manifest["tuning_graphs"] == []
    assert manifest["graphs"][0]["tuned_bindings"] == []


def test_memory_report_deduplicates_shared_producer_allocations():
    from b12x.preparation import MemoryRequirements, PersistentMemory
    from ignition.preparation.startup_tuning import CollectiveStartupTuning

    shared = PersistentMemory("eviction", 4096, 4096)
    def producer(key):
        memory = MemoryRequirements(persistent=(shared, PersistentMemory(key, 128, 128)))
        return SimpleNamespace(memory_requirements=lambda: memory)
    registry = CollectiveStartupTuning(object())
    registry.groups = {"a": producer("a"), "b": producer("b")}
    assert registry.memory_report() == dict(startup_tuning_bytes=4352, startup_tuning_producers=2)
