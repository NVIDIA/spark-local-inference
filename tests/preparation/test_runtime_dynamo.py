# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Dynamo guards retain invalidation and input-mutation ordering checks."""

from types import SimpleNamespace
import gc
import os
import weakref

import pytest
import torch
from torch._dynamo import exc
from torch._dynamo.output_graph import OutputGraph
from torch._dynamo.symbolic_convert import SpeculationLog

from ignition.runtime_patches import dynamo


class GuardedForward:
    def __init__(self):
        self.value = torch.ones(2)
        self.offset = 2

    def __call__(self):
        return self.value + self.offset


def _execute_guarded(forward):
    return forward()


def _load_guarded_forward():
    from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable

    forward = GuardedForward()
    compiled = torch.compile(_execute_guarded,
        backend=lambda graph, inputs: GraphModuleSerializableCallable(graph), fullgraph=True,
        dynamic=False).aot_compile(((forward,), {}))
    data = type(compiled).serialize(compiled).serialized_data
    return type(compiled).deserialize(data), forward


def _manager_ref(wrapper):
    from torch._dynamo.guards import CheckFunctionManager

    managers = [owner for owner in gc.get_referrers(wrapper) if isinstance(owner, CheckFunctionManager)]
    assert len(managers) == 1
    return weakref.ref(managers[0])


@pytest.fixture
def installed(monkeypatch):
    replacements = dynamo.replacements()
    for owner, name, replacement in replacements:
        monkeypatch.setattr(owner, name, replacement)
    yield replacements
    torch._dynamo.reset()


def test_discarded_loaded_forward_releases_guards_and_runtime_backend(installed):
    loaded, forward = _load_guarded_forward()
    torch.testing.assert_close(loaded(forward), torch.full((2,), 3.0))
    wrapper = loaded._artifacts.guard_manager
    backend = loaded._artifacts.compiled_fn
    wrapper_ref, backend_ref = weakref.ref(wrapper), weakref.ref(backend)
    manager_ref = _manager_ref(wrapper)
    assert any(value is backend for value in manager_ref().runtime_global_scope.values())
    del loaded, wrapper, backend
    gc.collect()
    assert wrapper_ref() is None
    assert manager_ref() is None
    assert backend_ref() is None
    assert type(forward) is GuardedForward


def test_live_guard_wrapper_retains_manager_and_runtime_backend(installed):
    loaded, forward = _load_guarded_forward()
    wrapper = loaded._artifacts.guard_manager
    wrapper_ref = weakref.ref(wrapper)
    backend_ref = weakref.ref(loaded._artifacts.compiled_fn)
    manager_ref = _manager_ref(wrapper)
    del loaded
    gc.collect()
    assert manager_ref() is not None
    assert backend_ref() is not None
    assert wrapper.check({"forward": forward})
    forward.offset = 7
    assert not wrapper.check({"forward": forward})
    del wrapper
    gc.collect()
    assert wrapper_ref() is None
    assert manager_ref() is None
    assert backend_ref() is None


def test_guarded_type_death_invalidates_real_dynamo_cache_entry(installed):
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    from torch._dynamo.guards import DeletedGuardManagerWrapper

    def execute(value, owner):
        return value + owner.offset

    def make_owner():
        class Owner:
            offset = 2

        return Owner(), weakref.ref(Owner)

    owner, type_ref = make_owner()
    compiled = torch.compile(execute, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(torch.ones(2), owner), torch.full((2,), 3.0))
    entry, = _debug_get_cache_entry_list(execute.__code__)
    wrapper = entry.guard_manager
    manager_ref = _manager_ref(wrapper)
    gc.collect()
    assert manager_ref() is not None
    assert wrapper.cache_entry is entry
    del owner
    gc.collect()
    assert type_ref() is None
    assert isinstance(entry.guard_manager, DeletedGuardManagerWrapper)
    assert "got deallocated" in entry.guard_manager.invalidation_reason
    assert wrapper.cache_entry is None
    assert wrapper.extra_state is None
    replacement, _ = make_owner()
    torch.testing.assert_close(compiled(torch.ones(2), replacement), torch.full((2,), 3.0))
    assert any(not isinstance(item.guard_manager, DeletedGuardManagerWrapper)
               for item in _debug_get_cache_entry_list(execute.__code__))


def test_preparing_dynamo_replacements_leaves_methods_unchanged():
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in dynamo.replacements()]
    dynamo.replacements()
    for owner, name, original in originals:
        assert getattr(owner, name) is original


def test_speculation_flag_belongs_to_one_trace_and_survives_retries(installed):
    first, second = SpeculationLog(), SpeculationLog()
    assert first.requires_input_mutation_stream_checks is False
    assert second.requires_input_mutation_stream_checks is False
    first.requires_input_mutation_stream_checks = True
    first.index = 3
    first.restart()
    assert first.index == 0
    assert first.requires_input_mutation_stream_checks is True
    first.entries.append(object())
    first.clear()
    assert first.entries == []
    assert first.requires_input_mutation_stream_checks is True
    assert second.requires_input_mutation_stream_checks is False
    assert SpeculationLog().requires_input_mutation_stream_checks is False


def test_mutation_scan_is_lazy_and_retains_stream_identity(installed):
    tensor = torch.ones(2)
    graph = torch.fx.Graph()
    placeholder = graph.placeholder('input')
    placeholder.meta['example_value'] = tensor
    log = SpeculationLog()
    stream = SimpleNamespace(index=4)
    state = SimpleNamespace(in_stream_context=lambda: True, cur_stream_id=lambda: stream.index)
    tx = SimpleNamespace(speculation_log=log, symbolic_stream_state=state)
    output = SimpleNamespace(
        root_tx=tx,
        root_tracer=SimpleNamespace(graph=graph, _input_versions_at_beginning=[tensor._version]),
        _last_checked_input_versions=None,
        _input_mutation_streams={},
        _EVENT_INPUT_MUTATION_FIX=OutputGraph._EVENT_INPUT_MUTATION_FIX,
    )
    tensor.add_(1)
    OutputGraph.check_input_mutation_on_current_stream(output, tx)
    assert output._last_checked_input_versions is None
    assert output._input_mutation_streams == {}
    with pytest.raises(exc.SpeculationRestartAnalysis) as restart:
        OutputGraph.check_event_record_after_input_mutation(output, 4)
    assert restart.value.restart_reason == 'event recording requires input-mutation stream checks'
    assert log.requires_input_mutation_stream_checks is True
    OutputGraph.check_input_mutation_on_current_stream(output, tx)
    assert set(output._input_mutation_streams) == {4}
    OutputGraph.check_event_record_after_input_mutation(output, 5)
    with pytest.raises(RuntimeError, match='previously mutated'):
        OutputGraph.check_event_record_after_input_mutation(output, 4)
    stream.index = 5
    tensor.add_(1)
    OutputGraph.check_input_mutation_on_current_stream(output, tx)
    assert set(output._input_mutation_streams) == {4, 5}


@pytest.fixture
def trace_state(monkeypatch, installed):
    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("CUDA stream tracing requires explicitly assigned devices")
    torch._dynamo.reset()
    monkeypatch.setattr(torch.accelerator, 'is_available', lambda: False)
    stream = torch.cuda.Stream(stream_id=0, device_index=0, device_type=1)
    other_stream = torch.cuda.Stream(stream_id=1, device_index=0, device_type=1)
    event = torch.cuda.Event()
    logs, scans = [], []
    initialize = SpeculationLog.__init__
    check_mutation = OutputGraph.check_input_mutation_on_current_stream

    def capture_log(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        logs.append(self)

    def capture_scan(self, tx):
        check_mutation(self, tx)
        scans.append((self, tx.speculation_log.requires_input_mutation_stream_checks))

    monkeypatch.setattr(SpeculationLog, '__init__', capture_log)
    monkeypatch.setattr(OutputGraph, 'check_input_mutation_on_current_stream', capture_scan)
    return SimpleNamespace(stream=stream, other_stream=other_stream, event=event, logs=logs, scans=scans)


def test_real_dynamo_trace_without_events_never_scans_inputs(trace_state):
    state = trace_state

    def fn(x):
        with torch.cuda.stream(state.stream):
            x.add_(1)
            x.mul_(2)
        return x.sin()

    result = torch._dynamo.export(fn)(torch.ones(2))
    assert any(node.target == 'add_' for node in result.graph_module.graph.nodes)
    assert state.scans
    assert all(not enabled for _, enabled in state.scans)
    assert all(output._last_checked_input_versions is None for output, _ in state.scans)
    assert state.logs and not any(log.requires_input_mutation_stream_checks for log in state.logs)


@pytest.mark.parametrize('record_on_stream', [False, True])
def test_real_dynamo_trace_restarts_and_rejects_event_after_mutation(trace_state, record_on_stream):
    state = trace_state

    def fn(x):
        with torch.cuda.stream(state.stream):
            x.add_(1)
            if record_on_stream:
                state.stream.record_event(state.event)
            else:
                state.event.record(state.stream)
        return x.sin()

    with pytest.raises(exc.InternalTorchDynamoError, match='previously mutated'):
        torch._dynamo.export(fn)(torch.ones(2))
    assert any(not enabled for _, enabled in state.scans)
    assert any(enabled for _, enabled in state.scans)
    assert sum(log.requires_input_mutation_stream_checks for log in state.logs) == 1


@pytest.mark.parametrize('record_on_stream', [False, True])
def test_real_dynamo_trace_accepts_event_before_mutation(trace_state, record_on_stream):
    state = trace_state

    def fn(x):
        with torch.cuda.stream(state.stream):
            if record_on_stream:
                state.stream.record_event(state.event)
            else:
                state.event.record(state.stream)
            x.add_(1)
        return x.sin()

    result = torch._dynamo.export(fn)(torch.ones(2))
    targets = [node.target for node in result.graph_module.graph.nodes]
    assert torch.ops.streams.record_event in targets
    assert 'add_' in targets
    assert any(log.requires_input_mutation_stream_checks for log in state.logs)
    assert any(enabled for _, enabled in state.scans)


def test_real_dynamo_trace_accepts_event_on_unmutated_stream(trace_state):
    state = trace_state

    def fn(x):
        with torch.cuda.stream(state.stream):
            x.add_(1)
            state.event.record(state.other_stream)
        return x.sin()

    result = torch._dynamo.export(fn)(torch.ones(2))
    assert any(node.target == torch.ops.streams.record_event for node in result.graph_module.graph.nodes)
    assert any(enabled for _, enabled in state.scans)


def test_real_dynamo_trace_keeps_checks_after_an_earlier_event(trace_state):
    state = trace_state

    def record():
        state.event.record(state.stream)

    def fn(x):
        with torch.cuda.stream(state.stream):
            record()
            x.add_(1)
            record()
        return x.sin()

    with pytest.raises(exc.InternalTorchDynamoError, match='previously mutated'):
        torch._dynamo.export(fn)(torch.ones(2))
    assert sum(log.requires_input_mutation_stream_checks for log in state.logs) == 1
    assert any(enabled for _, enabled in state.scans)
