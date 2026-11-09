# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Dynamo retains input-mutation ordering checks after event-triggered restarts."""

from types import SimpleNamespace
import os

import pytest
import torch
from torch._dynamo import exc
from torch._dynamo.output_graph import OutputGraph
from torch._dynamo.symbolic_convert import SpeculationLog

from ignition.runtime_patches import dynamo


@pytest.fixture
def installed(monkeypatch):
    replacements = dynamo.replacements()
    for owner, name, replacement in replacements:
        monkeypatch.setattr(owner, name, replacement)
    yield replacements
    torch._dynamo.reset()


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
