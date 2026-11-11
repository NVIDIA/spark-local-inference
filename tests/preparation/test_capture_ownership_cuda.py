# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured forwards release Python guards and retain live kernel resources."""

from contextlib import contextmanager, nullcontext
import gc
import os
from types import SimpleNamespace
import weakref

import pytest
import torch

from ignition.components.linear import UnquantizedProjection
from ignition.preparation.aot import BindingMismatch, BoundForward, ForwardCache, aot_forward
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


class Forward:
    def __init__(self, source, output, operation=None):
        self.source, self.output, self.operation = source, output, operation
        self.offset = True

    def __call__(self):
        value = self.source.square() if self.operation is None else self.operation(self.source)
        self.output.copy_(value + (1 if self.offset else 0))


@pytest.mark.parametrize("compile_decoder", [False, True])
@pytest.mark.parametrize("autotune_effort", ["off", "default"])
@torch.no_grad()
def test_forward_warmup_captures_unprimed_plans_and_cached_restart(tmp_path, compile_decoder, autotune_effort):
    device = torch.device("cuda:0")
    source = (torch.randn((4, 128), device=device) * .1).bfloat16()
    weight = (torch.randn((128, 128), device=device) * .1).bfloat16()
    projection_output = torch.empty_like(source)
    output = torch.empty_like(source)
    signature = GraphSignature(kind="decode", batch_capacity=4, token_capacity=1)
    for restart in (False, True):
        projection_output.fill_(float("nan"))
        with RecordedPreparationSession(device=device, checkpoint="unprimed-capture",
                workload_schema="projection", numerical_policy="bf16", compile_workers=1,
                autotune_effort=autotune_effort) as session:
            session.cache_dir = tmp_path / "kernels"
            assert session.prime is False
            operation = UnquantizedProjection(source, weight, projection_output,
                session=session, name="projection")
            session.tune(show_progress=False)
            assert projection_output.isnan().all()
            forward = Forward(source, output, operation)
            graphs, compiled = capture_forwards({"decode": forward}, {"decode": signature},
                session=session, device=device, stream=torch.cuda.Stream(device=device),
                pool=torch.cuda.graph_pool_handle(), compile_decoder=compile_decoder,
                cache_root=tmp_path / "forwards")
            try:
                if compile_decoder:
                    assert compiled["decode"].cache_hit is restart
                session.freeze()
                for value in (.125, -.25):
                    source.fill_(value)
                    before = torch.cuda.memory_stats(device)
                    graphs["decode"].replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    for key in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key]
                    expected = (source.float() @ weight.float().T).bfloat16() + 1
                    torch.testing.assert_close(output, expected, atol=.01, rtol=.01)
            finally:
                graphs["decode"].reset()
                compiled.clear()


@pytest.mark.parametrize("opaque", [False, True])
@torch.no_grad()
def test_capture_releases_guards_and_serialized_artifact_and_preserves_replay(tmp_path, opaque):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from ignition.preparation.inductor_tuning import _kernel_globals

    device = torch.device("cuda:0")
    source = (torch.randn((4, 128), device=device) * .1).bfloat16()
    output = torch.empty_like(source)
    scope = (RecordedPreparationSession(device=device, checkpoint="capture-ownership",
        workload_schema="projection", numerical_policy="bf16", compile_workers=1) if opaque else nullcontext())
    with scope as session:
        weight = (torch.randn((128, 128), device=device) * .1).bfloat16() if opaque else None
        operation = None
        if opaque:
            operation = UnquantizedProjection(source, weight, torch.empty_like(source),
                session=session, name="projection")
            session.tune(show_progress=False)
        forward = Forward(source, output, operation)
        artifact = aot_forward(forward)
        artifact(forward)
        path = tmp_path / "compiled.bin"
        artifact.save_compiled_function(str(path))
        del artifact
        with path.open("rb") as file:
            artifact = torch.compiler.load_compiled_function(file)
        bound = BoundForward(artifact, forward, True)
        bound()
        compiled = bound.backend.artifact.inner_fn.compiled_fn
        execution = compiled.__wrapped__.compiled_fn
        graph_state = bound.backend.artifact._flare_graph
        current_callable = graph_state.current_callable
        module = _kernel_globals(graph_state)
        tuners = {name: value for name, value in module.items() if isinstance(value, CachingAutotuner)}
        bundle_ref = weakref.ref(graph_state._triton_bundle)
        assert graph_state.source_code
        reference = weakref.ref(artifact)
        del artifact
        assert bound.prepare_capture() > 0
        assert bound.prepare_capture() == 0
        gc.collect()
        assert reference() is None
        assert bound.artifact.backend._serialized_artifact is None
        assert bound.backend.artifact.inner_fn.compiled_fn is compiled
        assert compiled.__wrapped__.compiled_fn is execution
        assert graph_state.current_callable is current_callable
        assert graph_state.source_code == "" and graph_state._triton_bundle is None
        assert bundle_ref() is None
        assert _kernel_globals(graph_state) is module
        assert all(module[name] is tuner for name, tuner in tuners.items())
        forward.offset = False
        with pytest.raises(BindingMismatch, match="compiled guards"):
            bound()
        forward.offset = True
        forward.source = source[:, :64]
        with pytest.raises(BindingMismatch, match="compiled guards"):
            bound()
        forward.source = source
        guards_ref = weakref.ref(bound.artifact.guards)
        manager_ref = weakref.ref(bound.artifact.guards._check_function_manager)
        backend = bound.backend
        bindings = backend._bound
        invocations = backend.invocations
        graph = torch.cuda.CUDAGraph()
        try:
            bound()
            with (session.capture() if opaque else nullcontext()), torch.cuda.graph(graph):
                bound()
            bound.retire_after_capture()
            gc.collect()
            assert guards_ref() is manager_ref() is None
            assert bound.backend is backend
            assert backend._bound is bindings and backend.invocations is invocations
            assert backend.artifact.inner_fn.compiled_fn is compiled
            assert compiled.__wrapped__.compiled_fn is execution
            assert graph_state.current_callable is current_callable
            assert _kernel_globals(graph_state) is module
            assert all(module[name] is tuner for name, tuner in tuners.items())
            del backend, bindings, invocations, compiled, execution, graph_state, current_callable, module, tuners
            gc.collect()
            with pytest.raises(RuntimeError, match="no longer supports Python execution"):
                bound()
            with pytest.raises(RuntimeError, match="no longer supports Python execution"):
                bound.prepare_capture()
            for value in (.125, -.25, .5):
                source.fill_(value)
                before = torch.cuda.memory_stats(device)
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                expected = ((source.float() @ weight.float().T).bfloat16() if opaque else source.square()) + 1
                torch.testing.assert_close(output, expected, atol=.01 if opaque else 0, rtol=.01 if opaque else 0)
        finally:
            graph.reset()


@pytest.mark.parametrize("retain_forwards", [False, True])
@torch.no_grad()
def test_grouped_capture_retires_guards_after_last_owner_and_preserves_eager_diagnostics(
        tmp_path, monkeypatch, retain_forwards):
    from ignition.preparation import aot

    device = torch.device("cuda:0")
    functions = {name: Forward(torch.randn(128, device=device), torch.empty(128, device=device))
                 for name in ("first", "second")}
    signature = GraphSignature("decode", 1, 1)
    guard_refs, manager_refs, artifact_refs, warmed, retired = [], [], [], [], []
    prepare_many = ForwardCache.prepare_many
    bind_group = ForwardCache.bind_group
    retire = BoundForward.retire_after_capture
    release = aot.release_unused

    def prepare(self, *args, **kwargs):
        result = prepare_many(self, *args, **kwargs)
        bound = next(iter(result.values()))
        guard_refs.append(weakref.ref(bound.artifact.guards))
        manager_refs.append(weakref.ref(bound.artifact.guards._check_function_manager))
        return result

    def bind(self, *args, **kwargs):
        result = bind_group(self, *args, **kwargs)
        artifact_refs.append(weakref.ref(result["first"].artifact))
        return result

    @contextmanager
    def scope():
        assert warmed == ["first", "second"]
        assert guard_refs[0]() is not None and manager_refs[0]() is not None
        yield

    def collect(**kwargs):
        release(**kwargs)
        if retired == ["first", "second"]:
            assert guard_refs[0]() is manager_refs[0]() is artifact_refs[0]() is None

    def retire_owner(self):
        name = "first" if not retired else "second"
        assert self.owner is functions[name]
        retired.append(name)
        retire(self)
        assert (guard_refs[0]() is not None) is (len(retired) == 1)

    monkeypatch.setattr(ForwardCache, "prepare_many", prepare)
    monkeypatch.setattr(ForwardCache, "bind_group", bind)
    monkeypatch.setattr(BoundForward, "retire_after_capture", retire_owner)
    monkeypatch.setattr(aot, "release_unused", collect)
    session = SimpleNamespace(capture=scope, namespace=SimpleNamespace(to_dict=lambda: {}),
                              requested_compile_workers=1)
    def restore(name):
        if artifact_refs and len(warmed) < 2:
            warmed.append(name)

    graphs, compiled = capture_forwards(functions, dict.fromkeys(functions, signature),
        session=session, device=device, stream=torch.cuda.Stream(device=device),
        pool=torch.cuda.graph_pool_handle(), cache_root=tmp_path, compile_group="elementwise",
        before_run=restore, retain_graphs=True, retain_forwards=retain_forwards)
    try:
        assert retired == ([] if retain_forwards else ["first", "second"])
        assert compiled["first"].backend is compiled["second"].backend
        assert (guard_refs[0]() is not None) is retain_forwards
        for name, forward in functions.items():
            forward.source.fill_(.5)
            forward.output.fill_(float("nan"))
            before = torch.cuda.memory_stats(device)
            graphs[name].replay()
            torch.cuda.synchronize(device)
            after = torch.cuda.memory_stats(device)
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            torch.testing.assert_close(forward.output, forward.source.square() + 1, rtol=0, atol=0)
            if retain_forwards:
                forward.output.zero_()
                compiled[name]()
                torch.testing.assert_close(forward.output, forward.source.square() + 1, rtol=0, atol=0)
                forward.offset = False
                with pytest.raises(BindingMismatch):
                    compiled[name]()
                forward.offset = True
            else:
                with pytest.raises(RuntimeError, match="no longer supports Python execution"):
                    compiled[name]()
    finally:
        for graph in graphs.values():
            graph.reset()


@pytest.mark.parametrize("failure", ["capture", "instantiate"])
@torch.no_grad()
def test_capture_failure_preserves_guarded_python_execution(tmp_path, monkeypatch, failure):
    from ignition.preparation import capture

    device = torch.device("cuda:0")
    forward = Forward(torch.randn(128, device=device), torch.empty(128, device=device))
    bound_forwards, graphs = [], []
    prepare_many = ForwardCache.prepare_many
    graph_type = torch.cuda.CUDAGraph

    def prepare(self, *args, **kwargs):
        result = prepare_many(self, *args, **kwargs)
        bound_forwards.extend(result.values())
        return result

    @contextmanager
    def scope():
        yield
        if failure == "capture":
            raise RuntimeError("capture scope failed")

    class Graph(graph_type):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.fail_instantiation = failure == "instantiate" and kwargs.get("keep_graph", False)
            graphs.append(self)

        def instantiate(self):
            if self.fail_instantiation:
                raise RuntimeError("graph instantiation failed")
            return super().instantiate()

    monkeypatch.setattr(ForwardCache, "prepare_many", prepare)
    monkeypatch.setattr(capture.torch.cuda, "CUDAGraph", Graph)
    session = SimpleNamespace(capture=scope, namespace=SimpleNamespace(to_dict=lambda: {}),
                              requested_compile_workers=1)
    try:
        with pytest.raises(RuntimeError, match="capture scope failed|graph instantiation failed"):
            capture_forwards({"decode": forward}, {"decode": GraphSignature("decode", 1, 1)},
                session=session, device=device, stream=torch.cuda.Stream(device=device),
                pool=torch.cuda.graph_pool_handle(), cache_root=tmp_path, retain_graphs=True)
        assert len(bound_forwards) == 1
        gc.collect()
        forward.source.fill_(.5)
        bound_forwards[0]()
        torch.testing.assert_close(forward.output, forward.source.square() + 1, rtol=0, atol=0)
        forward.offset = False
        with pytest.raises(BindingMismatch):
            bound_forwards[0]()
    finally:
        for captured in graphs:
            captured.reset()
