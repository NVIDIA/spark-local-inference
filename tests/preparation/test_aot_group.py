# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Grouped forwards retain ordinary guards and storage-alias checks."""

from types import SimpleNamespace
import gc
import pickle
import weakref

import pytest
import torch

from ignition.preparation.aot import (
    BindingMismatch, BoundForward, ForwardCache, StorageBackend,
    _BindingsChecked, _CHECK_BINDINGS, _GuardedForwardCall, _guard_code_cache, _load_inductor,
)


class Artifact(_GuardedForwardCall):
    def __init__(self):
        self.checked = []
        super().__init__(self.check_storage,
            SimpleNamespace(check=lambda args: args["forward"].guard,
                            check_verbose=lambda args: "incompatible owner"),
            object.__new__(StorageBackend))

    def check_storage(self, owner):
        if not _CHECK_BINDINGS.get():
            raise AssertionError("binding validation must not execute a forward")
        self.checked.append(owner)
        if not owner.storage_ok:
            raise BindingMismatch("compiled tensor aliases differ")
        raise _BindingsChecked


def owner(*, guard=True, storage_ok=True):
    return SimpleNamespace(guard=guard, storage_ok=storage_ok)


class GuardedForward:
    def __init__(self):
        self.value = torch.ones(2)
        self.offset = 1
        self.ready = True

    def __call__(self):
        return self.value + self.offset


def _execute_guarded(forward):
    return forward()


def _compiled_forward():
    forward = GuardedForward()
    artifact = torch.compile(_execute_guarded,
        backend=lambda graph, inputs: StorageBackend(graph, inputs, options={"compile_threads": 1}),
        fullgraph=True, dynamic=False).aot_compile(((forward,), {}))
    return artifact, forward


def _compiled_guard_state(tmp_path):
    from torch._dynamo import package

    artifact, _ = _compiled_forward()
    path = tmp_path / "forward.bin"
    artifact.save_compiled_function(str(path))
    with path.open("rb") as file:
        return package.load_guards_state(pickle.load(file)["guards_state"])


def _native_guard_diagnostics(wrapper):
    result = []
    pending = [wrapper.root]
    while pending:
        manager = pending.pop()
        result.extend(leaf.verbose_code_parts() for leaf in manager.get_leaf_guards())
        pending.extend(reversed(manager.get_child_managers()))
    result.extend(leaf.verbose_code_parts() for leaf in wrapper.root.get_epilogue_lambda_guards())
    return result


def test_restored_guards_share_exact_native_diagnostic_strings(tmp_path, monkeypatch):
    from torch._dynamo import guards, package

    state = _compiled_guard_state(tmp_path)
    for guard in state.output_graph._guards:
        guard.stack = None
        guard.user_stack = None
    arguments = (state, _execute_guarded.__code__, _execute_guarded.__globals__)
    original_parts = guards.get_verbose_code_parts
    formatted = set()

    def record_parts(*args, **kwargs):
        parts = original_parts(*args, **kwargs)
        formatted.update(parts)
        return parts

    with monkeypatch.context() as patch:
        patch.setattr(guards, "get_verbose_code_parts", record_parts)
        baseline = package.load_guard_manager(*arguments)
    with _guard_code_cache(tmp_path / "guards"):
        first = package.load_guard_manager(*arguments)
        assert guards.get_verbose_code_parts is original_parts
        second = package.load_guard_manager(*arguments)
        assert guards.get_verbose_code_parts is original_parts
    assert first.root is not second.root
    diagnostics = [_native_guard_diagnostics(manager) for manager in (baseline, first, second)]
    assert diagnostics[0] == diagnostics[1] == diagnostics[2]
    assert all(isinstance(parts, list) for lists in diagnostics for parts in lists)
    strings = [[text for parts in lists for text in parts if text in formatted]
               for lists in diagnostics]
    assert strings[0]
    assert all(left is right for left, right in zip(strings[1], strings[2]))

    for manager in (baseline, first, second):
        forward = GuardedForward()
        assert manager.check({"forward": forward})
        for attribute, value in (("offset", 7), ("value", torch.ones(3)),
                                 ("value", torch.ones(2, dtype=torch.float64))):
            changed = GuardedForward()
            setattr(changed, attribute, value)
            assert not manager.check({"forward": changed})
            actual = manager.check_verbose({"forward": changed})
            expected = baseline.check_verbose({"forward": changed})
            assert not actual.result
            assert actual.verbose_code_parts == expected.verbose_code_parts


def test_guard_diagnostic_formatter_restores_after_load_failure(tmp_path, monkeypatch):
    from torch._dynamo import guards, package

    state = _compiled_guard_state(tmp_path)
    original_parts = guards.get_verbose_code_parts
    formatted = []

    def fail_formatting(*args, **kwargs):
        formatted.append(original_parts(*args, **kwargs))
        raise RuntimeError("guard diagnostic formatter failed")

    monkeypatch.setattr(guards, "get_verbose_code_parts", fail_formatting)
    with _guard_code_cache(tmp_path / "guards"):
        with pytest.raises(RuntimeError, match="guard diagnostic formatter failed"):
            package.load_guard_manager(
                state, _execute_guarded.__code__, _execute_guarded.__globals__)
        assert formatted
        assert guards.get_verbose_code_parts is fail_formatting
    assert guards.get_verbose_code_parts is fail_formatting


def _metadata_owners(backend):
    owners = []
    wrapped = backend.artifact.inner_fn.compiled_fn
    while wrapped is not None and all(wrapped is not owner for owner in owners):
        owners.append(wrapped)
        wrapped = getattr(wrapped, "__wrapped__", None)
    graph = getattr(backend.artifact, "_flare_graph", None)
    if graph is not None and all(graph is not owner for owner in owners):
        owners.append(graph)
    return owners


def test_group_reuses_one_artifact_after_each_existing_guard_and_alias_check():
    artifact = Artifact()
    first, second = owner(), owner()
    template = BoundForward(artifact, first, cache_hit=False)
    cache = object.__new__(ForwardCache)
    bound = cache.bind_group(template, {"first": first, "second": second}, signature="same")
    assert tuple(bound) == ("first", "second")
    assert all(value.artifact is artifact and value.cache_hit is False for value in bound.values())
    assert artifact.checked == [first, second]


def test_group_rejects_storage_mismatch_on_every_rank_without_recompilation(monkeypatch):
    import torch.distributed as dist

    artifact = Artifact()
    first, bad = owner(), owner(storage_ok=False)
    template = BoundForward(artifact, first, cache_hit=False)
    coordinator = SimpleNamespace(ranks=(0, 1), group=object())
    def exchange(results, local, *, group):
        results[:] = [local, (local[0], local[1], {"bad": "peer guard failure"})]
    monkeypatch.setattr(dist, "all_gather_object", exchange)
    with pytest.raises(BindingMismatch, match="compiled tensor aliases differ"):
        object.__new__(ForwardCache).bind_group(template,
            {"first": first, "bad": bad}, signature="same", coordinator=coordinator)
    assert artifact.checked == [first, bad]


def test_group_rejects_different_rank_signatures(monkeypatch):
    import torch.distributed as dist

    artifact = Artifact()
    first = owner()
    template = BoundForward(artifact, first, cache_hit=False)
    coordinator = SimpleNamespace(ranks=(0, 1), group=object())
    monkeypatch.setattr(dist, "all_gather_object",
        lambda results, local, *, group: results.__setitem__(slice(None),
            [local, (("first",), "different", {})]))
    with pytest.raises(BindingMismatch, match="different grouped forward owners or signatures"):
        object.__new__(ForwardCache).bind_group(template,
            {"first": first}, signature="same", coordinator=coordinator)


def test_finalized_group_rejects_guard_mismatch_before_checking_storage():
    artifact = Artifact()
    first, bad = owner(), owner(guard=False)
    template = BoundForward(artifact, first, cache_hit=True)
    with pytest.raises(BindingMismatch, match="incompatible owner"):
        object.__new__(ForwardCache).bind_group(template,
            {"first": first, "bad": bad}, signature="same")
    assert artifact.checked == [first]


@pytest.mark.parametrize("restored", [False, True])
def test_finalization_releases_portable_owners_and_retains_live_checks(tmp_path, restored):
    class Callback:
        def check(self, arguments):
            return arguments["forward"].ready

    artifact, forward = _compiled_forward()
    backend = artifact._artifacts.compiled_fn
    portable_refs = ()
    execution_refs = ()
    if restored:
        backend.artifact, backend._serialized_artifact = _load_inductor(
            backend.artifact.serialize(), torch.device("cpu"))
        compiled = backend.artifact.inner_fn.compiled_fn
        portable = compiled.serialize()
        portable_refs = (weakref.ref(portable), weakref.ref(portable.compiled_fw.result),
                         weakref.ref(compiled.serialize_fn))
        execution_refs = (weakref.ref(compiled.serialize.__self__.compiled_fn),
                          weakref.ref(backend.artifact._flare_graph))
        torch.testing.assert_close(artifact(forward), torch.full((2,), 2.0))
        artifact.save_compiled_function(str(tmp_path / "forward.bin"))
        del compiled, portable
    metadata = _metadata_owners(backend)
    assert len(metadata) > 1
    assert all(value.source_code for value in metadata)
    bundle_refs = [weakref.ref(value._triton_bundle) for value in metadata
                   if value._triton_bundle is not None]
    assert bundle_refs
    structure = [tuple(id(getattr(value, field, None))
                       for field in ("__wrapped__", "compiled_fn", "current_callable", "__closure__"))
                 for value in metadata]
    del backend
    wrapper = artifact._artifacts.guard_manager
    callback = Callback()
    wrapper.root.add_lambda_guard(callback.check, ["forward.ready must be True"], None)
    wrapper.diff_guard_root = wrapper.root.clone_manager(lambda manager: True)
    wrapper.populate_code_parts_for_debugging()
    clone_ref = weakref.ref(wrapper.diff_guard_root)
    wrapper_ref, callback_ref = weakref.ref(wrapper), weakref.ref(callback)
    manager_ref = weakref.ref(wrapper._check_function_manager)
    backend_ref = weakref.ref(artifact._artifacts.compiled_fn)
    bound = BoundForward(artifact, forward, cache_hit=True)
    del artifact, wrapper, callback
    assert bound.prepare_capture() > 0
    gc.collect()
    assert clone_ref() is None
    assert wrapper_ref() is not None
    assert manager_ref() is not None
    assert not wrapper_ref().code_parts
    assert callback_ref() is not None
    assert all(reference() is None for reference in portable_refs)
    assert all(reference() is not None for reference in execution_refs)
    assert all(reference() is None for reference in bundle_refs)
    assert all(value.source_code == "" and value._triton_bundle is None
               and value.runnable_graph_str == ""
               and value.inductor_provenance_mapping_str is None
               and value.inductor_provenance_stack_traces_str is None for value in metadata)
    assert structure == [tuple(id(getattr(value, field, None))
                               for field in ("__wrapped__", "compiled_fn", "current_callable", "__closure__"))
                         for value in metadata]
    del metadata
    assert bound.prepare_capture() == 0
    torch.testing.assert_close(bound(), torch.full((2,), 2.0))
    forward.ready = False
    with pytest.raises(BindingMismatch, match="forward.ready must be True"):
        bound()
    del bound
    gc.collect()
    assert wrapper_ref() is None
    assert manager_ref() is None
    assert backend_ref() is None
    assert callback_ref() is None
    assert all(reference() is None for reference in execution_refs)
    assert type(forward) is GuardedForward


@pytest.mark.parametrize("restored", [False, True])
def test_group_retains_guard_manager_until_last_bound_owner_drops(restored):
    artifact, first = _compiled_forward()
    if restored:
        backend = artifact._artifacts.compiled_fn
        backend.artifact, backend._serialized_artifact = _load_inductor(
            backend.artifact.serialize(), torch.device("cpu"))
        del backend
    second = GuardedForward()
    wrapper = artifact._artifacts.guard_manager
    wrapper_ref = weakref.ref(wrapper)
    manager_ref = weakref.ref(wrapper._check_function_manager)
    backend_ref = weakref.ref(artifact._artifacts.compiled_fn)
    metadata = _metadata_owners(artifact._artifacts.compiled_fn)
    assert all(value.source_code for value in metadata)
    bundle_refs = [weakref.ref(value._triton_bundle) for value in metadata
                   if value._triton_bundle is not None]
    assert bundle_refs
    template = BoundForward(artifact, first, cache_hit=True)
    bound = object.__new__(ForwardCache).bind_group(template,
        {"first": first, "second": second}, signature="same")
    assert all(value.source_code == "" and value._triton_bundle is None for value in metadata)
    del metadata
    retained = bound.pop("second")
    del bound, artifact, template, wrapper
    gc.collect()
    assert wrapper_ref() is not None
    assert manager_ref() is not None
    assert backend_ref() is not None
    assert all(reference() is None for reference in bundle_refs)
    torch.testing.assert_close(retained(), torch.full((2,), 2.0))
    second.offset = 7
    with pytest.raises(BindingMismatch):
        retained()
    del retained
    gc.collect()
    assert wrapper_ref() is None
    assert manager_ref() is None
    assert backend_ref() is None
    assert type(first) is type(second) is GuardedForward
