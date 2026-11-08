# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Grouped forwards retain ordinary guards and storage-alias checks."""

from types import SimpleNamespace
import gc
from inspect import signature
import weakref

import pytest

from ignition.preparation.aot import (
    BindingMismatch, BoundForward, ForwardCache, StorageBackend,
    _BindingsChecked, _CHECK_BINDINGS, _GuardedForwardCall,
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


def test_finalization_releases_duplicate_guards_despite_type_finalizer_retention():
    from torch._dynamo.guards import CheckFunctionManager, GuardManagerWrapper

    class GuardedType:
        pass

    class Callback:
        def check(self, arguments):
            return arguments["forward"].ready

    callback = Callback()
    wrapper = GuardManagerWrapper()
    wrapper.root.add_lambda_guard(callback.check, ["forward.ready must be True"], None)
    wrapper.finalize()
    wrapper.diff_guard_root = wrapper.root.clone_manager(lambda manager: True)
    wrapper.populate_code_parts_for_debugging()
    manager = object.__new__(CheckFunctionManager)
    manager.guard_manager = wrapper
    manager._weakrefs = {}
    manager.id_ref(GuardedType, "GuardedType")
    manager._weakrefs.clear()
    clone_ref = weakref.ref(wrapper.diff_guard_root)
    wrapper_ref, callback_ref = weakref.ref(wrapper), weakref.ref(callback)
    backend = object.__new__(StorageBackend)
    backend._serialized_artifact = b"portable backend"
    artifact = SimpleNamespace(fn=lambda forward: forward,
        _artifacts=SimpleNamespace(signature=signature(lambda forward: None),
            runtime_env=SimpleNamespace(closure=()), guard_manager=wrapper,
            compiled_fn=backend, guards_state=b"portable guards"))
    forward = SimpleNamespace(ready=True)
    bound = BoundForward(artifact, forward, cache_hit=True)
    del artifact, wrapper, callback, manager
    assert bound.prepare_capture() > 0
    gc.collect()
    assert clone_ref() is None
    assert wrapper_ref() is not None
    assert not wrapper_ref().code_parts
    assert callback_ref() is not None
    assert bound() is forward
    forward.ready = False
    with pytest.raises(BindingMismatch, match="forward.ready must be True"):
        bound()
    del bound
    gc.collect()
    assert wrapper_ref() is not None
    assert callback_ref() is not None
    del GuardedType
    gc.collect()
    assert wrapper_ref() is None
    assert callback_ref() is None
