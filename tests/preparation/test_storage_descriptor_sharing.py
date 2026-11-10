# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Loaded forwards share immutable descriptors and preserve storage behavior."""

import pickle
from types import SimpleNamespace

import pytest
import torch
from torch.utils._pytree import tree_flatten, treespec_dumps

from ignition.preparation import aot
from ignition.preparation.aot import (
    BindingMismatch, StorageBackend, _Input, _share_storage_descriptors, _signature,
)
from ignition.preparation.compilation import _Invocation


def _view(offset=0):
    return torch.float32, tuple([2, 3]), tuple([4, 1]), offset


def _call(view, *, mutable=True):
    leaves, tree = tree_flatten(((None, None), {"alpha": 2}))
    operands = [(0, mutable, 0, view), (1, False, 0, _view())]
    return ("op", "aten", "add_", "Tensor", None), leaves, treespec_dumps(tree), operands, []


def test_sharing_preserves_mutable_descriptors_and_separates_artifacts():
    inputs = [_Input(int("1024"), _view(), tuple([0, 128])),
              _Input(int("1024"), _view(), tuple([0, 128]))]
    signature = ((tuple([2, 3]), tuple([4, 1]), 0, torch.float32, torch.device("cpu"), 0),) * 2
    calls = [_call(_view()), _call(_view())]
    owner = SimpleNamespace(resource="kernel owner")
    calls[0][1][0] = owner
    mutable = [inputs, *inputs, calls, *(call[1] for call in calls),
               *(call[3] for call in calls), *(call[4] for call in calls)]
    identities = list(map(id, mutable))
    expected = pickle.loads(pickle.dumps((inputs, signature, calls)))

    shared = _share_storage_descriptors(inputs, signature, calls)

    assert (inputs, shared, calls) == expected
    retained = [inputs, *inputs, calls, *(call[1] for call in calls),
                *(call[3] for call in calls), *(call[4] for call in calls)]
    assert list(map(id, retained)) == identities
    assert calls[0][1][0] is owner
    assert inputs[0] is not inputs[1]
    assert inputs[0].view is inputs[1].view
    assert inputs[0].region is inputs[1].region
    assert inputs[0].index is inputs[1].index
    assert shared[0] is shared[1]
    assert inputs[0].view[1] is shared[0][0]
    assert calls[0][3][0] is calls[1][3][0]
    assert calls[0][3][0][3] is inputs[0].view
    assert calls[0][3] is not calls[1][3]
    calls[0][3].clear()
    assert len(calls[1][3]) == 2
    inputs[0].view = None
    assert inputs[1].view is not None

    other = [_Input(0, _view()), _Input(0, _view())]
    _share_storage_descriptors(other, (), [])
    assert other[0].view is other[1].view
    assert other[0].view is not inputs[1].view


def test_boolean_and_integer_values_keep_distinct_descriptor_types():
    views = [_view(True), _view(1), (torch.float32, (True, 3), (4, 1), 1), _view(1)]
    inputs = [_Input(index, view) for index, view in enumerate(views)]
    calls = [_call(_view(), mutable=True), _call(_view(), mutable=1)]

    _share_storage_descriptors(inputs, (), calls)

    assert type(inputs[0].view[3]) is bool
    assert type(inputs[1].view[3]) is int
    assert inputs[0].view is not inputs[1].view
    assert type(inputs[2].view[1][0]) is bool
    assert inputs[2].view is not inputs[3].view
    assert type(calls[0][3][0][1]) is bool
    assert type(calls[1][3][0][1]) is int
    assert calls[0][3][0] is not calls[1][3][0]


def test_descriptors_with_mutable_values_keep_their_objects():
    dimensions = [2, 3]
    view = torch.float32, dimensions, (4, 1), 0
    inputs = [_Input(0, view)]
    _share_storage_descriptors(inputs, (), [])
    assert inputs[0].view is view
    assert inputs[0].view[1] is dimensions


def test_shared_input_views_preserve_binding_and_alias_validation():
    storage = torch.arange(24, dtype=torch.float32)
    source, alias = storage[:12], storage[4:16]
    inputs = [_Input(0, _view()), _Input(1, _view())]
    signature = _signature((source, alias))
    expected = [item.bind((source, alias)) for item in inputs]
    signature = _share_storage_descriptors(inputs, signature, [])
    backend = object.__new__(StorageBackend)
    backend.inputs, backend.signature, backend.invocations, backend._bound = inputs, signature, [], None
    backend.artifact = lambda *values: values

    actual = backend(source, alias)
    for wanted, value in zip(expected, actual, strict=True):
        torch.testing.assert_close(value, wanted, atol=0, rtol=0)
        assert value.data_ptr() == wanted.data_ptr()
        assert value.stride() == wanted.stride()
    actual[0].fill_(-3)
    torch.testing.assert_close(source.as_strided((2, 3), (4, 1)), actual[0], atol=0, rtol=0)
    with pytest.raises(BindingMismatch, match="storage aliases changed"):
        backend(source, alias.clone())
    with pytest.raises(BindingMismatch, match="storage aliases changed"):
        backend(source.reshape(3, 4), alias)


def test_shared_invocation_operands_preserve_replay_values_and_mutation():
    output, source = torch.arange(12, dtype=torch.float32), torch.ones(12)
    call = _call(_view())
    _, leaves, tree, operands, positions = call
    spec = aot.treespec_loads(tree)
    invocation = _Invocation(torch.ops.aten.add_.Tensor, leaves, spec, operands, positions)
    original = pickle.loads(pickle.dumps(operands))
    _share_storage_descriptors([], (), [call])
    assert invocation.operands is operands and invocation.leaves is leaves and invocation.tree is spec

    for value in (1., 3., -2.):
        source.fill_(value)
        expected = output.clone()
        expected.as_strided((2, 3), (4, 1)).add_(source.as_strided((2, 3), (4, 1)), alpha=2)
        invocation.run([source], [output])
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert operands == original


def test_deserialization_shares_descriptors_and_preserves_serialized_schema(monkeypatch):
    calls = [_call(_view()), _call(_view())]
    inputs = [_Input(0, _view()), _Input(1, _view())]
    row = tuple([2, 3]), tuple([4, 1]), 0, torch.float32, torch.device("cuda:0"), 0
    signature = (row, tuple(list(row)))
    payload = b"opaque Inductor artifact"
    wire = payload, inputs, calls, signature, True
    owner = SimpleNamespace(serialize=lambda: payload)
    loads = []

    def load(data, device, *, frozen):
        loads.append((data, device, frozen))
        return owner, None

    monkeypatch.setattr(aot, "_load_inductor", load)
    backend = StorageBackend.deserialize_compile_artifacts(pickle.dumps(wire))
    assert loads == [(payload, torch.device("cuda:0"), True)]
    assert backend.artifact is owner and backend._bound is None
    assert backend.inputs[0].view is backend.inputs[1].view
    assert backend.invocations[0].operands[0] is backend.invocations[1].operands[0]
    assert backend.invocations[0].leaves is not backend.invocations[1].leaves
    assert backend.invocations[0].target is torch.ops.aten.add_.Tensor
    assert pickle.loads(StorageBackend.serialize_compile_artifacts(backend)) == wire
