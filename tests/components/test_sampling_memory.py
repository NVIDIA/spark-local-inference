# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Conditional sampling owners share request-scoped vocabulary constants."""

from dataclasses import fields

import pytest
import torch
from torch._subclasses import FakeTensor

from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.sampling import DeviceRequestState
from ignition.components.speculation import PreparedSpeculativeDistributions
from ignition.preparation.forward_metadata import dumps, loads
from ignition.preparation.tensor_metadata import NumericStorage


def _distributions(state):
    device, vocabulary = state.histogram.device, state.vocab_size
    operations = []
    for batch, width in ((1, 1), (2, 3), (4, 5)):
        slots = torch.arange(batch, dtype=torch.int32, device=device)
        generations = torch.zeros(batch, dtype=torch.int64, device=device)
        live = torch.tensor([batch], dtype=torch.int32, device=device)
        candidates = torch.zeros((batch, width - 1), dtype=torch.int64, device=device)
        operations.append(PreparedSpeculativeDistributions(
            torch.empty((batch, width, vocabulary), device=device), state, slots,
            generations, live, candidates))
        for prefix in range(width - 1):
            operations.append(PreparedDraftDistribution(
                torch.empty((batch, 1, vocabulary), device=device), state, slots,
                generations, live, candidates, prefix_offset=prefix,
                seed=torch.zeros(batch, dtype=torch.int64, device=device),
                offset=torch.zeros(batch, dtype=torch.int64, device=device),
                sampled=torch.zeros(batch, dtype=torch.int64, device=device)))
    return operations


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_target_and_draft_shapes_share_one_request_vocabulary(device):
    state = DeviceRequestState.allocate(device, vocab_size=257)
    operations = _distributions(state)
    indices = state.vocab_ids
    assert indices.shape == (257,)
    assert indices.dtype == torch.int64 and indices.device == state.histogram.device
    assert all(operation.vocab_ids is indices for operation in operations)
    assert "vocab_ids" not in {field.name for field in fields(DeviceRequestState)}
    if device == "cpu":
        torch.testing.assert_close(indices, torch.arange(257), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_independent_request_states_own_separate_vocabulary_constants(device):
    left = DeviceRequestState.allocate(device, vocab_size=257)
    right = DeviceRequestState.allocate(device, vocab_size=257)
    assert left.vocab_ids is not right.vocab_ids
    assert left.vocab_ids.untyped_storage()._cdata != right.vocab_ids.untyped_storage()._cdata


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_forward_metadata_preserves_request_vocabulary_aliases(device):
    state = DeviceRequestState.allocate(device, vocab_size=257)
    operations = _distributions(state)
    restored_state, restored_operations = loads(dumps((state, operations)))
    indices = restored_state.vocab_ids
    assert isinstance(indices, FakeTensor)
    assert (indices.shape, indices.dtype, indices.device) == (
        state.vocab_ids.shape, state.vocab_ids.dtype, state.vocab_ids.device)
    assert all(operation.state is restored_state for operation in restored_operations)
    assert all(operation.vocab_ids is indices for operation in restored_operations)


def test_cached_vocabulary_is_registered_in_each_numeric_storage_context():
    state = DeviceRequestState.allocate("cpu", vocab_size=257)
    indices = state.vocab_ids
    for _ in range(2):
        with NumericStorage() as numeric:
            operations = _distributions(state)
        assert all(operation.vocab_ids is indices for operation in operations)
        assert indices.untyped_storage()._cdata in numeric.entries
        assert any(start <= indices.data_ptr() and indices.data_ptr() + indices.nbytes <= end
                   for start, end in numeric.ranges())


def test_request_vocabulary_uses_one_persistent_destination(monkeypatch):
    import ignition.components.sampling as sampling
    from ignition.preparation.persistent import persistent_copy

    destinations = []
    def copy(source):
        output = persistent_copy(source)
        destinations.append(output)
        return output

    monkeypatch.setattr(sampling, "persistent_copy", copy)
    state = DeviceRequestState.allocate("cpu", vocab_size=257)
    actual = state.vocab_ids
    assert actual is state.vocab_ids and destinations == [actual]
    assert actual.dtype == torch.int64 and actual.device.type == "cpu" and actual.stride() == (1,)
    assert torch.equal(actual, torch.arange(257, dtype=torch.int64))


def _sampler_inputs():
    from ignition.components.completion import CompletionOutputs

    device = "cuda:0"
    return (torch.empty((2, 64), dtype=torch.float32, device=device),
        DeviceRequestState.allocate(device, vocab_size=64, capacity=2),
        torch.arange(2, dtype=torch.int32, device=device),
        torch.ones(2, dtype=torch.int64, device=device),
        torch.tensor([2], dtype=torch.int32, device=device),
        CompletionOutputs.allocate((2, 2, 24), device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.int64, device=device))


def test_default_sampling_mask_values_from_constructor_graph():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from ignition.components.sampling import PreparedSampling

    with FakeTensorMode():
        inputs = _sampler_inputs()
        graph = make_fx(lambda: PreparedSampling(*inputs).sample_enabled)()
    graph.graph.eliminate_dead_code()
    for node in graph.graph.nodes:
        if "device" in node.kwargs:
            node.kwargs = dict(node.kwargs, device=torch.device("cpu"))
    graph.recompile()
    mask = graph()
    assert mask.device.type == "cpu" and mask.dtype == torch.int32 and mask.stride() == (1,)
    assert torch.equal(mask, torch.ones(2, dtype=torch.int32))


def test_explicit_sampling_mask_retains_input_identity(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    import ignition.components.sampling as sampling

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("an explicit sampling mask requires no default destination")

    monkeypatch.setattr(sampling, "persistent_empty", unexpected_allocation)
    with FakeTensorMode():
        inputs = _sampler_inputs()
        mask = torch.zeros(2, dtype=torch.int32, device="cuda:0")
        operation = sampling.PreparedSampling(*inputs, sample_enabled=mask)
        assert operation.sample_enabled is mask and operation._complete_args[7] is mask
