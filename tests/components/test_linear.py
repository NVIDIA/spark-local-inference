# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared projection tuning fixtures and scratch finalization."""

from dataclasses import dataclass
from types import SimpleNamespace
import gc
import weakref

import pytest
import torch

from ignition.components import linear
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.fixtures import bf16_gemv, blockscaled_gemm
from ignition.preparation.fixtures.projection import input_factory


@dataclass
class Query:
    workspace_form: str = "provided"
    workspace_nbytes: int | None = None


def projection_fixture(monkeypatch):
    from b12x.gemm.blockscaled import _a16
    from b12x.gemm.blockscaled import _linear

    requests = []
    calls = []
    values = torch.arange(30, dtype=torch.float32).view(5, 6).to(torch.bfloat16) / 32
    weight = SimpleNamespace(out_features=values.shape[0])

    def check_fixture_tensor(name, tensor):
        assert isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu", name

    monkeypatch.setattr(_linear, "_check_gpu_tensor", check_fixture_tensor)
    monkeypatch.setattr(_a16, "_weight_parts", lambda value: (values, None, None, False))
    monkeypatch.setattr(linear.blockscaled, "query_from_call", lambda *args, **kwargs: Query())
    plan = SimpleNamespace(request=lambda **kwargs: SimpleNamespace(**kwargs),
                           memory_requirements=lambda: SimpleNamespace(scratch_nbytes=12))
    monkeypatch.setattr(linear.blockscaled, "plan", lambda query: plan)

    def run(source, values, scales, gain, *, out, workspace, activation_scale=None):
        calls.append((source, out, workspace))
        out.copy_(source @ values.T)
        return out

    state = SimpleNamespace(required_workspace=12, run=run)

    def prepare(batch):
        requests.extend(batch)
        for request in batch:
            request.prepare_call(state).invoke()

    session = SimpleNamespace(prepare=prepare)
    return session, weight, values, state, requests, calls


@pytest.mark.parametrize("storage", ["arena", "tensor", "none"])
def test_dense_benchmark_uses_finite_private_strided_inputs_and_planned_workspace(monkeypatch, storage):
    session, weight, values, state, requests, calls = projection_fixture(monkeypatch)
    source = torch.zeros((4, 12), dtype=torch.bfloat16)[:, ::2]
    output = torch.empty((4, 10), dtype=torch.bfloat16)[:, :5]
    bias = torch.arange(5, dtype=torch.bfloat16)
    workspace = (ScratchArena("cpu") if storage == "arena" else
                 torch.empty(64, dtype=torch.uint8) if storage == "tensor" else None)
    linear.QuantizedProjection(source, weight, output, session=session, name="projection",
                               workspace=workspace, bias=bias, activation_mode="a16")
    trial = requests[0].benchmark_call(state)
    operands, owned_workspace, owned_weight = trial.owners
    seed, owned_source, owned_output = operands.seed, operands.source, operands.output
    assert owned_weight is weight
    expected_seed = torch.randn(source.shape, dtype=source.dtype,
        generator=torch.Generator(device=source.device).manual_seed(0xB12)).mul_(0.125)
    assert torch.equal(seed, expected_seed)
    assert trial.output is owned_output
    assert trial.reset is None and trial.restore is None
    trial.produce()
    actual = trial.invoke()
    trial_source, trial_output, trial_workspace = calls[-1]
    assert trial_source is owned_source and trial_output is owned_output
    assert trial_workspace is owned_workspace
    assert trial_source.stride() == source.stride()
    assert trial_output.stride() == output.stride()
    assert trial_source.data_ptr() != source.data_ptr()
    if isinstance(workspace, ScratchArena):
        assert trial_workspace is workspace.tensor and workspace.nbytes == 256
    elif workspace is not None:
        assert trial_workspace is workspace
    assert trial_workspace is not None
    assert trial_workspace.nbytes >= state.required_workspace
    assert torch.isfinite(trial_source).all() and trial_source.count_nonzero() > 0
    torch.testing.assert_close(actual, trial_source @ values.T + bias, rtol=0, atol=0)
    assert source.count_nonzero() == 0 and output.count_nonzero() == 0
    saved = trial_source.clone()
    trial_source.zero_()
    trial.produce()
    torch.testing.assert_close(trial_source, saved, rtol=0, atol=0)


def test_selected_prime_and_finalizer_resolve_grown_arena(monkeypatch):
    session, weight, values, state, requests, calls = projection_fixture(monkeypatch)
    arena = ScratchArena("cpu")
    source = torch.zeros((4, 6), dtype=torch.bfloat16)
    output = torch.empty((4, 5), dtype=torch.bfloat16)
    projection = linear.QuantizedProjection(source, weight, output, session=session,
                                            name="projection", workspace=arena)
    original = arena.tensor
    assert calls[-1][2] is original and projection.workspace is original
    grown = arena.reserve(512)
    assert grown is not original
    requests[0].prepare_call(state).invoke()
    arena.finalize()
    arena.freeze()
    assert calls[-1][2] is grown and projection.workspace is grown
    too_large = SimpleNamespace(required_workspace=513, run=state.run)
    with pytest.raises(RuntimeError, match="frozen scratch arena"):
        requests[0].benchmark_call(too_large)


@pytest.mark.parametrize("family", ["bf16", "blockscaled"])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_projection_candidates_share_restored_operands_until_last_call_releases(family, output_dtype):
    source = torch.full((4, 12), 7, dtype=torch.bfloat16)[:, ::2]
    output = torch.full((4, 10), 13, dtype=output_dtype)[:, :5]
    weight = torch.arange(30, dtype=torch.bfloat16).view(5, 6) / 32
    bias = torch.arange(5, dtype=output_dtype)
    acquire = input_factory(source, output)
    workspaces = {}

    def candidate(multiplier):
        def run(values, matrix, *unused, out, bias=None, workspace=None, **kwargs):
            expected = (values.float() @ matrix.float().T * multiplier).to(out.dtype)
            out.copy_(expected if bias is None else expected + bias)
            if family == "blockscaled":
                assert workspace is workspaces[multiplier]
            return out
        state = SimpleNamespace(required_workspace=multiplier, run=run)
        if family == "bf16":
            return bf16_gemv.benchmark_call(state, inputs=acquire, weight=weight, bias=bias)
        workspaces[multiplier] = torch.empty(multiplier, dtype=torch.uint8)
        return blockscaled_gemm.benchmark_call(state, inputs=acquire, values=weight,
            scales=None, gain=None, bias=bias, weight=weight,
            selected_workspace=workspaces.__getitem__)

    champion, challenger = candidate(1), candidate(2)
    assert champion.owners[0] is challenger.owners[0]
    fixture = champion.owners[0]
    assert fixture.source.stride() == source.stride()
    assert fixture.output.stride() == output.stride()
    assert fixture.source.data_ptr() != source.data_ptr()
    assert fixture.output.data_ptr() != output.data_ptr()
    other = input_factory(source, output)()
    assert other.source.data_ptr() != fixture.source.data_ptr()
    for call, multiplier in ((champion, 1), (challenger, 2), (champion, 1)):
        fixture.source.fill_(float("nan"))
        fixture.output.fill_(float("nan"))
        call.produce()
        result = call.invoke()
        expected = (fixture.seed.float() @ weight.float().T * multiplier).to(output_dtype) + bias
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.all(source == 7) and torch.all(output == 13)
    reference = weakref.ref(fixture)
    del fixture, challenger, call, result
    gc.collect()
    assert reference() is champion.owners[0]
    del champion
    gc.collect()
    assert reference() is None
    assert acquire().source.stride() == source.stride()


def test_projection_candidate_binding_failure_does_not_retain_fixture():
    source = torch.zeros((4, 6), dtype=torch.bfloat16)
    acquire = input_factory(source, source)
    references = []

    def observe():
        value = acquire()
        references.append(weakref.ref(value))
        return value

    def fail(_):
        raise RuntimeError("candidate workspace binding failed")

    with pytest.raises(RuntimeError, match="workspace binding failed"):
        blockscaled_gemm.benchmark_call(SimpleNamespace(required_workspace=12), inputs=observe,
            values=None, scales=None, gain=None, bias=None, weight=None, selected_workspace=fail)
    gc.collect()
    assert references and references[0]() is None


@pytest.mark.parametrize("native,requested,expected", [
    (False, "auto", "auto"), (True, "auto", "quantized"),
    (True, "a16", "a16"), (True, "quantized", "quantized"),
])
def test_native_dense_activations_only_override_auto(monkeypatch, native, requested, expected):
    session, weight, *_ = projection_fixture(monkeypatch)
    session.native_dense_activations = native
    modes = []

    def query(*args, activation_mode, **kwargs):
        modes.append(activation_mode)
        return Query()

    monkeypatch.setattr(linear.blockscaled, "query_from_call", query)
    source = torch.zeros((4, 6), dtype=torch.bfloat16)
    output = torch.empty((4, 5), dtype=torch.bfloat16)
    projection = linear.QuantizedProjection(source, weight, output, session=session,
                                            name="projection", activation_mode=requested)
    assert modes == [expected]
    assert projection.activation_mode == expected
