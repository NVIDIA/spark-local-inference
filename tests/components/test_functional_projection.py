# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Forward-local projection storage preserves bindings, layouts, and mutations."""

import gc
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from ignition.components import linear


@torch.library.custom_op("flare_test::functional_projection_out", mutates_args=("output", "workspace"))
def _project(source: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
             workspace: torch.Tensor) -> None:
    torch.mm(source, weight.T, out=output)
    workspace.fill_(17)


@_project.register_fake
def _fake_project(source, weight, output, workspace):
    pass


class _Plan:
    def __init__(self, nbytes=64, handle=7):
        self.nbytes, self.handle = nbytes, handle

    def memory_requirements(self):
        return SimpleNamespace(scratch_nbytes=self.nbytes)


def _quantized(monkeypatch, *, encoded=False, quantizer=False, stride=(11, 1)):
    projection = object.__new__(linear.QuantizedProjection)
    projection.output = torch.empty_strided((2, 3), stride)
    projection.workspace = torch.empty(64, dtype=torch.uint8)
    projection.weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    projection.plan = _Plan()
    projection.bias = torch.arange(3, dtype=torch.float32)
    projection.sm103 = False
    projection.quantizer_plan = _Plan() if quantizer else None
    projection.encoded = object() if encoded else None
    projection.encoded_layout = None
    projection.parts = (projection.weight, torch.empty(1), None)
    if encoded:
        projection.encoded_bases = (torch.zeros(2, 4), torch.zeros(2), torch.zeros(2))

    def mm(source, weight, *, out, plan, workspace):
        _project(source, weight, out, workspace)
        return out

    monkeypatch.setattr(linear.blockscaled, "mm", mm)
    return projection


def test_functional_projection_releases_bound_storage_and_preserves_strides(monkeypatch):
    projection = _quantized(monkeypatch)
    output_ref, workspace_ref = weakref.ref(projection.output), weakref.ref(projection.workspace)
    assert projection.functional(session=None) is projection
    gc.collect()
    assert output_ref() is None and workspace_ref() is None
    source = torch.arange(8, dtype=torch.float32).view(2, 4)
    first = projection(source)
    second = projection(source + 1)
    assert first.stride() == second.stride() == (11, 1)
    assert first.untyped_storage()._cdata != second.untyped_storage()._cdata
    torch.testing.assert_close(first, source @ projection.weight.T + projection.bias, rtol=0, atol=0)
    torch.testing.assert_close(second, (source + 1) @ projection.weight.T + projection.bias, rtol=0, atol=0)
    assert projection.output is None and projection.workspace is None


def test_functional_projection_uses_final_selected_workspace(monkeypatch):
    projection = _quantized(monkeypatch)
    finalizers = []
    monkeypatch.setattr(linear, "register_finalizer", lambda session, callback: finalizers.append(callback))
    projection.functional(session=None)
    with pytest.raises(RuntimeError, match="preparation before execution"):
        projection(torch.zeros(2, 4))
    projection.plan = _Plan(nbytes=128)
    for finalize in finalizers:
        finalize()
    assert projection.workspace_nbytes == 128
    projection(torch.ones(2, 4))
    assert projection.workspace is None


def test_forward_local_projection_allocations_survive_storage_lowering(monkeypatch):
    from ignition.preparation.compilation import _INVOCATIONS, _lower_storage

    projection = _quantized(monkeypatch).functional(session=None)
    source = torch.arange(8, dtype=torch.float32).view(2, 4)
    module = make_fx(lambda: projection(source))()
    allocations = [node for node in module.graph.nodes if node.op == "call_function"
                   and node.target in (torch.ops.aten.empty_strided.default, torch.ops.aten.empty.memory_format)]
    assert len(allocations) == 2
    calls = _lower_storage(module)
    module.recompile()
    token = _INVOCATIONS.set(calls)
    try:
        actual = module()
    finally:
        _INVOCATIONS.reset(token)
    torch.testing.assert_close(actual, source @ projection.weight.T + projection.bias, rtol=0, atol=0)


@pytest.mark.parametrize("owned", [False, True])
def test_functional_encoded_projection_uses_call_owned_bases(monkeypatch, owned):
    from b12x import gemm
    from b12x.quantization import mxfp8

    projection = _quantized(monkeypatch, encoded=True, quantizer=owned)
    projection.bias = None
    old_bases = [weakref.ref(base) for base in projection.encoded_bases]
    projection.functional(session=None)
    gc.collect()
    assert all(reference() is None for reference in old_bases)

    def quantize(source, values, rows, mma, *, plan):
        values.copy_(source)
        rows.fill_(19)
        mma.fill_(23)

    def mm(values, mma, weight, weight_scales, output, workspace, *, plan):
        _project(values, weight, output, workspace)

    monkeypatch.setattr(mxfp8, "quantize_dense_rows", quantize)
    monkeypatch.setattr(gemm, "mxfp8_mm_bases", mm)
    source = torch.arange(8, dtype=torch.float32).view(2, 4)
    if owned:
        actual = projection(source)
    else:
        with pytest.raises(ValueError, match="requires encoded_bases"):
            projection(source)
        actual = projection(source, encoded_bases=(source, torch.zeros(2), torch.zeros(2)))
    torch.testing.assert_close(actual, source @ projection.weight.T, rtol=0, atol=0)
    assert projection.encoded is None and projection.encoded_bases is None


def test_unquantized_projection_preserves_functional_output_layout(monkeypatch):
    projection = object.__new__(linear.UnquantizedProjection)
    projection.output = torch.empty_strided((2, 3), (9, 1))
    projection.weight, projection.bias = torch.ones(3, 4), None
    projection.plan, projection.dense = _Plan(), False
    monkeypatch.setattr(linear.bf16_gemv, "mm", lambda source, weight, *, out, plan, bias:
                        torch.mm(source, weight.T, out=out))
    output_ref = weakref.ref(projection.output)
    projection.functional(session=None)
    assert output_ref() is None
    result = projection(torch.ones(2, 4))
    assert result.stride() == (9, 1)
    assert result.eq(4).all()


def test_parallel_projection_keeps_local_branches_disjoint_through_join(monkeypatch):
    left, right = _quantized(monkeypatch), _quantized(monkeypatch, stride=(13, 1))
    operation = object.__new__(linear.ParallelProjections)
    operation.projections = (left, right)
    operation.outputs = [left.output, right.output]
    operation.workspaces = [left.workspace, right.workspace]
    operation.arena = object()
    operation.values = [left.weight, right.weight]
    operation.scales, operation.plans, operation.stream_handle = [None, None], [7, 7], 11

    def parallel(source, values, scales, outputs, workspaces, plans, stream_handle):
        assert outputs[0].untyped_storage()._cdata != outputs[1].untyped_storage()._cdata
        assert workspaces[0].untyped_storage()._cdata != workspaces[1].untyped_storage()._cdata
        assert plans == [7, 7] and stream_handle == 11
        for weight, output, workspace in zip(values, outputs, workspaces, strict=True):
            _project(source, weight, output, workspace)

    monkeypatch.setattr(linear, "_parallel_mx_projections", parallel)
    operation.functional(session=None)
    operation.refresh()
    outputs = operation(torch.ones(2, 4))
    assert [output.stride() for output in outputs] == [(11, 1), (13, 1)]
    for actual, weight in zip(outputs, operation.values, strict=True):
        torch.testing.assert_close(actual, torch.ones(2, 4) @ weight.T, rtol=0, atol=0)
    assert operation.outputs is None and operation.workspaces is None and operation.arena is None
