# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint transfers fill final destinations without retaining conversion inputs."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import weights


class CopySession:
    def __init__(self):
        self.destinations = []
        self.flushes = 0

    def __call__(self, destination, source):
        destination.copy_(source)
        self.destinations.append(destination)
        return True

    def flush(self):
        self.flushes += 1


@pytest.fixture
def loader():
    result = weights.DirectWeightLoader(SimpleNamespace(), "cpu")
    result.quantization = {}
    result._session = CopySession()
    return result


@pytest.fixture
def allocations(monkeypatch):
    outputs = []

    def empty(shape, *, dtype, device):
        output = torch.empty(shape, dtype=dtype, device=device)
        outputs.append(output)
        return output

    monkeypatch.setattr(weights, "persistent_empty", empty)
    monkeypatch.setattr(weights, "persistent_zeros", lambda *args, **kwargs: empty(*args, **kwargs).zero_())
    return outputs


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("temporary", [False, True])
def test_group_loads_selected_rows_and_columns_into_one_destination(loader, allocations, dtype, temporary):
    first = torch.arange(48).reshape(6, 8).to(dtype)
    second = (100 + torch.arange(32)).reshape(4, 8).to(dtype)
    loader._sources = {"a.weight": first, "b.weight": second}
    actual = loader.load_linear_group(("a", "b"), row_ranges=((1, 5), (0, 3)),
                                      column_range=(2, 7), temporary=temporary)
    assert torch.equal(actual, torch.cat((first[1:5, 2:7], second[:3, 2:7])))
    assert actual.stride() == (5, 1)
    assert len(allocations) == (0 if temporary else 1)
    assert all(view.untyped_storage().data_ptr() == actual.untyped_storage().data_ptr()
               for view in loader._session.destinations)
    assert loader._session.flushes == 1


@pytest.mark.parametrize("scale_rank", [2, 4])
def test_block_fp8_group_uses_one_value_and_one_scale_destination(loader, allocations, scale_rank):
    loader.quantization = {name: {"quant_algo": "FP8_PB_WO", "weight_block_size": [128, 128]}
                           for name in ("a", "b")}
    for index, name in enumerate(("a", "b")):
        loader._sources[name + ".weight"] = torch.full((256, 256), index + 1.).to(torch.float8_e4m3fn)
        scales = torch.arange(4, dtype=torch.float32).reshape(2, 2) + index * 10
        loader._sources[name + ".weight_scale"] = scales if scale_rank == 2 else scales.reshape(2, 1, 2, 1)
    actual = loader.load_linear_group(("a", "b"), row_ranges=((128, 256), (0, 256)), column_range=(128, 256))
    assert torch.equal(actual.values.float(), torch.cat((torch.ones(128, 128), torch.full((256, 128), 2.))))
    assert torch.equal(actual.scales, torch.tensor([[3.], [11.], [13.]]))
    assert len(allocations) == 2
    assert {tensor.untyped_storage().data_ptr() for tensor in allocations} == {
        actual.values.untyped_storage().data_ptr(), actual.scales.untyped_storage().data_ptr()}


@pytest.mark.parametrize("method", ["tensor", "tensors", "padded", "linear", "mixed"])
@pytest.mark.parametrize("temporary", [False, True])
def test_temporary_inputs_bypass_persistent_destinations(loader, allocations, method, temporary):
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    loader._sources = {"p.weight": source}
    if method == "tensor":
        actual = loader.load_tensor("p.weight", extent=(slice(1, 3), slice(None)), temporary=temporary)
        expected = source[1:3]
    elif method == "tensors":
        actual, = loader.load_tensors(("p.weight",), temporary=temporary)
        expected = source
    elif method == "padded":
        actual = loader.load_padded_tensor("p.weight", rows=(2, 6), temporary=temporary)
        expected = torch.cat((source[2:], torch.zeros(2, 6)))
    elif method == "linear":
        actual = loader.load_linear("p", temporary=temporary)
        expected = source
    else:
        actual = loader.load_mixed_linear_group(("p",), temporary=temporary)
        expected = source
    assert torch.equal(actual, expected)
    assert len(allocations) == (0 if temporary else 1)


@pytest.mark.parametrize("columns", [128, 160])
@pytest.mark.parametrize("temporary", [False, True])
def test_mxfp8_padded_and_borrowed_planes_are_final_destinations(loader, allocations, monkeypatch, columns, temporary):
    from b12x.gemm.blockscaled import _linear
    from b12x.gemm._shared import wo_mxfp8

    monkeypatch.setattr(_linear, "_check_gpu_tensor", lambda *args: None)
    monkeypatch.setattr(wo_mxfp8, "_check_gpu_tensor", lambda *args: None)
    loader.quantization = {"p": {"quant_algo": "MXFP8", "group_size": 32}}
    source = (torch.arange(8 * columns).reshape(8, columns) % 16).to(torch.float8_e4m3fn)
    scales = torch.arange(8 * columns // 32, dtype=torch.uint8).reshape(8, columns // 32) + 100
    loader._sources = {"p.weight": source, "p.weight_scale": scales}
    expected = _linear.pack_weight(source, scales, recipe="mxfp8")
    actual = loader.load_linear("p", temporary=temporary)
    final = []
    for field in ("values", "scale_rows", "scale_mma"):
        left, right = getattr(actual.weight, field), getattr(expected.weight, field)
        assert left.shape == right.shape and left.stride() == right.stride() and left.dtype == right.dtype
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
        final.append(left.untyped_storage().data_ptr())
    assert len(allocations) == (0 if temporary else 3)
    assert all(tensor.untyped_storage().data_ptr() in final for tensor in allocations)
