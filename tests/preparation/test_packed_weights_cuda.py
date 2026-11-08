"""Immutable weight arenas preserve raw bytes, strides, and export exclusions."""

from dataclasses import dataclass
import os

import pytest
import torch

from ignition.preparation.package_storage import PackageStorage


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an assigned CUDA GPU")


@pytest.fixture(scope="module")
def storage():
    # The native preparation arena is process-wide; all tests share its owner.
    with PackageStorage(device=0) as owner:
        yield owner


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.bfloat16, torch.float16,
                                  torch.float32, torch.float8_e4m3fn, torch.float8_e8m0fnu])
@torch.no_grad()
def test_weight_packing_preserves_raw_bits_and_transposed_storage(dtype, storage):
    with torch.cuda.device(0):
        packed = storage.packed_weights()
        raw = torch.randint(0, 256, (131, 96 * dtype.itemsize), device="cuda", dtype=torch.uint8)
        # Nonzero source offset, transposed dense storage, and arbitrary NaN bits.
        source = raw.view(dtype)[3:].transpose(0, 1)
        expected = source.contiguous().view(torch.uint8).cpu()
        output = packed.tensor(source)
        assert (output.shape, output.stride()) == (source.shape, source.stride())
        assert output.data_ptr() % 512 == 0
        assert packed.tensor(source) is output
        assert packed.tensor(output) is output
        del source, raw
        torch.cuda.synchronize()
        assert torch.equal(output.contiguous().view(torch.uint8).cpu(), expected)
        assert packed.inactive_ranges() == []
        packed.freeze()
        excluded = storage.unused_ranges()
        assert any(left == output.data_ptr() + output.nbytes and right > left for left, right in excluded)
        assert not any(left < output.data_ptr() + output.nbytes and output.data_ptr() < right
                       for left, right in excluded)
        with pytest.raises(RuntimeError, match="frozen"):
            packed.tensor(output)


@torch.no_grad()
def test_weight_tree_identity_and_dense_layout_validation(storage):
    @dataclass(frozen=True)
    class Weight:
        tensor: torch.Tensor
        reference: torch.Tensor

    with torch.cuda.device(0):
        packed = storage.packed_weights()
        source = torch.arange(16, dtype=torch.float32, device="cuda").view(4, 4)
        cpu = source.cpu()
        result = packed.tree({"layer": Weight(source, cpu), "aliases": (source, [source])})
        assert result["layer"].tensor is result["aliases"][0] is result["aliases"][1][0]
        assert result["layer"].reference is cpu
        with pytest.raises(ValueError, match="nonoverlapping dense"):
            packed.tensor(source[:, ::2])
        with pytest.raises(ValueError, match="nonoverlapping dense"):
            packed.tensor(source[:1].expand(4, 4))
        with pytest.raises(ValueError, match="inference tensors"):
            packed.tensor(source.clone().requires_grad_(True))


@torch.no_grad()
def test_weight_packing_preserves_recorded_checkpoint_sources(storage):
    from ignition.preparation.provenance import lookup, record_sources

    with torch.cuda.device(0):
        packed = storage.packed_weights()
        source = torch.arange(96, device="cuda", dtype=torch.bfloat16).view(12, 8).T
        record_sources(source, [
            dict(tensor="encoder.q.weight", ranges=((4, 8), (0, 12)), concatenation_axis=0),
            dict(tensor="encoder.k.weight", ranges=((4, 8), (0, 12)), concatenation_axis=0),
        ], packing="fused-column-shards", statistics_domain="checkpoint",
            packing_parameters={"axis": 0, "transpose": True})
        expected = lookup(source)
        output = packed.tensor(source)
        assert lookup(output) == expected
        assert output.shape == source.shape and output.stride() == source.stride()
        assert torch.equal(output.contiguous().view(torch.uint8), source.contiguous().view(torch.uint8))
        del source
        assert lookup(output) == expected
        assert packed.tensor(output) is output
