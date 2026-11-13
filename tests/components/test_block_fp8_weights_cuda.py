# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Column gathering retains FP8 checkpoint bytes and square-block scale values."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ignition.components.block_fp8_weights import BlockFP8WeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.reference.qwen38_quant import decode_mxfp8


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("block", [32, 128])
@torch.no_grad()
def test_interleaved_projection_columns_preserve_values_and_scales(tmp_path, block):
    generator = torch.Generator().manual_seed(41017)
    weight = torch.randn((256, 6144), generator=generator).to(torch.float8_e4m3fn)
    scales = torch.randint(123, 130, (256 // block, 6144 // block), generator=generator, dtype=torch.uint8)
    tensors = {"projection.weight": weight, "projection.scale": scales.view(torch.float8_e8m0fnu)}
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    (tmp_path / "config.json").write_text("{}")
    manifest = inspect_indexed_checkpoint(tmp_path)
    loader = BlockFP8WeightLoader(manifest, "cuda:0")
    loader.config = SimpleNamespace(quantization={"weight_block_size": [block, block]})
    expanded = scales.repeat_interleave(block, 0).repeat_interleave(block // 32, 1)
    reference = decode_mxfp8(weight, expanded)
    source = torch.randn((9, 6144), generator=generator)
    partials = []
    with loader:
        for rank in range(4):
            ranges = [(head * 256, (head + 1) * 256) for head in range(rank, 24, 4)]
            columns = torch.tensor([column for first, last in ranges for column in range(first, last)])
            packed = loader.load_linear_columns("projection", ranges)
            actual = decode_mxfp8(packed.weight.values.cpu(), packed.weight.scale_rows[0].view(torch.uint8).cpu())
            torch.testing.assert_close(actual, reference[:, columns], rtol=0, atol=0)
            assert torch.equal(packed.weight.values.cpu().view(torch.uint8), weight.view(torch.uint8)[:, columns])
            assert torch.equal(packed.weight.scale_rows[0].cpu().view(torch.uint8),
                               expanded[:, columns[::32] // 32])
            partials.append(source[:, columns] @ actual.T)
    torch.testing.assert_close(sum(partials), source @ reference.T, rtol=2e-5, atol=2e-3)


@pytest.mark.parametrize("block", [32, 128])
@torch.no_grad()
def test_padded_partitions_preserve_checkpoint_bits_and_zero_only_the_tail(tmp_path, block):
    generator = torch.Generator().manual_seed(41018)
    weight = torch.randn((384, 384), generator=generator).to(torch.float8_e4m3fn)
    scales = torch.randint(123, 130, (384 // block, 384 // block), generator=generator, dtype=torch.uint8)
    head = torch.randn((5, 8), generator=generator).bfloat16()
    tensors = {"projection.weight": weight, "projection.scale": scales.view(torch.float8_e8m0fnu),
               "head.weight": head}
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    (tmp_path / "config.json").write_text("{}")
    loader = BlockFP8WeightLoader(inspect_indexed_checkpoint(tmp_path), "cuda:0")
    loader.config = SimpleNamespace(quantization={"weight_block_size": [block, block]})
    with loader:
        with pytest.raises(ValueError, match="in range"):
            loader.load_linear_group(("projection",), row_ranges=((256, 512),), column_range=(256, 512))
        packed = loader.load_linear_group(("projection",), row_ranges=((256, 512),),
            column_range=(256, 512), pad=True)
        padded_head = loader.load_padded_tensor("head.weight", rows=(4, 8))
    values = packed.weight.values.cpu().view(torch.uint8)
    assert torch.equal(values[:128, :128], weight.view(torch.uint8)[256:, 256:])
    assert not values[128:].count_nonzero() and not values[:, 128:].count_nonzero()
    expanded = scales.repeat_interleave(block, 0).repeat_interleave(block // 32, 1)
    actual_scales = packed.weight.scale_rows[0].cpu().view(torch.uint8)
    assert torch.equal(actual_scales[:128, :4], expanded[256:, 8:])
    assert actual_scales[128:].eq(127).all() and actual_scales[:, 4:].eq(127).all()
    assert torch.equal(padded_head[0].cpu(), head[4])
    assert not padded_head[1:].count_nonzero()
    actual = decode_mxfp8(packed.weight.values.cpu(), actual_scales)
    expected = torch.zeros((256, 256))
    expected[:128, :128] = decode_mxfp8(weight, expanded)[256:, 256:]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
