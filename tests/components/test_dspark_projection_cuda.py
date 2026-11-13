# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packing draft-layer projections preserves checkpoint values and scale rows."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ignition.components.block_fp8_weights import BlockFP8WeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("hidden,block", [(4096, 128), (5120, 32)])
@torch.no_grad()
def test_packed_context_weights_preserve_layer_order_and_quantization(tmp_path, hidden, block):
    generator = torch.Generator().manual_seed(41019)
    tensors = {}
    names = tuple(f"mtp.{layer}.attn.wkv" for layer in range(3))
    for name in names:
        tensors[name + ".weight"] = torch.randn((512, hidden), generator=generator).to(torch.float8_e4m3fn)
        tensors[name + ".scale"] = torch.randint(120, 132, (512 // block, hidden // block),
            generator=generator, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(tensor.nbytes for tensor in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    (tmp_path / "config.json").write_text("{}")
    loader = BlockFP8WeightLoader(inspect_indexed_checkpoint(tmp_path), "cuda:0")
    loader.config = SimpleNamespace(quantization={"weight_block_size": [block, block]})
    with loader:
        packed = loader.load_linear_group(names).weight
    values = packed.values.cpu().view(torch.uint8)
    scales = packed.scale_rows[0].cpu().view(torch.uint8)
    for layer, name in enumerate(names):
        assert torch.equal(values[layer * 512:(layer + 1) * 512], tensors[name + ".weight"].view(torch.uint8))
        expected = tensors[name + ".scale"].view(torch.uint8).repeat_interleave(block, 0).repeat_interleave(block // 32, 1)
        assert torch.equal(scales[layer * 512:(layer + 1) * 512], expected)
