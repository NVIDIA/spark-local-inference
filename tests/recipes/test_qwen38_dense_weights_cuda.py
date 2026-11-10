# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe-selected GDN groups preserve real checkpoint projection rows."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.qwen38_dense.config import ModelConfig, TEXT_PREFIX
from ignition.recipes.qwen38_dense.weights import _gdn_projection_groups
from ignition.reference.qwen38_quant import CheckpointTensors


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU")
def test_hybrid_recipe_groups_preserve_serialized_rows():
    snapshot = Path(os.environ.get("FLARE_QWEN38_DENSE_CHECKPOINT",
        "/data/models/Qwen3.8-27B-NVIDIA-NVFP4-QAD-MXFP8"))
    if not snapshot.is_dir():
        pytest.skip("requires the NVIDIA NVFP4 checkpoint with QAD MXFP8 projections")
    manifest = inspect_indexed_checkpoint(snapshot)
    config = ModelConfig.load(snapshot)
    with DirectWeightLoader(manifest, "cuda:0", prefixes=(TEXT_PREFIX + ".layers.",)) as loader:
        groups = {name: parts for name, parts in _gdn_projection_groups(config, loader).items()
                  if name.startswith("layers.0.")}
        assert set(groups) == {"layers.0.linear_attn.in_proj_ba", "layers.0.linear_attn.in_proj_qkvz"}
        with CheckpointTensors(manifest, byte_budget=128 << 20) as reader:
            for parts in groups.values():
                combined = loader.load_linear_group(parts)
                originals = [reader.read(part + ".weight") for part in parts]
                expected = torch.cat(originals)
                if isinstance(combined, torch.Tensor):
                    assert combined.dtype == expected.dtype == torch.bfloat16
                    assert torch.equal(combined.cpu(), expected)
                else:
                    values = combined.weight.values.view(torch.uint8).cpu().reshape(combined.out_features, -1)
                    assert torch.equal(values[:, :expected.shape[1]], expected.view(torch.uint8))
                    scales = combined.weight.scale_rows.view(torch.uint8).cpu().reshape(combined.out_features, -1)
                    expected_scales = torch.cat([reader.read(part + ".weight_scale").view(torch.uint8)
                                                [:original.shape[0], :original.shape[1] // 32]
                                                for part, original in zip(parts, originals, strict=True)])
                    assert torch.equal(scales[:, :expected_scales.shape[1]], expected_scales)
