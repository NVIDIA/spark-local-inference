# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical module geometry of the local GLM 5.3 Flash checkpoint."""

import os
from pathlib import Path

import pytest
import torch

from ignition.recipes.glm53 import create_model_tree


def test_checkpoint_constructs_canonical_meta_tree_without_cuda():
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    initialized = torch.cuda.is_initialized()
    model = create_model_tree(snapshot)
    layers = model.model.language_model.layers
    assert len(layers) == 45
    assert sum(layer.block_type == "linear_attention" for layer in layers) == 34
    assert sum(layer.block_type == "indexed_attention" for layer in layers) == 11
    assert sum(hasattr(layer.mlp, "experts") for layer in layers) == 42
    assert layers[0].self_attn.q_proj.weight.shape == (8192, 4096)
    assert layers[3].self_attn.q_b_proj.weight.shape == (16384, 1536)
    assert layers[3].mlp.experts.gate_up_proj.shape == (288, 4096, 4096)
    assert layers[3].attn_hc.fn.shape == (24, 16384)
    assert model.lm_head.weight.shape == (154880, 4096)
    assert all(p.is_meta and not p.requires_grad for p in model.parameters())
    assert all(p.is_meta for p in model.buffers())
    assert not model.training
    assert torch.cuda.is_initialized() == initialized
