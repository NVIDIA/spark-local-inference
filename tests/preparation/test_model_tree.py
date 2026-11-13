# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.recipes.qwen38 import create_model_tree


def test_exact_checkpoint_constructs_canonical_meta_tree_without_cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    snapshot = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    if not snapshot.exists():
        pytest.skip("exact Qwen3.8 checkpoint is not cached")
    initialized = torch.cuda.is_initialized()
    model = create_model_tree(inspect_qwen38_checkpoint(snapshot))
    assert type(model).__name__ == "Qwen4ExpForConditionalGeneration"
    assert len(model.model.language_model.layers) == 48
    assert len(model.model.visual.blocks) == 27
    assert model.model.language_model.layers[1].ple is not None
    assert model.model.language_model.layers[3].self_attn.head_dim == 256
    assert sum(p.numel() for p in model.parameters()) == 177392830576
    assert all(p.device.type == "meta" and not p.requires_grad for p in model.parameters())
    assert all(p.device.type == "meta" for p in model.buffers())
    assert not model.training
    assert torch.cuda.is_initialized() == initialized
