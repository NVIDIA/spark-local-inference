# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.qwen38 import TEXT_PREFIX
from ignition.recipes.qwen38.weights import (
    QwenWeightLoader,
    _expert_ids,
    _expert_scale_vectors,
    allocate_expert_scale_scratch,
    expert_memory_plan,
)
from ignition.reference.qwen38_quant import CheckpointTensors


@pytest.fixture(scope="module")
def manifest():
    default = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    snapshot = Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", default))
    if not snapshot.exists():
        pytest.skip("exact Qwen checkpoint snapshot is unavailable")
    return inspect_qwen38_checkpoint(snapshot)


def _scalars():
    values = {}
    for expert, base in ((3, 1.0), (5, 2.0)):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts.{expert}.{projection}"
            values[f"{prefix}.weight_scale_2"] = torch.tensor(base * .01)
            values[f"{prefix}.input_scale"] = torch.tensor(base * .125)
    return values


def test_expert_scale_vectors_preserve_order_and_per_expert_values():
    values = _expert_scale_vectors(_scalars(), 0, (5, 3))
    torch.testing.assert_close(values[0], torch.tensor([.02, .01]), rtol=0, atol=0)
    torch.testing.assert_close(values[1], torch.tensor([.02, .01]), rtol=0, atol=0)
    assert values[2].tolist() == [4, 8]
    assert values[3].tolist() == [4, 8]


@pytest.mark.parametrize("suffix", ["weight_scale_2", "input_scale"])
def test_fused_experts_reject_differing_gate_up_scales(suffix):
    sources = _scalars()
    sources[f"{TEXT_PREFIX}.layers.0.mlp.experts.3.gate_proj.{suffix}"] *= 2
    with pytest.raises(ValueError, match="gate/up.*differ"):
        _expert_scale_vectors(sources, 0, (3, 5))


@pytest.mark.parametrize("invalid", [0, -1, float("inf"), float("nan")])
def test_expert_scales_reject_invalid_values(invalid):
    sources = _scalars()
    sources[f"{TEXT_PREFIX}.layers.0.mlp.experts.5.down_proj.input_scale"] = torch.tensor(float(invalid))
    with pytest.raises(ValueError, match="finite and positive"):
        _expert_scale_vectors(sources, 0, (3, 5))


def test_expert_scales_reject_reciprocal_overflow():
    sources = _scalars()
    sources[f"{TEXT_PREFIX}.layers.0.mlp.experts.5.down_proj.input_scale"] = torch.tensor(1e-40)
    with pytest.raises(ValueError, match="reciprocal scales must remain finite"):
        _expert_scale_vectors(sources, 0, (3, 5))


@pytest.mark.parametrize("layer, ids", [(48, None), (True, None), (0, []), (0, [0, 0]), (0, [512]), (0, [False])])
def test_invalid_expert_selection(layer, ids):
    with pytest.raises(ValueError):
        _expert_ids(layer, ids)


def test_loader_construction_and_memory_plan_are_cpu_only(manifest):
    loader = QwenWeightLoader(manifest, "cuda:2")
    with pytest.raises(RuntimeError, match="open QwenWeightLoader"):
        loader.load_tensor(f"{TEXT_PREFIX}.embed_tokens.weight")
    plan = expert_memory_plan(manifest, 0)
    assert plan.expert_ids == tuple(range(512))
    assert plan.packed_bytes == 1_258_291_200
    assert plan.scalar_bytes == 12_288
    assert plan.scale_scratch_bytes == 157_286_400
    assert 75 << 20 < plan.csf_bytes < 90 << 20
    with pytest.raises(ValueError, match="CUDA"):
        allocate_expert_scale_scratch("cpu")


def test_actual_layer_scales(manifest):
    sources = {}
    with CheckpointTensors(manifest, byte_budget=2 << 20) as reader:
        for expert in range(512):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts.{expert}.{projection}"
                for suffix in ("weight_scale_2", "input_scale"):
                    name = f"{prefix}.{suffix}"
                    sources[name] = reader.read(name)
        scale_vectors = _expert_scale_vectors(sources, 0, tuple(range(512)))
        assert scale_vectors[0].unique().numel() > 1
        assert scale_vectors[3].unique().numel() > 1
        assert scale_vectors[2].shape == (512,)


def test_calibrated_experts_reject_missing_activation_scale():
    sources = _scalars()
    del sources[f"{TEXT_PREFIX}.layers.0.mlp.experts.5.down_proj.input_scale"]
    with pytest.raises(ValueError, match="input_scale is required for calibrated NVFP4 activations"):
        _expert_scale_vectors(sources, 0, (3, 5))
