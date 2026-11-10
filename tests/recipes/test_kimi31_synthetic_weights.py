# SPDX-License-Identifier: Apache-2.0

"""Synthetic Kimi values retain prepared layouts without opening weight payloads."""

from dataclasses import replace
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.checkpoint import TensorMetadata
from ignition.recipes.kimi31.synthetic_weights import (
    SyntheticDraftWeightLoader, _write,
)


cuda = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires an explicitly assigned CUDA device")


@pytest.fixture
def no_payload(monkeypatch):
    class Forbidden:
        def __init__(self, *args, **kwargs):
            raise AssertionError("checkpoint payload session was opened")

    import b12x.loader
    from ignition.preparation import remote_weights

    monkeypatch.setattr(b12x.loader, "DirectWeightSession", Forbidden)
    monkeypatch.setattr(remote_weights, "RemoteWeightSession", Forbidden)


def _manifest(tensors):
    entries = {}
    widths = {"BF16": 2, "F32": 4, "U8": 1}
    for name, (shape, dtype) in tensors.items():
        entries[name] = TensorMetadata(name, "model.safetensors", 0,
            math.prod(shape) * widths[dtype], shape, dtype)
    return SimpleNamespace(tensors=entries, snapshot=Path("/unused"), metadata_path=Path("/unused"))


def test_generated_values_are_finite_and_tensor_specific():
    first = torch.empty((64, 8), dtype=torch.bfloat16)
    second = torch.empty_like(first)
    _write(first, "layers.0.router.weight", seed=123)
    _write(second, "layers.0.router.weight", seed=123)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all() and first.std() > 0
    _write(second, "layers.1.router.weight", seed=123)
    assert not torch.equal(first, second)
    _write(second, "layers.0.router.weight", seed=124)
    assert not torch.equal(first, second)

    norm = torch.empty(16, dtype=torch.bfloat16)
    recurrence = torch.empty(16, dtype=torch.float32)
    scales = torch.empty(16, dtype=torch.uint8)
    _write(norm, "self_attn.q_a_layernorm.weight")
    _write(recurrence, "self_attn.A_log")
    _write(scales, "experts.0.w1.weight_scale")
    assert torch.equal(norm, torch.ones_like(norm))
    assert torch.equal(recurrence, torch.full_like(recurrence, -1))
    assert torch.equal(scales, torch.full_like(scales, 127))


def test_scales_have_calibrated_csf_exception_density():
    scales = torch.empty((2000, 96), dtype=torch.uint8)
    _write(scales, "experts.0.w1.weight_scale", seed=123)
    counts = torch.stack([(scales == value).sum(1) for value in range(256)], dim=1)
    exceptions = 96 - (counts[:, :-1] + counts[:, 1:]).max(1).values
    assert abs(float((exceptions == 0).float().mean()) - 0.708) < 0.01
    assert abs(float(exceptions.float().mean()) - 0.368) < 0.01
    assert int(exceptions.max()) == 4


def test_metadata_sources_do_not_open_payloads(no_payload, monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    manifest = _manifest({
        "projection.weight": ((32, 16), "BF16"),
        "projection.weight_scale_2": ((), "F32"),
        "norm.weight": ((16,), "BF16"),
    })
    loader = SyntheticDraftWeightLoader(manifest, "cuda", synthetic_seed=0)
    with loader:
        assert loader._sources["projection.weight"].device.type == "meta"
        assert loader._sources["projection.weight"].shape == (32, 16)
        assert loader._sources["projection.weight"].dtype == torch.bfloat16
        assert loader._sources["projection.weight_scale_2"].device.type == "cpu"
        assert loader._sources["projection.weight_scale_2"].item() == 1
        assert loader._sources["norm.weight"].device.type == "meta"
    assert loader._session is None and not loader._sources


@cuda
def test_dflash2_loader_geometry_without_payloads(no_payload, monkeypatch):
    target = os.environ.get("FLARE_KIMI31_CHECKPOINT")
    draft = os.environ.get("FLARE_KIMI31_DFLASH_CHECKPOINT")
    if not target or not draft:
        pytest.skip("set FLARE_KIMI31_CHECKPOINT and FLARE_KIMI31_DFLASH_CHECKPOINT")
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.kimi3.config import DSparkConfig, ModelConfig
    from ignition.recipes.kimi31.dflash_weights import DraftWeights

    torch.cuda.set_device(0)
    config = ModelConfig.load(target)
    config = replace(config, draft=DSparkConfig.load(draft, config))
    weights = DraftWeights.load(draft, config, torch.device("cuda", 0), TensorParallelShard(0, 16),
                                target_head=None, synthetic_seed=123)
    assert len(weights.layers) == 5
    assert weights.projection.shape == (448, 43008)
    assert torch.isfinite(weights.projection).all()
    assert not torch.equal(weights.selector["predecessor"], weights.selector["successor"])


@cuda
def test_native_mxfp4_scales_prepare_without_payloads(no_payload):
    tensors = {}
    for projection in ("up_proj", "gate_proj", "down_proj"):
        tensors[f"experts.0.{projection}.weight"] = ((256, 128), "U8")
        tensors[f"experts.0.{projection}.weight_scale"] = ((256, 8), "U8")
    loader = SyntheticDraftWeightLoader(_manifest(tensors), torch.device("cuda", 0), synthetic_seed=0)
    torch.cuda.set_device(0)
    with loader:
        prepared = loader.load_mxfp4_experts(("experts.0",), hidden_size=256,
            intermediate_size=256, scale_scratch={}, scale_compression="csf")
    assert prepared._impl.w1_fp4.shape[0] == 1
    assert prepared._impl.w2_fp4.shape[0] == 1
    assert prepared._impl.w1_blockscale.numel() > 0
    torch.cuda.synchronize()
