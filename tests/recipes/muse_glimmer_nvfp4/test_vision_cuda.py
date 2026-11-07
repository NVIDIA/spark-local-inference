# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Muse Glimmer vision graph against ``MuseGlimmerModel.get_image_features``.

Metadata (window order, cells, taps, reverse index) is produced here with torch from
the canonical ``transformers.vision_utils`` helpers so the graph is tested on its own;
Helios' parity with those helpers is the frontend profile's test.

The 50-layer tower is BF16-sensitive: from layer 31 the canonical BF16 path itself
drifts about 10% (relative L2) from an FP32 run of the same weights. The checkpoint
test therefore measures the prepared graph against the FP32 run and requires it to
land within the canonical BF16 path's own distance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.muse_glimmer_nvfp4.vision import (
    PERCEPTION_NORM_EPS, PreparedVision, VisionMetadata, VisionScratch, VisionWeights, window_capacity,
)
from tests.recipes.muse_glimmer_nvfp4.test_reference_cuda import CHECKPOINT, CUDA, SNAPSHOT, device as _device

EVALUATION_IMAGES = ("color-bars.png", "glimmer-sign.png", "green-triangles.png", "shapes-left-right.png")
RAGGED = (980, 532)  # width, height -> a 70x38 patch grid: 32x32, 32x6, 6x32 and 6x6 cells


def test_vision_buffer_geometry():
    metadata = VisionMetadata.allocate(1024, 1, torch.device("cpu"))
    scratch = VisionScratch.allocate(1024, torch.device("cpu"))
    assert scratch.patches.shape == (1024, 1176)
    assert scratch.query.shape == (1024, 16, 96)
    assert scratch.merged.shape == (256, 6144)
    assert scratch.output.shape == (256, 6656)
    assert metadata.position_ids.shape == (1024, 2)
    assert metadata.cu_seqlens.shape == (2,)
    assert metadata.cu_window_seqlens.shape == (window_capacity(1024, 1) + 1,)
    assert metadata.reverse_index.shape == (1024,)
    assert metadata.num_patches.item() == 0
    with pytest.raises(ValueError):
        VisionScratch.allocate(1023, torch.device("cpu"))


def _vision_config(**overrides):
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig
    config = MuseGlimmerVisionConfig(**overrides)
    config._attn_implementation = "eager"
    return config


def _meta_tower(config):
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionModel
    with torch.device("meta"):
        module = MuseGlimmerVisionModel(config)
    return module.eval().requires_grad_(False)


def _adapter_config():
    return SimpleNamespace(out_hidden_size=6144, projector_hidden_size=4096, projector_hidden_act="gelu")


def _random_weights(config, device, seed=0):
    """Random BF16 tensors keyed the way ``VisionWeights.load`` keys them."""
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionAdapter, MuseGlimmerVisionModel
    torch.manual_seed(seed)
    tower = MuseGlimmerVisionModel(config)
    adapter = MuseGlimmerVisionAdapter(_adapter_config())
    projection = torch.nn.Linear(4096, 6656, bias=False)
    parameters, projections = {}, {}
    for path, value in tower.named_parameters():
        kind = path.split(".")[-2]
        value = value.detach().to(device, torch.bfloat16)
        if kind in ("q_proj", "k_proj", "v_proj"):
            continue
        (projections if kind in ("patch_embedding", "proj", "fc1", "fc2") else parameters)[path] = value
    for i, layer in enumerate(tower.layers):
        for part in ("weight", "bias"):
            projections[f"layers.{i}.attn.qkv.{part}"] = torch.cat([
                getattr(getattr(layer.attn, f"{name}_proj"), part).detach() for name in ("q", "k", "v")
            ]).to(device, torch.bfloat16)
    projections["adapter.fc1.weight"] = adapter.fc1.weight.detach().to(device, torch.bfloat16)
    projections["adapter.fc2.weight"] = adapter.fc2.weight.detach().to(device, torch.bfloat16)
    projections["projection.weight"] = projection.weight.detach().to(device, torch.bfloat16)
    return VisionWeights(parameters, projections)


def _reference(config, weights, device, dtype=torch.bfloat16):
    """Canonical modules over the prepared weights, driven by ``MuseGlimmerModel.get_image_features``.

    ``torch.float32`` upcasts the small modules outright and each encoder layer
    for the duration of its own forward, so the FP32 run never holds a second
    copy of the tower.
    """
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerRMSNorm, MuseGlimmerVisionAdapter, MuseGlimmerVisionRotaryEmbedding,
    )
    tower = _meta_tower(config)
    def assign(module, path, value):
        parent, _, attribute = path.rpartition(".")
        setattr(module.get_submodule(parent) if parent else module, attribute,
                torch.nn.Parameter(value, requires_grad=False))
    for path, value in weights.parameters.items():
        assign(tower, path, value)
    p = weights.projections
    assign(tower, "patch_embedder.patch_embedding.weight", p["patch_embedder.patch_embedding.weight"])
    for i in range(len(tower.layers)):
        for part in ("weight", "bias"):
            for index, name in enumerate(("q", "k", "v")):
                assign(tower, f"layers.{i}.attn.{name}_proj.{part}",
                       p[f"layers.{i}.attn.qkv.{part}"][index * 1536:(index + 1) * 1536])
            assign(tower, f"layers.{i}.attn.proj.{part}", p[f"layers.{i}.attn.proj.{part}"])
            for key in ("fc1", "fc2"):
                assign(tower, f"layers.{i}.mlp.{key}.{part}", p[f"layers.{i}.mlp.{key}.{part}"])
    tower.rotary_emb = MuseGlimmerVisionRotaryEmbedding(config).to(device)
    with torch.device("meta"):
        adapter = MuseGlimmerVisionAdapter(_adapter_config())
        projection = torch.nn.Linear(4096, 6656, bias=False)
    assign(adapter, "fc1.weight", p["adapter.fc1.weight"])
    assign(adapter, "fc2.weight", p["adapter.fc2.weight"])
    assign(projection, "weight", p["projection.weight"])
    if dtype == torch.float32:
        for module in (tower.patch_embedder, tower.ln_pre, tower.ln_post, adapter, projection):
            module.float()
        for layer in tower.layers:
            stored = {name: parameter.data for name, parameter in layer.named_parameters()}
            def upcast(module, _, stored=stored):
                for name, parameter in module.named_parameters():
                    parameter.data = stored[name].float()
            def restore(module, _, __, stored=stored):
                for name, parameter in module.named_parameters():
                    parameter.data = stored[name]
            layer.register_forward_pre_hook(upcast)
            layer.register_forward_hook(restore)
    return SimpleNamespace(
        vision_tower=tower, vision_adapter=adapter, vision_projection=projection,
        perception_emb_norm=MuseGlimmerRMSNorm(eps=PERCEPTION_NORM_EPS, with_scale=False),
        config=SimpleNamespace(vision_config=config, return_dict=True, output_hidden_states=False,
                               output_attentions=False))


def _canonical(reference, pixels, grid):
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerModel
    return MuseGlimmerModel.get_image_features(reference, pixels, grid).pooler_output[0].float()


def _pack(metadata, pixels, grid, device):
    """Window-order pixels and metadata from the canonical helpers, padded to the metadata capacity."""
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights, get_vision_position_ids, get_vision_window_index,
    )
    capacity = metadata.reverse_index.shape[0]
    count = int(grid.prod())
    window_index, cu_window = get_vision_window_index(grid, spatial_merge_size=1, window_size=448, patch_size=14)
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=32, mode="bilinear", align_corners=False, spatial_merge_size=1, padding="zeros")
    windows = metadata.cu_window_seqlens.shape[0]
    assert cu_window.shape[0] <= windows, "cell capacity bound violated"
    metadata.interpolation_indices[:count].copy_(indices[window_index])
    metadata.interpolation_weights[:count].copy_(weights[window_index])
    metadata.position_ids[:count].copy_(get_vision_position_ids(grid, 1)[window_index])
    metadata.reverse_index[:count].copy_(torch.argsort(window_index))
    metadata.cu_seqlens.copy_(torch.tensor([0, count], dtype=torch.int32))
    metadata.cu_window_seqlens.copy_(torch.nn.functional.pad(cu_window, (0, windows - cu_window.shape[0]),
                                                             value=count))
    metadata.num_patches.fill_(count)
    packed = torch.zeros((capacity, 1176), dtype=torch.float32, device=device)
    packed[:count].copy_(pixels[window_index])
    return packed


def _prepare(config, weights, capacity, device):
    from b12x.preparation import PreparationSession
    scratch = VisionScratch.allocate(capacity, device)
    metadata = VisionMetadata.allocate(capacity, 1, device)
    session = PreparationSession(device=device, autotune=False, compile_workers=0)
    session.__enter__()
    prepared = PreparedVision(_meta_tower(config), scratch, metadata, weights, session=session)
    session.freeze()
    return prepared, scratch, metadata, session


def _replay(prepared, scratch, metadata, pixels, grid, device):
    """Capture the prepared graph on one image, replay it, and return its live feature rows."""
    count = int(grid.prod())
    packed = _pack(metadata, pixels, grid, device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        prepared(packed)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        prepared(packed)
    scratch.output.fill_(float("nan"))
    torch.cuda.synchronize(device)
    graph.replay()
    torch.cuda.synchronize(device)
    actual = scratch.output.float()
    assert torch.count_nonzero(actual[count // 4:]) == 0
    return actual[:count // 4]


def _distance(actual, expected):
    difference = (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-12)
    return {"relative_l2": float((actual - expected).norm() / expected.norm().clamp_min(1e-12)),
            "worst_token_relative_l2": float(difference.max()),
            "max_absolute": float((actual - expected).abs().max())}


@CUDA
@pytest.mark.parametrize("capacity,grid", [(1024, (16, 16)), (4096, (38, 70))])
@torch.no_grad()
def test_synthetic_parity_on_random_weights(capacity, grid, record_property):
    """Four random layers at checkpoint widths; the ragged grid spans 32x32, 32x6, 6x32 and 6x6 cells."""
    device = _device()
    config = _vision_config(num_hidden_layers=4)
    assert config.layer_types == ["window_attention"] * 3 + ["full_attention"]
    weights = _random_weights(config, device)
    reference = _reference(config, weights, device)
    prepared, scratch, metadata, session = _prepare(config, weights, capacity, device)
    try:
        torch.manual_seed(1)
        grid = torch.tensor([[1, *grid]], device=device)
        pixels = torch.randn((int(grid.prod()), 1176), dtype=torch.float32, device=device)
        expected = _canonical(reference, pixels, grid)
        actual = _replay(prepared, scratch, metadata, pixels, grid, device)
        errors = _distance(actual, expected)
        record_property("numerical_errors", json.dumps(errors))
        assert errors["relative_l2"] < .03, errors
        torch.testing.assert_close(actual, expected, rtol=.05, atol=float(expected.abs().max()) * .03)
    finally:
        session.__exit__(None, None, None)


@pytest.fixture(scope="module")
def snapshot():
    path = Path(SNAPSHOT)
    if not (path / "model.safetensors.index.json").is_file():
        pytest.skip(f"{path} holds no checkpoint index")
    return path


class _ShardTensors:
    """``load_tensors`` over the one shard holding the vision tower, in place of ``DirectWeightLoader``."""

    def __init__(self, snapshot, device):
        from safetensors import safe_open
        weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        shards = {shard for name, shard in weight_map.items() if name.startswith("model.vision")}
        assert len(shards) == 1, shards
        self.file = safe_open(snapshot / shards.pop(), "pt", device=str(device))

    def load_tensors(self, names):
        return tuple(self.file.get_tensor(name) for name in names)


def _evaluation_images(snapshot):
    from PIL import Image
    directory = Path(os.environ.get("FLARE_MUSE_GLIMMER_IMAGES", snapshot / "evaluation/core-workloads/images"))
    missing = [name for name in EVALUATION_IMAGES if not (directory / name).is_file()]
    if missing:
        pytest.skip(f"{directory} lacks {missing}")
    images = {name: Image.open(directory / name).convert("RGB") for name in EVALUATION_IMAGES}
    width, height = RAGGED
    pixels = bytes(component for y in range(height) for x in range(width)
                   for component in ((x * 7) % 256, (y * 11) % 256, (x + y * 3) % 256))
    images["ragged-980x532"] = Image.frombytes("RGB", (width, height), pixels)
    return images


def _check_checkpoint_parity(config, weights, snapshot, device, record_property):
    from transformers import AutoProcessor
    from transformers.models.muse_glimmer.image_processing_muse_glimmer import smart_resize
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    bf16 = _reference(config, weights, device)
    fp32 = _reference(config, weights, device, torch.float32)
    prepared, scratch, metadata, session = _prepare(config, weights, 4096, device)
    errors = []
    try:
        for name, image in _evaluation_images(snapshot).items():
            inputs = processor.image_processor(images=[image], return_tensors="pt")
            grid, pixels = inputs.image_grid_thw.to(device), inputs.pixel_values.to(device)
            height, width = smart_resize(image.height, image.width, patch_size=28, max_tokens=4096)
            assert grid.tolist() == [[1, height // 14, width // 14]], name
            if name.startswith("ragged"):
                assert grid.tolist() == [[1, 38, 70]]
            truth = _canonical(fp32, pixels, grid)
            canonical = _distance(_canonical(bf16, pixels, grid), truth)
            actual = _replay(prepared, scratch, metadata, pixels, grid, device)
            assert actual.shape == truth.shape == (int(grid.prod()) // 4, 6656)
            prepared_error = _distance(actual, truth)
            errors.append({"image": name, "tokens": actual.shape[0], "canonical_bf16": canonical,
                           "prepared": prepared_error})
            # Within the canonical BF16 path's own distance from FP32: 25% headroom on the
            # whole image, 2x on the single worst token (an extreme, not a bias).
            assert prepared_error["relative_l2"] <= max(.03, 1.25 * canonical["relative_l2"]), errors[-1]
            assert prepared_error["worst_token_relative_l2"] <= max(.03, 2 * canonical["worst_token_relative_l2"]), errors[-1]
            assert prepared_error["relative_l2"] < .2, errors[-1]
    finally:
        session.__exit__(None, None, None)
    record_property("numerical_errors", json.dumps(errors))


@CUDA
@CHECKPOINT
@torch.no_grad()
def test_checkpoint_parity_on_evaluation_images(snapshot, record_property):
    """Prepared graph vs ``get_image_features`` on the card's four images plus a ragged 980x532 grid.

    Measured on the pinned checkpoint: the canonical BF16 path is 5-11% relative L2
    from an FP32 run of the same weights (the drift starts at layer 31 of 50), and the
    prepared graph lands in the same band (5-11%, nearer FP32 on some images). A
    direct comparison with the canonical BF16 features therefore cannot be tighter
    than that noise, so the criterion is relative to the canonical BF16 error against
    the FP32 run; the synthetic test holds the strict 3% check on a short tower.
    """
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig

    device = _device()
    model_config = MuseGlimmerConfig.from_pretrained(snapshot)
    assert model_config.text_config.rms_norm_eps == PERCEPTION_NORM_EPS
    config = _vision_config(**json.loads((snapshot / "config.json").read_text())["vision_config"])
    assert config.num_hidden_layers == 50 and config.layer_types.count("full_attention") == 13
    weights = VisionWeights.load(_meta_tower(config), _ShardTensors(snapshot, device))
    _check_checkpoint_parity(config, weights, snapshot, device, record_property)
