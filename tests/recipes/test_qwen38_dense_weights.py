# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense GDN loading groups, transfer counts, and canonical BF16 parameter views."""

import json
from types import SimpleNamespace

import pytest
import torch

from ignition.components.model import canonical_tree
from ignition.components.weights import DirectWeightLoader
from ignition.recipes.qwen38_dense.config import ModelConfig, TEXT_PREFIX
from ignition.recipes.qwen38_dense import weights


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    config = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
        layer_types=["linear_attention"], num_attention_heads=1, num_key_value_heads=1,
        linear_num_key_heads=16, linear_num_value_heads=48, vocab_size=128, head_dim=256)
    raw = dict(architectures=["Qwen3_5ForConditionalGeneration"], text_config=config.to_dict(),
        quantization_config=dict(quant_method="modelopt", quant_algo="MIXED_PRECISION", quantized_layers={}))
    (tmp_path / "config.json").write_text(json.dumps(raw))
    tree = canonical_tree(Qwen3_5TextModel, config)
    sources = {TEXT_PREFIX + "." + name: torch.full(parameter.shape, (index + 1) / 16,
                                                   dtype=torch.bfloat16)
               for index, (name, parameter) in enumerate(tree.named_parameters())}
    sources["lm_head.weight"] = torch.ones((128, 128), dtype=torch.bfloat16)
    manifest = SimpleNamespace(snapshot=tmp_path, metadata_path=tmp_path,
        tensors={name: SimpleNamespace(shape=tensor.shape,
            dtype="BF16" if tensor.dtype == torch.bfloat16 else "F32") for name, tensor in sources.items()})
    copies = []

    def enter(loader):
        loader._sources = sources
        loader._session = SimpleNamespace(flush=lambda: None)

        def copy(destination, source, **kwargs):
            copies.append(next(name for name, tensor in sources.items()
                               if tensor.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()))
            destination.copy_(source)

        loader._copy = copy
        return loader

    monkeypatch.setattr(DirectWeightLoader, "__enter__", enter)
    monkeypatch.setattr(DirectWeightLoader, "__exit__", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    return manifest, sources, copies


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_grouped_ba_loads_each_source_once_and_shares_canonical_views(checkpoint, dtype):
    manifest, sources, copies = checkpoint
    names = tuple(f"{TEXT_PREFIX}.layers.0.linear_attn.in_proj_{field}.weight" for field in ("b", "a"))
    for name in names:
        sources[name] = sources[name].to(dtype)
    loaded = weights.ModelWeights.load(manifest, "cpu")
    combined = loaded.projections["layers.0.linear_attn.in_proj_ba"]
    assert combined.dtype == dtype
    assert torch.equal(combined, torch.cat([sources[name] for name in names]))
    for field, name in zip(("b", "a"), names, strict=True):
        assert copies.count(name) == 1
        assert f"layers.0.linear_attn.in_proj_{field}" not in loaded.projections
        parameter = getattr(loaded.tree.layers[0].linear_attn, "in_proj_" + field).weight
        assert torch.equal(parameter, sources[name])
        assert parameter.untyped_storage().data_ptr() == combined.untyped_storage().data_ptr()


def test_online_ba_conversion_receives_only_the_grouped_matrix(checkpoint, monkeypatch):
    manifest, sources, copies = checkpoint
    converted = []

    def convert(tensor):
        converted.append(tensor)
        return SimpleNamespace(out_features=tensor.shape[0], in_features=tensor.shape[1], source=tensor)

    monkeypatch.setattr(weights, "quantize_bf16_linear_as_mxfp8", convert)
    loaded = weights.ModelWeights.load(manifest, "cpu", online_mxfp8=True)
    combined = loaded.projections["layers.0.linear_attn.in_proj_ba"].source
    names = [f"{TEXT_PREFIX}.layers.0.linear_attn.in_proj_{field}.weight" for field in ("b", "a")]
    assert torch.equal(combined, torch.cat([sources[name] for name in names]))
    assert all(copies.count(name) == 1 for name in names)
    assert [tensor.shape[0] for tensor in converted].count(48) == 0
    assert [tensor.shape[0] for tensor in converted].count(96) == 1
    assert loaded.tree.layers[0].linear_attn.in_proj_a.weight.is_meta
    assert loaded.tree.layers[0].linear_attn.in_proj_b.weight.is_meta


@pytest.mark.parametrize("shapes", [((49, 128), (47, 128)), ((48, 64), (48, 64)), ((48, 128, 1), (48, 128))])
def test_grouped_components_reject_individual_shape_mismatches(checkpoint, shapes):
    manifest, sources, _ = checkpoint
    for field, shape in zip(("b", "a"), shapes, strict=True):
        name = f"{TEXT_PREFIX}.layers.0.linear_attn.in_proj_{field}.weight"
        sources[name] = torch.zeros(shape, dtype=torch.bfloat16)
        manifest.tensors[name] = SimpleNamespace(shape=shape)
    with pytest.raises(ValueError, match="checkpoint projection shape differs|missing quantized_layers declaration"):
        weights.ModelWeights.load(manifest, "cpu")


@pytest.mark.parametrize("fmt,fields,shapes", [
    ("MXFP8", ("qkv", "z"), ((10241, 128), (6143, 128))),
    ("NVFP4", ("b", "a"), ((48, 128), (48, 128))),
    ("MXFP4", ("b", "a"), ((48, 128), (48, 128))),
])
def test_quantized_group_members_validate_logical_geometry(checkpoint, fmt, fields, shapes):
    manifest, sources, _ = checkpoint
    path = manifest.snapshot / "config.json"
    raw = json.loads(path.read_text())
    quantization = raw["quantization_config"]["quantized_layers"]
    for field, shape in zip(fields, shapes, strict=True):
        prefix = f"{TEXT_PREFIX}.layers.0.linear_attn.in_proj_{field}"
        quantization[prefix] = dict(quant_algo=fmt, group_size=16 if fmt == "NVFP4" else 32)
        sources[prefix + ".weight"] = torch.empty(shape, device="meta", dtype=(
            torch.float8_e4m3fn if fmt == "MXFP8" else torch.uint8))
        manifest.tensors[prefix + ".weight"] = SimpleNamespace(shape=shape)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="checkpoint projection shape differs"):
        weights.ModelWeights.load(manifest, "cpu")


@pytest.mark.parametrize("formats,expected", [
    (("MXFP8", "MXFP8", None, None), ("ba", "qkvz")),
    (("FP8", "FP8", None, None), ("ba",)),
    (("MXFP8", "FP8", None, None), ("ba",)),
    ((None, None, None, None), ("ba",)),
    (("MXFP8", "MXFP8", "MXFP8", "MXFP8"), ("qkvzba",)),
])
def test_gdn_group_plan_preserves_format_eligibility(formats, expected):
    prefix = f"{TEXT_PREFIX}.layers.0.linear_attn.in_proj_"
    quantization = {prefix + field: dict(quant_algo=fmt, **({"group_size": 32} if fmt == "MXFP8" else {}))
                    for field, fmt in zip(("qkv", "z", "b", "a"), formats, strict=True) if fmt is not None}
    config = ModelConfig(dict(layer_types=["linear_attention"]), quantization)
    loader = DirectWeightLoader(SimpleNamespace(), "meta")
    loader.quantization = quantization
    loader._sources = {prefix + field + ".weight": torch.empty((128, 128), device="meta", dtype=torch.bfloat16)
                       for field in ("qkv", "z", "b", "a")}
    groups = weights._gdn_projection_groups(config, loader)
    assert tuple(groups) == tuple("layers.0.linear_attn.in_proj_" + field for field in expected)
    assert tuple(part for parts in groups.values() for part in parts) == tuple(
        prefix + field for group in expected for field in {
            "ba": ("b", "a"), "qkvz": ("qkv", "z"), "qkvzba": ("qkv", "z", "b", "a")}[group])


def test_four_way_grouping_requires_all_gdn_layers_to_have_mxfp8_inputs():
    fields = ("qkv", "z", "b", "a")
    quantization = {f"{TEXT_PREFIX}.layers.0.linear_attn.in_proj_{field}":
                    dict(quant_algo="MXFP8", group_size=32) for field in fields}
    config = ModelConfig(dict(layer_types=["linear_attention", "linear_attention", "full_attention"]), quantization)
    loader = DirectWeightLoader(SimpleNamespace(), "meta")
    loader.quantization = quantization
    loader._sources = {f"{TEXT_PREFIX}.layers.{layer}.linear_attn.in_proj_{field}.weight":
                       torch.empty((128, 128), device="meta", dtype=torch.bfloat16)
                       for layer in (0, 1) for field in fields}
    assert set(weights._gdn_projection_groups(config, loader)) == {
        "layers.0.linear_attn.in_proj_ba", "layers.0.linear_attn.in_proj_qkvz",
        "layers.1.linear_attn.in_proj_ba"}


@pytest.mark.parametrize("formats,expected", [
    (("MXFP8", "MXFP8", None, None), ("ba", "qkvz")),
    (("FP8", "FP8", None, None), ("qkv", "z", "ba")),
    (("MXFP8", "FP8", None, None), ("qkv", "z", "ba")),
    (("MXFP8", "MXFP8", "MXFP8", "MXFP8"), ("qkvzba",)),
    ((None, None, "NVFP4", "NVFP4"), ("qkv", "z", "ba")),
])
def test_quantized_load_plan_never_loads_replaced_individual_projections(checkpoint, monkeypatch, formats, expected):
    from ignition.preparation.quantization import projection_group_format

    manifest, sources, copies = checkpoint
    path = manifest.snapshot / "config.json"
    raw = json.loads(path.read_text())
    prefix = f"{TEXT_PREFIX}.layers.0.linear_attn.in_proj_"
    quantization = {}
    for field, fmt in zip(("qkv", "z", "b", "a"), formats, strict=True):
        if fmt is None:
            continue
        quantization[prefix + field] = dict(quant_algo=fmt)
        if fmt in ("MXFP8", "NVFP4"):
            quantization[prefix + field]["group_size"] = 32 if fmt == "MXFP8" else 16
        name = prefix + field + ".weight"
        rows, columns = sources[name].shape
        shape = (rows, columns // 2 if fmt == "NVFP4" else columns)
        sources[name] = torch.empty(shape, device="meta", dtype=torch.uint8 if fmt == "NVFP4" else torch.float8_e4m3fn)
        manifest.tensors[name] = SimpleNamespace(shape=shape)
    raw["quantization_config"]["quantized_layers"] = quantization
    path.write_text(json.dumps(raw))
    original = DirectWeightLoader.load_linear_group
    groups = []

    def pack(loader, parts, **kwargs):
        if not any(part in quantization for part in parts):
            return original(loader, parts, **kwargs)
        fmt = projection_group_format(quantization, parts)
        groups.append(tuple(part.removeprefix(prefix) for part in parts))
        copies.extend(part + ".weight" for part in parts)
        return SimpleNamespace(out_features=sum(sources[part + ".weight"].shape[0] for part in parts),
            in_features=sources[parts[0] + ".weight"].shape[1] * (2 if fmt == "nvfp4" else 1))

    monkeypatch.setattr(DirectWeightLoader, "load_linear_group", pack)
    loaded = weights.ModelWeights.load(manifest, "cpu")
    projected = tuple(name.removeprefix("layers.0.linear_attn.in_proj_")
                      for name in loaded.projections if name.startswith("layers.0.linear_attn.in_proj_"))
    assert projected == expected
    assert all(copies.count(prefix + field + ".weight") == 1 for field in ("qkv", "z", "b", "a"))
    if "qkvz" in expected:
        assert ("qkv", "z") in groups
        assert ("qkv",) not in groups and ("z",) not in groups
    if "qkvzba" in expected:
        assert groups == [("qkv", "z", "b", "a")]
