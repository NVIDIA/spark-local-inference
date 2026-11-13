# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inspect the logical source inventory and physical lossless scale container."""

import hashlib
import json

import pytest

from ignition.preparation.checkpoint import CheckpointError, inspect_indexed_checkpoint


def container(root, *, change=None):
    metadata, tensors = root / "metadata", root / "tensors"
    metadata.mkdir()
    tensors.mkdir()
    file = "weights.safetensors"
    index = {"weight_map": {"experts.0.w1.weight": file, "experts.0.w1.scale": file}}
    assets = {"config.json": {"architectures": ["DeepseekV41ForCausalLM"]}, "tokenizer.json": {}, "tokenizer_config.json": {},
              "model.safetensors.index.json": index}
    hashes = {}
    for name, value in assets.items():
        data = json.dumps(value).encode()
        (metadata / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    header = {
        "experts.0.w1.weight": {"dtype": "I8", "shape": [32, 64], "data_offsets": [0, 2048]},
        "experts.0.w1.scale.mxfp4_csf_fixed": {"dtype": "U8", "shape": [64], "data_offsets": [2048, 2112]},
        "experts.0.w1.scale.mxfp4_csf_exceptions": {"dtype": "U32", "shape": [1], "data_offsets": [2112, 2116]},
    }
    if change is not None:
        change(header)
    raw = json.dumps(header).encode()
    (tensors / file).write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(2116))
    manifest = dict(schema="lil-mxfp4-csf-checkpoint/1", codec="row-base-offset1-u24-exceptions/1",
        compressed_scale_count=1, metadata_sha256=hashes,
        compressed_weight_file_bytes=(tensors / file).stat().st_size,
        shards=[dict(file=file, target_file_bytes=(tensors / file).stat().st_size)])
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_container_keeps_physical_paths_and_logical_scale_identity(tmp_path):
    container(tmp_path)
    result = inspect_indexed_checkpoint(tmp_path)
    assert result.compressed_scales == {"experts.0.w1.scale"}
    assert result.metadata_path == tmp_path / "metadata"
    assert result.index_path is None
    assert result.shard_paths == (tmp_path / "tensors/weights.safetensors",)
    assert result.tensors["experts.0.w1.scale.mxfp4_csf_fixed"].shape == (64,)


@pytest.mark.parametrize("field,value", [("compressed_scale_count", 2), ("codec", "unknown")])
def test_container_rejects_manifest_inconsistency(tmp_path, field, value):
    manifest = container(tmp_path)
    manifest[field] = value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CheckpointError):
        inspect_indexed_checkpoint(tmp_path)


def test_container_rejects_changed_source_metadata(tmp_path):
    container(tmp_path)
    (tmp_path / "metadata/config.json").write_text('{"changed": true}')
    with pytest.raises(CheckpointError, match="metadata differs"):
        inspect_indexed_checkpoint(tmp_path)


def test_container_rejects_scale_geometry(tmp_path):
    container(tmp_path, change=lambda h: h["experts.0.w1.scale.mxfp4_csf_fixed"].update(shape=[32, 2]))
    with pytest.raises(CheckpointError, match="scale plane geometry"):
        inspect_indexed_checkpoint(tmp_path)


def test_container_rejects_unmapped_physical_tensor(tmp_path):
    def rename(header):
        header["unexpected"] = header.pop("experts.0.w1.scale.mxfp4_csf_exceptions")
    container(tmp_path, change=rename)
    with pytest.raises(CheckpointError, match="encoded pair|source inventory"):
        inspect_indexed_checkpoint(tmp_path)


def test_local_model_resolution_uses_container_metadata_and_checks_missing_shards(tmp_path):
    from ignition.recipes import resolve

    declared = container(tmp_path)
    model = resolve(str(tmp_path))
    assert model.recipe.name == "deepseek41"
    assert model.checkpoint.stored_bytes == declared["compressed_weight_file_bytes"]
    model.verify(tmp_path)
    assert model.cached(tmp_path)
    (tmp_path / "tensors/weights.safetensors").unlink()
    assert not model.cached(tmp_path)
    # Metadata verification precedes downloading missing tensors.
    model.verify(tmp_path)
