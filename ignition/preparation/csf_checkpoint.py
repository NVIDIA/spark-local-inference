# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Read lossless MXFP4-CSF containers without restoring expanded checkpoints."""

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from .checkpoint import CheckpointError, IndexedCheckpoint, _parse_json, _read_header


@dataclass(frozen=True, slots=True)
class Mxfp4CsfCheckpoint(IndexedCheckpoint):
    compressed_scales: frozenset[str]

    @property
    def metadata_path(self) -> Path:
        return self.snapshot / "metadata"

    @property
    def index_path(self) -> None:
        # The source index describes logical scales; stored headers describe their encoded planes.
        return None


def inspect_mxfp4_csf(snapshot: Path, description: dict) -> Mxfp4CsfCheckpoint:
    if description.get("codec") != "row-base-offset1-u24-exceptions/1":
        raise CheckpointError("unsupported MXFP4-CSF scale codec")
    metadata = snapshot / "metadata"
    hashes = description.get("metadata_sha256")
    if not isinstance(hashes, dict):
        raise CheckpointError("MXFP4-CSF metadata fingerprints are required")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        raw = (metadata / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != hashes.get(name):
            raise CheckpointError(f"MXFP4-CSF metadata differs from its manifest: {name}")
    index = _parse_json((metadata / "model.safetensors.index.json").read_bytes(), "source index")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise CheckpointError("MXFP4-CSF requires a nonempty source weight map")
    files = set()
    for name, file in mapping.items():
        if not isinstance(name, str) or not name or not isinstance(file, str):
            raise CheckpointError("invalid MXFP4-CSF source index entry")
        path = PurePosixPath(file)
        if path.is_absolute() or ".." in path.parts or "\\" in file or not file.endswith(".safetensors"):
            raise CheckpointError("MXFP4-CSF shard path must stay inside the tensor directory")
        files.add(file)
    inventory = description.get("shards")
    if (not isinstance(inventory, list) or len(inventory) != len(files)
            or any(not isinstance(item, dict) or not isinstance(item.get("file"), str) for item in inventory)):
        raise CheckpointError("MXFP4-CSF shard inventory differs from its source index")
    declared = {item["file"]: item for item in inventory}
    if set(declared) != files:
        raise CheckpointError("MXFP4-CSF shard names differ from its source index")
    tensors, shards, logical, compressed = {}, [], set(), set()
    fixed_suffix, exceptions_suffix = ".mxfp4_csf_fixed", ".mxfp4_csf_exceptions"
    for file in sorted(files):
        relative = "tensors/" + file
        record, entries = _read_header(snapshot / relative, relative)
        if record.file_size != declared[file].get("target_file_bytes"):
            raise CheckpointError(f"MXFP4-CSF shard size differs from its manifest: {file}")
        for name, tensor in entries.items():
            if name in tensors:
                raise CheckpointError(f"duplicate MXFP4-CSF tensor: {name}")
            encoded = name.endswith((fixed_suffix, exceptions_suffix))
            original = name.rsplit(".mxfp4_csf_", 1)[0] if encoded else name
            if mapping.get(original) != file:
                raise CheckpointError(f"MXFP4-CSF tensor is outside its source inventory: {name}")
            logical.add(original)
            if not name.endswith(fixed_suffix):
                continue
            other = entries.get(original + exceptions_suffix)
            weight = entries.get(original.removesuffix(".scale") + ".weight")
            if (not original.endswith(".scale") or original in entries or other is None
                    or weight is None or weight.dtype not in ("I8", "U8") or len(weight.shape) != 2):
                raise CheckpointError(f"MXFP4-CSF scale needs a unique packed weight and encoded pair: {original}")
            rows, packed_columns = weight.shape
            if rows <= 0 or rows % 32 or packed_columns <= 0 or packed_columns % 16:
                raise CheckpointError(f"invalid MXFP4-CSF matrix geometry: {original}")
            columns = packed_columns // 16
            expected = rows * (1 + (columns + 7) // 8)
            if tensor.dtype != "U8" or tensor.shape != (expected,) or other.dtype != "U32" or len(other.shape) != 1:
                raise CheckpointError(f"invalid MXFP4-CSF scale plane geometry: {original}")
            compressed.add(original)
        tensors.update(entries)
        shards.append(record)
    if logical != mapping.keys() or len(compressed) != description.get("compressed_scale_count"):
        raise CheckpointError("MXFP4-CSF tensor coverage differs from its declared source inventory")
    if any(name.endswith(exceptions_suffix) and name.removesuffix(exceptions_suffix) not in compressed for name in tensors):
        raise CheckpointError("MXFP4-CSF exception plane lacks its fixed plane")
    if sum(shard.file_size for shard in shards) != description.get("compressed_weight_file_bytes"):
        raise CheckpointError("MXFP4-CSF stored bytes differ from its shard inventory")
    return Mxfp4CsfCheckpoint(snapshot, MappingProxyType(tensors), tuple(shards), frozenset(compressed))
