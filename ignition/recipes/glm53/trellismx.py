# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 Flash TrellisMX release: detection, sidecar experts, and MTP layer 45.

The release is a routed-expert overlay on a pinned carrier. Every tensor except
the routed experts is read from ``<root>/carrier`` exactly as stored, and each
routed layer takes its experts from ``<root>/sidecars``: one file per layer and
tensor-parallel rank, carrying the eight tensors of the coupled H512/H128
contract. The two trellis payloads are already b12x's assembled ``w13``/``w2``
int16 carriers, so nothing here re-encodes, requantizes, rounds or casts a
codeword or a scale: the only numeric-adjacent operations are ``.contiguous()``
and host-to-device copies.

The codec is procedural NPK (multiplier ``0xCBAC1FED``, mask/or
``0x8FFF8FFF``/``0x3B603B60``, two fp16 halves added), which the sidecar
metadata records under its own older label. Preparation of the prepared expert
bundle belongs to b12x's fused-MoE ``trellismx`` source format; this module
supplies the validated sidecar tensors and metadata and hands them over.

Every sidecar is read once during preparation, checked against the size and
SHA-256 the release manifest declares, and parsed and copied from those same
bytes. Serving never reads a sidecar: an exported package holds the prepared
tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ignition.preparation.trellismx_release import (
    CARRIER_DIRECTORY, MANIFEST_NAME, MANIFEST_SCHEMA, TRANSFORM_PATH, sidecar_path,
)


if TYPE_CHECKING:
    import torch
    from b12x.moe.fused_moe import PreparedExperts

CHECKPOINT_ID = "codec-v2-uniform-k4-mtp45-k4"
SIDECAR_SCHEMA = "glm53-p8-coupled-h512-h128-tp4-rank.v1"

# Geometry the recipe and every sidecar's metadata agree on.
HIDDEN_SIZE = 4096
GLOBAL_INTERMEDIATE = 2048
NUM_EXPERTS = 288
TOP_K = 8
WORLD_SIZE = 4
BITS = 4
FIRST_ROUTED_LAYER = 3
MTP_LAYER = 45
SLOT_CHANNELS = 32
# GLM-5.3's SwiGLU clamp. The sidecars were encoded for exactly this activation
# (contract ``activation: silu-cap10``), so preparation accepts no other value.
SWIGLU_LIMIT = 10.0

# One sidecar holds exactly these tensors, with these safetensors dtypes.
TENSOR_DTYPES = {
    "w13_trellis": "I16",
    "w2_trellis": "I16",
    "w13_scale_ue8m0": "U8",
    "w2_scale_ue8m0": "U8",
    "intermediate_scales_fp16": "F16",
    "coupled_sign_draw_u8": "U8",
    "gate_up_suh_fp16": "F16",
    "down_svh_fp16": "F16",
}
_NUMPY_DTYPES = {"I16": "int16", "U8": "uint8", "F16": "float16"}
_ELEMENT_BYTES = {"I16": 2, "U8": 1, "F16": 2}
# A sidecar header is a few KiB; anything near this bound is not one.
_MAX_HEADER_BYTES = 100 << 20

# Release gates 5: the per-sidecar metadata the release's own loader checks.
SIDECAR_IDENTITY = {
    "schema": SIDECAR_SCHEMA,
    "bits": str(BITS),
    "world_size": str(WORLD_SIZE),
    "tp_slice": "contiguous-atom32-v1",
    "global_intermediate": str(GLOBAL_INTERMEDIATE),
}

# The coupled transform the sidecar tensors are encoded under. b12x's trellismx
# preparation implements exactly this contract, so a different value here means
# the payloads are not what that preparation consumes.
SIDECAR_CONTRACT = {
    "law": "procedural-mcg-alpha2",
    "alphabet": "e4m3",
    "scale": "ue8m0-k32",
    "signed_scales": "true",
    "ldlq": "false",
    "activation": "silu-cap10",
    "h512": "normalized-sylvester-512-v1",
    "h128": "normalized-sylvester-128-v1",
    "boundary": "coupled-h512-h128-suh-svh-v1",
    "composition_target": "coupled-h512-h128-suh-svh-v1",
    "sign_generator": "qsrt-coupled-signs-v1",
    "sign_draw": "0",
    "sign_pre_axis": "1",
    "sign_post_axis": "2",
    "coupled_signs_shared": "true",
    "gate_up_suh_shared": "true",
    "down_svh_shared": "true",
    "coupled_scale_order": "gate_svh-up_svh-down_suh",
    "fc1_scale_plane_order": "up-gate",
    "fc1_trellis_slot_order": "gate-up",
    "fc1_interleave": "slot0-atom32-slot1-atom32-v1",
    "quantized_input_order": "bf16-cvt-rn-fp16-h512-fp32-suh-fp32-h128-fp32-e4m3-ue8m0-k32",
    "quantized_down_order": "silu-cap10-fp32-sign-h128-fp32-down-suh-fp32-h128-fp32-e4m3-ue8m0-k32",
    # The P8 runtime's full-coupled validator requires these too; checking them here keeps a
    # sidecar from passing CPU preparation only to fail when the runtime is attached.
    "component": "p8-coupled-h512-h128-sign-v1",
    "cast_order": "luke-qsrt-coupled-reference-v1",
    "full_coupled": "true",
    "transform_id": "normalized-h512-outer-h128-inner-sign-draw0-v1",
}

# The quantized_layers entries the carrier's config.json declares, exactly: only the
# routed experts are quantized, so the sidecars replace exactly those tensors and every
# other projection is stored. The MTP layer's declared MXFP8 is superseded by its sidecar
# exactly as the main layers' NVFP4 is.
_QUANTIZED_LAYERS = {layer: {"quant_algo": "NVFP4", "group_size": 16}
                     for layer in range(FIRST_ROUTED_LAYER, MTP_LAYER)}
_QUANTIZED_LAYERS[MTP_LAYER] = {"quant_algo": "MXFP8", "group_size": 32}


@dataclass(frozen=True)
class TrellisMxSidecar:
    """One layer and rank's sidecar file as the release manifest declares it."""

    layer: int
    rank: int
    path: Path
    bytes: int
    sha256: str
    source_design_sha256: str


@dataclass(frozen=True)
class TrellisMxRelease:
    """A validated TrellisMX overlay release and where each of its files lives."""

    root: Path
    manifest_sha256: str
    layers: tuple[int, ...]
    world_size: int
    bits: int
    mtp_layer: int
    carrier_repository: str
    carrier_revision: str
    files: tuple[TrellisMxSidecar, ...]
    # SHA-256 of design/transform.json, the encoder transform every sidecar records.
    transform_sha256: str

    @property
    def carrier_root(self) -> Path:
        """The directory holding every tensor the sidecars do not replace."""
        return self.root / CARRIER_DIRECTORY

    @property
    def routed_layers(self) -> tuple[int, ...]:
        return tuple(layer for layer in self.layers if layer != self.mtp_layer)

    def sidecar(self, layer: int, rank: int) -> TrellisMxSidecar:
        for entry in self.files:
            if entry.layer == layer and entry.rank == rank:
                return entry
        raise ValueError(f"trellismx release {self.root} has no sidecar for layer {layer} rank {rank}")

    def local_intermediate_size(self) -> int:
        """Intermediate channels one rank holds; the intermediate axis is cut into 32-channel slots."""
        return GLOBAL_INTERMEDIATE // self.world_size


def _gate(number, message: str) -> ValueError:
    return ValueError(f"trellismx release gate {number}: {message}")


def _read_bytes(path: Path, what: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"{what} at {path} is unreadable: {error}") from error


def _json_document(raw: bytes, path: Path, what: str) -> dict:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{what} at {path} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{what} at {path} must be a JSON object")
    return value


def _json_object(path: Path, what: str) -> dict:
    return _json_document(_read_bytes(path, what), path, what)


def _sha256_field(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _int_field(value, what: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{what} must be an integer, got {value!r}")
    return value


def _manifest_files(manifest: dict, root: Path, layers: tuple[int, ...], world_size: int) -> tuple[TrellisMxSidecar, ...]:
    """Gate 4: one entry per layer and rank, named and sized as the release declares."""
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise _gate(4, "manifest files must be a list")
    if len(entries) != len(layers) * world_size:
        raise _gate(4, f"manifest lists {len(entries)} sidecars for {len(layers)} layers"
                       f" and {world_size} ranks, expected {len(layers) * world_size}")
    seen, files = set(), []
    for entry in entries:
        if not isinstance(entry, dict):
            raise _gate(4, f"sidecar entry must be an object, got {entry!r}")
        layer = _int_field(entry.get("layer"), "sidecar entry layer")
        rank = _int_field(entry.get("rank"), "sidecar entry rank")
        expected_path = sidecar_path(layer, rank, world_size)
        if entry.get("path") != expected_path:
            raise _gate(4, f"sidecar path {entry.get('path')!r} does not name layer {layer} rank {rank}"
                           f" ({expected_path})")
        if (layer, rank) in seen:
            raise _gate(4, f"duplicate sidecar for layer {layer} rank {rank}")
        if layer not in layers or not 0 <= rank < world_size:
            raise _gate(4, f"sidecar for layer {layer} rank {rank} is outside the declared allocation")
        if entry.get("bits") != BITS:
            raise _gate(4, f"{expected_path}: bits must be {BITS}, got {entry.get('bits')!r}")
        size = _int_field(entry.get("bytes"), f"{expected_path} bytes")
        digest = entry.get("sha256")
        design = entry.get("source_design_sha256")
        if not _sha256_field(digest) or not _sha256_field(design):
            raise _gate(4, f"{expected_path}: file and source-design SHA-256 must be 64 lowercase hex digits")
        path = root / expected_path
        if not path.is_file():
            raise _gate(4, f"missing sidecar {path}")
        actual = path.stat().st_size
        if actual != size:
            raise _gate(4, f"{path}: {actual} bytes, manifest declares {size}")
        seen.add((layer, rank))
        files.append(TrellisMxSidecar(layer, rank, path, size, digest, design))
    files.sort(key=lambda entry: (entry.layer, entry.rank))
    return tuple(files)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _header_length(prefix: bytes, size: int, path) -> int:
    length = int.from_bytes(prefix, "little")
    if len(prefix) != 8 or not 0 < length <= min(_MAX_HEADER_BYTES, size - 8):
        raise ValueError(f"sidecar {path} has an invalid safetensors header length")
    return length


def _header_entries(raw: bytes, size: int, path) -> tuple[dict[str, str], dict[str, tuple[str, tuple[int, ...], int]]]:
    """Validate one safetensors header against a file of ``size`` bytes.

    Returns the string metadata and each tensor's dtype, shape and absolute byte
    offset. The tensors must cover the payload exactly, in order, without holes
    or overlap, so no byte of the file is outside a declared tensor.
    """
    try:
        header = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"sidecar {path} header is unreadable: {error}") from error
    if not isinstance(header, dict):
        raise ValueError(f"sidecar {path} header must be a JSON object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(value, str) for value in metadata.values()):
        raise ValueError(f"sidecar {path} metadata must map names to strings")
    begin, entries = 8 + len(raw), {}
    for name, entry in header.items():
        dtype, shape, offsets = ((entry.get("dtype"), entry.get("shape"), entry.get("data_offsets"))
                                 if isinstance(entry, dict) else (None, None, None))
        if (dtype not in _ELEMENT_BYTES or not isinstance(shape, list)
                or any(type(value) is not int or value < 0 for value in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int or value < 0 for value in offsets)
                or offsets[1] - offsets[0] != math.prod(shape) * _ELEMENT_BYTES[dtype]):
            raise ValueError(f"sidecar {path}: invalid dtype, shape or byte range for tensor {name!r}")
        entries[name] = (dtype, tuple(shape), begin + offsets[0])
    end = begin
    for name, (dtype, shape, offset) in sorted(entries.items(), key=lambda item: item[1][2]):
        if offset != end:
            raise ValueError(f"sidecar {path}: overlap or hole before tensor {name!r}")
        end += math.prod(shape) * _ELEMENT_BYTES[dtype]
    if end != size:
        raise ValueError(f"sidecar {path}: tensors cover {end} of its {size} bytes")
    return metadata, entries


def sidecar_header(path: Path) -> tuple[dict[str, str], dict[str, tuple[str, tuple[int, ...]]]]:
    """Read one sidecar's metadata and tensor geometry without reading any payload."""
    try:
        with Path(path).open("rb", buffering=0) as source:
            size = os.fstat(source.fileno()).st_size
            length = _header_length(source.read(8), size, path)
            raw = source.read(length)
    except OSError as error:
        raise ValueError(f"sidecar {path} is unreadable: {error}") from error
    if len(raw) != length:
        raise ValueError(f"sidecar {path} has a truncated safetensors header")
    metadata, entries = _header_entries(raw, size, path)
    return metadata, {name: (dtype, shape) for name, (dtype, shape, _) in entries.items()}


def validate_sidecar_metadata(metadata: dict[str, str], *, layer: int, rank: int, world_size: int,
                              expected_design_sha256: str | None = None,
                              expected_transform_sha256: str | None = None) -> None:
    """Gate 5 and the coupled-transform contract, for one sidecar's metadata."""
    for name, expected in SIDECAR_IDENTITY.items():
        if metadata.get(name) != expected:
            raise _gate(5, f"layer {layer} rank {rank}: metadata {name} is {metadata.get(name)!r},"
                           f" expected {expected!r}")
    if metadata.get("layer") != str(layer) or metadata.get("rank") != str(rank):
        raise _gate(5, f"sidecar metadata names layer {metadata.get('layer')!r} rank {metadata.get('rank')!r},"
                       f" not layer {layer} rank {rank}")
    if metadata.get("local_atom_begin") != str(rank * (GLOBAL_INTERMEDIATE // world_size) // SLOT_CHANNELS):
        raise _gate(5, f"layer {layer} rank {rank}: local_atom_begin is {metadata.get('local_atom_begin')!r},"
                       f" expected {rank * (GLOBAL_INTERMEDIATE // world_size) // SLOT_CHANNELS}")
    for name, expected in SIDECAR_CONTRACT.items():
        if metadata.get(name) != expected:
            raise ValueError(f"trellismx sidecar layer {layer} rank {rank}: {name} is {metadata.get(name)!r},"
                             f" but the coupled contract requires {expected!r}")
    if expected_design_sha256 is not None and metadata.get("source_design_sha256") != expected_design_sha256:
        raise _gate(5, f"layer {layer} rank {rank}: source_design_sha256 is"
                       f" {metadata.get('source_design_sha256')!r}, manifest declares {expected_design_sha256!r}")
    if expected_transform_sha256 is not None and metadata.get("encoder_transform_sha256") != expected_transform_sha256:
        raise _gate(5, f"layer {layer} rank {rank}: encoder_transform_sha256 is"
                       f" {metadata.get('encoder_transform_sha256')!r}, but {TRANSFORM_PATH} hashes to"
                       f" {expected_transform_sha256!r}")


def expected_tensor_shapes(local_intermediate_size: int) -> dict[str, tuple[int, ...]]:
    """The eight sidecar tensors' shapes for one rank's contiguous intermediate slice."""
    if (type(local_intermediate_size) is not int or local_intermediate_size <= 0
            or local_intermediate_size % SLOT_CHANNELS or GLOBAL_INTERMEDIATE % local_intermediate_size):
        raise ValueError(f"trellismx local intermediate size must divide {GLOBAL_INTERMEDIATE} into"
                         f" {SLOT_CHANNELS}-channel slots, got {local_intermediate_size!r}")
    row_groups = 2 * local_intermediate_size // SLOT_CHANNELS
    hidden_tiles = HIDDEN_SIZE // 16
    return {
        "w13_trellis": (2, NUM_EXPERTS, hidden_tiles, row_groups, 16 * BITS),
        "w2_trellis": (NUM_EXPERTS, row_groups, hidden_tiles, 16 * BITS),
        "w13_scale_ue8m0": (NUM_EXPERTS, 2 * local_intermediate_size, HIDDEN_SIZE // 32),
        "w2_scale_ue8m0": (NUM_EXPERTS, HIDDEN_SIZE, local_intermediate_size // 32),
        "intermediate_scales_fp16": (NUM_EXPERTS, 3 * local_intermediate_size),
        "coupled_sign_draw_u8": (NUM_EXPERTS,),
        "gate_up_suh_fp16": (HIDDEN_SIZE,),
        "down_svh_fp16": (HIDDEN_SIZE,),
    }


def validate_sidecar_tensors(tensors: dict[str, tuple[str, tuple[int, ...]]], *, layer: int, rank: int,
                             local_intermediate_size: int) -> None:
    """The shape and dtype gate of the release contract, without reading payloads."""
    if set(tensors) != set(TENSOR_DTYPES):
        missing = sorted(set(TENSOR_DTYPES) - set(tensors))
        extra = sorted(set(tensors) - set(TENSOR_DTYPES))
        raise ValueError(f"trellismx sidecar layer {layer} rank {rank} tensor set differs from the release"
                         f" contract: missing {missing}, unexpected {extra}")
    for name, shape in expected_tensor_shapes(local_intermediate_size).items():
        dtype, actual = tensors[name]
        if dtype != TENSOR_DTYPES[name] or actual != shape:
            raise ValueError(f"trellismx sidecar layer {layer} rank {rank}: {name} is {dtype} {actual},"
                             f" expected {TENSOR_DTYPES[name]} {shape}")


def _config_gates(config: dict) -> None:
    """Gate 6: the architecture and geometry the recipe prepares, and which tensors are quantized."""
    if config.get("architectures") != ["Glm5NextForConditionalGeneration"]:
        raise _gate(6, f"config.json architectures is {config.get('architectures')!r},"
                       " expected ['Glm5NextForConditionalGeneration']")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise _gate(6, "config.json has no text_config")
    for name, expected in {"hidden_size": HIDDEN_SIZE, "moe_intermediate_size": GLOBAL_INTERMEDIATE,
                           "n_routed_experts": NUM_EXPERTS, "swiglu_limit": SWIGLU_LIMIT}.items():
        if text.get(name) != expected:
            raise _gate(6, f"config.json text_config.{name} is {text.get(name)!r}, expected {expected!r}")
    from ignition.preparation.quantization import quantized_layers

    # ModelOpt MIXED_PRECISION with unquantized KV, the form Flare's loaders read.
    try:
        quantized = quantized_layers(config)
    except ValueError as error:
        raise _gate(6, f"config.json quantization_config: {error}") from error
    expected = {f"model.language_model.layers.{layer}.mlp.experts": entry
                for layer, entry in _QUANTIZED_LAYERS.items()}
    # Every entry is compared whole, so a malformed or extended declaration is rejected too.
    actual = dict(quantized)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
        raise _gate(6, "quantized_layers must declare exactly the routed expert modules the sidecars replace,"
                       f" each as {_QUANTIZED_LAYERS[FIRST_ROUTED_LAYER]} (layer {MTP_LAYER}:"
                       f" {_QUANTIZED_LAYERS[MTP_LAYER]}): missing {missing}, unexpected {unexpected},"
                       f" changed {changed}")


def is_trellismx_overlay(root: str | Path) -> TrellisMxRelease | None:
    """Return the validated release rooted at ``root``, or None when ``root`` is not one.

    None means only that ``root`` carries no TrellisMX overlay manifest, or one
    another recipe declares (the manifest's ``schema`` names its owner). Every
    other failure of the release contract raises, naming the gate: a directory
    that declares this schema but does not satisfy it must never fall through to
    another weight format. Detection reads every sidecar's header but no payload;
    each payload's digest is checked when preparation loads it.
    """
    root = Path(root).expanduser()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    # One read: the recorded identity is the digest of the very bytes every gate parses.
    manifest_bytes = _read_bytes(manifest_path, "trellismx manifest")
    manifest = _json_document(manifest_bytes, manifest_path, "trellismx manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        return None

    # Gate 2: the uniform K4 checkpoint identity and its per-layer rank allocation.
    if manifest.get("checkpoint") != CHECKPOINT_ID:
        raise _gate(2, f"checkpoint is {manifest.get('checkpoint')!r}, expected {CHECKPOINT_ID!r}")
    allocation = manifest.get("allocation")
    if not isinstance(allocation, dict) or not allocation:
        raise _gate(2, "allocation must be a nonempty object of layer indices")
    allocated = {}
    for layer, ranks in allocation.items():
        try:
            index = int(layer)
        except (TypeError, ValueError) as error:
            raise _gate(2, f"allocation keys must be layer indices, got {layer!r}") from error
        # One spelling per layer: "03" beside "3" would silently merge two allocations.
        if layer != str(index):
            raise _gate(2, f"allocation key {layer!r} is not the canonical layer index {str(index)!r}")
        allocated[index] = _int_field(ranks, f"allocation[{layer}]")
    if set(allocated) != set(range(FIRST_ROUTED_LAYER, MTP_LAYER + 1)):
        raise _gate(2, f"allocation covers layers {sorted(allocated)}, expected"
                       f" {FIRST_ROUTED_LAYER} through {MTP_LAYER}")
    if set(allocated.values()) != {WORLD_SIZE}:
        raise _gate(2, f"allocation world sizes are {sorted(set(allocated.values()))}, expected {WORLD_SIZE}")
    mtp = manifest.get("mtp")
    if not isinstance(mtp, dict) or mtp.get("layer") != MTP_LAYER or mtp.get("bits") != BITS:
        raise _gate(2, f"mtp must declare layer {MTP_LAYER} at {BITS} bits, got {mtp!r}")

    # Gate 3: the pinned carrier this release overlays.
    carrier = manifest.get("carrier")
    if not isinstance(carrier, dict):
        raise _gate(3, "manifest carrier must be an object")
    repository, revision = carrier.get("repo_id"), carrier.get("revision")
    if (not isinstance(repository, str) or not repository
            or not isinstance(revision, str) or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)):
        raise _gate(3, f"manifest carrier must pin a repository and a 40-digit revision, got {carrier!r}")
    carrier_root = root / CARRIER_DIRECTORY
    for name in ("model.safetensors.index.json", "config.json"):
        if not (carrier_root / name).is_file():
            raise _gate(3, f"carrier directory {carrier_root} has no {name}")
    carrier_config = _json_object(carrier_root / "config.json", "carrier config.json")
    root_config = _json_object(root / "config.json", "release config.json")
    if carrier_config != root_config:
        raise _gate(3, "the release's config.json differs from the carrier's: the root does not describe"
                       " the carrier the sidecars overlay")

    # Gate 6 before the 172 sidecar headers: a wrong config rejects the release fastest.
    _config_gates(root_config)

    layers = tuple(sorted(allocated))
    files = _manifest_files(manifest, root, layers, WORLD_SIZE)
    if not (root / TRANSFORM_PATH).is_file():
        raise _gate(4, f"missing {root / TRANSFORM_PATH}, the encoder transform the sidecars pin")
    transform_sha256 = hashlib.sha256((root / TRANSFORM_PATH).read_bytes()).hexdigest()

    # Gate 5: every sidecar's own metadata, checked against the manifest entry.
    for entry in files:
        metadata, tensors = sidecar_header(entry.path)
        validate_sidecar_metadata(metadata, layer=entry.layer, rank=entry.rank, world_size=WORLD_SIZE,
                                  expected_design_sha256=entry.source_design_sha256,
                                  expected_transform_sha256=transform_sha256)
        validate_sidecar_tensors(tensors, layer=entry.layer, rank=entry.rank,
                                 local_intermediate_size=GLOBAL_INTERMEDIATE // WORLD_SIZE)

    return TrellisMxRelease(
        root=root, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        layers=layers, world_size=WORLD_SIZE, bits=BITS, mtp_layer=MTP_LAYER,
        carrier_repository=repository, carrier_revision=revision, files=files, transform_sha256=transform_sha256)


def _read_verified(entry: TrellisMxSidecar) -> bytearray:
    """The sidecar's bytes, read once and checked against the manifest's size and SHA-256.

    The digest covers the header, the metadata and every payload, including the
    two trellis weight payloads, and the caller parses and copies these same
    bytes, so nothing is used that was not verified.
    """
    try:
        with entry.path.open("rb", buffering=0) as source:
            size = os.fstat(source.fileno()).st_size
            if size != entry.bytes:
                raise ValueError(f"trellismx sidecar {entry.path} is {size} bytes, but the release manifest"
                                 f" declares {entry.bytes}")
            buffer = bytearray(size)
            with memoryview(buffer) as view:
                filled = 0
                while filled < size:
                    count = source.readinto(view[filled:])
                    if not count:
                        raise ValueError(f"trellismx sidecar {entry.path} ended after {filled} of {size} bytes")
                    filled += count
    except OSError as error:
        raise ValueError(f"trellismx sidecar {entry.path} is unreadable: {error}") from error
    digest = hashlib.sha256(buffer).hexdigest()
    if digest != entry.sha256:
        raise ValueError(f"trellismx sidecar {entry.path} SHA-256 is {digest}, but the release manifest"
                         f" declares {entry.sha256}")
    return buffer


def _verified_sidecar(release: TrellisMxRelease, layer: int, rank: int):
    """One sidecar verified against the manifest and validated: its entry, metadata and payload arrays.

    The arrays are views of the verified buffer, which they keep alive.
    """
    import numpy

    entry = release.sidecar(layer, rank)
    buffer = _read_verified(entry)
    raw = bytes(buffer[8:8 + _header_length(bytes(buffer[:8]), len(buffer), entry.path)])
    metadata, entries = _header_entries(raw, len(buffer), entry.path)
    validate_sidecar_metadata(metadata, layer=layer, rank=rank, world_size=release.world_size,
                              expected_design_sha256=entry.source_design_sha256,
                              expected_transform_sha256=release.transform_sha256)
    validate_sidecar_tensors({name: (dtype, shape) for name, (dtype, shape, _) in entries.items()},
                             layer=layer, rank=rank, local_intermediate_size=release.local_intermediate_size())
    arrays = {}
    for name, (dtype, shape, offset) in entries.items():
        array = numpy.frombuffer(buffer, dtype=numpy.dtype(_NUMPY_DTYPES[dtype]).newbyteorder("<"),
                                 count=math.prod(shape), offset=offset).reshape(shape)
        # Safetensors stores little-endian values at any byte offset; use them in native order and alignment.
        if not (array.dtype.isnative and array.flags.aligned):
            array = array.astype(array.dtype.newbyteorder("="))
        arrays[name] = array
    return entry, metadata, arrays


def load_sidecar_tensors(release: TrellisMxRelease, *, layer: int, rank: int, device,
                         names: tuple[str, ...] | None = None) -> dict[str, "torch.Tensor"]:
    """Read one sidecar's tensors after verifying the whole file against the release manifest.

    ``names`` selects a subset; the expert preparation uses all eight. Payloads
    are copied to ``device`` unchanged, and CPU tensors view the verified bytes.
    """
    import torch

    if not isinstance(release, TrellisMxRelease):
        raise TypeError("trellismx sidecar loading requires a TrellisMxRelease")
    if layer not in release.layers:
        raise ValueError(f"trellismx release {release.root} has no sidecar for layer {layer}")
    if type(rank) is not int or not 0 <= rank < release.world_size:
        raise ValueError(f"trellismx sidecar rank must be in [0, {release.world_size}), got {rank!r}")
    selected = tuple(TENSOR_DTYPES) if names is None else tuple(names)
    if not selected or any(name not in TENSOR_DTYPES for name in selected):
        raise ValueError(f"trellismx sidecar tensors must be drawn from {sorted(TENSOR_DTYPES)}, got {names!r}")
    _, _, arrays = _verified_sidecar(release, layer, rank)
    return {name: torch.from_numpy(arrays[name]).to(device) for name in selected}


def prepared_metadata(metadata: dict[str, str], *, global_intermediate_size: int) -> dict[str, str]:
    """The sidecar metadata with its rank slice in the source format's units.

    The release names each rank's slice as ``local_atom_begin``, a 32-channel atom
    index (0, 16, 32, 48 at TP4); the source format wants the channel offset. Every
    other field the format validates is either already in the sidecar or follows
    from the geometry the caller declares. A key the sidecar carries is never
    silently replaced: a conflict raises.
    """
    local = global_intermediate_size // WORLD_SIZE
    begin = int(metadata["local_atom_begin"]) * SLOT_CHANNELS
    derived = {"slice_begin": str(begin), "slice_end": str(begin + local)}
    for name, value in derived.items():
        if name in metadata and metadata[name] != value:
            raise ValueError(f"trellismx sidecar metadata {name} is {metadata[name]!r},"
                             f" but the release's own fields imply {value!r}")
    return {**derived, **metadata}


def prepare_trellismx_experts(tensors: dict[str, "torch.Tensor"], metadata: dict[str, str], *,
                              num_experts: int, hidden_size: int, intermediate_size: int, top_k: int,
                              device) -> "PreparedExperts":
    """Hand the validated sidecar to b12x's fused-MoE ``trellismx`` source format.

    Until that format lands in this b12x tree the import fails, naming the module:
    a release whose expert format has no prepared execution must not be served.
    The format fixes its activation; preparation rejects a bundle whose activation
    is not the BF16 SiLU capped at GLM-5.3's clamp that the sidecars encode.
    """
    try:
        from b12x.moe.fused_moe.trellismx import prepare_trellismx_experts as prepare
    except ImportError as error:
        raise RuntimeError(
            "this b12x build has no fused-MoE trellismx source format"
            " (b12x.moe.fused_moe.trellismx.prepare_trellismx_experts), so a TrellisMX release's"
            f" routed experts cannot be prepared: {error}") from error
    prepared = prepare(tensors, metadata, num_experts=num_experts, hidden_size=hidden_size,
                       intermediate_size=intermediate_size, top_k=top_k, device=device)
    activation = prepared.plan.activation
    if (activation.nonlinearity, activation.swiglu_limit, activation.swiglu_clamp_after_silu) != (
            "silu", SWIGLU_LIMIT, False):
        raise ValueError(f"b12x prepared TrellisMX experts with activation {activation}, but the sidecars"
                         f" encode SiLU with GLM-5.3's SwiGLU clamp {SWIGLU_LIMIT}")
    return prepared


def load_trellis_experts(release: TrellisMxRelease, *, layer: int, rank: int, world_size: int,
                         hidden_size: int, global_intermediate_size: int, num_experts: int, top_k: int,
                         swiglu_limit: float, device, params_dtype) -> "PreparedExperts":
    """Prepare one layer and rank's routed experts from its sidecar.

    The sidecar stores the checkpoint-global intermediate axis cut into contiguous
    32-channel atoms: rank ``r`` owns atoms ``[16r, 16r + 16)``. That is exactly
    the contiguous slice this rank's routed contribution covers, so the prepared
    experts describe one rank's partial sum and need no further slicing. The
    sidecar is verified against the release manifest before any of it is used.
    """
    import torch

    if not isinstance(release, TrellisMxRelease):
        raise TypeError("trellismx expert loading requires a TrellisMxRelease")
    for name, actual, expected in (("world_size", world_size, release.world_size),
                                   ("hidden_size", hidden_size, HIDDEN_SIZE),
                                   ("global_intermediate_size", global_intermediate_size, GLOBAL_INTERMEDIATE),
                                   ("num_experts", num_experts, NUM_EXPERTS),
                                   ("top_k", top_k, TOP_K),
                                   ("swiglu_limit", swiglu_limit, SWIGLU_LIMIT)):
        if actual != expected:
            raise ValueError(f"trellismx experts require {name} {expected}, got {actual!r}")
    if params_dtype is not torch.bfloat16:
        raise ValueError(f"trellismx experts execute BF16 activations, got {params_dtype}")
    local = global_intermediate_size // world_size
    if local != release.local_intermediate_size():
        raise ValueError(f"trellismx rank width {local} is not the release's"
                         f" {release.local_intermediate_size()}")
    if layer not in release.layers:
        raise ValueError(f"trellismx release {release.root} has no sidecar for layer {layer}")
    entry, metadata, arrays = _verified_sidecar(release, layer, rank)
    payload = {name: torch.from_numpy(array).to(device) for name, array in arrays.items()}
    del arrays
    prepared = prepare_trellismx_experts(
        payload, prepared_metadata(metadata, global_intermediate_size=global_intermediate_size),
        num_experts=num_experts, hidden_size=hidden_size, intermediate_size=local, top_k=top_k,
        device=torch.device(device))
    for name, expected in (("num_experts", num_experts), ("hidden_size", hidden_size),
                           ("intermediate_size", local)):
        if getattr(prepared, name, None) != expected:
            raise ValueError(f"prepared trellismx experts report {name}"
                             f" {getattr(prepared, name, None)!r}, expected {expected}")
    # The P8 kernels execute on CUDA only; loading onto the CPU validates the release and
    # its sidecars without executing experts, so it has no runtime to attach.
    if torch.device(device).type == "cuda":
        prepared = attach_p8_runtime(prepared, release, entry, metadata, payload, layer=layer, rank=rank,
                                     world_size=world_size, local_intermediate_size=local, device=device)
    return prepared


def p8_settings() -> dict[str, dict]:
    """The P8 settings preparation records: the in-tree backend declaration b12x pins the runtime by
    and b12x's declared GLM-5.3 Flash serving configuration, which every runtime is built with."""
    from b12x.moe.fused_moe.trellismx_p8_runtime import GLM53_SERVING_OPTIONS, in_tree_p8_backend

    backend = in_tree_p8_backend()
    return {"backend": {"package": backend.package, "identity": backend.identity, "upstream": backend.upstream},
            "options": GLM53_SERVING_OPTIONS.as_dict()}


def p8_runtime(release: TrellisMxRelease, entry: TrellisMxSidecar, metadata: dict[str, str],
               payload: dict[str, "torch.Tensor"], *, layer: int, rank: int, world_size: int,
               local_intermediate_size: int, device):
    """b12x's in-tree P8 runtime for one layer and rank, built with b12x's serving configuration.

    The trellis stream is decoded to E4M3 inside the MMA and the UE8M0/32 scales are
    consumed by the tensor core; activations enter as E4M3 with their own UE8M0/32
    scales. The runtime reads the payload already on ``device`` in place. Its launch
    geometry is not a construction option: each capacity's plan races a
    ``P8LaunchConfig`` and every prepared call names the winner.
    """
    import torch
    from b12x.moe._shared.p8.p8_coupled_scales import SCALE_NAMES
    from b12x.moe.fused_moe.trellismx_p8_runtime import (
        GLM53_SERVING_OPTIONS, P8NativeTPMoE, P8TensorSource,
    )

    # The scale-component validator assembles its packed vectors on the CPU, exactly as
    # from the sidecar file; the large planes stay where they are.
    tensors = {**payload, **{name: payload[name].cpu() for name in SCALE_NAMES}}
    # The runtime checks the sidecar's recorded encoder transform against this file's digest,
    # which detection computed and checked every sidecar against.
    return P8NativeTPMoE(
        P8TensorSource(tensors, metadata), device=torch.device(device), tp_rank=rank, world_size=world_size,
        layer=layer, topk=TOP_K, hidden=HIDDEN_SIZE, intermediate=local_intermediate_size,
        swiglu_limit=SWIGLU_LIMIT, expected_design_sha256=entry.source_design_sha256,
        expected_transform_sha256=release.transform_sha256, options=GLM53_SERVING_OPTIONS)


def attach_p8(prepared, runtime):
    """``prepared`` attached to ``runtime`` under b12x's in-tree backend declaration.

    The runtime must have resolved b12x's declared GLM-5.3 Flash serving configuration;
    b12x then records every setting it resolved and verifies the runtime against the bundle.
    """
    from b12x.moe.fused_moe.trellismx_p8 import attach_trellismx_p8_runtime
    from b12x.moe.fused_moe.trellismx_p8_runtime import GLM53_SERVING_OPTIONS, in_tree_p8_backend

    if runtime.options != GLM53_SERVING_OPTIONS:
        raise ValueError("the P8 runtime resolved a configuration other than b12x's GLM-5.3 Flash"
                         f" serving one: {runtime.options!r}")
    return attach_trellismx_p8_runtime(prepared, runtime, backend=in_tree_p8_backend())


def attach_p8_runtime(prepared, release: TrellisMxRelease, entry: TrellisMxSidecar, metadata: dict[str, str],
                      payload: dict[str, "torch.Tensor"], *, layer: int, rank: int, world_size: int,
                      local_intermediate_size: int, device):
    """Return ``prepared`` attached to b12x's in-tree P8 runtime for this layer and rank.

    Ownership: b12x returns a new prepared bundle that owns the runtime, and every
    plan, binding and captured graph made from that bundle runs it. The GLM weights
    hold the bundle for the prepared model's whole life, so the runtime, its kernels
    and its persistent tensors outlive every graph that replays them and are released
    with the model after native work drains. A call takes its whole arena from the
    caller: the plan declares the bytes and preparation sizes them in the shared
    scratch arena, so a captured graph holds no P8 scratch of its own.
    """
    import torch

    from ignition.preparation.tensor_metadata import record_numeric_tensor

    device = torch.device(device)
    runtime = p8_runtime(release, entry, metadata, payload, layer=layer, rank=rank, world_size=world_size,
                         local_intermediate_size=local_intermediate_size, device=device)
    # Nothing launches here. The runtime declares every program a call of 1..capacity rows
    # reaches and the metadata each capacity's arena is sized from; b12x preparation
    # compiles the programs of every raced launch configuration and installs the winner
    # before it freezes kernel resolution, and a launch whose program was not installed fails.
    # The export stores these values. The trellis carriers and logical scale planes alias the
    # bundle's; the runtime's repacked planes replace the bundle's byte-identical copies when
    # it is attached, and its packed boundary vectors are its own.
    for tensor in (runtime.w13_stream, runtime.w2_stream, runtime.w13_scale_mx, runtime.w2_scale_mx,
                   runtime.w13_sfb, runtime.w2_sfb, runtime.scale_component_packed, runtime.ones):
        record_numeric_tensor(tensor)
    return attach_p8(prepared, runtime)


def agree_on_release(release: TrellisMxRelease | None, group) -> None:
    """Fail on every rank unless all tensor-parallel ranks detected the same release bytes.

    Ranks detect the release independently, so each compares its manifest and transform
    digests with the others' before any of them loads. A checkpoint that is not a release
    takes part as an all-zero identity. Two small all-reduces over the preparation group.
    """
    if group is None:
        return
    import torch
    import torch.distributed as dist

    identity = (bytes(64) if release is None
                else bytes.fromhex(release.manifest_sha256) + bytes.fromhex(release.transform_sha256))
    low = torch.tensor(list(identity), dtype=torch.int64)
    high = low.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(low, high):
        raise ValueError("tensor-parallel ranks detected different TrellisMX release manifests or encoder"
                         " transforms; the release changed while preparation started, or ranks read different"
                         " copies")


def p8_attachments(weights) -> dict[str, list]:
    """This rank's attached P8 runtimes as b12x holds them: the attachment identities (backend
    pin, contract and resolved settings) and every setting each runtime resolved, its serving
    launch geometry included. Each capacity's raced launch configuration belongs to its plan.

    Raises unless every TrellisMX expert layer, the MTP layer's included, carries a live
    runtime: a layer without one would run b12x's own TrellisMX arms instead of the P8 kernels.
    """
    from b12x.moe.fused_moe.trellismx_p8 import trellismx_p8_attachment

    from .weights import MoEWeights

    mlps = [layer.mlp for layer in weights.layers] + ([] if weights.mtp is None else [weights.mtp.mlp])
    entries = [trellismx_p8_attachment(mlp.experts) for mlp in mlps
               if isinstance(mlp, MoEWeights) and mlp.trellismx_release is not None]
    if not entries or any(entry is None or not entry.live for entry in entries):
        raise RuntimeError("a TrellisMX expert layer was prepared without a live P8 runtime attached")
    options = {json.dumps(entry.options.as_dict(), sort_keys=True) for entry in entries}
    return {"identities": sorted({entry.identity for entry in entries}),
            "options": [json.loads(value) for value in sorted(options)]}


def sidecar_bytes(release: TrellisMxRelease, layers: tuple[int, ...]) -> int:
    """Stored bytes of the named layers' sidecars on every rank."""
    return sum(entry.bytes for entry in release.files if entry.layer in layers)


__all__ = [
    "BITS", "CHECKPOINT_ID", "FIRST_ROUTED_LAYER", "GLOBAL_INTERMEDIATE", "HIDDEN_SIZE", "MANIFEST_NAME",
    "MANIFEST_SCHEMA", "MTP_LAYER", "NUM_EXPERTS", "SIDECAR_SCHEMA",
    "SWIGLU_LIMIT", "TOP_K", "TrellisMxRelease", "TrellisMxSidecar", "WORLD_SIZE", "agree_on_release", "attach_p8",
    "attach_p8_runtime", "expected_tensor_shapes", "is_trellismx_overlay", "load_sidecar_tensors",
    "load_trellis_experts", "p8_attachments", "p8_runtime", "p8_settings", "prepare_trellismx_experts",
    "sidecar_bytes", "sidecar_header", "validate_sidecar_metadata", "validate_sidecar_tensors",
]
