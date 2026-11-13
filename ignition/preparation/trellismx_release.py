# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Locate a checkpoint's assets when a TrellisMX release overlays a carrier.

A TrellisMX release is a routed-expert overlay: its root holds the manifest, the
per-layer sidecars and the design record, while the safetensors index, tokenizer,
chat template and generation config stay in ``carrier/`` with every tensor the
sidecars do not replace. Shared code that resolves or inspects a checkpoint asks
this module where those files are, without importing a recipe.

Only the manifest is read, and no payload. This module does not validate a release;
the recipe that serves one owns every gate. A root without this manifest is treated
exactly as before so no other checkpoint's behaviour moves, and a manifest that
cannot be read raises instead of letting its root pass for a plain checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_NAME = "trellismx-manifest.json"
MANIFEST_SCHEMA = "trellismx.hf-overlay-release.v1"
CARRIER_DIRECTORY = "carrier"
SIDECAR_DIRECTORY = "sidecars"
# The encoder transform the sidecars pin, which preparation reads beside them.
TRANSFORM_PATH = "design/transform.json"


def sidecar_path(layer: int, rank: int, world_size: int) -> str:
    """The release-relative path of one layer and tensor-parallel rank's sidecar."""
    return f"{SIDECAR_DIRECTORY}/p8-layer-{layer:03d}-tp{world_size}-rank-{rank}.safetensors"


def _manifest(root: Path) -> dict | None:
    """``root``'s overlay manifest when it declares this schema, else None.

    Raises ``ValueError`` when the manifest exists but is not a readable JSON object.
    """
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{manifest_path} is unreadable: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} is not a JSON object")
    return manifest if manifest.get("schema") == MANIFEST_SCHEMA else None


def _carrier_index(root: Path) -> Path:
    return root / CARRIER_DIRECTORY / "model.safetensors.index.json"


def trellismx_assets_root(root: str | Path) -> Path:
    """The directory holding ``root``'s index and frontend assets.

    The carrier directory when ``root`` declares this overlay schema and carries a
    carrier index, and ``root`` itself otherwise.
    """
    root = Path(root).expanduser()
    return root / CARRIER_DIRECTORY if _manifest(root) is not None and _carrier_index(root).is_file() else root


def overlay_schema(root: str | Path) -> str | None:
    """The overlay manifest schema ``root`` is packaged with, or None for any other checkpoint."""
    return MANIFEST_SCHEMA if _manifest(Path(root).expanduser()) is not None else None


def _allocation(manifest: dict) -> dict[int, int]:
    """The manifest's allocation: each layer index and the tensor-parallel size it is sliced for."""
    allocation = manifest.get("allocation")
    if not isinstance(allocation, dict) or not allocation:
        raise ValueError("TrellisMX manifest allocation must be a nonempty object of layer indices")
    layers = {}
    for key, world_size in allocation.items():
        try:
            layer = int(key)
        except ValueError:
            layer = None
        if layer is None or key != str(layer) or layer < 0 or type(world_size) is not int or world_size <= 0:
            raise ValueError(f"TrellisMX manifest allocation entry {key!r}: {world_size!r} is invalid")
        layers[layer] = world_size
    return layers


def _manifest_files(manifest: dict) -> list[tuple[str, int]]:
    """Every file the manifest lists, which must be exactly the release's canonical inventory.

    That inventory is one rank-named sidecar for every layer and rank the allocation
    declares, so a list that omits, repeats, renames or adds an entry is rejected
    instead of making an incomplete download look complete.
    """
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("TrellisMX manifest lists no files")
    entries = []
    for entry in files:
        path = entry.get("path") if isinstance(entry, dict) else None
        size = entry.get("bytes") if isinstance(entry, dict) else None
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts \
                or type(size) is not int or size <= 0:
            raise ValueError(f"TrellisMX manifest file entry {entry!r} is invalid")
        entries.append((path, size))
    inventory = {sidecar_path(layer, rank, world_size)
                 for layer, world_size in _allocation(manifest).items() for rank in range(world_size)}
    listed = [path for path, _ in entries]
    if len(listed) != len(set(listed)) or set(listed) != inventory:
        missing, unexpected = sorted(inventory - set(listed)), sorted(set(listed) - inventory)
        repeated = sorted({path for path in listed if listed.count(path) > 1})
        raise ValueError("TrellisMX manifest files must be one sidecar per allocated layer and rank:"
                         f" missing {missing}, unexpected {unexpected}, repeated {repeated}")
    return entries


def trellismx_files_present(root: str | Path) -> bool:
    """Whether every file ``root``'s release needs beside its carrier is present at its declared size.

    Those files are the canonical inventory (one sidecar per allocated layer and rank)
    and the encoder transform. True for a root that is not a release, which needs
    nothing more; a manifest whose file list is not that inventory raises
    ``ValueError``. Only sizes are compared; preparation verifies each sidecar's
    digest when it loads it.
    """
    root = Path(root).expanduser()
    manifest = _manifest(root)
    if manifest is None:
        return True
    return _carrier_index(root).is_file() and (root / TRANSFORM_PATH).is_file() and all(
        (root / path).is_file() and (root / path).stat().st_size == size for path, size in _manifest_files(manifest))


def trellismx_stored_bytes(root: str | Path) -> int | None:
    """Stored tensor bytes of ``root``'s release: the carrier's payload plus every manifest file.

    None for a root that is not a release.
    """
    root = Path(root).expanduser()
    manifest = _manifest(root)
    if manifest is None:
        return None
    try:
        index = json.loads(_carrier_index(root).read_text())
    except OSError as error:
        raise ValueError(f"{_carrier_index(root)} is unreadable: {error}") from error
    carrier = index.get("metadata", {}).get("total_size") if isinstance(index, dict) else None
    if type(carrier) is not int or carrier <= 0:
        raise ValueError(f"{root / CARRIER_DIRECTORY} has an invalid safetensors total_size: {carrier!r}")
    return carrier + sum(size for _, size in _manifest_files(manifest))


__all__ = [
    "CARRIER_DIRECTORY", "MANIFEST_NAME", "MANIFEST_SCHEMA", "SIDECAR_DIRECTORY", "TRANSFORM_PATH", "overlay_schema",
    "sidecar_path", "trellismx_assets_root", "trellismx_files_present", "trellismx_stored_bytes",
]
