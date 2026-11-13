# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Public test credentials shared by Python exporters and native fixture readers."""

from pathlib import Path
from ignition.preparation.runtime_core import export_model
from ignition.preparation.memory_inventory import print_inventory

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ignition.preparation.protection import Publisher
if __package__:
    from . import legacy_core
else:
    import legacy_core

DEFAULT_SHARD_SIZE = legacy_core.DEFAULT_SHARD_SIZE
publish_legacy = legacy_core.publish_legacy
extend_cluster_legacy = legacy_core.extend_cluster
zstd_compress = legacy_core.zstd_compress
zstd_decompress = legacy_core.zstd_decompress


def publish_captured_legacy(path, sources, *, shard_size=DEFAULT_SHARD_SIZE):
    """Serialize captured native sources with the compatibility fixture writer."""
    import json
    import flare_heliopause
    from ignition.preparation import runtime_core_pb2 as proto

    record = json.loads(flare_heliopause.export_read_sources(str(sources)))
    index = proto.ModelIndex.FromString(bytes(record['index']))
    payload = [(offset, Path(source), start, size)
               for offset, source, start, size in record['payload']]
    physical = Path(sources).with_suffix('.safetensors')
    publish_legacy(physical, index, payload, publisher=publisher(), shard_size=shard_size)
    legacy_core.snapshot_fixture(physical, path)


def publisher():
    return Publisher(Ed25519PrivateKey.from_private_bytes(bytes([42]) * 32),
                     'flare-tests', 'https://huggingface.co/flare-tests/fixture')


def export_cluster_core(prepared, images, storage, path, *, peer_files, options, export_id, world,
                           publisher, peer_inventories, progress=None, shard_size=DEFAULT_SHARD_SIZE):
    """Create signed cluster containers for version 4/5 compatibility fixtures."""

    if publisher is None:
        raise ValueError("final model publication requires a signing publisher")

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)

    try:
        core = export_model(prepared, images, storage, partial, options=options, progress=progress)
        extend_cluster_legacy(partial, peer_files, (world, export_id))
        index, start = legacy_core.read_index(partial)
        payload = sorted((span.offset, partial, start + span.offset, span.nbytes)
                         for _, _, span in legacy_core.resource_roles(index))
        publish_legacy(path, index, payload, publisher=publisher, shard_size=shard_size)
        print_inventory([core.memory_inventory, *peer_inventories], console=getattr(progress, "console", None))
    finally:
        partial.unlink(missing_ok=True)
