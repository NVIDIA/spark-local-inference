# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish a model package containing one native resource bundle per rank."""

from pathlib import Path

from ignition.progress import phase

from .runtime_package import export_model
from .package_file import combine
from .memory_inventory import print_inventory


def export_package(prepared, images, storage, path, *, options, publisher, progress=None, console=None,
                   staging=None, form='directory'):
    """Export every rank, close preparation resources, and assemble the model file.

    Ranks reuse immutable allocation blobs and stage execution resources under `staging`
    (default: beside `path`). Publication links payload blobs beside the metadata primary.
    """
    if publisher is None:
        raise ValueError("final model publication requires a signing publisher")
    import flare_heliopause
    import torch.distributed as dist

    cache_only = path is None
    from .package_snapshot import _OPTIONS

    settings = _OPTIONS.get()
    if settings is None:
        raise RuntimeError("export requires the configured package cache")
    # Native chooses the workspace; a cache-only export has no destination.
    native_stage = flare_heliopause.NativeExportStage(
        None if cache_only else Path(path), staging, settings.cache)
    stage = Path(native_stage.path)
    token = _OPTIONS.set(settings)
    result = None
    try:
        if not cache_only and form == 'directory':
            flare_heliopause.export_preflight_directory(str(path))
        roots = [Path(native_stage.rank_path(rank)) for rank in range(storage.world)]
        if storage.world > 1:
            for rank, control in enumerate(prepared.controls, 1):
                control.send(('export', roots[rank], options, _OPTIONS.get()))
            dist.barrier()
            storage.peer_storage.reset()
            dist.barrier()
        model = prepared if storage.world == 1 else prepared.model
        package = export_model(model, images, storage, roots[0], options=options, progress=progress,
                               sources=True)
        inventories = [package.memory_inventory]
        memory = [[]] if storage.world == 1 else [list(storage.peer_storage.owned.items())]
        if storage.world > 1:
            with phase(progress, "Wait for rank exports", total=storage.world - 1) as stage:
                for control in prepared.controls:
                    reply = control.recv()
                    memory.append(reply['peer_memory'])
                    inventories.append(reply['memory_inventory'])
                    stage.update(advance=1)
        peers = []
        for rank, allocations in enumerate(memory):
            for address, size in allocations:
                peers.append(dict(owner=rank, address=address, nbytes=size, readers=list(range(storage.world))))
        # The stage leases every cache payload the rank sources name, so peer
        # preparation leases can drop before combine reads them.
        with phase(progress, "Release preparation resources"):
            native_stage.retain_sources(roots)
            prepared.close()
            for region in storage.host_regions:
                region.close()
        result = combine(None if cache_only else path, roots, peers,
                         publisher=publisher, form=form, cache_only=cache_only)
        print_inventory(inventories, console=console or getattr(progress, "console", None))
    finally:
        _OPTIONS.reset(token)
        with phase(progress, "Clear export staging"):
            native_stage.close()
    return result
