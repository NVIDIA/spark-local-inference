# SPDX-License-Identifier: Apache-2.0

"""Context subgroup memory sharing releases every exported descriptor."""

import ctypes as C
import os
import pickle
from multiprocessing import resource_sharer
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from ignition.preparation.peer_storage import PeerStorage
from ignition.preparation import peer_storage


@pytest.mark.skipif(os.name != "posix", reason="CUDA VMM descriptor sharing requires POSIX")
@pytest.mark.parametrize("start,size", [(0, 16), (0, 8), (8, 8)])
def test_subgroup_cache_export_creates_descriptors_only_for_receivers(monkeypatch, start, size):
    descriptor = os.open(os.devnull, os.O_RDONLY)

    class Cuda:
        def call(self, name, *arguments):
            if name == "cuMemExportToShareableHandle":
                C.cast(arguments[0], C.POINTER(C.c_int))[0] = os.dup(descriptor)

    owner = object.__new__(PeerStorage)
    owner.storage = SimpleNamespace(rank=start, world=16, cuda=Cuda())
    owner.dynamic, owner.owned = {}, {}
    owner.cudaIpcOpenMemHandleBytes = lambda payload: 999
    ranks = tuple(range(start, start + size))
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: ranks)
    registrations = resource_sharer._resource_sharer._cache
    original = set(registrations)
    exported = []

    def gather(output, payloads, *, group):
        for payload in payloads:
            _, _, parts = pickle.loads(payload)
            exported.extend(item for _, _, descriptors in parts for item in descriptors if item is not None)
            for _, _, descriptors in parts:
                for rank in ranks:
                    if rank != start:
                        os.close(descriptors[rank].detach())
        output[:] = [payloads] * len(ranks)

    monkeypatch.setattr(dist, "all_gather_object", gather)
    try:
        owner.share_regions(object(), [SimpleNamespace(address=123, extent=128, mappings=[(123, 128)])])
        assert not set(registrations) - original, "nonmember ranks retain unclaimed exported descriptors"
    finally:
        for item in exported:
            if item._id[1] in set(registrations) - original:
                os.close(item.detach())
        os.close(descriptor)


@pytest.mark.skipif(os.name != "posix", reason="CUDA VMM descriptor sharing requires POSIX")
@pytest.mark.parametrize("ranks", [tuple(range(16)), tuple(range(8)), tuple(range(8, 16))])
def test_runtime_allocator_group_view_exports_only_receivers_and_shares_ownership(monkeypatch, ranks):
    descriptor = os.open(os.devnull, os.O_RDONLY)

    class Cuda:
        def call(self, name, *arguments):
            if name == "cuMemExportToShareableHandle":
                C.cast(arguments[0], C.POINTER(C.c_int))[0] = os.dup(descriptor)

    owner = object.__new__(PeerStorage)
    owner.storage = SimpleNamespace(rank=ranks[0], world=16, cuda=Cuda(), device=0, error=None,
        library=SimpleNamespace(flare_core_allocate=lambda *args: 123, flare_core_free=lambda *args: None),
        mappings={123: SimpleNamespace(nbytes=128)})
    owner._export_ranks = tuple(range(16))
    owner.dynamic, owner.owned, owner.imported, owner.imported_mappings = {}, {}, {}, {}
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: ranks)
    monkeypatch.setattr(peer_storage, "DupFd", lambda fd: "exported-descriptor")
    scoped = owner.for_group(object())
    try:
        assert scoped.storage is owner.storage
        for field in ("dynamic", "owned", "imported", "imported_mappings"):
            assert getattr(scoped, field) is getattr(owner, field)
        pointer = scoped.cudaMalloc(128)
        assert owner.owned[pointer] == 128
        _, _, parts = pickle.loads(scoped.cudaIpcGetMemHandleBytes(pointer))
        receivers = {rank for rank, item in enumerate(parts[0][2]) if item is not None}
        assert receivers == set(ranks) - {owner.storage.rank}
        _, _, unscoped = pickle.loads(owner.cudaIpcGetMemHandleBytes(pointer))
        default_receivers = {rank for rank, item in enumerate(unscoped[0][2]) if item is not None}
        assert default_receivers == set(range(16)) - {owner.storage.rank}
        scoped.cudaFree(pointer)
        assert pointer not in owner.owned
    finally:
        os.close(descriptor)
