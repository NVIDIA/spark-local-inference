# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-rank access to fixed collective buffers and incrementally backed cache regions."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _rank(rank, rendezvous):
    from b12x.sequence._shared.disk_table import _tensor_from_pointer
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.core_storage import CoreStorage

    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    with ModuleImages() as images, CoreStorage(rank=rank, world=2, device=rank) as storage:
        peer = storage.peer_storage
        size = storage.granularity
        pointer = peer.cudaMalloc(size)
        peer.cudaMemset(pointer, rank + 11, size)
        handles = [None, None]
        dist.all_gather_object(handles, peer.cudaIpcGetMemHandleBytes(pointer))
        imported = peer.cudaIpcOpenMemHandleBytes(handles[1 - rank])
        cache = storage.device_allocate("cache", (4, size), torch.uint8,
                                        group="KV_PAGES", maximum=4, units=1)
        storage.grow("KV_PAGES", 3)
        cache[:3].fill_(rank + 21)
        torch.cuda.synchronize()
        region = storage.dynamic_regions[0]
        addresses = peer.share_regions(dist.group.WORLD, [region])[0]
        assert peer.imported[addresses[1 - rank]] == 4 * size
        assert len(peer.imported_mappings[addresses[1 - rank]]) == 2
        for parts in peer.imported_mappings.values():
            for address, nbytes in parts:
                assert images.allocations[address] == (nbytes, "device")

        def remote(address, count):
            return _tensor_from_pointer(address, shape=(count,), dtype=torch.uint8,
                                        device=torch.device("cuda", rank), nbytes=count)

        assert (remote(imported, size).clone() == 12 - rank).all().item()
        assert (remote(addresses[1 - rank], 3 * size).clone() == 22 - rank).all().item()
        dist.barrier()
        peer.reset()
        torch.cuda.synchronize()
        dist.barrier()
        assert remote(imported, size).clone().count_nonzero().item() == 0
        assert (remote(addresses[1 - rank], 3 * size).clone() == 22 - rank).all().item()
        torch.cuda.synchronize()
        dist.barrier()
        peer.cudaIpcCloseMemHandle(addresses[1 - rank])
        peer.cudaIpcCloseMemHandle(imported)
        dist.barrier()
        peer.cudaFree(pointer)
    dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                    reason="requires two assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1")
def test_fixed_and_dynamic_peer_mappings(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign two CUDA devices explicitly"
    assert torch.cuda.device_count() >= 2
    mp.spawn(_rank, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)
