# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gathering latent records over a network, window by window, reproduces the slab CUDA IPC pulls on one host."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("end", [1, 5, 37, 1000, 2048])
def test_windowed_records_match_the_ipc_pull(world, end):
    from ignition.recipes.glm53.dcp import (ContextShards, pack_records, pull_context, scatter_records,
                                            window_counts)

    torch.manual_seed(end + world)
    pages, columns, window = 6, 4, 192
    positions = columns * 256 * world
    table = torch.tensor([3, 0, 5, 1], dtype=torch.int32, device="cuda")
    caches = [torch.randint(-2**31, 2**31 - 1, (pages * 256, 132), dtype=torch.int32, device="cuda")
              for _ in range(world)]
    final = torch.tensor([end], dtype=torch.int64, device="cuda")
    expected = torch.zeros(positions, 132, dtype=torch.int32, device="cuda")
    addresses = torch.tensor([cache.data_ptr() for cache in caches], dtype=torch.int64, device="cuda")
    pull_context(addresses, table, final, expected, ContextShards.on(0, world, "cuda"))
    slab = torch.zeros_like(expected)
    total = 0
    for first in range(0, columns * 256, window):
        counts = window_counts(final, world, first, window)
        total += int(counts.sum())
        # Each rank packs its own row; the gather then delivers each row's counted bytes to every rank.
        gathered = torch.full((world, window * 528), 255, dtype=torch.uint8, device="cuda")
        for rank in range(world):
            records = torch.full((world, window * 528), 255, dtype=torch.uint8, device="cuda")
            pack_records(caches[rank], table, final, records, ContextShards.on(rank, world, "cuda"), first)
            gathered[rank, :int(counts[rank])] = records[rank, :int(counts[rank])]
        scatter_records(gathered.view(torch.bfloat16), final, slab, ContextShards.on(1, world, "cuda"), first)
    assert total == end * 528
    assert torch.equal(slab[:end], expected[:end])
    assert not slab[end:].any()
