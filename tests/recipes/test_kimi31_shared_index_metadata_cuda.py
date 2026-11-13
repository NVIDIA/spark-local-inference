# SPDX-License-Identifier: Apache-2.0

"""Shared DSA metadata matches the original per-layer operations on replay."""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 5, 8, 32])
@torch.no_grad()
def test_shared_index_metadata_matches_per_layer_operations(rows):
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.kimi31.dcp import ContextShards, index_pages
    from ignition.recipes.kimi31.indexer import SharedIndexMetadata

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    shards = ContextShards.on(5, 16, device)
    width = 17
    positions = 32768 + torch.arange(rows, dtype=torch.int64, device=device) * 791
    if rows > 1:
        positions[-1] = -1
    tables = (torch.arange(rows * width, dtype=torch.int32, device=device).view(rows, width) % 251) + 1
    tables[-1, -1] = -1
    metadata = SimpleNamespace(positions=positions, token_page_table=tables)

    def forward():
        return SharedIndexMetadata.create(metadata, shards, rows)

    def reference():
        pages = torch.empty((rows, width * 2), dtype=torch.int32, device=device)
        index_pages(tables, pages)
        lengths = shards.local_tokens((positions + 1).clamp(min=0)).to(torch.int32)
        slots = shards.token_slots(positions, tables)
        return pages, lengths, slots

    def check(actual):
        expected = reference()
        torch.cuda.synchronize(device)
        for value, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, wanted, rtol=0, atol=0)

    compiled = compile_forward(forward)
    check(compiled())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled()
    positions.copy_(65536 + torch.arange(rows, dtype=torch.int64, device=device) * 397)
    if rows > 1:
        positions[-1] = -1
    tables.copy_((tables + 3) % 251)
    for value in actual:
        value.fill_(-91)
    before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
    check(actual)
    graph.reset()
