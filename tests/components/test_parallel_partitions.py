# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit recipe partitions preserve fused projections and grouped KV heads."""

import pytest

from ignition.components.parallel import TensorParallelShard


@pytest.mark.parametrize("world", (1, 2, 3, 4, 6, 8, 16))
def test_fused_projections_partition_each_component(world):
    shards = [TensorParallelShard(rank, world) for rank in range(world)]
    extents = (3072, 6144, 12288)
    partitions = [shard.fused_partitions(extents) for shard in shards]
    offset = 0
    for index, extent in enumerate(extents):
        assert [partitions[rank][index] for rank in range(world)] == [
            (offset + rank * extent // world, offset + (rank + 1) * extent // world)
            for rank in range(world)
        ]
        offset += extent


@pytest.mark.parametrize("world,expected", [(1, [(0, 2)]), (2, [(0, 1), (1, 2)]),
                                           (4, [(0, 1), (0, 1), (1, 2), (1, 2)])])
def test_query_rank_groups_share_kv_heads(world, expected):
    assert [TensorParallelShard(rank, world).replicated_partition(2) for rank in range(world)] == expected


def test_unrepresentable_partitions_fail_before_loading():
    with pytest.raises(ValueError):
        TensorParallelShard(0, 4).partition(640, alignment=128)
    with pytest.raises(ValueError):
        TensorParallelShard(0, 4).replicated_partition(3)
    assert TensorParallelShard(3, 4).partition(640, alignment=16) == (480, 640)


@pytest.mark.parametrize("extent,alignment,expected", [
    (8, 1, [(0, 3), (3, 6), (6, 9)]),
    (2304, 32, [(0, 768), (768, 1536), (1536, 2304)]),
    (129280, 64, [(0, 43136), (43136, 86272), (86272, 129408)]),
])
def test_three_rank_padding_is_minimal_and_preserves_contiguous_coverage(extent, alignment, expected):
    bounds = [TensorParallelShard(rank, 3).padded_partition(extent, alignment=alignment) for rank in range(3)]
    assert bounds == expected
    assert extent <= bounds[-1][1] < extent + 3 * alignment
    assert list(range(extent)) == [i for first, last in bounds for i in range(first, min(last, extent))]


@pytest.mark.parametrize("world,rank", [(0, 0), (-1, 0), (True, 0), (8, -1), (8, 8), (8, True)])
def test_invalid_rank_identity(world, rank):
    with pytest.raises(ValueError):
        TensorParallelShard(rank, world)


@pytest.mark.parametrize("world", (3, 8, 16))
def test_replicated_kv_partitions_cover_query_rank_groups(world):
    for heads in (1, world):
        ranges = [TensorParallelShard(rank, world).replicated_partition(heads) for rank in range(world)]
        assert ranges == [(rank // (world // heads), rank // (world // heads) + 1)
                          for rank in range(world)]
