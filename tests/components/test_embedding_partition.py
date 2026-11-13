# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Interleaved embedding partitions preserve each head's complete row range."""

import pytest

from ignition.components.embedding_partition import EmbeddingHeadPartition


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_interleaved_heads_cover_every_row_and_balance_ngram_orders(world):
    sizes = [101 + head * 2 for head in range(24)]
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += size
    seen = []
    for rank in range(world):
        partition = EmbeddingHeadPartition.for_rank(offsets, total, rank, world)
        assert [sum(head // 8 == order for head in partition.heads) for order in range(3)] == [8 // world] * 3
        local = 0
        for head in range(24):
            first, last, destination = partition.row_mapping[head]
            assert (first, last) == (offsets[head], offsets[head] + sizes[head])
            if head in partition.heads:
                assert destination == local
                local += sizes[head]
                seen.extend(range(first, last))
            else:
                assert destination == -1
        assert local == partition.rows
        assert partition.source_rows == tuple((offsets[head], offsets[head] + sizes[head])
                                             for head in partition.heads)
    assert sorted(seen) == list(range(total))


@pytest.mark.parametrize("offsets,rows,rank,world", [
    ((), 3, 0, 1), ((1, 2), 3, 0, 1), ((0, 0), 3, 0, 1),
    ((0, 2), 2, 0, 1), ((0, 2), 4, 0, 3), ((0, 2), 4, 2, 2),
])
def test_invalid_head_partitions_are_rejected(offsets, rows, rank, world):
    with pytest.raises(ValueError, match="head boundaries"):
        EmbeddingHeadPartition.for_rank(offsets, rows, rank, world)
