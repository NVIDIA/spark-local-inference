# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Interleaved ownership of complete embedding heads and their checkpoint rows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingHeadPartition:
    heads: tuple[int, ...]
    source_rows: tuple[tuple[int, int], ...]
    row_mapping: tuple[tuple[int, int, int], ...]
    rows: int

    @classmethod
    def for_rank(cls, offsets, rows, rank, world):
        offsets = tuple(offsets)
        if (world < 1 or not 0 <= rank < world or not offsets or len(offsets) % world
                or offsets[0] != 0 or any(a >= b for a, b in zip(offsets, (*offsets[1:], rows)))):
            raise ValueError("embedding head boundaries must cover the table and divide across ranks")
        heads = tuple(range(rank, len(offsets), world))
        bounds = tuple(zip(offsets, (*offsets[1:], rows), strict=True))
        mapping = [(first, last, -1) for first, last in bounds]
        cursor = 0
        for head in heads:
            first, last = bounds[head]
            mapping[head] = first, last, cursor
            cursor += last - first
        return cls(heads, tuple(bounds[head] for head in heads), tuple(mapping), cursor)
