# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit tensor partitions for single-rank and b12x tensor-parallel recipes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorParallelShard:
    rank: int
    world_size: int

    def __post_init__(self):
        if (type(self.world_size) is not int or self.world_size < 1
                or type(self.rank) is not int or not 0 <= self.rank < self.world_size):
            raise ValueError("tensor parallelism requires a positive world size and a rank within it")

    def partition(self, extent: int, *, alignment: int = 1) -> tuple[int, int]:
        if (type(extent) is not int or extent <= 0 or type(alignment) is not int
                or alignment <= 0 or extent % (self.world_size * alignment)):
            raise ValueError("tensor extent must divide into aligned rank partitions")
        width = extent // self.world_size
        return self.rank * width, (self.rank + 1) * width

    def fused_partitions(self, extents: tuple[int, ...], *, alignment: int = 1) -> tuple[tuple[int, int], ...]:
        """Slice each independently partitioned projection inside a fused axis."""
        offset, ranges = 0, []
        for extent in extents:
            first, last = self.partition(extent, alignment=alignment)
            ranges.append((offset + first, offset + last))
            offset += extent
        return tuple(ranges)

    def padded_partition(self, extent: int, *, alignment: int = 1) -> tuple[int, int]:
        """Partition an axis into equal aligned widths, including a zero-filled tail."""
        if (type(extent) is not int or extent <= 0 or type(alignment) is not int or alignment <= 0):
            raise ValueError("padded tensor partitions require positive integer dimensions")
        width = -(-extent // (self.world_size * alignment)) * alignment
        return self.rank * width, (self.rank + 1) * width

    def replicated_partition(self, extent: int) -> tuple[int, int]:
        """Partition an axis, replicating each element when ranks outnumber it.

        Grouped-query attention uses this for KV heads: contiguous rank groups
        share a KV head while each rank owns distinct query heads.
        """
        if type(extent) is not int or extent <= 0:
            raise ValueError("replicated extent must be a positive integer")
        if extent >= self.world_size:
            return self.partition(extent)
        if self.world_size % extent:
            raise ValueError("replicated extent must divide the rank count")
        first = self.rank // (self.world_size // extent)
        return first, first + 1
