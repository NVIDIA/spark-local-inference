# SPDX-License-Identifier: Apache-2.0

"""Preparation-only, byte-preserving storage for immutable checkpoint tensors."""

from dataclasses import fields, is_dataclass, replace
from math import ceil
import weakref

import torch

from .provenance import lookup, record_sources


class PackedWeights:
    """Pack dense weight views into aligned arenas without retaining their sources.

    The package storage owns this arena through export and execution drain.
    Callers may supply only immutable numeric weights, never prepared operators,
    pointer tables, or mutable state. Tensor shape, strides and bytes are preserved;
    repeated references to the same tensor return the same packed view.
    """

    def __init__(self, storage, *, chunk_bytes=32 << 20):
        if (type(chunk_bytes) is not int or chunk_bytes <= 0
                or chunk_bytes % storage.granularity):
            raise ValueError("weight arena chunks must contain whole allocation granules")
        self.storage = storage
        self.device = torch.device("cuda", storage.device)
        self.chunk_bytes = chunk_bytes
        self.pool = torch.cuda.MemPool(storage._allocator.allocator())
        storage._preparation_pools.add(self.pool.id)
        self.chunks, self.offsets, self.padding = [], [], []
        self.memo = {}
        self.frozen = False

    def tensor(self, source):
        if self.frozen:
            raise RuntimeError("immutable weight arena is frozen")
        if source.device.type == "cpu" or not source.numel():
            return source
        if (source.device != self.device or source.requires_grad
                or source.is_conj() or source.is_neg()):
            raise ValueError("weight packing requires inference tensors on the owning GPU")
        previous = self.memo.get(id(source))
        if previous is not None and previous[0]() is source:
            return previous[1]
        width = 1
        for stride, size in sorted((stride, size) for size, stride in zip(source.shape, source.stride()) if size > 1):
            if stride != width:
                raise ValueError("weight packing requires nonoverlapping dense storage")
            width *= size
        extent = source.nbytes
        for index, chunk in enumerate(self.chunks):
            first = (self.offsets[index] + 511) // 512 * 512
            if first + extent <= chunk.numel():
                break
        else:
            index, first = len(self.chunks), 0
            capacity = max(self.chunk_bytes, ceil(extent / self.storage.granularity) * self.storage.granularity)
            with torch.cuda.use_mem_pool(self.pool, device=self.device.index):
                chunk = torch.zeros(capacity, device=self.device, dtype=torch.uint8)
            self.storage.numeric_storage.record(chunk)
            self.chunks.append(chunk)
            self.offsets.append(0)
        if first > self.offsets[index]:
            self.padding.append((chunk.data_ptr() + self.offsets[index], chunk.data_ptr() + first))
        # Copy raw storage, including NaN payloads and signed zeros, without a
        # numeric cast. Dense transposes keep their original element strides.
        physical = source.as_strided((source.numel(),), (1,), source.storage_offset())
        chunk[first:first + extent].copy_(physical.view(torch.uint8))
        view = torch.empty(0, dtype=source.dtype, device=source.device).set_(
            chunk.untyped_storage()[first:first + extent], 0, source.shape, source.stride())
        self.storage.numeric_storage.record(view)
        origin = lookup(source)
        if origin is not None:
            record_sources(view, **origin)
        self.offsets[index] = first + extent
        self.memo[id(source)] = (weakref.ref(source), view)
        self.memo[id(view)] = (weakref.ref(view), view)
        return view

    def tree(self, value):
        """Copy a caller-selected tree of immutable numeric weights."""
        if isinstance(value, torch.Tensor):
            return self.tensor(value)
        if isinstance(value, dict):
            return {key: self.tree(child) for key, child in value.items()}
        if isinstance(value, tuple):
            return tuple(self.tree(child) for child in value)
        if isinstance(value, list):
            return [self.tree(child) for child in value]
        if is_dataclass(value):
            return replace(value, **{field.name: self.tree(getattr(value, field.name))
                                     for field in fields(value) if field.init})
        return value

    def freeze(self):
        self.frozen = True

    def inactive_ranges(self):
        """Exclude only immutable alignment holes and the frozen unused tails."""
        if not self.frozen:
            return []
        return [*self.padding, *((chunk.data_ptr() + used, chunk.data_ptr() + chunk.nbytes)
                                for chunk, used in zip(self.chunks, self.offsets) if used < chunk.nbytes)]
