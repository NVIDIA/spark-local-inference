# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Final tensor destinations packed into package-owned device mappings."""

from contextvars import ContextVar
import math

import torch


_ACTIVE_STORAGE = ContextVar("flare_persistent_storage", default=None)


def active_package_storage():
    """Return the package storage owning persistent destinations in this context."""
    return _ACTIVE_STORAGE.get()


def allocated_tensor_bytes(device=None):
    """Count active Torch allocations and packed model tensor destinations."""
    allocated = torch.cuda.memory_allocated(device)
    storage = active_package_storage()
    if storage is None or storage._persistent_arena is None or storage._persistent_arena.closed:
        return allocated
    target = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if target.type == "cuda" and (target.index is None or target.index == storage.device):
        allocated += storage._persistent_arena.used
    return allocated


def _layout(shape, dtype, stride):
    shape = (shape,) if type(shape) is int else tuple(shape)
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("persistent tensor dimensions must be nonnegative integers")
    if not isinstance(dtype, torch.dtype):
        raise TypeError("persistent tensors require a torch dtype")
    if stride is None:
        strides, step = [], 1
        for size in reversed(shape):
            strides.append(step)
            step *= max(1, size)
        stride = tuple(reversed(strides))
    else:
        stride = tuple(stride)
        if len(stride) != len(shape) or any(type(step) is not int or step < 0 for step in stride):
            raise ValueError("persistent tensor strides must match the shape and be nonnegative integers")
    if not math.prod(shape):
        return shape, stride, 0
    span = 1
    for step, size in sorted((step, size) for size, step in zip(shape, stride) if size > 1):
        if step < span:
            raise ValueError("persistent tensors require nonoverlapping strides")
        span += (size - 1) * step
    return shape, stride, span * dtype.itemsize


def persistent_empty(shape, dtype, device, *, stride=None, alignment=256):
    """Allocate a final destination, using packed storage during package preparation."""
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("persistent tensor alignment must be a positive power of two")
    shape, stride, nbytes = _layout(shape, dtype, stride)
    device = torch.device(device)
    storage = active_package_storage()
    if storage is None or device.type != "cuda" or not nbytes:
        return torch.empty_strided(shape, stride, dtype=dtype, device=device)
    ordinal = torch.cuda.current_device() if device.index is None else device.index
    if ordinal != storage.device:
        raise ValueError("persistent tensor device must match the package rank")
    return storage._persistent_empty(shape, stride, dtype, nbytes, alignment)


def persistent_zeros(shape, dtype, device, *, stride=None, alignment=256):
    """Allocate and zero a final tensor destination."""
    return persistent_empty(shape, dtype, device, stride=stride, alignment=alignment).zero_()


def persistent_copy(tensor, *, alignment=256):
    """Copy a final tensor into packed storage while preserving its shape and strides."""
    return persistent_empty(tensor.shape, tensor.dtype, tensor.device,
                            stride=tensor.stride(), alignment=alignment).copy_(tensor)


def persistent_cat(tensors, dim=0):
    """Concatenate matching final tensors directly into a packed destination."""
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError("persistent concatenation requires at least one tensor")
    first = tensors[0]
    if type(dim) is not int or first.ndim == 0 or not -first.ndim <= dim < first.ndim:
        raise ValueError("persistent concatenation requires a valid tensor dimension")
    dim %= first.ndim
    if any(tensor.ndim != first.ndim or tensor.dtype != first.dtype or tensor.device != first.device
           or any(size != other for axis, (size, other) in enumerate(zip(first.shape, tensor.shape))
                  if axis != dim) for tensor in tensors):
        raise ValueError("persistent concatenation requires matching shapes, dtypes, and devices")
    shape = list(first.shape)
    shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    output = persistent_empty(shape, first.dtype, first.device)
    offset = 0
    for tensor in tensors:
        output.narrow(dim, offset, tensor.shape[dim]).copy_(tensor)
        offset += tensor.shape[dim]
    return output


class PersistentTensorArena:
    """Map a stable virtual prefix and pack final, nonoverlapping tensor destinations."""

    def __init__(self, region, capacity, granularity):
        self.region, self.capacity, self.granularity = region, capacity, granularity
        self.used = 0
        self.padding = []
        self.closed = False

    def empty(self, shape, stride, dtype, nbytes, alignment):
        from b12x.sequence._shared.disk_table import _tensor_from_pointer

        if self.closed:
            raise RuntimeError("persistent tensor storage is closed")
        alignment = max(alignment, dtype.itemsize)
        start = -(-(self.region.address + self.used) // alignment) * alignment
        end = start + nbytes - self.region.address
        if end > self.capacity:
            raise MemoryError("persistent tensor storage exceeds its reserved device capacity")
        backed = -(-end // self.granularity) * self.granularity
        self.region.grow(backed)
        tensor = _tensor_from_pointer(start, shape=(nbytes // dtype.itemsize,), dtype=dtype,
            device=torch.device("cuda", self.region.device), nbytes=nbytes).as_strided(shape, stride)
        if start > self.region.address + self.used:
            self.padding.append((self.region.address + self.used, start))
        self.used = end
        return tensor

    def unused_ranges(self):
        if self.closed:
            return []
        result = list(self.padding)
        if self.used < self.region.nbytes:
            result.append((self.region.address + self.used, self.region.address + self.region.nbytes))
        return result

    def close(self):
        self.region.close()
        self.closed = True
