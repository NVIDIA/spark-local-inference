# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation workspace and typed invocation-local scratch views."""

from __future__ import annotations

from contextlib import nullcontext
from math import prod
from weakref import finalize

import torch


def scratch_views(backing, shapes):
    """Partition scratch into disjoint tensors with independent storage descriptors.

    Compilers can track each partition independently. Operations using the
    unpartitioned backing must execute separately from operations using these views.
    """
    if backing.ndim != 1 or not backing.is_contiguous() or backing.storage_offset():
        raise ValueError("scratch partitions require a contiguous one-dimensional backing")
    offset, views = 0, []
    for shape in shapes:
        shape = tuple(shape)
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("scratch partition dimensions must be positive integers")
        offset = (offset + 255) // 256 * 256
        end = offset + prod(shape) * backing.element_size()
        if end > backing.nbytes:
            raise ValueError("scratch partitions exceed their backing")
        storage = backing.untyped_storage()[offset:end]
        views.append(torch.empty(0, dtype=backing.dtype, device=backing.device).set_(storage, 0, shape))
        offset = end
    return tuple(views)


class ScratchArena:
    """Share tuning workspace and retire declaration storage before compilation.

    With ``overlap`` false, preparation creates no auxiliary stream and accepts
    no concurrent plans. Serving forwards allocate their own temporary tensors.
    """

    def __init__(self, device, *, overlap=True, allocation_pool=None):
        if type(overlap) is not bool:
            raise TypeError("overlap must be a boolean")
        self.tensor = torch.empty(0, dtype=torch.uint8, device=device)
        self.allocation_pool = allocation_pool
        self.overlap = overlap
        self.frozen = False
        self._released = False
        self._finalizers = []
        self._preparation_releases = []
        self._parallel_plans = []
        self._auxiliary_stream = None
        self._recorded_views = None
        self._binding_views = None

    @property
    def device(self):
        return self.tensor.device

    @property
    def nbytes(self):
        return self.tensor.nbytes

    @property
    def auxiliary_stream(self):
        self._require_active()
        if not self.overlap:
            raise RuntimeError("stream overlap is disabled; compose this branch on the main stream")
        if self._auxiliary_stream is None:
            from cuda.bindings import driver as cuda

            # PyTorch's round-robin stream pool can alias a later capture stream.
            with torch.cuda.device(self.device):
                status, handle = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING)
                if status != cuda.CUresult.CUDA_SUCCESS:
                    raise RuntimeError(f"cannot create overlap stream: {status}")
                stream = torch.cuda.ExternalStream(int(handle), device=self.device)
            finalize(stream, cuda.cuStreamDestroy, handle).atexit = False
            self._auxiliary_stream = stream
        return self._auxiliary_stream

    def register_parallel(self, *plans):
        """Declare plans that need disjoint workspace during preparation."""
        if not self.overlap:
            raise RuntimeError("stream overlap is disabled; serial plans share scratch")
        if self.frozen:
            raise RuntimeError("scratch declarations are frozen")
        if len(plans) < 2:
            raise ValueError("parallel scratch requires at least two plans")
        self._parallel_plans.append(plans)

    def replace_parallel_plans(self, replacements):
        """Rebind declared forks when prepared plan choices change before capture."""
        if self.frozen:
            raise RuntimeError("scratch declarations are frozen")
        self._parallel_plans = [tuple(replacements.get(plan, plan) for plan in group)
                                for group in self._parallel_plans]

    def parallel_nbytes(self, requirement):
        sizes = {}
        def aligned(plan):
            key = id(plan)
            if key not in sizes:
                sizes[key] = (requirement(plan) + 255) // 256 * 256
            return sizes[key]
        return max((sum(aligned(plan) for plan in group)
                    for group in self._parallel_plans), default=0)

    def reserve(self, nbytes):
        self._require_active()
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("scratch size must be a nonnegative integer")
        if nbytes > self.nbytes:
            if self.frozen:
                raise RuntimeError(f"frozen scratch arena has {self.nbytes} bytes, requires {nbytes}")
            with (nullcontext() if self.allocation_pool is None else
                  torch.cuda.use_mem_pool(self.allocation_pool, device=self.device.index)):
                self.tensor = torch.empty((nbytes + 255) // 256 * 256,
                                          dtype=torch.uint8, device=self.device)
        return self.tensor

    def freeze(self):
        self.frozen = True

    def release_preparation(self):
        """Retire finalized workspace before capture when all consumers use local scratch.

        The caller must finish tuning and binding finalizers, synchronize their
        work, and ensure no execution depends on this arena's storage. Existing
        tensor views retain their own storage ownership; this method does not
        inspect consumers or infer whether release is safe.
        """
        if self._released:
            return
        if not self.frozen or self._finalizers:
            raise RuntimeError("scratch preparation release requires completed binding finalization")
        for callback in self._preparation_releases:
            callback()
        self._preparation_releases.clear()
        self.tensor = torch.empty(0, dtype=torch.uint8, device=self.device)
        self._parallel_plans.clear()
        self._released = True

    def register_preparation_release(self, callback):
        """Release shared declaration owners after every binding has finalized."""
        self._require_active()
        if not callable(callback):
            raise TypeError("preparation release must be callable")
        self._preparation_releases.append(callback)

    def _require_active(self):
        if self._released:
            raise RuntimeError("scratch preparation storage has been released")

    def trim(self, nbytes):
        self._require_active()
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("scratch size must be a nonnegative integer")
        capacity = (nbytes + 255) // 256 * 256
        if capacity > self.nbytes:
            raise ValueError("selected scratch exceeds the reserved candidate capacity")
        if capacity < self.nbytes:
            if self.frozen:
                raise RuntimeError("cannot trim a frozen scratch arena")
            with (nullcontext() if self.allocation_pool is None else
                  torch.cuda.use_mem_pool(self.allocation_pool, device=self.device.index)):
                self.tensor = torch.empty(capacity, dtype=torch.uint8, device=self.device)

    def views(self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]) -> tuple[torch.Tensor, ...]:
        """Return disjoint views for simultaneous use within one serialized call.

        Calls reuse the same offsets. Preparation must refresh retained views
        after growth and freeze the arena before recording CUDA graph pointers.
        """
        if self._recorded_views is not None:
            self._recorded_views.append(shapes_and_dtypes)
        if self._binding_views is not None:
            shapes_and_dtypes = next(self._binding_views)
        sizes = []
        for shape, dtype in shapes_and_dtypes:
            if any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError("scratch dimensions must be nonnegative integers")
            if not isinstance(dtype, torch.dtype):
                raise TypeError("scratch dtype must be a torch dtype")
            sizes.append(prod(shape) * dtype.itemsize)
        storage = self.reserve(sum((size + 255) // 256 * 256 for size in sizes))
        result, offset = [], 0
        for (shape, dtype), size in zip(shapes_and_dtypes, sizes, strict=True):
            result.append(storage.narrow(0, offset, size).view(dtype).reshape(shape))
            offset += (size + 255) // 256 * 256
        return tuple(result)

    def register_finalizer(self, callback):
        if self.frozen:
            raise RuntimeError("scratch binding declarations are frozen")
        if not callable(callback):
            raise TypeError("scratch binding finalizer must be callable")
        self._finalizers.append(callback)

    def finalize(self, *, group=None, retain=False):
        """Bind scratch views, retaining callbacks only for preparation trial graphs."""
        self._require_active()
        if group is None:
            for callback in self._finalizers:
                callback()
        else:
            import torch.distributed as dist

            self._recorded_views = []
            try:
                for callback in self._finalizers:
                    callback()
                layouts = self._recorded_views
            finally:
                self._recorded_views = None
            ranks = [None] * dist.get_world_size(group)
            dist.all_gather_object(ranks, (self.nbytes, layouts), group=group)
            shared = _shared_views([views for _, views in ranks])
            required = max((sum((prod(shape) * dtype.itemsize + 255) // 256 * 256
                                for shape, dtype in views) for views in shared), default=0)
            before = self.tensor
            self.reserve(max(required, *(size for size, _ in ranks)))
            if shared != layouts or self.tensor is not before:
                self._binding_views = iter(shared)
                try:
                    for callback in self._finalizers:
                        callback()
                finally:
                    self._binding_views = None
        if not retain:
            self._finalizers.clear()


def _shared_views(layouts):
    """Give equivalent rank forwards identical scratch sizes and offsets."""
    result = []
    for calls in zip(*layouts, strict=True):
        views = []
        for specs in zip(*calls, strict=True):
            shape, dtype = specs[0]
            if any(spec != specs[0] for spec in specs[1:]):
                if any(len(other) != 1 or kind != torch.uint8 for other, kind in specs):
                    raise ValueError("tensor-parallel scratch layouts differ beyond byte-buffer capacity")
                shape = (max(other[0] for other, _ in specs),)
            views.append((shape, dtype))
        result.append(tuple(views))
    return result


def scratch_storage(workspace, *, nbytes, device):
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError("scratch size must be a nonnegative integer")
    device = torch.device(device)
    if isinstance(workspace, ScratchArena):
        if workspace.device != device:
            raise ValueError("scratch arena and operator must share a device")
        return workspace.reserve(nbytes)
    if workspace is None:
        return torch.empty(nbytes, dtype=torch.uint8, device=device)
    if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8:
        raise TypeError("planned scratch must be a uint8 tensor or ScratchArena")
    if workspace.device != device or workspace.ndim != 1 or not workspace.is_contiguous():
        raise ValueError("planned scratch requires contiguous one-dimensional storage on the operator device")
    if workspace.nbytes < nbytes:
        raise ValueError(f"planned scratch has {workspace.nbytes} bytes, requires {nbytes}")
    return workspace


def bind_scratch(layout, *, device, workspace=None):
    """Bind contiguous operand prefixes from caller-owned, typed capacity buffers."""
    if workspace is None:
        return {name: torch.empty(shape, dtype=dtype, device=device)
                for name, (shape, dtype) in layout.items()}
    result = {}
    for name, (shape, dtype) in layout.items():
        tensor = workspace[name]
        if tensor.dtype != dtype or tensor.device != torch.device(device) or not tensor.is_contiguous():
            raise ValueError(f"scratch {name} requires contiguous {dtype} storage on {device}")
        if tensor.numel() < prod(shape):
            raise ValueError(f"scratch {name} has {tensor.numel()} elements, requires {prod(shape)}")
        result[name] = tensor.view(-1)[:prod(shape)].view(shape)
    return result


def planned_scratch(specs, workspace=None):
    specs = tuple(specs)
    if isinstance(workspace, ScratchArena):
        if not specs or any(spec.device != workspace.device for spec in specs):
            raise ValueError("scratch specs must be nonempty and share the arena device")
        views = workspace.views(*((spec.shape, spec.dtype) for spec in specs))
        return views[0] if len(views) == 1 else views
    if len(specs) != 1 or specs[0].dtype != torch.uint8 or len(specs[0].shape) != 1:
        raise ValueError("operator scratch plan must expose one uint8 byte buffer")
    spec, = specs
    return scratch_storage(workspace, nbytes=spec.nbytes, device=spec.device)


def prime_owners(workspace, *owners):
    """Keep preparation storage without retaining disposable declaration views.

    b12x retains priming owners until a plan is released. Functional execution
    supplies invocation tensors; standalone calls retain their concrete owners.
    """
    return (workspace,) if isinstance(workspace, ScratchArena) else owners
