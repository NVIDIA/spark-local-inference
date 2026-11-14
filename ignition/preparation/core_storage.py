# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation storage whose virtual addresses can be reserved by native restore."""

import ctypes as C
from contextlib import contextmanager
from dataclasses import dataclass
import math
import json
import mmap
import os

from .runtime_core import _Cuda


@dataclass
class Mapping:
    address: int
    nbytes: int
    kind: str


class CoreStorage:
    """Back a PyTorch caching pool with CUDA VMM and own mapped host regions.

    The pool preserves PyTorch's capture and allocation-lifetime machinery.
    Only backing segments are allocated here. Callbacks and reservations remain
    alive until all tensors and graphs using the pool have been released.
    """

    # Above 2-6 TiB on Linux, where PyTorch's mimalloc CPU allocator places its 1 GiB arenas;
    # a peer rank allocates CPU tensors before it reserves these ranges.
    DEVICE_BASE = 0x40000000000 if os.name == "nt" else 0x80000000000
    # The native allocator selects a higher base when the device ranks need this range.
    HOST_BASE = 0xc0000000000
    RESERVATION_BYTES = 1 << 40
    # Each rank's host range; independent of an explicit device window.
    HOST_RESERVATION_BYTES = 1 << 40
    # Keep library-owned device storage above the ranges occupied by Linux and WDDM contexts.
    EXTERNAL_FLOOR = 0x4000000000
    # Where sealing below the floor starts; a context's own storage lies above it.
    LOW_SCAN_START = 1 << 32

    def __init__(self, *, rank=0, world=1, device=0, local_ranks=None, device_window=None):
        """``device_window=(base, extent)`` places rank ``r``'s fixed device window at
        ``base + r * extent`` instead of the platform default ``(DEVICE_BASE, RESERVATION_BYTES)``.
        A GPU whose virtual address space ends below the default window needs one.
        ``base`` must lie at or above ``EXTERNAL_FLOOR``; the native allocator validates
        alignment, the host ranges, and the GPU's address width."""
        import torch

        self.rank, self.world, self.device = rank, world, device
        self.local_ranks = world if local_ranks is None else local_ranks
        if type(self.local_ranks) is not int or not 1 <= self.local_ranks <= world:
            raise ValueError("local preparation ranks must be positive and fit the rank group")
        if device_window is not None:
            base, extent = (int(value) for value in device_window)
            if base < self.EXTERNAL_FLOOR or extent <= 0:
                raise ValueError(f"device window {base:#x}+{extent:#x} must start at or above "
                                 f"{self.EXTERNAL_FLOOR:#x} and be nonempty")
            self.DEVICE_BASE, self.RESERVATION_BYTES = base, extent
        properties = torch.cuda.get_device_properties(device)
        self.integrated = properties.is_integrated
        self._persistent_total_bytes = properties.total_memory
        self._retention_scope = False
        self._retention_pool = None
        window = (self.DEVICE_BASE, self.RESERVATION_BYTES)
        self.DEVICE_BASE += rank * self.RESERVATION_BYTES
        self.cuda = _Cuda()
        from flare_heliopause import flare_heliopause as native

        self.host_regions = []
        self.dynamic_regions = []
        self.persistent_regions = []
        self._persistent_arena = None
        self._persistent_closed = False
        self._persistent_model_owned = False
        self._context_tokens = []
        from .tensor_metadata import NumericStorage

        self.numeric_storage = NumericStorage()
        self._device_cursor = self.DEVICE_BASE + self.RESERVATION_BYTES // 2
        self.library = C.CDLL(native.__file__)
        self.library.flare_core_error.restype = C.c_char_p
        self.library.flare_core_snapshot.restype = C.c_void_p
        self.library.flare_core_initialize_window.argtypes = [
            C.c_uint32, C.c_uint32, C.c_int32, C.c_uint64, C.c_uint64]
        if self.library.flare_core_initialize_window(rank, world, device, *window):
            raise self.error or RuntimeError("cannot initialize native core allocator")
        layout = self._snapshot()
        self.granularity = layout["granularity"]
        gpudirect = C.c_int()
        self.cuda.call("cuDeviceGetAttribute", C.byref(gpudirect), C.c_int(116), C.c_int(device))
        self.gpudirect = bool(gpudirect.value)
        self.HOST_BASE = layout["host_base"]
        self._host_cursor = self.HOST_BASE
        self.low_reservations = []
        # WDDM retains its low allocation pool; export rejects retained storage below the floor.
        if self.integrated and os.name != "nt":
            self._reserve_low_window()
        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            native.__file__, "flare_core_allocate", "flare_core_free")
        self._allocator = allocator
        self._transient_allocator = torch.cuda.memory.CUDAPluggableAllocator(
            native.__file__, "flare_core_transient_allocate", "flare_core_transient_free")
        self.pool = torch.cuda.MemPool(allocator.allocator())
        self.graph_pool = torch.cuda.MemPool(allocator.allocator())
        self._preparation_pools = {self.pool.id}
        self._weight_arenas = []
        self._scope = torch.cuda.use_mem_pool(self.pool, device=device)
        self.peer_storage = None
        # RoCEnante runtimes whose regions this storage holds; export records them for native ranks.
        self.transports = []
        if world > 1:
            from .peer_storage import PeerStorage

            self.peer_storage = PeerStorage(self)

    def _reserve_low_window(self):
        """Keep library allocations outside the core above the addresses a serving context occupies.

        A unified-memory GPU such as GB10 places a context's own storage at low device addresses,
        and ordinary allocations such as cuBLAS's directly after it. That storage's extent differs
        between processes, so a serving process could not reserve the addresses those allocations
        had here. Reserving every free span below EXTERNAL_FLOOR moves them above it.
        """
        self.seal_low_window()
        probe = C.c_uint64()
        self.cuda.call("cuMemAlloc_v2", C.byref(probe), C.c_size_t(self.granularity))
        above = probe.value
        self.cuda.call("cuMemFree_v2", probe)
        if above < self.EXTERNAL_FLOOR:
            raise RuntimeError(f"device allocations still start at {above:#x} below {self.EXTERNAL_FLOOR:#x}")

    def _reserve_at(self, address, size):
        """Reserve exactly [address, address + size), or nothing when any of it is taken."""
        window = C.c_uint64()
        try:
            self.cuda.call("cuMemAddressReserve", C.byref(window), C.c_size_t(size), C.c_size_t(self.granularity),
                           C.c_uint64(address), C.c_uint64(0))
        except RuntimeError:
            return False
        if window.value != address:
            self.cuda.call("cuMemAddressFree", C.c_uint64(window.value), C.c_size_t(size))
            return False
        self.low_reservations.append((address, size))
        return True

    def _release_last(self):
        address, size = self.low_reservations.pop()
        self.cuda.call("cuMemAddressFree", C.c_uint64(address), C.c_size_t(size))

    def seal_low_window(self):
        """Reserve every span below EXTERNAL_FLOOR that is free now, including spans frees reopened.

        Storage occupied when the window was first reserved and freed later would otherwise let
        an external allocation land below the floor, where a serving context's own storage lies.
        """
        if not self.integrated or os.name == "nt":
            return
        granule = self.granularity
        address = -(-self.LOW_SCAN_START // granule) * granule
        while address < self.EXTERNAL_FLOOR:
            held = next((start + length for start, length in self.low_reservations
                         if start <= address < start + length), None)
            if held is not None:
                address = held
                continue
            # A taken granule costs one probe; a free one starts the longest free run found by halving.
            if not self._reserve_at(address, granule):
                address += granule
                continue
            self._release_last()
            end = min((start for start, _ in self.low_reservations if start > address),
                      default=self.EXTERNAL_FLOOR)
            size = end - address
            while not self._reserve_at(address, size) and size > granule:
                size = max(granule, size // 2 // granule * granule)
            # A granule taken since the probe is skipped.
            address += size

    def _snapshot(self):
        pointer = self.library.flare_core_snapshot()
        if not pointer:
            raise RuntimeError("native core allocator is not initialized")
        try:
            return json.loads(C.string_at(pointer))
        finally:
            self.library.flare_core_string_free(C.c_void_p(pointer))

    @property
    def mappings(self):
        result = {int(address): Mapping(int(address), size, "device")
                  for address, size in self._snapshot()["mappings"].items()}
        result.update((region.address, Mapping(region.address, region.nbytes, "device"))
                      for region in self.persistent_regions if region.nbytes)
        return result

    @property
    def error(self):
        message = self.library.flare_core_error()
        return None if message is None else RuntimeError(message.decode())

    @contextmanager
    def retain_transient_backing(self):
        """Keep transient physical allocations alive across compiled-forward warming."""
        if self._retention_scope:
            raise RuntimeError("nested transient physical retention scope")
        self._retention_scope = True
        retired_pool = False
        try:
            if self.library.flare_core_retention_begin():
                raise self.error or RuntimeError("cannot begin transient physical retention")
            try:
                yield
                if self._retention_pool is not None:
                    import torch

                    torch.cuda.synchronize(self.device)
                    retired_pool = True
                self._retire_retention_pool()
                if self.error is not None:
                    raise self.error
                if self.library.flare_core_retention_finish():
                    raise self.error or RuntimeError("cannot release transient physical backing")
            except BaseException as error:
                try:
                    if self._retention_pool is not None:
                        import torch

                        torch.cuda.synchronize(self.device)
                except BaseException as cleanup_error:
                    error.add_note(f"transient pool synchronization failed: {cleanup_error!r}")
                else:
                    try:
                        self._retire_retention_pool()
                    except BaseException as cleanup_error:
                        error.add_note(f"transient pool cleanup failed: {cleanup_error!r}")
                self.library.flare_core_retention_abort()
                raise
            else:
                if retired_pool:
                    self.seal_low_window()
        finally:
            self._retention_scope = False

    def _retire_retention_pool(self):
        pool, self._retention_pool = self._retention_pool, None
        if pool is None:
            return
        import torch

        del pool
        torch.cuda.empty_cache()

    @contextmanager
    def transient(self):
        """Route allocations through a core-backed transient pool.

        Outside physical retention, the pool is destroyed on exit so its cached
        segments return to the device. Inside physical retention, contexts share
        one pool that is retired before retained backing is released.
        """
        import torch

        shared = self._retention_scope
        if shared:
            pool = self._retention_pool
            if pool is None:
                pool = torch.cuda.MemPool(self._transient_allocator.allocator())
                self._retention_pool = pool
                self._preparation_pools.add(pool.id)
        else:
            pool = torch.cuda.MemPool(self._allocator.allocator())
            self._preparation_pools.add(pool.id)
        try:
            with torch.cuda.use_mem_pool(pool, device=self.device):
                yield
        finally:
            torch.cuda.synchronize(self.device)
            if not shared:
                del pool
                torch.cuda.empty_cache()
                # Returned segments may reopen low addresses that later library allocations would take.
                self.seal_low_window()

    def scratch_pool(self):
        """Own zero-initialized scratch whose physical backing can be released during compilation."""
        pool = ScratchPool(self)
        self._preparation_pools.add(pool.pool.id)
        return pool

    def packed_weights(self):
        """Own aligned immutable weight storage until graph users have drained."""
        from .packed_weights import PackedWeights

        arena = PackedWeights(self)
        self._weight_arenas.append(arena)
        return arena

    def unused_ranges(self):
        """Exclude free blocks and allocator padding from retained preparation pools."""
        import torch

        ranges = [span for arena in self._weight_arenas for span in arena.inactive_ranges()]
        for segment in torch.cuda.memory_snapshot(include_traces=False):
            if (segment["device"] != self.device
                    or tuple(segment["segment_pool_id"]) not in self._preparation_pools):
                continue
            for block in segment["blocks"]:
                live = 0 if block["state"] == "inactive" else block["requested_size"]
                if live < block["size"]:
                    ranges.append((block["address"] + live, block["address"] + block["size"]))
        if getattr(self, "_persistent_arena", None) is not None:
            ranges.extend(self._persistent_arena.unused_ranges())
        return ranges

    def _persistent_empty(self, shape, stride, dtype, nbytes, alignment):
        from .persistent import PersistentTensorArena

        if self._persistent_closed:
            raise RuntimeError("persistent tensor storage is closed")
        if self._persistent_arena is None:
            capacity = -(-self._persistent_total_bytes // self.granularity) * self.granularity
            address = self._device_cursor
            if address + capacity > self.DEVICE_BASE + self.RESERVATION_BYTES:
                raise MemoryError("persistent storage address range exhausted")
            region = DeviceRegion(self.cuda, address, 0, self.device, gpudirect=self.gpudirect)
            self._persistent_arena = PersistentTensorArena(region, capacity, self.granularity)
            self.persistent_regions.append(region)
            self._device_cursor += capacity
        return self._persistent_arena.empty(shape, stride, dtype, nbytes, alignment)

    def close(self):
        """Release persistent mappings after every referencing graph and tensor has drained."""
        if not self._persistent_closed:
            self.cuda.call("cuCtxSynchronize")
            if self._persistent_arena is not None:
                self._persistent_arena.close()
            self._persistent_closed = True

    def host_allocate(self, name, shape, dtype):
        size = math.prod(shape) * dtype.itemsize
        # Host mappings are whole host pages; Grace kernels may use 64 KiB pages.
        extent = -(-size // mmap.PAGESIZE) * mmap.PAGESIZE
        region_type = WindowsHostRegion if os.name == "nt" else HostRegion
        if self._host_cursor + extent > self.HOST_BASE + self.HOST_RESERVATION_BYTES:
            raise MemoryError("native core host-address range exhausted")
        region = region_type(self.cuda, self._host_cursor, extent, shape, dtype, self.device,
                             lock=self.integrated)
        if self._host_cursor + region.extent > self.HOST_BASE + self.HOST_RESERVATION_BYTES:
            region.close()
            raise MemoryError("native core host-address range exhausted")
        self._host_cursor += region.extent
        self.host_regions.append(region)
        return region

    def device_allocate(self, name, shape, dtype, *, group, maximum, units=None):
        """Map a prefix of a stable virtual range for runtime-sized storage."""
        import torch
        from b12x.sequence._shared.disk_table import _tensor_from_pointer

        units = shape[0] if units is None else units
        unit_bytes = math.prod(shape[1:]) * dtype.itemsize
        if not 0 < units <= maximum or shape[0] > maximum:
            raise ValueError("dynamic storage exceeds its virtual extent")
        extent = (maximum * unit_bytes + self.granularity - 1) // self.granularity * self.granularity
        nbytes = (units * unit_bytes + self.granularity - 1) // self.granularity * self.granularity
        address = self._device_cursor
        if address + extent > self.DEVICE_BASE + self.RESERVATION_BYTES:
            raise MemoryError("dynamic storage address range exhausted")
        region = DeviceRegion(self.cuda, address, nbytes, self.device)
        region.name, region.group = name, group
        region.unit_bytes, region.maximum, region.extent = unit_bytes, maximum, extent
        self.dynamic_regions.append(region)
        self._device_cursor += extent
        return _tensor_from_pointer(address, shape=shape, dtype=dtype,
                                    device=torch.device("cuda", self.device), nbytes=math.prod(shape) * dtype.itemsize)

    def grow(self, group, units):
        """Back additional units after compiler contexts have exited."""
        for region in self.dynamic_regions:
            if region.group == group:
                if not 0 < units <= region.maximum:
                    raise ValueError("dynamic storage exceeds its virtual extent")
                size = (units * region.unit_bytes + self.granularity - 1) // self.granularity * self.granularity
                region.grow(size)

    def __enter__(self):
        from .persistent import _ACTIVE_STORAGE

        self._scope.__enter__()
        self.numeric_storage.__enter__()
        self._context_tokens.append(_ACTIVE_STORAGE.set(self))
        return self

    def __exit__(self, kind, value, traceback):
        from .persistent import _ACTIVE_STORAGE

        try:
            self._scope.__exit__(kind, value, traceback)
        finally:
            try:
                self.numeric_storage.__exit__(kind, value, traceback)
            finally:
                _ACTIVE_STORAGE.reset(self._context_tokens.pop())
        native_error = self.error if kind is None else None
        if (self._persistent_arena is not None and not self._persistent_model_owned
                and (kind is not None or native_error is not None)):
            try:
                self.close()
            except BaseException as cleanup:
                if kind is not None:
                    value.add_note(f"Persistent storage cleanup failed; mappings retained: {cleanup}")
                else:
                    failure = RuntimeError("native core allocator failed")
                    failure.add_note(f"Persistent storage cleanup failed; mappings retained: {cleanup}")
                    raise failure from native_error
        if native_error is not None:
            raise RuntimeError("native core allocator failed") from native_error

class HostRegion:
    """Registered storage with identical CPU and device addresses."""

    def __init__(self, cuda, address, extent, shape, dtype, device, *, lock=False):
        """`lock`, for a unified-memory GPU, keeps the region resident: registering host memory there
        does not pin it, and a swapped-out page turns each GPU access into a fault."""
        import torch
        from b12x.sequence._shared.disk_table import _tensor_from_pointer

        self.cuda, self.address, self.extent = cuda, address, extent
        self.nbytes = math.prod(shape) * dtype.itemsize
        self.closed = True
        self.libc = C.CDLL(None, use_errno=True)
        self.libc.mmap.restype = C.c_void_p
        pointer = self.libc.mmap(C.c_void_p(address), C.c_size_t(extent), C.c_int(3),
                                 C.c_int(0x100022), C.c_int(-1), C.c_long(0))
        if pointer != address:
            raise OSError(C.get_errno(), f"cannot reserve native core host address {address:#x}")
        try:
            if lock and self.libc.mlock(C.c_void_p(address), C.c_size_t(extent)):
                raise OSError(C.get_errno(), f"cannot lock {extent} bytes of mapped host storage; raise RLIMIT_MEMLOCK")
            cuda.call("cuMemHostRegister_v2", C.c_void_p(address), C.c_size_t(extent), C.c_uint(2))
        except BaseException:
            self.libc.munmap(C.c_void_p(address), C.c_size_t(extent))
            raise
        self.closed = False
        try:
            gpu = C.c_uint64()
            cuda.call("cuMemHostGetDevicePointer_v2", C.byref(gpu), C.c_void_p(address), C.c_uint(0))
            if gpu.value != address:
                raise RuntimeError("mapped host storage requires identical CPU and CUDA addresses")
            views = [_tensor_from_pointer(address, shape=shape, dtype=dtype, device=device, nbytes=self.nbytes)
                     for device in (torch.device("cpu"), torch.device("cuda", device))]
            self.host_view, self.device_view = views
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.cuda.call("cuCtxSynchronize")
            self.cuda.call("cuMemHostUnregister", C.c_void_p(self.address))
            if self.libc.munmap(C.c_void_p(self.address), C.c_size_t(self.extent)):
                raise OSError(C.get_errno(), "cannot unmap native core host storage")
            self.closed = True


class _Location(C.Structure):
    _fields_ = [("kind", C.c_int), ("device", C.c_int)]


class _Properties(C.Structure):
    _fields_ = [("kind", C.c_int), ("handles", C.c_int), ("location", _Location),
                ("windows", C.c_void_p), ("flags", C.c_ubyte * 8)]


class _Access(C.Structure):
    _fields_ = [("location", _Location), ("flags", C.c_int)]


class WindowsHostRegion:
    """Host-backed CUDA VMM storage at one recorded CPU/GPU virtual address."""

    def __init__(self, cuda, address, extent, shape, dtype, device, *, lock=False):
        import torch
        from b12x.sequence._shared.disk_table import _tensor_from_pointer

        self.cuda, self.address = cuda, address
        self.nbytes = math.prod(shape) * dtype.itemsize
        self.closed = True
        numa = C.c_int()
        cuda.call("cuDeviceGetAttribute", C.byref(numa), C.c_int(134), C.c_int(device))
        properties = _Properties(1, 0, _Location(3, max(0, numa.value)), None, (C.c_ubyte * 8)())
        granularity = C.c_size_t()
        cuda.call("cuMemGetAllocationGranularity", C.byref(granularity), C.byref(properties), C.c_int(0))
        self.extent = -(-extent // granularity.value) * granularity.value
        pointer = C.c_uint64()
        cuda.call("cuMemAddressReserve", C.byref(pointer), C.c_size_t(self.extent),
                  C.c_size_t(granularity.value), C.c_uint64(address), C.c_uint64(0))
        mapped = False
        try:
            if pointer.value != address:
                raise RuntimeError(f"cannot reserve mapped host CUDA address {address:#x}")
            physical = C.c_uint64()
            cuda.call("cuMemCreate", C.byref(physical), C.c_size_t(self.extent), C.byref(properties), C.c_uint64(0))
            try:
                cuda.call("cuMemMap", pointer, C.c_size_t(self.extent), C.c_size_t(0), physical, C.c_uint64(0))
                mapped = True
            finally:
                cuda.call("cuMemRelease", physical)
            access = (_Access * 2)(_Access(properties.location, 3), _Access(_Location(1, device), 3))
            cuda.call("cuMemSetAccess", pointer, C.c_size_t(self.extent), access, C.c_size_t(2))
            C.memset(address, 0, self.extent)
            self.host_view, self.device_view = [
                _tensor_from_pointer(address, shape=shape, dtype=dtype, device=location, nbytes=self.nbytes)
                for location in (torch.device("cpu"), torch.device("cuda", device))]
        except BaseException:
            if mapped:
                cuda.call("cuMemUnmap", pointer, C.c_size_t(self.extent))
            cuda.call("cuMemAddressFree", pointer, C.c_size_t(self.extent))
            raise
        self.closed = False

    def close(self):
        if not self.closed:
            self.cuda.call("cuCtxSynchronize")
            self.cuda.call("cuMemUnmap", C.c_uint64(self.address), C.c_size_t(self.extent))
            self.cuda.call("cuMemAddressFree", C.c_uint64(self.address), C.c_size_t(self.extent))
            self.closed = True


class ScratchPool:
    """Keep scratch addresses and tensor aliases while compiler workers use only their metadata.

    All tensors allocated in this pool must be disposable scratch. No GPU work
    may access them between release_backing and materialize. Materialization
    restores zeroed storage before warmup, capture, or export.
    """

    def __init__(self, storage):
        import torch

        self.storage = storage
        self.pool = torch.cuda.MemPool(storage._allocator.allocator())
        self.pool_id = self.pool.id
        self.unbacked = []

    def __enter__(self):
        import torch

        if self.pool is None:
            raise RuntimeError("scratch allocations are closed")
        self.scope = torch.cuda.use_mem_pool(self.pool, device=self.storage.device)
        self.scope.__enter__()
        return self

    def __exit__(self, *exc):
        self.scope.__exit__(*exc)

    def retire(self):
        """Close allocations and return cached segments after all allocation contexts have exited."""
        import torch

        self.pool = None
        torch.cuda.empty_cache()

    def release_backing(self):
        """Discard scratch contents after all GPU users have completed."""
        import torch

        if self.unbacked:
            raise RuntimeError("scratch backing is already released")
        torch.cuda.synchronize(self.storage.device)
        mappings = self.storage.mappings
        try:
            for segment in torch.cuda.memory_snapshot(self.pool_id, include_traces=False):
                address = segment["address"]
                size = mappings[address].nbytes
                self.storage.cuda.call("cuMemUnmap", C.c_uint64(address), C.c_size_t(size))
                self.unbacked.append((address, size))
        except BaseException:
            self.materialize()
            raise

    def materialize(self):
        """Restore the scratch pool's physical storage at its original addresses."""
        while self.unbacked:
            address, size = self.unbacked[-1]
            DeviceRegion(self.storage.cuda, address, size, self.storage.device)
            self.unbacked.pop()


class DeviceRegion:
    def __init__(self, cuda, address, nbytes, device, *, gpudirect=False):
        self.cuda, self.device, self.address, self.nbytes = cuda, device, address, 0
        self.gpudirect = gpudirect
        self.closed = False
        self.mappings = []
        self.grow(nbytes)

    def grow(self, nbytes):
        if self.closed:
            raise RuntimeError("device mapping is closed")
        if nbytes < self.nbytes:
            raise ValueError("preparation storage cannot shrink")
        if nbytes == self.nbytes:
            return
        cuda, device = self.cuda, self.device
        address, size = self.address + self.nbytes, nbytes - self.nbytes
        handle = C.c_uint64()
        properties = _Properties(1, 0 if os.name == "nt" else 1, _Location(1, device), None,
                                 (C.c_ubyte * 8)(0, int(self.gpudirect), 0, 0, 0, 0, 0, 0))
        cuda.call("cuMemCreate", C.byref(handle), C.c_size_t(size), C.byref(properties), C.c_uint64(0))
        mapped = False
        try:
            try:
                cuda.call("cuMemMap", C.c_uint64(address), C.c_size_t(size), C.c_size_t(0), handle, C.c_uint64(0))
                mapped = True
            finally:
                cuda.call("cuMemRelease", handle)
            access = _Access(_Location(1, device), 3)
            cuda.call("cuMemSetAccess", C.c_uint64(address), C.c_size_t(size), C.byref(access), C.c_size_t(1))
            cuda.call("cuMemsetD8_v2", C.c_uint64(address), C.c_ubyte(0), C.c_size_t(size))
        except BaseException:
            if mapped:
                cuda.call("cuMemUnmap", C.c_uint64(address), C.c_size_t(size))
            raise
        self.nbytes = nbytes
        self.mappings.append((address, size))

    def close(self):
        while self.mappings:
            address, size = self.mappings[-1]
            self.cuda.call("cuMemUnmap", C.c_uint64(address), C.c_size_t(size))
            self.mappings.pop()
            self.nbytes -= size
        self.closed = True
