# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""In-process device storage with a fixed virtual extent whose physical prefix grows."""

import ctypes as C
import math

from .core_storage import DeviceRegion, _Location, _Properties
from .runtime_core import _Cuda


class VirtualStorage:
    """Reserve each runtime-sized tensor's full extent and back only a measured prefix.

    Compiled forwards bind tensor shapes. A core therefore compiles against the
    full virtual extent and backs the measured working set afterwards; this gives an
    in-process preparation the same shapes, so its compiled forwards do not depend on
    the memory one launch happens to find free. It offers CoreStorage's
    `device_allocate` and `grow`; `close` unmaps everything after native replay drains.
    """

    def __init__(self, device):
        import torch

        self.device = torch.device(device).index
        self.cuda = _Cuda()
        granularity = C.c_size_t()
        properties = _Properties(1, 1, _Location(1, self.device), None, (C.c_ubyte * 8)())
        self.cuda.call("cuMemGetAllocationGranularity", C.byref(granularity), C.byref(properties), C.c_int(0))
        self.granularity = granularity.value
        self.dynamic_regions = []
        self.closed = False

    def _round(self, nbytes):
        return -(-nbytes // self.granularity) * self.granularity

    def device_allocate(self, name, shape, dtype, *, group, maximum, units=None):
        """Reserve `maximum` units of address space and back the first `units`."""
        import torch
        from b12x.sequence._shared.disk_table import _tensor_from_pointer

        units = shape[0] if units is None else units
        unit_bytes = math.prod(shape[1:]) * dtype.itemsize
        if not 0 < units <= maximum or shape[0] > maximum:
            raise ValueError("dynamic storage exceeds its virtual extent")
        extent = self._round(maximum * unit_bytes)
        address = C.c_uint64()
        self.cuda.call("cuMemAddressReserve", C.byref(address), C.c_size_t(extent), C.c_size_t(self.granularity),
                       C.c_uint64(0), C.c_uint64(0))
        try:
            region = DeviceRegion(self.cuda, address.value, self._round(units * unit_bytes), self.device)
        except BaseException:
            self.cuda.call("cuMemAddressFree", C.c_uint64(address.value), C.c_size_t(extent))
            raise
        region.name, region.group = name, group
        region.unit_bytes, region.maximum, region.extent = unit_bytes, maximum, extent
        self.dynamic_regions.append(region)
        return _tensor_from_pointer(address.value, shape=shape, dtype=dtype,
                                    device=torch.device("cuda", self.device), nbytes=math.prod(shape) * dtype.itemsize)

    def grow(self, group, units):
        """Back additional units after compiler contexts have exited."""
        for region in self.dynamic_regions:
            if region.group == group:
                if not 0 < units <= region.maximum:
                    raise ValueError("dynamic storage exceeds its virtual extent")
                region.grow(self._round(units * region.unit_bytes))

    @property
    def nbytes(self):
        return sum(region.nbytes for region in self.dynamic_regions)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cuda.call("cuCtxSynchronize")
        for region in reversed(self.dynamic_regions):
            for address, size in reversed(region.mappings):
                self.cuda.call("cuMemUnmap", C.c_uint64(address), C.c_size_t(size))
            self.cuda.call("cuMemAddressFree", C.c_uint64(region.address), C.c_size_t(region.extent))
        self.dynamic_regions.clear()
