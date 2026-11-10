# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Device storage every rank of an exchange group can read through CUDA IPC."""

from contextlib import suppress

import torch
import torch.distributed as dist


class _DeviceArray:
    def __init__(self, pointer, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (pointer, False),
                                         "version": 3, "strides": None}


class SharedPages:
    """One byte buffer per rank, exported to and mapped by every other rank.

    `tensor` is this rank's buffer; `addresses` holds every rank's buffer address in
    this process, as int64 device values that kernels convert to pointers. Peer
    buffers are written only by their owners; readers must order reads after the
    owner's writes through a collective. All ranks allocate and close together,
    after native replay has drained. `ipc` is the CudaRTLibrary-compatible allocator.
    """

    def __init__(self, group, device, nbytes, *, ipc=None):
        from b12x.comm.pcie._cuda_ipc import CudaRTLibrary
        from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReduce

        if type(nbytes) is not int or nbytes <= 0 or nbytes % 16:
            raise ValueError("shared pages require a positive size of whole 16-byte packs")
        self.group, self.device, self.nbytes = group, torch.device(device), nbytes
        # Core preparation supplies symmetric mappings that native restore reproduces.
        if ipc is None:
            ipc = CudaRTLibrary()
            ipc.cudaSetDevice(self.device.index)
        self._ipc = ipc
        self._buffer = PCIeOneshotAllReduce._allocate_shared_buffer(group, nbytes, zero_fill=True, ipc=self._ipc)
        self.tensor = torch.as_tensor(_DeviceArray(self._buffer.local_ptr, nbytes), device=self.device)
        self.addresses = torch.tensor(self._buffer.peer_ptrs, dtype=torch.int64, device=self.device)
        self._closed = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize(self.device)
        # Every importer unmaps before any owner frees its export.
        dist.barrier(group=self.group)
        for pointer in self._buffer.remote_ptrs:
            with suppress(Exception):
                self._ipc.cudaIpcCloseMemHandle(pointer)
        dist.barrier(group=self.group)
        with suppress(Exception):
            self._ipc.cudaFree(self._buffer.local_ptr)
        dist.barrier(group=self.group)
