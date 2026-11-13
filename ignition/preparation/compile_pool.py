# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Trace and compile prepared forwards in workers without model storage."""

import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from ignition import runtime_patches

from . import host_memory
from .compilation import _OPTIONS, _defer_full_collection
from .host_memory import cpu_workers, device_memory_info

# Device memory one compiler worker can hold: its CUDA context plus Inductor's constant folding.
_WORKER_DEVICE_BYTES = 2 << 30
# A discrete GPU's worker holds only device memory, about 0.7 GiB for Kimi K3's 93-layer forwards.
_DISCRETE_WORKER_DEVICE_BYTES = 1 << 30


@dataclass(frozen=True)
class CompilerResources:
    capacity: int
    available: int
    limit: int
    worker_bytes: int
    headroom: int


def compiler_resources(device, *, headroom_bytes=0, options=None, compile_workers=None):
    """Sample a rank's device capacity and its host's compiler memory budget."""
    if type(headroom_bytes) is not int or headroom_bytes < 0:
        raise ValueError("headroom_bytes must be a nonnegative byte count")
    options = _OPTIONS if options is None else options
    limit = min(cpu_workers(compile_workers), cpu_workers())
    if compile_workers is None:
        limit = min(limit, options['compile_threads'])
    available = host_memory.available_bytes()
    free = device_memory_info(device)[0]
    shared = torch.cuda.get_device_properties(device).is_integrated
    device_bytes = _WORKER_DEVICE_BYTES if shared else _DISCRETE_WORKER_DEVICE_BYTES
    capacity = max(0, (free - headroom_bytes) // device_bytes)
    if compile_workers is None:
        capacity = min(8, capacity)
    if shared:
        available = min(available, free)
    return CompilerResources(capacity, available, limit,
        host_memory.FORWARD_WORKER_BYTES + (device_bytes if shared else 0),
        max(host_memory._HEADROOM_BYTES, headroom_bytes if shared else 0))


def allocate_workers(resources, counts, hosts):
    """Admit forward compiler workers once per host, including zero-worker owners."""
    workers = [0] * len(resources)
    for host in dict.fromkeys(hosts):
        members = [rank for rank, name in enumerate(hosts) if name == host]
        jobs = sum(counts[rank] for rank in members)
        if not jobs:
            continue
        available = min(resources[rank].available for rank in members)
        headroom = max(resources[rank].headroom for rank in members)
        limit = min(jobs, min(resources[rank].limit for rank in members))
        remaining = available - headroom
        for _ in range(limit):
            able = [rank for rank in members if workers[rank] < resources[rank].capacity
                    and resources[rank].worker_bytes <= remaining]
            if not able:
                break
            rank = min(able, key=lambda rank: (workers[rank], rank))
            workers[rank] += 1
            remaining -= resources[rank].worker_bytes
        if not any(workers[rank] for rank in members):
            # Small forwards can fit below the concurrency estimate. Preserve one worker per host
            # under the minimum allowance without multiplying that allowance by the rank count.
            able = [rank for rank in members if resources[rank].capacity
                    and resources[rank].worker_bytes - host_memory.FORWARD_WORKER_BYTES
                    + host_memory._WORKER_BYTES <= remaining]
            if able:
                workers[able[0]] = 1
            else:
                raise MemoryError(f"host {host!r} has {jobs} missing forwards but no capacity for a forward compiler "
                                  f"({available >> 20} MiB available host memory, {headroom >> 20} MiB headroom)")
    return workers


@_defer_full_collection()
def _compile_forward(payload, device, options, path, num_threads, grad_enabled):
    runtime_patches.apply()
    from .aot import aot_forward
    from .device_binding import bind_device
    from .forward_metadata import loads

    torch.set_num_threads(num_threads)
    torch.set_grad_enabled(grad_enabled)
    torch.cuda.set_device(device)
    # A lent forward's metadata names its owner's GPU; it loads bound to this worker's.
    with bind_device(device):
        forward = loads(payload)
    artifact = aot_forward(forward, options=options)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
    try:
        artifact.save_compiled_function(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CompilePool:
    def __init__(self, device, *, headroom_bytes=0, options=None, compile_workers=None, workers=None):
        """Use admitted workers, or size one rank's pool from its host and device resources."""
        self.device = torch.device(device)
        self.pool = None
        options = dict(_OPTIONS if options is None else options)
        if workers is None:
            resources = compiler_resources(self.device, headroom_bytes=headroom_bytes, options=options,
                                           compile_workers=compile_workers)
            workers = allocate_workers([resources], [resources.limit], ["local"])[0]
        if type(workers) is not int or workers < 1:
            raise ValueError("forward compiler pool requires a positive admitted worker count")
        self.workers = workers
        # Nested Inductor pools would escape the host's admitted process budget.
        self.options = dict(options, compile_threads=1)

    def submit(self, forward, path):
        from .forward_metadata import dumps

        return self.submit_payload(dumps(forward), path)

    def submit_payload(self, payload, path):
        """Compile a forward serialized by `forward_metadata.dumps`, possibly on another rank, into `path`."""
        from torch._inductor.compile_worker.subproc_pool import SubprocKind, SubprocPool

        if self.pool is None:
            if os.name == "nt":
                self.pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"))
            else:
                self.pool = SubprocPool(self.workers, kind=SubprocKind.SPAWN)
        return self.pool.submit(_compile_forward, payload, self.device, self.options, str(path),
                                torch.get_num_threads(), torch.is_grad_enabled())

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.pool is not None:
            self.pool.shutdown()
