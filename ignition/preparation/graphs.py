# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation-owned CUDA graphs transferred once to a native replay worker."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ignition.planning import SUPPORTED_ARCHITECTURES, SUPPORTED_CAPABILITIES

if TYPE_CHECKING:
    import torch
    from .runtime_package import RecordedExportGraph


def _named_copy(values: Mapping[str, Any], kind: str) -> MappingProxyType:
    copied = dict(values)
    if any(not isinstance(name, str) or not name.strip() for name in copied):
        raise ValueError(f"{kind} names must be nonempty strings")
    return MappingProxyType(copied)


class PreparedGraphBundle:
    """Retain captured graphs, persistent CUDA buffers, and additional resources.

    ``device`` is the process-local CUDA ordinal used during capture. PyTorch
    does not expose a graph's capture device, so capture on that device remains
    a preparation invariant. Registration checks its capability and context,
    every declared buffer's device and extent, and instantiated graph handles.
    Export-only graphs retain native metadata and external pool owners after
    releasing their CUDA handles. Native replay still requires executables.

    Graphs must not use PyTorch generator-backed random operations: raw native
    launch bypasses PyTorch RNG bookkeeping. After handoff, callers must not
    reinstantiate graphs, resize storage, or mutate buffers while replay is in
    flight. Outputs that survive another graph require persistent storage;
    sharing a graph memory pool additionally requires valid allocation lifetimes
    for the intended variant order.

    Register a contiguous backing tensor for a noncontiguous view. Mapped host
    storage, streams, modules, and other owners belong in ``resources``. The
    native worker retains the complete bundle until it confirms stream drain.
    """

    def __init__(
        self,
        *,
        device: int,
        graphs: Mapping[str, torch.cuda.CUDAGraph | RecordedExportGraph],
        buffers: Mapping[str, torch.Tensor],
        resources: Iterable[Any] = (),
    ) -> None:
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError("device must be a nonnegative process-local CUDA ordinal")
        self._device = device
        self._graphs = _named_copy(graphs, "graph")
        if not self._graphs:
            raise ValueError("at least one captured graph is required")
        self._buffers = _named_copy(buffers, "buffer")
        self._resources = tuple(resources)
        self._graph_indices = MappingProxyType({name: index for index, name in enumerate(self._graphs)})
        self._handoff_lock = Lock()
        self._consumed = False

    @property
    def device(self) -> int:
        return self._device

    @property
    def graphs(self) -> Mapping[str, torch.cuda.CUDAGraph | RecordedExportGraph]:
        return self._graphs

    @property
    def buffers(self) -> Mapping[str, torch.Tensor]:
        return self._buffers

    @property
    def resources(self) -> tuple[Any, ...]:
        return self._resources

    @property
    def graph_indices(self) -> Mapping[str, int]:
        return self._graph_indices

    @property
    def consumed(self) -> bool:
        with self._handoff_lock:
            return self._consumed

    def start_worker(self, *, queue_capacity: int = 8) -> Any:
        """Consume this bundle into a qualification-only raw replay worker.

        Requires a native extension built with the ``qualification`` feature.
        Validation or native initialization failure leaves the bundle retryable.
        Successful registration consumes it permanently, including after close.
        The caller must close the worker before interpreter finalization.
        """
        if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int) or queue_capacity <= 0:
            raise ValueError("queue_capacity must be a positive integer")
        with self._handoff_lock:
            if self._consumed:
                raise RuntimeError("graph bundle has already been consumed by a native worker")
            worker = self._register(queue_capacity)
            self._consumed = True
            return worker

    def _register(self, queue_capacity: int) -> Any:
        import flare_heliopause

        if not hasattr(flare_heliopause, "GraphReplayWorker"):
            raise RuntimeError("raw replay requires the native qualification Cargo feature")
        import torch

        context, handles, spans = self._validate_resources()
        torch.cuda.synchronize(self._device)
        return flare_heliopause.GraphReplayWorker(
            self, context, self._device, handles, spans, queue_capacity=queue_capacity,
        )

    def _validate_resources(self, *, export_graphs=False) -> tuple[int, list[int], list[tuple[int, int]]]:
        import torch
        import flare_heliopause
        from .runtime_package import RecordedExportGraph

        with torch.cuda.device(self._device):
            capability = torch.cuda.get_device_capability(self._device)
            if capability not in SUPPORTED_CAPABILITIES:
                raise ValueError(f"CUDA device {self._device} is SM{capability[0]}{capability[1]};"
                                 f" {SUPPORTED_ARCHITECTURES} is required")
            handles = []
            identities = []
            for name, graph in self._graphs.items():
                if export_graphs and isinstance(graph, RecordedExportGraph):
                    context, device = flare_heliopause.current_context()
                    if graph.context != context or device != self._device:
                        raise ValueError(f"graph {name!r} export metadata belongs to another CUDA context")
                    if not graph.recorded:
                        raise ValueError(f"graph {name!r} export metadata has been released")
                    handles.append(graph.recorded)
                    identities.append(("recorded", id(graph.recorder), graph.recorded))
                    continue
                if not isinstance(graph, torch.cuda.CUDAGraph):
                    raise TypeError(f"graph {name!r} must be a torch.cuda.CUDAGraph")
                try:
                    handle = graph.raw_cuda_graph_exec()
                except RuntimeError as error:
                    raise ValueError(f"graph {name!r} must already be instantiated: {error}") from error
                if not isinstance(handle, int) or handle <= 0:
                    raise ValueError(f"graph {name!r} has no instantiated executable handle")
                handles.append(handle)
                identities.append(("executable", handle))
            if len(set(identities)) != len(identities):
                raise ValueError("named graphs must have unique executable handles")

            spans = []
            for name, tensor in self._buffers.items():
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"buffer {name!r} must be a torch.Tensor")
                if tensor.device.type != "cuda" or tensor.device.index != self._device:
                    raise ValueError(f"buffer {name!r} must reside on CUDA device {self._device}")
                if not tensor.is_contiguous():
                    raise ValueError(f"buffer {name!r} must be contiguous; register its contiguous backing tensor")
                nbytes = tensor.numel() * tensor.element_size()
                address = tensor.data_ptr()
                if nbytes <= 0 or address == 0:
                    raise ValueError(f"buffer {name!r} must have nonempty allocated storage")
                storage = tensor.untyped_storage()
                offset = address - storage.data_ptr()
                if offset < 0 or offset + nbytes > storage.nbytes():
                    raise ValueError(f"buffer {name!r} extends outside its backing storage")
                spans.append((address, nbytes))

            context, device = flare_heliopause.current_context()
            if device != self._device:
                raise RuntimeError(f"native CUDA context belongs to device {device}, expected {self._device}")
            return context, handles, spans
