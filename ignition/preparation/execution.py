# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed graph I/O metadata for the native two-frame execution ABI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ignition.components.capacity import BATCH_CAPACITIES, request_capacity
from ignition.components.completion import validate_completion_buffers

from .graphs import PreparedGraphBundle


@dataclass(frozen=True)
class GraphSignature:
    kind: str
    batch_capacity: int
    token_capacity: int
    inputs: tuple[str, ...] = ()
    sampling_mode: str = "general"
    speculation_enabled: bool | None = None
    prefill_capacity: int = 0
    packed_token_capacity: int = 0
    draft_tokens: int | None = None
    confidence_width: int = 0

    def __post_init__(self):
        if self.kind not in ("init", "restore", "snapshot", "prefill", "scored_prefill", "decode", "resume",
                             "vision", "speculative", "mixed"):
            raise ValueError("unknown prepared graph operation")
        if self.kind == "scored_prefill" and (self.batch_capacity != 1
                or self.token_capacity < 64 or self.token_capacity % 64
                or "prompt_target_tokens" not in self.inputs):
            raise ValueError("scored prefill requires aligned prompt rows and next-token targets")
        if "prompt_logprobs" in self.inputs:
            raise ValueError("prompt logprob outputs cannot be uploaded")
        if self.sampling_mode != "general":
            raise ValueError("graphs use the general stochastic sampling path")
        if self.speculation_enabled is not None and type(self.speculation_enabled) is not bool:
            raise TypeError("graph speculation mode must be a boolean or None")
        if self.kind == "speculative" and self.speculation_enabled is False:
            raise ValueError("speculative graphs require speculation")
        if self.kind == "decode" and self.speculation_enabled is True:
            raise ValueError("ordinary decode graphs require disabled speculation")
        if any(type(value) is not int or value <= 0 for value in (self.batch_capacity, self.token_capacity)):
            raise ValueError("graph capacities must be positive integers")
        if self.kind == "speculative" and (
                self.batch_capacity not in BATCH_CAPACITIES or not 1 <= self.token_capacity < 1 << 31):
            raise ValueError("speculative graphs require power-of-two batch capacities through 64 and positive Int32 verification widths")
        if type(self.prefill_capacity) is not int or self.prefill_capacity < 0:
            raise ValueError("prefill capacity must be a nonnegative integer")
        if self.kind == "mixed":
            if (self.batch_capacity not in BATCH_CAPACITIES[1:]
                    or not 1 <= self.token_capacity < 1 << 31
                    or self.prefill_capacity < 64 or self.prefill_capacity % 64
                    or self.prefill_capacity + (self.batch_capacity - 1) * self.token_capacity >= 1 << 31
                    or self.speculation_enabled is None
                    or (not self.speculation_enabled and self.token_capacity != 1)):
                raise ValueError("mixed graphs require a prompt capacity, at least two rows, and an explicit decode mode")
            if "prefill_row" not in self.inputs:
                raise ValueError("mixed graphs require the prefill_row input")
        elif self.prefill_capacity:
            raise ValueError("only mixed graphs declare a separate prefill capacity")
        if (type(self.packed_token_capacity) is not int or self.packed_token_capacity < 0
                or (self.packed_token_capacity and (self.kind != "speculative"
                    or not self.batch_capacity <= self.packed_token_capacity <= self.batch_capacity * self.token_capacity))):
            raise ValueError("packed verification capacity must fit the batch and per-request prefix bounds")
        # A speculative round and the round a mixed graph embeds both propose width - 1 tokens;
        # the package binding requires that depth and Helios selects both kinds by it.
        proposing = self.kind == "speculative" or (self.kind == "mixed" and self.speculation_enabled)
        if self.kind == "speculative" and "speculative_prefix_limits" in self.inputs and (
                self.draft_tokens is None or not self.packed_token_capacity):
            raise ValueError("device prefix allocation requires explicit draft depth and positive target capacity")
        if proposing:
            depth = self.token_capacity - 1 if self.draft_tokens is None else self.draft_tokens
            if type(depth) is not int or not 0 <= depth < (1 << 31) - 1 or (self.token_capacity == 1 and depth != 0):
                raise ValueError("proposing rounds require nonnegative Int32 draft depth and width-one rounds require zero")
            if self.kind == "mixed" and depth != self.token_capacity - 1:
                raise ValueError("a mixed round's embedded round proposes width - 1 tokens; declare that depth or none")
            object.__setattr__(self, "draft_tokens", depth)
        elif self.draft_tokens is not None:
            raise ValueError("explicit draft depth belongs to proposing rounds")
        if self.kind == "speculative":
            packed = self.batch_capacity * self.token_capacity if not self.packed_token_capacity else self.packed_token_capacity
            object.__setattr__(self, "packed_token_capacity", packed)
            if "query_lengths" in self.inputs and "speculative_prefix_limits" in self.inputs:
                raise ValueError("speculative rounds cannot bind both query lengths and prefix limits")
        if (type(self.confidence_width) is not int or self.confidence_width < 0
                or (self.confidence_width and (self.kind != "speculative"
                    or self.confidence_width != self.draft_tokens))):
            raise ValueError("confidence width must equal the producing round's draft depth")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("graph input names must be unique")
        object.__setattr__(self, "inputs", tuple(field for field in self.inputs
            if not (self.kind in ("decode", "speculative") and field == "token_ids")))
        if any(name in self.inputs for name in ("frame_index", "submission_sequence", "completion_records",
                                               "completion_logprobs", "logprob_requested", "logprob_token_ids",
                                               "logprob_token_id_counts", "readiness_timestamps",
                                               "completion_speculative_survival")):
            raise ValueError("reserved execution buffers are bound automatically")


@dataclass(frozen=True)
class TunedBinding:
    buffer: str
    tuning_graph: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in (self.buffer, self.tuning_graph)):
            raise ValueError("tuned bindings require buffer and tuning graph names")


@dataclass(frozen=True)
class TuningGraph:
    graph: Any
    outputs: tuple[str, ...]
    status_buffer: str

    def __post_init__(self):
        if not self.outputs or len(set(self.outputs)) != len(self.outputs):
            raise ValueError("tuning graphs require distinct persistent outputs")
        if self.status_buffer in self.outputs:
            raise ValueError("tuning status must be separate from selected outputs")


class PreparedExecution(PreparedGraphBundle):
    """Retain immutable graph ownership and export the matching native ABI.

    Graph I/O and declared persistent-state views belong in ``buffers``. Their
    storage owners, packed weights, and graph scratch belong in ``resources``. Native
    validation rejects overlapping I/O spans, incompatible graph capacities,
    and malformed reserved buffers before its worker becomes ready.
    """

    def __init__(self, *, signatures: Mapping[str, GraphSignature], max_active: int = 4,
                 estimated_durations_ns=None, export_zero_ranges=(), export_events=None,
                 tuning_graphs=None, tuned_bindings=None, **kwargs):
        serving = dict(kwargs.pop("graphs"))
        tuners = dict(tuning_graphs or {})
        if set(serving).intersection(tuners):
            raise ValueError("serving and tuning graph names must be distinct")
        if any(not isinstance(value, TuningGraph) for value in tuners.values()):
            raise TypeError("startup tuning entries must be TuningGraph objects")
        super().__init__(graphs={**serving, **{name: value.graph for name, value in tuners.items()}}, **kwargs)
        self.serving_graphs = MappingProxyType(serving)
        self.tuning_graphs = MappingProxyType(tuners)
        bindings = {name: tuple(values) for name, values in (tuned_bindings or {}).items()}
        if not bindings.keys() <= serving.keys():
            raise ValueError("tuned bindings must name serving graphs")
        self.tuned_bindings = MappingProxyType(bindings)
        request_capacity(max_active)
        if signatures.keys() != self.serving_graphs.keys():
            raise ValueError("every prepared graph requires exactly one signature")
        if any(not isinstance(value, GraphSignature) for value in signatures.values()):
            raise TypeError("prepared graph signatures must be GraphSignature objects")
        required = {"frame_index", "submission_sequence", "completion_records"}
        if not required <= self.buffers.keys():
            raise ValueError("execution requires frame index, submission sequence, and completion buffers")
        self.completion_token_capacity = validate_completion_buffers(self.buffers, request_capacity(max_active))
        for signature in signatures.values():
            if not set(signature.inputs) <= self.buffers.keys():
                raise ValueError("graph signature references an unregistered input buffer")
        for signature in signatures.values():
            if (signature.kind in ("speculative", "mixed") and signature.token_capacity > self.completion_token_capacity
                    or signature.draft_tokens is not None and signature.draft_tokens >= self.completion_token_capacity
                    or signature.confidence_width >= self.completion_token_capacity):
                raise ValueError("graph width, draft depth, or confidence exceeds completion token capacity")
        uploads = {"frame_index", "submission_sequence", "logprob_requested", "logprob_token_ids", "logprob_token_id_counts"}
        uploads.update(field for signature in signatures.values() for field in signature.inputs)
        for name, tuner in tuners.items():
            if not {*tuner.outputs, tuner.status_buffer} <= self.buffers.keys():
                raise ValueError("tuning graphs reference unregistered persistent buffers")
            status = self.buffers[tuner.status_buffer]
            if str(status.dtype) != "torch.int32" or tuple(status.shape) != (1,):
                raise ValueError("tuning graph status must be int32[1]")
            if uploads.intersection((*tuner.outputs, tuner.status_buffer)):
                raise ValueError("tuning outputs and status cannot receive host uploads")
        for values in bindings.values():
            if any(not isinstance(value, TunedBinding) for value in values):
                raise TypeError("serving tuning dependencies must be TunedBinding objects")
            if len({value.buffer for value in values}) != len(values):
                raise ValueError("serving graph has duplicate tuned buffers")
            for value in values:
                if value.tuning_graph not in tuners or value.buffer not in tuners[value.tuning_graph].outputs:
                    raise ValueError("tuned binding must name an output of its tuning graph")
        self.signatures = MappingProxyType(dict(signatures))
        durations = dict.fromkeys(self.serving_graphs, 0) if estimated_durations_ns is None else dict(estimated_durations_ns)
        if durations.keys() != self.serving_graphs.keys() or any(
                type(value) is not int or not 0 <= value < 1 << 64 for value in durations.values()):
            raise ValueError("graph duration estimates must cover the catalog with unsigned nanoseconds")
        self.estimated_durations_ns = MappingProxyType(durations)
        self.max_active = max_active
        self.export_zero_ranges = tuple(export_zero_ranges)
        self.export_events = dict(export_events or {})

    def manifest(self, *, all_variants=False) -> dict[str, Any]:
        context, handles, spans = self._validate_resources()
        descriptors = []
        offsets = {}
        cursor = 0
        reserved = ("frame_index", "submission_sequence")
        if "logprob_requested" in self.buffers:
            reserved += tuple(name for name in ("logprob_requested", "logprob_token_ids",
                                                "logprob_token_id_counts") if name in self.buffers)
        input_names = set(reserved)
        for signature in self.signatures.values():
            input_names.update(signature.inputs)
        for (name, tensor), (address, nbytes) in zip(self.buffers.items(), spans, strict=True):
            descriptors.append({"name": name, "address": address, "nbytes": nbytes,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape), "strides": list(tensor.stride())})
            if name in input_names:
                cursor = (cursor + 7) // 8 * 8
                offsets[name] = (cursor, nbytes)
                cursor += nbytes
        graphs = []
        speculative = any(signature.kind == "speculative" for signature in self.signatures.values())
        graph_handles = dict(zip(self.graphs, handles, strict=True))
        for name in self.serving_graphs:
            handle = graph_handles[name]
            signature = self.signatures[name]
            if not all_variants and signature.speculation_enabled not in (None, speculative):
                continue
            inputs = (*reserved, *signature.inputs)
            graphs.append({"name": name, "kind": signature.kind, "handle": handle,
                "sampling_mode": signature.sampling_mode,
                "batch_capacity": signature.batch_capacity, "token_capacity": signature.token_capacity,
                "prefill_capacity": signature.prefill_capacity,
                "confidence_width": signature.confidence_width,
                "estimated_duration_ns": self.estimated_durations_ns[name],
                "input_bindings": [{"buffer_name": field, "host_offset": offsets[field][0],
                                    "nbytes": offsets[field][1]} for field in inputs]})
            graphs[-1]["tuned_bindings"] = [dict(buffer=value.buffer, tuning_graph=value.tuning_graph)
                for value in self.tuned_bindings.get(name, ())]
            if signature.speculation_enabled is not None:
                graphs[-1]["speculation_enabled"] = signature.speculation_enabled
            if signature.packed_token_capacity:
                graphs[-1]["packed_token_capacity"] = signature.packed_token_capacity
            if signature.draft_tokens is not None:
                graphs[-1].update(draft_tokens=signature.draft_tokens,
                                  verify_tokens=signature.token_capacity, draft_source=1)
        abi_version = 6
        return {"abi_version": abi_version, "completion_token_capacity": self.completion_token_capacity,
                "context": context, "device": self.device,
                "max_active": self.max_active, "input_frame_bytes": (cursor + 7) // 8 * 8,
                "buffers": descriptors, "graphs": graphs,
                "tuning_graphs": [dict(name=name, handle=graph_handles[name], outputs=list(value.outputs),
                                       status_buffer=value.status_buffer) for name, value in self.tuning_graphs.items()]}

    def start_executor(self, *, peers=None):
        """Synchronize preparation and consume the bundle into the native executor."""
        import flare_heliopause
        import torch

        with self._handoff_lock:
            if self._consumed:
                raise RuntimeError("graph bundle has already been consumed by a native worker")
            manifest = self.manifest()
            torch.cuda.synchronize(self.device)
            arguments = {} if peers is None else {"peer_fds": [peer.fileno() for peer in peers]}
            executor = flare_heliopause.GraphExecutor(self, manifest, **arguments)
            self._consumed = True
            return executor

    def serve_rank(self, socket, *, rank, world):
        """Hand a peer to Rust until every rank drains; Python does no step work."""
        import flare_heliopause
        import torch

        with self._handoff_lock:
            if self._consumed:
                raise RuntimeError("graph bundle has already been consumed by a native worker")
            manifest = self.manifest()
            torch.cuda.synchronize(self.device)
            self._consumed = True
        flare_heliopause.serve_tensor_parallel_rank(self, manifest, socket.fileno(), rank, world)
