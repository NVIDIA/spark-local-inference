# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep QSA selector execution beside QKV production through forward compilation."""

import itertools
import weakref

import torch
from b12x.attention import qsa

from ignition.components.linear import _parallel_mx_projections
from ignition.preparation.tuning import register_finalizer

_CONTEXTS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()
_MUTABLE_BINDING_FIELDS = (
    "compressed_k_cache", "raw_k_ring", "raw_logical_positions",
    "raw_rope_positions", "raw_interval_start_positions", "output", "selected_positions",
)
_DYNAMIC_FIELDS = ("query", "index_query", "raw_index_key", "request_ids", "query_positions",
                   "rope_positions", "sequence_lengths", "query_start_loc", "num_accepted_tokens", "is_prefilling")


@torch.library.custom_op("flare::parallel_qsa", mutates_args=("writes", "workspace"))
def _parallel_qsa(source: torch.Tensor, reads: list[torch.Tensor], writes: list[torch.Tensor],
                  workspace: torch.Tensor, handle: int) -> None:
    _CONTEXTS[handle].run(reads, writes, workspace)


@_parallel_qsa.register_fake
def _fake_parallel_qsa(source, reads, writes, workspace, handle):
    pass


class FunctionalQSA:
    def __init__(self, attention):
        self.plan = attention.plan
        self.read_names = tuple(name for name in attention.binding_kwargs if name not in _MUTABLE_BINDING_FIELDS)
        self.cache_only = attention.cache_only
        self.selection_only = attention.sparse is not None
        self.handle = next(_HANDLES)
        _CONTEXTS[self.handle] = self

    def __call__(self, source, bindings, dynamic, workspace):
        reads = [bindings[name] for name in self.read_names] + [dynamic[name] for name in _DYNAMIC_FIELDS]
        writes = [bindings[name] for name in _MUTABLE_BINDING_FIELDS]
        _parallel_qsa(source, reads, writes, workspace, self.handle)

    def run(self, reads, writes, workspace):
        count = len(self.read_names)
        bindings = dict(zip(self.read_names, reads[:count], strict=True))
        bindings.update(zip(_MUTABLE_BINDING_FIELDS, writes, strict=True))
        binding = qsa.bind(self.plan, scratch=workspace, **bindings)
        qsa.run(binding, **dict(zip(_DYNAMIC_FIELDS, reads[count:], strict=True)),
                selection_only=self.selection_only, cache_only=self.cache_only)


class ConcurrentAttention:
    """Prepared events and disjoint scratch for QKV, index projection, and QSA.

    The opaque QSA call reconstructs its binding from explicit tensor operands
    so compiler functionalization cannot detach mutations from their storage.
    Its source operand keeps projection input live through the selector join.
    b12x owns selection, attention, and the selection-stream join. The context
    registry retains only weak references; the prepared attention owns its
    context until all captured graphs have drained.
    """

    def __init__(self, attention, *, session, workspace):
        from b12x.gemm.blockscaled._a16 import _weight_parts

        self.attention, self.arena = attention, workspace
        self.projections = tuple(attention.projections[key]
                                 for key in ("qkv_proj", "indexer.index_qk_proj"))
        self.values, self.scales, self.outputs, self.plans = [], [], [], []
        for projection in self.projections:
            values, scales, gain, fp4 = _weight_parts(projection.weight)
            if (values.dtype not in (torch.float8_e4m3fn, torch.uint8) or gain is not None or fp4
                    or projection.bias is not None):
                raise ValueError("concurrent attention projections require unbiased MXFP8 or MXFP4 weights")
            self.values.append(values)
            self.scales.append(scales.view(torch.uint8))
            self.outputs.append(projection.output)
            self.plans.append(projection.plan.handle)
        workspace.register_parallel(attention.plan, *(projection.plan for projection in self.projections))
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        self.ready, self.done = torch.cuda.Event(), torch.cuda.Event()
        self.ready.record(self.stream)
        self.done.record(self.stream)
        self.read_names = tuple(name for name in attention.binding_kwargs if name not in _MUTABLE_BINDING_FIELDS)
        self.dynamic_names = tuple(attention.dynamic())
        self.handle = next(_HANDLES)
        _CONTEXTS[self.handle] = self
        register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        self.workspace, *self.projection_workspaces = self.arena.views(*(
            ((plan.memory_requirements().scratch_nbytes,), torch.uint8)
            for plan in (self.attention.plan, *(projection.plan for projection in self.projections))))

    def functional(self, *, session):
        self.output_layouts = tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in self.outputs)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.workspace_bytes = self.workspace.numel()
        self.projection_workspace_bytes = tuple(t.numel() for t in self.projection_workspaces)
        self.outputs = self.workspace = self.projection_workspaces = None
        self.arena = None

    def project_runtime(self, source):
        outputs = [torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
                   for shape, stride, dtype in self.output_layouts]
        workspaces = [torch.empty(size, dtype=torch.uint8, device=source.device)
                      for size in self.projection_workspace_bytes]
        _parallel_mx_projections(source, self.values, self.scales, outputs,
            workspaces, self.plans, self.stream_handle, False)
        return outputs, workspaces

    def attend_runtime(self, source, bindings, dynamic, projection_workspaces):
        reads = ([bindings[name] for name in self.read_names]
                 + [dynamic[name] for name in self.dynamic_names] + projection_workspaces)
        writes = [bindings[name] for name in _MUTABLE_BINDING_FIELDS]
        workspace = torch.empty(self.workspace_bytes, dtype=torch.uint8, device=source.device)
        _parallel_qsa(source, reads, writes, workspace, self.handle)

    def project(self, source):
        _parallel_mx_projections(source, self.values, self.scales, self.outputs,
            self.projection_workspaces, self.plans, self.stream_handle, False)

    def attend(self, source):
        bindings = self.attention.binding_kwargs
        dynamic = self.attention.dynamic()
        reads = [bindings[name] for name in self.read_names] + [dynamic[name] for name in self.dynamic_names]
        writes = [bindings[name] for name in _MUTABLE_BINDING_FIELDS]
        _parallel_qsa(source, reads, writes, self.workspace, self.handle)
        return self.attention.scratch.attended

    def run(self, reads, writes, workspace):
        count = len(self.read_names)
        bindings = dict(zip(self.read_names, reads[:count], strict=True))
        bindings.update(zip(_MUTABLE_BINDING_FIELDS, writes, strict=True))
        binding = qsa.bind(self.attention.plan, scratch=workspace, **bindings,
                           selection_stream=self.stream, selection_done=self.done)
        # Compiled metadata can be materialized after the projection-stream fork.
        self.stream.wait_stream(torch.cuda.current_stream(workspace.device))
        self.ready.record(self.stream)
        dynamic = reads[count:count + len(self.dynamic_names)]
        qsa.run(binding, **dict(zip(self.dynamic_names, dynamic, strict=True)), index_ready=self.ready)
