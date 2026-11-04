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


@torch.library.custom_op("flare::parallel_qsa", mutates_args=("writes", "workspace"))
def _parallel_qsa(source: torch.Tensor, reads: list[torch.Tensor], writes: list[torch.Tensor],
                  workspace: torch.Tensor, handle: int) -> None:
    _CONTEXTS[handle].run(reads, writes, workspace)


@_parallel_qsa.register_fake
def _fake_parallel_qsa(source, reads, writes, workspace, handle):
    pass


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
        qsa.run(binding, **dict(zip(self.dynamic_names, reads[count:], strict=True)), index_ready=self.ready)
