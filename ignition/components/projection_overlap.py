# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture a chain of dense projections alongside independent main-stream work."""

import torch

from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer


@torch.library.custom_op("flare::start_projection_chain", mutates_args=("outputs", "workspace"))
def _start(source: torch.Tensor, weights: list[torch.Tensor], scales: list[torch.Tensor | None],
           gains: list[torch.Tensor | None], activation_scales: list[torch.Tensor | None], scale_kinds: list[int],
           outputs: list[torch.Tensor], workspace: torch.Tensor, plans: list[int], stream: int) -> None:
    from ignition.components.sm103_projection import bf16_out, packed_out_workspace

    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        value = source
        for weight, scale, gain, activation_scale, kind, output, plan in zip(
                weights, scales, gains, activation_scales, scale_kinds, outputs, plans, strict=True):
            if scale is None:
                bf16_out(value, weight, output, plan)
            else:
                packed_out_workspace(value, weight, scale, gain, activation_scale, output, workspace,
                                      ("none", "multiplier", "reciprocal")[kind], None, plan, "any")
            value = output


@_start.register_fake
def _fake_start(source, weights, scales, gains, activation_scales, scale_kinds, outputs, workspace, plans, stream):
    pass


@torch.library.custom_op("flare::join_projection_chain", mutates_args=("outputs", "workspace"))
def _join(source: torch.Tensor, dependency: torch.Tensor, outputs: list[torch.Tensor],
          workspace: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join.register_fake
def _fake_join(source, dependency, outputs, workspace, stream):
    pass


@torch.library.custom_op("flare::start_encoded_projection", mutates_args=(
    "output", "workspace", "values_base", "row_base", "mma_base"))
def _start_encoded(source: torch.Tensor, weight_values: torch.Tensor, weight_mma_base: torch.Tensor,
                   output: torch.Tensor, values_base: torch.Tensor, row_base: torch.Tensor,
                   mma_base: torch.Tensor, workspace: torch.Tensor, quantizer: int, dense: int,
                   stream: int) -> None:
    from b12x import gemm
    from b12x.preparation.types import plan_from_handle
    from b12x.quantization import mxfp8

    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        mxfp8.quantize_dense_rows(source, values_base, row_base, mma_base,
                                  plan=plan_from_handle(quantizer))
        gemm.mxfp8_mm_bases(values_base, mma_base, weight_values, weight_mma_base,
                            output, workspace, plan=plan_from_handle(dense))


@_start_encoded.register_fake
def _fake_start_encoded(source, weight_values, weight_mma_base, output, values_base, row_base,
                        mma_base, workspace, quantizer, dense, stream):
    pass


@torch.library.custom_op("flare::start_preencoded_projection", mutates_args=("output", "workspace"))
def _start_preencoded(source: torch.Tensor, weight_values: torch.Tensor, weight_mma_base: torch.Tensor,
                      output: torch.Tensor, values_base: torch.Tensor, row_base: torch.Tensor,
                      mma_base: torch.Tensor, workspace: torch.Tensor, dense: int, stream: int) -> None:
    from b12x import gemm
    from b12x.preparation.types import plan_from_handle

    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        gemm.mxfp8_mm_bases(values_base, mma_base, weight_values, weight_mma_base,
                            output, workspace, plan=plan_from_handle(dense))


@_start_preencoded.register_fake
def _fake_start_preencoded(source, weight_values, weight_mma_base, output, values_base,
                           row_base, mma_base, workspace, dense, stream):
    pass


@torch.library.custom_op("flare::join_encoded_projection", mutates_args=("output", "workspace"))
def _join_encoded(source: torch.Tensor, dependency: torch.Tensor, output: torch.Tensor,
                  values_base: torch.Tensor, row_base: torch.Tensor, mma_base: torch.Tensor,
                  workspace: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join_encoded.register_fake
def _fake_join_encoded(source, dependency, output, values_base, row_base, mma_base, workspace, stream):
    pass


class ConcurrentProjectionChain:
    """Run dependent projections on an auxiliary stream with one scratch suffix.

    Each projection consumes its predecessor's complete output. The recipe
    declares every concurrent main-stream plan, starts the chain, and joins
    before consuming its output or reusing source and scratch storage.
    """

    def __init__(self, projections, *, parallel_plans, session, workspace: ScratchArena):
        from b12x.gemm.blockscaled._a16 import _weight_parts

        self.projections, self.parallel_plans = tuple(projections), tuple(parallel_plans)
        if not isinstance(workspace, ScratchArena) or not self.projections or not self.parallel_plans:
            raise ValueError("projection overlap requires an arena, a projection chain, and concurrent plans")
        self.arena = workspace
        self.encoded_projection = (self.projections[0] if len(self.projections) == 1
            and isinstance(self.projections[0], QuantizedProjection)
            and self.projections[0].encoded is not None else None)
        if self.encoded_projection is not None:
            projection = self.encoded_projection
            if projection.bias is not None:
                raise ValueError("overlapped MXFP8 projection cannot add bias")
            for main in self.parallel_plans:
                if projection.quantizer_plan is not None:
                    workspace.register_parallel(main, projection.quantizer_plan)
                workspace.register_parallel(main, projection.plan)
            self.stream = workspace.auxiliary_stream
            self.stream_handle = self.stream.cuda_stream
            register_finalizer(session, self.refresh, workspace=workspace)
            return
        self.weights, self.scales, self.outputs, self.plans = [], [], [], []
        self.gains, self.activation_scales, self.scale_kinds = [], [], []
        for projection in self.projections:
            if not isinstance(projection, (QuantizedProjection, UnquantizedProjection)) or projection.bias is not None:
                raise ValueError("projection overlap requires unbiased BF16 or blockscaled projections")
            if isinstance(projection, QuantizedProjection):
                values, scales, gain, fp4 = _weight_parts(projection.weight)
                if scales.dtype == torch.float8_e8m0fnu:
                    scales = scales.view(torch.uint8)
                width = projection.weight.in_features
                activation_scale = projection.activation_scale
                kind = projection.weight.global_scale_kind if fp4 else "none"
            else:
                values, scales, gain = projection.weight, None, None
                width, activation_scale, kind = values.shape[1], None, "none"
            if self.outputs and self.outputs[-1].shape[1] != width:
                raise ValueError("projection chain output width must match the following input width")
            self.weights.append(values)
            self.scales.append(scales)
            self.gains.append(gain)
            self.activation_scales.append(activation_scale)
            self.scale_kinds.append(("none", "multiplier", "reciprocal").index(kind))
            self.outputs.append(projection.output)
            self.plans.append(projection.plan.handle)
            for main in self.parallel_plans:
                workspace.register_parallel(main, projection.plan)
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        if self.arena is None:
            return
        main_bytes = max(plan.memory_requirements().scratch_nbytes for plan in self.parallel_plans)
        if self.encoded_projection is not None:
            projection = self.encoded_projection
            side_plans = ((projection.quantizer_plan, projection.plan)
                          if projection.quantizer_plan is not None else (projection.plan,))
            side_bytes = max(plan.memory_requirements().scratch_nbytes for plan in side_plans)
        else:
            side_bytes = max(p.plan.memory_requirements().scratch_nbytes for p in self.projections)
        _, self.workspace = self.arena.views(((main_bytes,), torch.uint8), ((side_bytes,), torch.uint8))

    def functional(self, *, session):
        self.output_layouts = tuple((tuple(p.output.shape), tuple(p.output.stride()), p.output.dtype)
                                   for p in self.projections)
        projection = self.encoded_projection
        self.encoded_layouts = None
        if projection is not None:
            self.encoded_layouts = tuple((tuple(t.shape), t.dtype) for t in projection.encoded_bases)
            self.encoded_weights = projection.parts[:2]
            self.quantizer_handle = None if projection.quantizer_plan is None else projection.quantizer_plan.handle
            self.dense_handle = projection.plan.handle
        self.side_plans = tuple(p.plan for p in self.projections)
        if projection is not None and projection.quantizer_plan is not None:
            self.side_plans += (projection.quantizer_plan,)
        self.plan_owners = tuple(p.functional(session=session) for p in self.projections)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.side_plans = tuple(p.plan for p in self.plan_owners)
        if self.encoded_layouts is not None:
            projection = self.plan_owners[0]
            self.dense_handle = projection.plan.handle
            self.quantizer_handle = None if projection.quantizer_plan is None else projection.quantizer_plan.handle
            if projection.quantizer_plan is not None:
                self.side_plans += (projection.quantizer_plan,)
        else:
            self.plans = [p.plan.handle for p in self.plan_owners]
        self.workspace_nbytes = max(plan.memory_requirements().scratch_nbytes for plan in self.side_plans)
        self.projections = self.parallel_plans = self.encoded_projection = self.arena = self.workspace = None
        self.outputs = None

    def start(self, source, *, encoded_bases=None):
        if self.projections is None:
            outputs = [torch.empty_strided(shape, stride, dtype=dtype, device=source.device).zero_()
                       for shape, stride, dtype in self.output_layouts]
            workspace = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=source.device)
            if self.encoded_layouts is not None:
                if self.quantizer_handle is None:
                    if encoded_bases is None:
                        raise ValueError("preencoded projection overlap requires runtime encoded bases")
                    _start_preencoded(source, *self.encoded_weights, outputs[0], *encoded_bases,
                                      workspace, self.dense_handle, self.stream_handle)
                else:
                    encoded_bases = tuple(torch.zeros(shape, dtype=dtype, device=source.device)
                                          for shape, dtype in self.encoded_layouts)
                    _start_encoded(source, *self.encoded_weights, outputs[0], *encoded_bases, workspace,
                                   self.quantizer_handle, self.dense_handle, self.stream_handle)
            else:
                _start(source, self.weights, self.scales, self.gains, self.activation_scales,
                       self.scale_kinds, outputs, workspace, self.plans, self.stream_handle)
            return source, outputs, workspace, encoded_bases
        if self.encoded_projection is not None:
            projection = self.encoded_projection
            if projection.quantizer_plan is None:
                _start_preencoded(source, projection.parts[0], projection.parts[1], projection.output,
                    *projection.encoded_bases, self.workspace, projection.plan.handle, self.stream_handle)
            else:
                _start_encoded(source, projection.parts[0], projection.parts[1], projection.output,
                    *projection.encoded_bases, self.workspace, projection.quantizer_plan.handle,
                    projection.plan.handle, self.stream_handle)
            return
        _start(source, self.weights, self.scales, self.gains, self.activation_scales, self.scale_kinds,
               self.outputs, self.workspace, self.plans, self.stream_handle)

    def finish(self, source, dependency):
        if self.projections is None:
            source, outputs, workspace, bases = source
            if bases is not None:
                _join_encoded(source, dependency, outputs[0], *bases, workspace, self.stream_handle)
            else:
                _join(source, dependency, outputs, workspace, self.stream_handle)
            return outputs[-1]
        if self.encoded_projection is not None:
            projection = self.encoded_projection
            _join_encoded(source, dependency, projection.output, *projection.encoded_bases,
                          self.workspace, self.stream_handle)
            return projection.output
        _join(source, dependency, self.outputs, self.workspace, self.stream_handle)
        return self.outputs[-1]
