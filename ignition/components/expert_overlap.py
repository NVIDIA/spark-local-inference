# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture shared-expert MLPs beside routing and routed-expert execution."""

from dataclasses import dataclass
from functools import partial

import torch

from ignition.components.linear import QuantizedProjection
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer


def _run_shared_expert(source, weights, scales, gains, scale_kinds, outputs,
                       workspace, plans, activation, limit, round_silu, auxiliary,
                       encoded, encoding_offsets, quantizers):
    from ignition.components.sm103_projection import bf16_out, packed_out_workspace

    def project(value, index, output):
        bases = encoded[encoding_offsets[index]:encoding_offsets[index + 1]]
        if quantizers[index]:
            from b12x import gemm
            from b12x.quantization import mxfp8
            from b12x.preparation.types import plan_from_handle

            mxfp8.quantize_dense_rows(value, *bases, plan=plan_from_handle(quantizers[index]))
            gemm.mxfp8_mm_bases(bases[0], bases[2], weights[index], scales[index],
                               output, workspace, plan=plan_from_handle(plans[index]))
            return
        if scales[index] is None:
            from b12x.preparation.types import plan_from_handle
            from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE
            from ignition.components.sm103_projection import DENSE_COMPONENT, _dense_out_workspace

            if (plan_from_handle(plans[index]).component_id == DENSE_COMPONENT
                    and workspace.numel() >= DEFAULT_WORKSPACE_SIZE):
                _dense_out_workspace(value, weights[index], output, workspace, plans[index])
            else:
                bf16_out(value, weights[index], output, plans[index])
        elif scale_kinds[index] == 3 and bases:
            from ignition.components.block_fp8 import _project_out, _quantize_out

            _quantize_out(value, bases[0], bases[1])
            activation_scales = bases[1]
            weight_scales = scales[index]
            if len(bases) == 4:
                bases[2].copy_(activation_scales.T)
                bases[3].copy_(weight_scales.T)
                activation_scales = bases[2]
                weight_scales = bases[3]
            _project_out(bases[0], activation_scales, weights[index], weight_scales,
                         output, workspace, plans[index])
        elif scale_kinds[index] == 3:
            from ignition.components.block_fp8 import _project

            output.copy_(_project(value, weights[index], scales[index], plans[index]))
        else:
            packed_out_workspace(
                value, weights[index], scales[index], gains[index], None, output,
                workspace, ("none", "multiplier", "reciprocal")[scale_kinds[index]], None, plans[index], "any")

    gate_up, intermediate, projected = outputs[:3]
    with torch.cuda.stream(auxiliary):
        project(source, 0, gate_up)
        if activation == "relu2":
            torch.ops.b12x.hyperconnection_relu2(gate_up, intermediate, plans[-1])
        else:
            torch.ops.b12x.hyperconnection_swiglu(gate_up, intermediate, limit, round_silu, plans[-1])
        project(intermediate, 1, projected)
        if len(weights) == 3:
            project(source, 2, outputs[3])


@torch.library.custom_op("flare::start_shared_expert", mutates_args=("outputs", "workspace", "encoded"))
def _start_shared_expert(
    source: torch.Tensor, weights: list[torch.Tensor],
    scales: list[torch.Tensor | None], gains: list[torch.Tensor | None],
    scale_kinds: list[int], outputs: list[torch.Tensor], workspace: torch.Tensor,
    plans: list[int], activation: str, limit: float, round_silu: bool, auxiliary_stream: int,
    launch_order: int,
    encoded: list[torch.Tensor], encoding_offsets: list[int], quantizers: list[int],
) -> None:
    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    auxiliary.wait_stream(main)
    if not launch_order:
        _run_shared_expert(source, weights, scales, gains, scale_kinds, outputs,
                           workspace, plans, activation, limit, round_silu, auxiliary,
                           encoded, encoding_offsets, quantizers)


@_start_shared_expert.register_fake
def _fake_start_shared_expert(source, weights, scales, gains, scale_kinds, outputs,
                             workspace, plans, activation, limit, round_silu, auxiliary_stream, launch_order,
                             encoded, encoding_offsets, quantizers):
    pass


@torch.library.custom_op("flare::join_shared_expert", mutates_args=("outputs", "workspace", "encoded"))
def _join_shared_expert(
    source: torch.Tensor, routed: torch.Tensor, weights: list[torch.Tensor],
    scales: list[torch.Tensor | None], gains: list[torch.Tensor | None],
    scale_kinds: list[int], outputs: list[torch.Tensor], workspace: torch.Tensor,
    plans: list[int], activation: str, limit: float, round_silu: bool, auxiliary_stream: int,
    launch_order: int,
    encoded: list[torch.Tensor], encoding_offsets: list[int], quantizers: list[int],
) -> None:
    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    if launch_order:
        _run_shared_expert(source, weights, scales, gains, scale_kinds, outputs,
                           workspace, plans, activation, limit, round_silu, auxiliary,
                           encoded, encoding_offsets, quantizers)
    main.wait_stream(auxiliary)


@_join_shared_expert.register_fake
def _fake_join_shared_expert(source, routed, weights, scales, gains, scale_kinds, outputs,
                            workspace, plans, activation, limit, round_silu, auxiliary_stream, launch_order,
                            encoded, encoding_offsets, quantizers):
    pass


@dataclass(frozen=True)
class PendingSharedExpert:
    source: torch.Tensor
    outputs: list[torch.Tensor]
    workspace: torch.Tensor
    encoded: list[torch.Tensor]

    @property
    def gate(self):
        return self.outputs[3] if len(self.outputs) > 3 else None


class ConcurrentSharedExpert:
    """Fork a prepared SwiGLU or ReLU-squared MLP and join before consumption.

    The recipe supplies every plan that can run concurrently on the main stream.
    Main-stream scratch occupies the arena prefix. MLP projections reuse one
    disjoint suffix, sized after autotuning. Explicit mutation operands keep the
    fork, join, and storage lifetimes ordered through forward compilation.
    An optional source projection computes a shared-expert gate on the same fork;
    the recipe applies its activation and combines outputs after ``finish``.
    With ``defer_launch``, ``start`` records the fork and ``finish`` enqueues the
    MLP before joining. The GPU branches still overlap, while main-stream kernels
    precede shared-expert kernels in graph construction order.
    Launch order is a rank-local integer binding during AOT compilation, so
    ranks can retain their configured order while sharing a compiled forward.
    """

    def __init__(self, mlp, *, parallel_plans, session, workspace: ScratchArena,
                 gate_projection=None, defer_launch=False):
        from b12x.gemm.blockscaled._a16 import _weight_parts
        from ignition.components.block_fp8 import BlockFP8Projection

        if not isinstance(workspace, ScratchArena) or not parallel_plans:
            raise ValueError("shared-expert overlap requires an arena and concurrent main-stream plans")
        self.activation = getattr(mlp, "activation", "swiglu")
        if self.activation not in ("swiglu", "relu2"):
            raise ValueError("shared-expert overlap requires SwiGLU or ReLU-squared activation")
        self.limit = getattr(mlp, "swiglu_limit", float("inf"))
        self.round_silu = getattr(mlp, "round_silu", True)
        self.mlp, self.parallel_plans, self.arena = mlp, tuple(parallel_plans), workspace
        self.defer_launch = bool(defer_launch)
        self.projections = (mlp.up_projection, mlp.down_projection)
        self.outputs = [mlp.up_projection.output, mlp.intermediate, mlp.down_projection.output]
        if gate_projection is not None:
            self.projections += (gate_projection,)
            self.outputs.append(gate_projection.output)
        self.weights, self.scales, self.gains, self.scale_kinds = [], [], [], []
        for projection in self.projections:
            if projection.bias is not None:
                raise ValueError("shared-expert overlap requires unbiased projections")
            if isinstance(projection, QuantizedProjection):
                values, scales, gain, fp4 = _weight_parts(projection.weight)
                if scales.dtype == torch.float8_e8m0fnu:
                    scales = scales.view(torch.uint8)
                kind = projection.weight.global_scale_kind if fp4 else "none"
            elif isinstance(projection, BlockFP8Projection):
                values, scales = projection.weight.values, projection.weight.scales
                gain, kind = None, "block_fp8"
            else:
                values, scales, gain, kind = projection.weight, None, None, "none"
            self.weights.append(values)
            self.scales.append(scales)
            self.gains.append(gain)
            self.scale_kinds.append(("none", "multiplier", "reciprocal", "block_fp8").index(kind))
            for main_plan in self.parallel_plans:
                workspace.register_parallel(main_plan, projection.plan)
        self.plans = [projection.plan.handle for projection in self.projections] + [mlp.activation_plan.handle]
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        self.encoding_offsets = [0] * (len(self.projections) + 1)
        self.quantizers = [0] * len(self.projections)
        register_finalizer(session, partial(self.refresh, self.parallel_plans, workspace), workspace=workspace)

    def functional(self, *, session):
        """Carry the private branch's local storage explicitly through its join."""
        if self.outputs is not None:
            self.output_layouts = tuple((tuple(t.shape), t.stride(), t.dtype) for t in self.outputs)
            self.outputs = None
            self.mlp.functional(session=session)
            for projection in self.projections:
                projection.functional(session=session)
            register_finalizer(session, self._finalize_functional, workspace=self.arena)
        return self

    def _finalize_functional(self):
        from ignition.components.block_fp8 import BlockFP8Projection, _workspace_bytes

        self.encoded_layouts = []
        self.encoding_offsets = [0]
        sizes = []
        for index, projection in enumerate(self.projections):
            if isinstance(projection, BlockFP8Projection):
                shape = self.output_layouts[1 if index == 1 else 0][0]
                rows = shape[0]
                columns = projection.weight.in_features
                self.encoded_layouts.extend((((rows, columns), torch.float8_e4m3fn),
                                            ((rows, columns // 128), torch.float32)))
                if projection.sm103 and rows <= 32:
                    self.encoded_layouts.append(((columns // 128, rows), torch.float32))
                    self.encoded_layouts.append((tuple(projection.weight.scales.T.shape), torch.float32))
                sizes.append(_workspace_bytes(projection.plan))
            else:
                layout = getattr(projection, "encoded_layout", None)
                if layout is not None:
                    self.encoded_layouts.extend(layout)
                    if projection.quantizer_plan is None:
                        raise ValueError("shared expert overlap requires projection-owned activation encoding")
                    self.quantizers[index] = projection.quantizer_plan.handle
                    self.weights[index], self.scales[index] = projection.parts[:2]
                if getattr(projection, "dense", False):
                    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE

                    sizes.append(DEFAULT_WORKSPACE_SIZE)
                else:
                    sizes.append(projection.plan.memory_requirements().scratch_nbytes)
            self.encoding_offsets.append(len(self.encoded_layouts))
        self.workspace_nbytes = max(sizes, default=0)
        self.workspace = self.arena = self.mlp = self.parallel_plans = None

    @property
    def defer_launch(self):
        return bool(self.launch_order)

    @defer_launch.setter
    def defer_launch(self, value):
        self.launch_order = int(bool(value))

    def refresh(self, parallel_plans, arena):
        main_bytes = max(plan.memory_requirements().scratch_nbytes for plan in parallel_plans)
        shared_bytes = max(projection.plan.memory_requirements().scratch_nbytes for projection in self.projections)
        _, self.workspace = arena.views(((main_bytes,), torch.uint8), ((shared_bytes,), torch.uint8))

    def start(self, source):
        if self.outputs is None:
            outputs = [torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
                       for shape, stride, dtype in self.output_layouts]
            workspace = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=source.device)
            encoded = [torch.zeros(shape, dtype=dtype, device=source.device) for shape, dtype in self.encoded_layouts]
            pending = PendingSharedExpert(source, outputs, workspace, encoded)
            _start_shared_expert(source, self.weights, self.scales, self.gains, self.scale_kinds,
                outputs, workspace, self.plans, self.activation, self.limit, self.round_silu,
                self.stream_handle, self.launch_order, encoded, self.encoding_offsets, self.quantizers)
            return pending
        _start_shared_expert(source, self.weights, self.scales, self.gains, self.scale_kinds,
                            self.outputs, self.workspace, self.plans, self.activation, self.limit,
                            self.round_silu, self.stream_handle, self.launch_order,
                            [], self.encoding_offsets, self.quantizers)

    def finish(self, source, routed):
        from ignition.components.collectives import WeightedGroupSource

        if isinstance(routed, WeightedGroupSource):
            routed = routed.values
        if self.outputs is None:
            pending = source
            _join_shared_expert(pending.source, routed, self.weights, self.scales, self.gains,
                self.scale_kinds, pending.outputs, pending.workspace, self.plans, self.activation,
                self.limit, self.round_silu, self.stream_handle, self.launch_order,
                pending.encoded, self.encoding_offsets, self.quantizers)
            return pending.outputs[2]
        _join_shared_expert(source, routed, self.weights, self.scales, self.gains, self.scale_kinds,
                           self.outputs, self.workspace, self.plans, self.activation, self.limit,
                           self.round_silu, self.stream_handle, self.launch_order,
                           [], self.encoding_offsets, self.quantizers)
        return self.mlp.down_projection.output
