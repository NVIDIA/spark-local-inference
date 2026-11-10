# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared FP8 projections with FP32 scales for 128-by-128 weight blocks."""

from dataclasses import dataclass

import torch
from b12x.gemm import blockscaled
from b12x.preparation import PreparedCall, plan_from_handle, require_prepared
from flashinfer.quantization import per_token_group_quant_8bit

from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer


@dataclass(frozen=True)
class BlockFP8Weight:
    values: torch.Tensor
    scales: torch.Tensor

    @property
    def in_features(self):
        return self.values.shape[1]

    @property
    def out_features(self):
        return self.values.shape[0]


@torch.library.custom_op("flare::block_fp8_quantize_out", mutates_args=("quantized", "scales"))
def _quantize_out(source: torch.Tensor, quantized: torch.Tensor, scales: torch.Tensor) -> None:
    per_token_group_quant_8bit(source, 128, out=(quantized, scales))


@_quantize_out.register_fake
def _fake_quantize_out(source, quantized, scales):
    pass


@torch.library.custom_op("flare::block_fp8_projection_out", mutates_args=("output", "workspace"))
def _project_out(quantized: torch.Tensor, activation_scales: torch.Tensor, values: torch.Tensor,
                 scales: torch.Tensor, output: torch.Tensor, workspace: torch.Tensor, handle: int) -> None:
    from .block_fp8_sm103 import COMPONENT

    plan = plan_from_handle(handle)
    if plan.component_id == COMPONENT:
        from flashinfer.gemm import gemm_fp8_nt_groupwise

        state = require_prepared(plan, COMPONENT, quantized.device)
        gemm_fp8_nt_groupwise(quantized, values, activation_scales, scales,
            scale_major_mode="MN" if quantized.shape[0] <= 32 else "K", mma_sm=state.mma_sm,
            out=output, backend="cutlass", workspace_buffer=workspace)
    else:
        require_prepared(plan, plan.component_id, quantized.device).run_serialized(
            quantized, activation_scales, values, scales, None,
            ab_dtype="float8_e4m3fn", sf_dtype="float32", c_dtype="bfloat16",
            sf_vec_size=128, block_fp8=True, stream=None, out=output,
            workspace=workspace.view(torch.float32))


@_project_out.register_fake
def _fake_project_out(quantized, activation_scales, values, scales, output, workspace, handle):
    pass


def _workspace_bytes(plan):
    from .block_fp8_sm103 import COMPONENT

    if plan.component_id == COMPONENT:
        from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE

        return DEFAULT_WORKSPACE_SIZE
    state = require_prepared(plan, plan.component_id)
    if state.gemv is not None:
        return 0
    p = state.dense.lowering
    return (p.policy.split_k_slices * p.m * p.n * 4
            if p.policy.split_k_slices > 1 and not p.policy.split_k_atomic_bf16 else 0)


def _project(source, values, scales, handle, *, workspace_nbytes=None):
    plan = plan_from_handle(handle)
    from .block_fp8_sm103 import COMPONENT

    quantized = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    activation_scales = torch.empty((source.shape[0], source.shape[1] // 128),
                                    dtype=torch.float32, device=source.device)
    _quantize_out(source, quantized, activation_scales)
    if plan.component_id == COMPONENT and source.shape[0] <= 32:
        activation_scales, scales = activation_scales.T.contiguous(), scales.T.contiguous()
    output = torch.empty((source.shape[0], values.shape[0]), dtype=torch.bfloat16, device=source.device)
    nbytes = _workspace_bytes(plan) if workspace_nbytes is None else workspace_nbytes
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=source.device)
    _project_out(quantized, activation_scales, values, scales, output, workspace, handle)
    return output


class BlockFP8Projection:
    def __init__(self, source, weight, output=None, *, session, name):
        n, k = weight.values.shape
        if (source.ndim != 2 or source.shape[1] != k or source.dtype != torch.bfloat16
                or weight.values.dtype != torch.float8_e4m3fn or weight.scales.dtype != torch.float32
                or k % 128 or n % 128 or weight.scales.shape != (n // 128, k // 128)
                or not weight.values.is_contiguous() or not weight.scales.is_contiguous()):
            raise ValueError("block-FP8 projection requires BF16 activations, E4M3 weights and FP32 block-128 scales")
        self.weight = weight
        self.bias = None
        for tensor in (weight.values, weight.scales):
            record_numeric_tensor(tensor)
        self.output = output
        self.output_layout = None
        if output is not None and (output.shape != (source.shape[0], n) or output.dtype != source.dtype
                                   or output.device != source.device):
            raise ValueError("block-FP8 output must match projection rows, columns, dtype and device")
        from .sm103 import is_sm103

        self.sm103 = is_sm103(source.device)
        if self.sm103:
            from .block_fp8_sm103 import plan

            self.plan = plan(source.shape[0], k, n, source.device)
        else:
            self.plan = blockscaled.plan(blockscaled.FixedBlockscaledQuery(
                recipe="block_fp8", call_kind="serialized", max_rows=source.shape[0],
                in_features=k, padded_in_features=k, out_features=n,
                input_dtype="float8_e4m3fn", output_dtype="bfloat16", expected_m=source.shape[0]))

        def prepare(state):
            generator = torch.Generator(device=source.device).manual_seed(0xB12)
            trial = torch.randn(source.shape, dtype=source.dtype, device=source.device,
                                generator=generator).mul_(0.125)
            values, scales = per_token_group_quant_8bit(trial, 128)

            def produce():
                produced, produced_scales = per_token_group_quant_8bit(trial, 128)
                values.copy_(produced)
                scales.copy_(produced_scales)

            # Call closures own trial inputs; prepared plans must not retain them after priming.
            return PreparedCall(run=lambda: state.run_serialized(values, scales, weight.values,
                weight.scales, None, ab_dtype="float8_e4m3fn", sf_dtype="float32",
                c_dtype="bfloat16", sf_vec_size=128, block_fp8=True, stream=None),
                produce=produce)

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=prepare),))
        register_finalizer(session, self._finalize_workspace)

    def _finalize_workspace(self):
        self.workspace_nbytes = _workspace_bytes(self.plan)

    def functional(self, *, session):
        """Allocate projection outputs inside the forward with the prepared layout."""
        del session
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
        self.output = None
        return self

    def __call__(self, source):
        result = _project(source, self.weight.values, self.weight.scales, self.plan.handle,
                          workspace_nbytes=self.workspace_nbytes)
        if self.output is None and self.output_layout is not None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
            return output.copy_(result)
        return result if self.output is None else self.output.copy_(result)
