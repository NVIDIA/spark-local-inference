# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared RMS normalization reading projection slices with explicit row strides."""

import torch
from b12x.norm.hyperconnection import api as hyperconnection
from b12x.preparation import PreparedCall

from ignition.preparation.fixtures.hyperconnection import HyperconnectionBenchmarkInputs
from ignition.preparation.tuning import benchmark_resource, declare


class PreparedRMSNorm:
    def __init__(self, source, weight, output, *, eps, session, name, zero_centered=False, residual=None, staging=None):
        if (source.ndim != 2 or source.shape != output.shape or weight.shape != source.shape[1:]
                or source.dtype not in (torch.bfloat16, torch.float32) or output.dtype != torch.bfloat16
                or weight.dtype not in (torch.bfloat16, torch.float32)
                or source.stride(1) != 1 or source.stride(0) < source.shape[1]
                or not output.is_contiguous() or not weight.is_contiguous()):
            raise ValueError("RMS normalization requires BF16/FP32 input, BF16 output, and contiguous affine weights")
        self.weight, self.output, self.eps = weight, output, float(eps)
        self.zero_centered, self.add_residual = bool(zero_centered), residual is not None
        if residual is not None and (residual.shape != source.shape or residual.dtype not in (torch.bfloat16, torch.float32)):
            raise ValueError("normalization residual must match the input matrix")
        self.staging = None
        if not source.is_contiguous():
            if staging is None:
                staging = torch.empty(source.shape, dtype=source.dtype, device=source.device)
            if (not staging.is_contiguous() or staging.dtype != source.dtype
                    or staging.device != source.device or staging.numel() < source.numel()):
                raise ValueError("RMS normalization staging must hold a contiguous input matrix")
            self.staging = staging.view(-1)[:source.numel()].view(source.shape)
            self.staging.copy_(source)
        prepared_source = source if self.staging is None else self.staging
        self.plan = hyperconnection.plan(hyperconnection.Caps(device=source.device,
            max_tokens=source.shape[0], hidden_size=source.shape[1], streams=1, lowrank=1, dtype=source.dtype),
            invocation={"operation": "grouped_rmsnorm", "zero_centered": self.zero_centered, "eps": self.eps,
                        "add_residual": self.add_residual,
                        "right_dtype": str(source.dtype if residual is None else residual.dtype).removeprefix("torch."),
                        "weight_dtype": str(weight.dtype).removeprefix("torch.")})
        kwargs = {"eps": self.eps} if residual is None else {"eps": self.eps, "residual": residual}
        def benchmark(state):
            arguments = ((prepared_source, weight, output) if residual is None
                         else (prepared_source, weight, residual, output))
            operands = benchmark_resource(session, ("rms_norm", name),
                lambda: HyperconnectionBenchmarkInputs("grouped_rmsnorm", arguments))
            if residual is None:
                return PreparedCall(run=lambda: state.launch(*operands.arguments, **kwargs),
                                    output=operands.output, produce=operands.produce, owners=(operands,))
            a, w, r, o = operands.arguments
            return PreparedCall(run=lambda: state.launch(a, w, o, eps=self.eps, residual=r),
                                output=o, produce=operands.produce, owners=(operands,))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.launch(prepared_source, weight, output, **kwargs)),
            benchmark_call=benchmark),))

    def __call__(self, source, residual=None):
        original_source = source
        if self.staging is not None:
            self.staging.copy_(source)
            source = self.staging
        if self.add_residual:
            torch.ops.b12x.hyperconnection_residual_rmsnorm(
                source, residual, self.weight, self.output, self.eps, self.plan.handle, self.zero_centered)
            if self.staging is not None:
                original_source.copy_(source)
        else:
            torch.ops.b12x.hyperconnection_grouped_rmsnorm(
                source, self.weight, self.output, self.eps, self.plan.handle, self.zero_centered)
        return self.output


class PreparedPairedRMSNorm:
    """Normalize two BF16 matrices into adjacent column blocks in one launch."""

    def __init__(self, left, right, weights, output, *, eps, session, name, zero_centered=False):
        if (left.ndim != 2 or right.shape != left.shape or output.shape != (left.shape[0], 2 * left.shape[1])
                or any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (left, right, output))
                or len(weights) != 2 or any(w.shape != left.shape[1:] for w in weights)
                or weights[0].dtype != weights[1].dtype):
            raise ValueError("paired normalization requires two BF16 matrices and matching gains")
        self.weight, self.output, self.eps = torch.cat(weights), output, float(eps)
        self.plan = hyperconnection.plan(hyperconnection.Caps(device=left.device,
            max_tokens=left.shape[0], hidden_size=left.shape[1], streams=2, lowrank=1),
            invocation={"operation": "grouped_rmsnorm", "zero_centered": zero_centered, "eps": self.eps,
                        "paired": True, "weight_dtype": str(self.weight.dtype).removeprefix("torch.")})

        def benchmark(state):
            operands = benchmark_resource(session, ("paired_rms_norm", name),
                lambda: HyperconnectionBenchmarkInputs("grouped_rmsnorm", (left, self.weight, right, output)))
            a, w, b, out = operands.arguments
            return PreparedCall(run=lambda: state.launch(a, w, out, eps=self.eps, paired=b), output=out,
                                produce=operands.produce, owners=(operands,))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.launch(
                left, self.weight, output, eps=self.eps, paired=right)), benchmark_call=benchmark),))

    def __call__(self, left, right):
        torch.ops.b12x.hyperconnection_paired_rmsnorm(left, right, self.weight, self.output, self.eps, self.plan.handle)
        return self.output
