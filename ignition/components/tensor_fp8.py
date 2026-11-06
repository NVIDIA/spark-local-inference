# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared tensor-scaled FP8 projections with checkpoint activation calibration."""

from dataclasses import dataclass

import torch
from b12x.gemm import blockscaled
from b12x.preparation import PreparedCall

from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.scratch import prime_owners, scratch_storage
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .sm103 import is_sm103


@dataclass(frozen=True)
class TensorFP8Weight:
    packed: object
    input_scale: torch.Tensor

    @property
    def out_features(self):
        return self.packed.out_features


class TensorFP8Projection:
    def __init__(self, source, weight, output, *, session, name, staging=None, workspace=None):
        if (source.ndim != 2 or source.dtype != torch.bfloat16
                or source.shape[1] != weight.packed.in_features
                or output.shape != (source.shape[0], weight.packed.out_features)
                or output.dtype != source.dtype or output.device != source.device):
            raise ValueError("tensor-FP8 projection requires matching BF16 input and output matrices")
        self.weight, self.output = weight, output
        for tensor in (weight.packed.values, weight.packed.scale_mma, weight.packed.block_scale,
                       weight.packed.output_scale, weight.input_scale):
            record_numeric_tensor(tensor)
        if staging is None:
            self.quantized = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        else:
            if (staging.dtype != torch.uint8 or staging.device != source.device
                    or not staging.is_contiguous() or staging.numel() < source.numel()):
                raise ValueError("tensor-FP8 staging requires sufficient contiguous device bytes")
            self.quantized = staging.flatten()[:source.numel()].view(torch.float8_e4m3fn).view(source.shape)
        self.sm103 = is_sm103(source.device)
        if self.sm103:
            from .tensor_fp8_sm103 import plan

            self.plan = plan(source.shape[0], weight.packed, source.device)

            def storage():
                return scratch_storage(workspace, nbytes=self.plan.memory_requirements().scratch_nbytes,
                                       device=source.device)

            def prepare_native(state):
                self.quantized.zero_()
                scratch = storage()
                return PreparedCall(run=lambda: state.run(self.quantized, weight.packed.values,
                    weight.packed.output_scale, output, scratch),
                    owners=(self.quantized, weight, output, *prime_owners(workspace, scratch)))

            def benchmark_native(state):
                def make_inputs():
                    seed = torch.randn(source.shape, device=source.device).clamp(-4, 4).to(torch.float8_e4m3fn)
                    return torch.empty_like(seed), seed, torch.empty_like(output)

                inputs, seed, result = benchmark_resource(session, ("tensor_fp8", name), make_inputs)
                scratch = storage()
                return PreparedCall(run=lambda: state.run(inputs, weight.packed.values,
                    weight.packed.output_scale, result, scratch), output=result,
                    produce=lambda: inputs.copy_(seed),
                    owners=(inputs, seed, weight, result, *prime_owners(workspace, scratch)))

            declare(session, (self.plan.request(name=name, prepare_call=prepare_native,
                                               benchmark_call=benchmark_native),))
            register_finalizer(session, lambda: setattr(self, "workspace", storage()), workspace=workspace)
            return
        self.plan = blockscaled.plan(blockscaled.query_from_call(self.quantized, weight.packed))

        def prepare(state):
            self.quantized.zero_()
            w = weight.packed
            return PreparedCall(run=lambda: state.run_tensor_fp8(
                self.quantized, w.values, w.scale_mma, w.block_scale, w.output_scale,
                out_dtype=output.dtype, stream=None), owners=(self.quantized, weight))

        def benchmark(state):
            inputs = benchmark_resource(session, ("tensor_fp8", name), lambda:
                torch.randn(source.shape, device=source.device).clamp(-4, 4).to(torch.float8_e4m3fn))
            w = weight.packed
            return PreparedCall(run=lambda: state.run_tensor_fp8(
                inputs, w.values, w.scale_mma, w.block_scale, w.output_scale,
                out_dtype=output.dtype, stream=None), owners=(inputs, weight))

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=benchmark),))

    def __call__(self, source):
        self.quantized.copy_((source.float() / self.weight.input_scale).clamp(-448, 448))
        if self.sm103:
            torch.ops.flare.sm103_tensor_fp8_out(self.quantized, self.weight.packed.values,
                self.weight.packed.output_scale, self.output, self.workspace, self.plan.handle)
            return self.output
        self.output.copy_(blockscaled.mm(self.quantized, self.weight.packed, plan=self.plan))
        return self.output
