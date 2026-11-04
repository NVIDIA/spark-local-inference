# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared tensor-scaled FP8 projections with checkpoint activation calibration."""

from dataclasses import dataclass

import torch
from b12x.gemm import blockscaled
from b12x.preparation import PreparedCall

from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import benchmark_resource, declare


@dataclass(frozen=True)
class TensorFP8Weight:
    packed: object
    input_scale: torch.Tensor

    @property
    def out_features(self):
        return self.packed.out_features

    @property
    def in_features(self):
        return self.packed.in_features


class TensorFP8Projection:
    def __init__(self, source, weight, output, *, session, name, staging=None):
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
        self.output.copy_(blockscaled.mm(self.quantized, self.weight.packed, plan=self.plan))
        return self.output
