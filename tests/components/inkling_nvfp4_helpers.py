# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw Inkling NVFP4 fixtures and an independent quantized arithmetic oracle."""

import torch
import triton
import triton.language as tl


@triton.jit
def _reciprocal_approximate_kernel(source, output, count: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + index, index < count, other=1.)
    result = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f",
        args=[value], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(output + index, result, index < count)


def _reciprocal_approximate(value):
    """The native quantizer's CUDA reciprocal primitive, independent of its kernels."""
    value = value.contiguous()
    output = torch.empty_like(value)
    _reciprocal_approximate_kernel[(triton.cdiv(value.numel(), 256),)](value, output, value.numel(), 256)
    return output


@triton.jit
def _exp2_approximate_kernel(source, output, count: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + index, index < count, other=0.)
    result = tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", constraints="=f,f",
        args=[value], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(output + index, result, index < count)


def _exp2_approximate(value):
    value = value.contiguous()
    output = torch.empty_like(value)
    _exp2_approximate_kernel[(triton.cdiv(value.numel(), 256),)](value, output, value.numel(), 256)
    return output


def unpack_weight(values, scales):
    """Decode low-nibble-first E2M1 and unswizzled checkpoint E4M3 block scales."""
    magnitudes = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=values.device)
    codes = torch.stack((values & 15, values >> 4), -1).flatten(-2).long()
    signed = magnitudes[codes & 7] * torch.where(codes < 8, 1., -1.)
    return signed * scales.float().repeat_interleave(16, -1)


def quantize_activation(value, amax, *, quant_scale_math="b12x"):
    """Return E2M1 values times E4M3 block scales, and separate FP32 calibration."""
    value = value.float()
    blocks = value.reshape(*value.shape[:-1], -1, 16)
    # Round amax/2688 before reciprocating, as the publisher does.
    dequant = torch.tensor(float(amax.item()) / 2688., dtype=torch.float32, device=value.device)
    encode = dequant.reciprocal()
    sixth = (_reciprocal_approximate(torch.full_like(encode, 6.))
             if quant_scale_math == "flashinfer" else torch.full_like(encode, 1. / 6.))
    scales = ((blocks.abs().amax(-1, keepdim=True) * sixth) * encode).clamp(max=448.)
    scales = scales.to(torch.float8_e4m3fn).float()
    if quant_scale_math == "b12x":
        inverse = scales.reciprocal() * encode
    elif quant_scale_math == "flashinfer":
        # Preserve rcp.approx.ftz ordering from quantization_utils.cuh; reassociation
        # or correctly-rounded reciprocal changes E2M1 ties, including amax=187.
        inverse = _reciprocal_approximate(scales * _reciprocal_approximate(encode))
    else:
        raise ValueError(f"unknown NVFP4 quantizer arithmetic: {quant_scale_math}")
    inverse = torch.where(scales != 0, inverse, 0.)
    normalized = blocks * inverse
    magnitude = normalized.abs()
    code = ((magnitude > .25).int() + (magnitude >= .75).int() + (magnitude > 1.25).int()
            + (magnitude >= 1.75).int() + (magnitude > 2.5).int() + (magnitude >= 3.5).int()
            + (magnitude > 5.).int())
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=value.device)
    return (levels[code] * normalized.sign() * scales).reshape_as(value), encode


def nvfp4_routes(source, ids, tensors, prefix, *, intermediate_range=None,
                 quant_scale_math="b12x"):
    """Unweighted routes: FP32 FC1/SwiGLU -> BF16 -> NVFP4 -> FP32 FC2 -> BF16."""
    first = prefix + ".w13_weight"
    second = prefix + ".w2_weight"
    tensors = {name: value.to(source.device) for name, value in tensors.items() if name.startswith(prefix + ".")}
    intermediate = tensors[first].shape[1] // 2
    begin, end = (0, intermediate) if intermediate_range is None else intermediate_range
    source_quantized, input_scale = quantize_activation(
        source, tensors[first + ".input_amax"], quant_scale_math=quant_scale_math)
    result = torch.zeros((*ids.shape, source.shape[-1]), dtype=torch.bfloat16, device=source.device)
    for expert in range(tensors[first].shape[0]):
        tokens, routes = torch.where(ids == expert)
        if tokens.numel() == 0:
            continue
        rows = slice(2 * begin, 2 * end)
        # Scaled FP4 MMA applies the outer multiplier after the dot product.
        up_gate = unpack_weight(tensors[first][expert, rows], tensors[first + ".scale"][expert, rows])
        projected = (source_quantized[tokens] @ up_gate.T) * (tensors[first + ".scale2"][expert] / input_scale)
        gate, up = projected[:, ::2], projected[:, 1::2]
        if quant_scale_math == "flashinfer":
            # CuTe's FP32 SwiGLU epilogue uses exp2.approx and rcp.approx.
            # Their rounding can cross a BF16 midpoint before FC2 quantization.
            sigmoid = _reciprocal_approximate(1. + _exp2_approximate(gate * -1.4426950408889634))
            activated = ((sigmoid * gate) * up).bfloat16()
        else:
            activated = (torch.nn.functional.silu(gate) * up).bfloat16()
        activated, intermediate_scale = quantize_activation(
            activated, tensors[second + ".input_amax"], quant_scale_math=quant_scale_math)
        down = unpack_weight(tensors[second][expert, :, begin // 2:end // 2],
                             tensors[second + ".scale"][expert, :, begin // 16:end // 16])
        result[tokens, routes] = ((activated @ down.T)
            * (tensors[second + ".scale2"][expert] / intermediate_scale)).bfloat16()
    return result
