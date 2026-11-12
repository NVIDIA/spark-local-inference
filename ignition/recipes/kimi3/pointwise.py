# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 single-kernel SiTU, RMSNorm and gating steps between projections.

Each kernel replaces a chain of small eager and generated kernels with one launch while keeping the
canonical rounding points: SiTU and gating compute in FP32 and round once; RMSNorm rounds the
normalized value to BF16 before applying its weight.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _situ(Source, Output, source_stride, output_stride, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0).to(tl.int64), tl.program_id(1)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    valid = columns < WIDTH
    gate = tl.load(Source + row * source_stride + columns, valid, other=0.).to(tl.float32)
    up = tl.load(Source + row * source_stride + WIDTH + columns, valid, other=0.).to(tl.float32)
    sigmoid = 1. / (1. + tl.exp(-gate))
    value = 4. * libdevice.tanh(gate / 4.) * sigmoid * (25. * libdevice.tanh(up / 25.))
    tl.store(Output + row * output_stride + columns, value.to(Output.dtype.element_ty), valid)


def situ(gate_up, out):
    """``out = situ(gate_up)`` for gate/up halves of each row, matching ``model.situ``."""
    rows, width = out.shape
    if gate_up.shape != (rows, 2 * width) or gate_up.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("SiTU takes [rows, 2 * width] gate/up rows and writes [rows, width]")
    block = min(1024, triton.next_power_of_2(width))
    torch.library.wrap_triton(_situ)[(rows, triton.cdiv(width, block))](
        gate_up, out, gate_up.stride(0), out.stride(0), width, block, num_warps=4)
    return out


@triton.jit
def _rms_norm(Source, Weight, Output, Begin, source_stride, output_stride, eps, WIDTH: tl.constexpr,
              OUTPUT_WIDTH: tl.constexpr, OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    valid = columns < WIDTH
    value = tl.load(Source + row * source_stride + columns, valid, other=0.).to(tl.float32)
    normalized = (value * tl.rsqrt(tl.sum(value * value) / WIDTH + eps)).to(Output.dtype.element_ty)
    begin = tl.load(Begin) if OFFSET else 0
    keep = (columns >= begin) & (columns < begin + OUTPUT_WIDTH)
    weight = tl.load(Weight + columns - begin, keep, other=0.).to(tl.float32)
    tl.store(Output + row * output_stride + columns - begin, (weight * normalized.to(tl.float32)).to(Output.dtype.element_ty), keep)


def rms_norm_columns(source, weight, out, eps, begin=None):
    """Canonical RMSNorm over each full row of ``source``, writing ``out.shape[1]`` columns from ``begin``.

    ``weight`` holds the weights of the written columns only. ``begin`` is a one-element integer tensor, so
    tensor-parallel ranks share one compiled forward; ``None`` writes the leading columns.
    """
    rows, width = source.shape
    if out.shape[0] != rows or not 0 < out.shape[1] <= width or weight.shape != (out.shape[1],):
        raise ValueError("RMSNorm writes a column range of its rows")
    if source.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("RMSNorm rows must be contiguous")
    if begin is not None and (begin.shape != (1,) or begin.dtype not in (torch.int32, torch.int64)):
        raise ValueError("RMSNorm column offsets are one-element integer tensors")
    block = triton.next_power_of_2(width)
    torch.library.wrap_triton(_rms_norm)[(rows,)](
        source, weight, out, source if begin is None else begin, source.stride(0), out.stride(0), float(eps),
        width, out.shape[1], begin is not None, block, num_warps=4 if block <= 2048 else 8)
    return out


@triton.jit
def _sigmoid_gate(Value, Gate, value_stride, gate_stride, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0).to(tl.int64), tl.program_id(1)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    valid = columns < WIDTH
    value = tl.load(Value + row * value_stride + columns, valid, other=0.).to(tl.float32)
    gate = tl.load(Gate + row * gate_stride + columns, valid, other=0.).to(tl.float32)
    gate = (1. / (1. + tl.exp(-gate))).to(Value.dtype.element_ty).to(tl.float32)
    tl.store(Value + row * value_stride + columns, (value * gate).to(Value.dtype.element_ty), valid)


def sigmoid_gate(value, gate):
    """``value.mul_(gate.sigmoid())`` with the sigmoid rounded to BF16 first, as the eager pair does."""
    rows, width = value.shape
    if gate.shape != value.shape or value.stride(-1) != 1 or gate.stride(-1) != 1:
        raise ValueError("sigmoid gating takes matching contiguous rows")
    block = min(1024, triton.next_power_of_2(width))
    torch.library.wrap_triton(_sigmoid_gate)[(rows, triton.cdiv(width, block))](
        value, gate, value.stride(0), gate.stride(0), width, block, num_warps=4)
    return value


@triton.jit
def _unpack_norm_q(Gathered, Weight, Out, g_row_stride, g_rank_stride, out_stride, eps,
                   LOCAL_Q: tl.constexpr, NORM_WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < NORM_WIDTH
    rank = cols // LOCAL_Q
    offset = cols % LOCAL_Q
    value = tl.load(Gathered + row * g_row_stride + rank * g_rank_stride + offset,
                    mask=valid, other=0.0).to(tl.float32)
    normalized = (value * tl.rsqrt(tl.sum(value * value) / NORM_WIDTH + eps)).to(Out.dtype.element_ty)
    weight = tl.load(Weight + cols, mask=valid, other=0.0).to(tl.float32)
    tl.store(Out + row * out_stride + cols, (weight * normalized.to(tl.float32)).to(Out.dtype.element_ty), mask=valid)


@triton.jit
def _unpack_norm_kv_and_tail(Gathered, Weight, LatentOut, g_row_stride, g_rank_stride, lat_stride, eps,
                             LOCAL_Q: tl.constexpr, LOCAL_KV: tl.constexpr,
                             KV_NORM_WIDTH: tl.constexpr, TAIL_WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < KV_NORM_WIDTH
    rank = cols // LOCAL_KV
    offset = LOCAL_Q + (cols % LOCAL_KV)
    value = tl.load(Gathered + row * g_row_stride + rank * g_rank_stride + offset,
                    mask=valid, other=0.0).to(tl.float32)
    normalized = (value * tl.rsqrt(tl.sum(value * value) / KV_NORM_WIDTH + eps)).to(LatentOut.dtype.element_ty)
    weight = tl.load(Weight + cols, mask=valid, other=0.0).to(tl.float32)
    tl.store(LatentOut + row * lat_stride + cols,
             (weight * normalized.to(tl.float32)).to(LatentOut.dtype.element_ty), mask=valid)
    if TAIL_WIDTH > 0:
        tail_cols = tl.arange(0, 64)
        tail_valid = tail_cols < TAIL_WIDTH
        tail_index = KV_NORM_WIDTH + tail_cols
        tail_rank = tail_index // LOCAL_KV
        tail_offset = LOCAL_Q + (tail_index % LOCAL_KV)
        tail_value = tl.load(Gathered + row * g_row_stride + tail_rank * g_rank_stride + tail_offset,
                             mask=tail_valid, other=0.0)
        tl.store(LatentOut + row * lat_stride + tail_index, tail_value, mask=tail_valid)


def unpack_and_rms_norm_qkv(gathered, local_q: int, local_kv: int, q_weight, kv_weight, q_norm_out, latent_out, eps: float):
    """Unpack rank-interleaved latents and apply canonical Q/KV RMSNorm and tail copy."""
    rows, world, width = gathered.shape
    q_width = q_norm_out.shape[1]
    kv_norm_width = kv_weight.shape[0]
    tail_width = latent_out.shape[1] - kv_norm_width
    if (local_q + local_kv > width or q_width > world * local_q or latent_out.shape[1] > world * local_kv
            or not 0 <= tail_width <= 64 or q_weight.shape != (q_width,)
            or q_norm_out.shape[0] != rows or latent_out.shape[0] != rows
            or gathered.stride(2) != 1 or q_norm_out.stride(1) != 1 or latent_out.stride(1) != 1):
        raise ValueError("fused latent unpacking requires complete rank shards, contiguous rows and at most 64 tail columns")
    q_block = triton.next_power_of_2(q_width)
    kv_block = triton.next_power_of_2(kv_norm_width)
    torch.library.wrap_triton(_unpack_norm_q)[(rows,)](
        gathered, q_weight, q_norm_out, gathered.stride(0), gathered.stride(1), q_norm_out.stride(0), float(eps),
        local_q, q_width, q_block, num_warps=4 if q_block <= 2048 else 8)
    torch.library.wrap_triton(_unpack_norm_kv_and_tail)[(rows,)](
        gathered, kv_weight, latent_out, gathered.stride(0), gathered.stride(1), latent_out.stride(0), float(eps),
        local_q, local_kv, kv_norm_width, tail_width, kv_block, num_warps=4 if kv_block <= 2048 else 8)