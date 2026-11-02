# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused gated RMS normalization of recurrent attention outputs before their output projection."""

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import launch_triton, warmup_triton
from b12x.gemm.blockscaled._quantize import _scale_offset


@triton.jit
def _gated_values(Hidden, Gate, Weight, Live, gate_row_stride, EPS: tl.constexpr, HEADS: tl.constexpr,
                    DIM: tl.constexpr, HEAD_BLOCK: tl.constexpr, ROUND_NORMALIZED: tl.constexpr,
                    BF16_WEIGHT: tl.constexpr, SILU_GATE: tl.constexpr,
                    NORM_BEFORE_GATE: tl.constexpr = True, WEIGHT_PER_HEAD: tl.constexpr = False):
    # FP32 RMS normalization per head with configurable gate placement.
    # ROUND_NORMALIZED rounds the normalized values to BF16 before weighting, and BF16_WEIGHT also
    # rounds their product with a BF16 weight, as a module computing in its input dtype does.
    # Rows at or after the live count are written as zeros.
    row = tl.program_id(0)
    head = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)[:, None]
    column = tl.arange(0, DIM)[None, :]
    mask = head < HEADS
    offset = row * (HEADS * DIM) + head * DIM + column
    hidden = tl.load(Hidden + offset, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(Gate + row.to(tl.int64) * gate_row_stride + head * DIM + column, mask=mask, other=0.0).to(tl.float32)
    gate = gate * tl.sigmoid(gate) if SILU_GATE else tl.sigmoid(gate)
    if not NORM_BEFORE_GATE:
        hidden = hidden * gate
    weight = tl.load(Weight + (head * DIM if WEIGHT_PER_HEAD else 0) + column, mask=mask, other=0).to(tl.float32)
    variance = tl.sum(hidden * hidden, axis=1)[:, None] / DIM
    # A constexpr epsilon keeps the arithmetic FP32 under whole-forward compilation,
    # which would pass a float argument as FP64.
    hidden = hidden * tl.rsqrt(variance + EPS)
    if ROUND_NORMALIZED:
        hidden = hidden.to(tl.bfloat16)
        if BF16_WEIGHT:
            hidden = (hidden * weight).to(tl.bfloat16).to(tl.float32)
        else:
            hidden = weight * hidden.to(tl.float32)
    else:
        hidden = weight * hidden
    value = hidden * gate if NORM_BEFORE_GATE else hidden
    live = row < tl.load(Live)
    return offset, mask, tl.where(live, value, 0.0).to(tl.bfloat16)


@triton.jit
def _gated_rms_norm(Hidden, Gate, Weight, Output, Live, gate_row_stride, EPS: tl.constexpr, HEADS: tl.constexpr,
                    DIM: tl.constexpr, HEAD_BLOCK: tl.constexpr, ROUND_NORMALIZED: tl.constexpr,
                    BF16_WEIGHT: tl.constexpr, SILU_GATE: tl.constexpr,
                    NORM_BEFORE_GATE: tl.constexpr = True, WEIGHT_PER_HEAD: tl.constexpr = False):
    offset, mask, value = _gated_values(Hidden, Gate, Weight, Live, gate_row_stride, EPS, HEADS,
        DIM, HEAD_BLOCK, ROUND_NORMALIZED, BF16_WEIGHT, SILU_GATE, NORM_BEFORE_GATE, WEIGHT_PER_HEAD)
    tl.store(Output + offset, value, mask=mask)


@triton.jit
def _gated_rms_norm_mxfp8(Hidden, Gate, Weight, Output, Live, gate_row_stride, Values, ScaleRows, ScaleMma,
                         EPS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr, HEAD_BLOCK: tl.constexpr,
                         ROUND_NORMALIZED: tl.constexpr, BF16_WEIGHT: tl.constexpr, SILU_GATE: tl.constexpr,
                         NORM_BEFORE_GATE: tl.constexpr, WEIGHT_PER_HEAD: tl.constexpr):
    offset, mask, value = _gated_values(Hidden, Gate, Weight, Live, gate_row_stride, EPS, HEADS,
        DIM, HEAD_BLOCK, ROUND_NORMALIZED, BF16_WEIGHT, SILU_GATE, NORM_BEFORE_GATE, WEIGHT_PER_HEAD)
    if Output is not None:
        tl.store(Output + offset, value, mask=mask)
    groups: tl.constexpr = HEADS * DIM // 32
    local_groups: tl.constexpr = HEAD_BLOCK * DIM // 32
    x = tl.reshape(value.to(tl.float32), (local_groups, 32))
    amax = tl.max(tl.abs(x), 1)
    ratio = tl.where(amax > 0, amax / 448.0, 1.0)
    exponent = tl.minimum(tl.maximum(tl.ceil(tl.log2(ratio)), -127.0), 127.0)
    scale = tl.exp2(exponent)
    quantized = (x / scale[:, None]).to(tl.float8e4nv)
    tl.store(Values + offset, tl.reshape(quantized, (HEAD_BLOCK, DIM)), mask=mask)
    row = tl.program_id(0)
    group = tl.program_id(1) * local_groups + tl.arange(0, local_groups)
    byte = (exponent + 127).to(tl.uint8)
    tl.store(ScaleRows + row * groups + group, byte, group < groups)
    tl.store(ScaleMma + _scale_offset(row, group, groups), byte, group < groups)


class PreparedGatedRMSNorm:
    """Normalize BF16 groups and gate them with sigmoid(gate) or silu(gate).

    Matches an FP32 gated RMSNorm (normalize, scale by the weight, multiply by
    the sigmoid gate, round to BF16) such as Transformers' RMSNormGated with a
    sigmoid activation. ``round_normalized`` instead matches a module that rounds
    the normalized values to its BF16 input dtype before weighting them, as
    Qwen's gated norms do. Rows at or after the device live count become zeros.
    Inputs and output are caller-owned storage with contiguous rows; the gate
    may be a column slice of a wider row. The output may feed a projection directly.
    ``norm_before_gate=False`` gates in FP32 before computing the variance.
    A [heads, dim] weight gives each group separate affine parameters. Passing a
    preparation session declares compilation and tuning with the model's batch.
    With an encoded destination, output=None omits the BF16 output entirely.
    """

    def __init__(self, hidden, gate, weight, output, live, *, eps, round_normalized=False, activation="sigmoid",
                 norm_before_gate=True, session=None, name=None, encoded=None):
        rows, heads, dim = hidden.shape
        gate_rows_contiguous = gate.stride()[1:] == (dim, 1)
        if output is None and encoded is None:
            raise ValueError("gated RMS norm requires a BF16 output or an encoded destination")
        outputs = () if output is None else (output,)
        if (gate.shape != hidden.shape or any(t.shape != hidden.shape for t in outputs)
                or weight.shape not in ((dim,), (heads, dim))
                or any(t.dtype != torch.bfloat16 for t in (hidden, gate, *outputs))
                or weight.dtype not in (torch.bfloat16, torch.float32)
                or not all(t.is_contiguous() for t in (hidden, weight, *outputs)) or not gate_rows_contiguous
                or live.shape != (1,) or dim & (dim - 1) or activation not in ("sigmoid", "silu")):
            raise ValueError("gated RMS norm requires BF16 [rows,heads,dim] tensors with contiguous rows "
                             "and a [dim] or [heads,dim] weight")
        self.head_block = min(8, triton.next_power_of_2(heads))
        self.args = (hidden, gate, weight, output, live, gate.stride(0))
        self.grid = (rows, triton.cdiv(heads, self.head_block), 1)
        self.options = {"EPS": float(eps), "HEADS": heads, "DIM": dim, "HEAD_BLOCK": self.head_block,
                        "ROUND_NORMALIZED": bool(round_normalized),
                        "BF16_WEIGHT": weight.dtype == torch.bfloat16, "SILU_GATE": activation == "silu",
                        "NORM_BEFORE_GATE": bool(norm_before_gate), "WEIGHT_PER_HEAD": weight.ndim == 2,
                        "num_warps": 4}
        self.kernel = _gated_rms_norm
        if encoded is not None:
            from b12x.quantization import mxfp8

            if (session is not None or dim % 32 or not isinstance(encoded, mxfp8.MXFP8Rows)
                    or encoded.values.shape != (rows, heads * dim) or encoded.values.device != hidden.device):
                raise ValueError("encoded gated norm requires prepared MXFP8 rows and 32-aligned head dimensions")
            self.kernel = _gated_rms_norm_mxfp8
            self.args += (encoded.values, encoded.scale_rows.view(torch.uint8),
                          mxfp8.dense_scale_storage(encoded.scale_mma))
        self.plan = None
        if session is None:
            self.compiled = warmup_triton(self.kernel, self.args, self.grid, **self.options)
        else:
            from .gated_norm_preparation import declare_norm

            self.plan = declare_norm(self.args, self.options, session=session, name=name)

    def __call__(self):
        if self.plan is None:
            launch_triton(self.kernel, self.compiled, self.args, self.grid, **self.options)
        else:
            from .gated_norm_preparation import execute

            execute(*self.args[:5], self.plan.handle)
