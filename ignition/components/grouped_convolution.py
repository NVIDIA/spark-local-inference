# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block-local dynamic depthwise convolution with explicit BF16 arithmetic boundaries."""

import torch
import triton
import triton.language as tl

from .linear import UnquantizedProjection


@triton.jit
def _round(value):
    return tl.inline_asm_elementwise("cvt.rn.bf16.f32 $0, $1;", constraints="=h,f", args=[value],
                                    dtype=tl.bfloat16, is_pure=True, pack=1)


@triton.jit
def _convolve(X, Delta, Base, Y, N: tl.constexpr, H: tl.constexpr, GROUP: tl.constexpr,
              WIDTH: tl.constexpr, TAPS: tl.constexpr, DS: tl.constexpr, SIDE: tl.constexpr,
              BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, feature = offset // H, offset % H
    valid = offset < N * H
    total = tl.full((BLOCK,), 0, tl.bfloat16)
    for tap in tl.static_range(TAPS):
        x = tl.load(X + offset - tap * H, valid & (row % WIDTH >= tap), other=0).to(tl.float32)
        delta = tl.load(Delta + row * DS + (SIDE * TAPS + tap) * (H // GROUP) + feature // GROUP,
                        valid, other=0).to(tl.float32)
        base = tl.load(Base + (SIDE * TAPS + tap) * H + feature, valid, other=0).to(tl.float32)
        coefficient = _round(base + delta)
        product = _round(coefficient.to(tl.float32) * x)
        total = _round(total.to(tl.float32) + product.to(tl.float32))
    tl.store(Y + offset, total, valid)


class PreparedGroupedConvolution:
    def __init__(self, source, weights, *, block_size, group_size, session, name, buffers, token_capacity=None):
        self.block_size, self.group_size = block_size, group_size
        self.base, self.tokens, self.hidden = weights["base"], *source.shape
        capacity = self.tokens if token_capacity is None else token_capacity
        if capacity < self.tokens:
            raise ValueError("dynamic convolution scratch capacity must cover its input rows")
        if self.base.ndim != 3 or self.base.shape[0] != 2 or self.base.shape[2] != self.hidden:
            raise ValueError("dynamic convolution requires two sides of hidden-width tap weights")
        self.taps = self.base.shape[1]
        if self.hidden % group_size or self.tokens % block_size or not 1 <= self.taps <= block_size:
            raise ValueError("dynamic convolution groups and blocks must divide their input extents")
        channels = 2 * self.taps * self.hidden // group_size
        for key, width in (("coefficients", channels), ("before", self.hidden), ("after", self.hidden)):
            if key not in buffers:
                buffers[key] = torch.zeros((capacity, width), dtype=torch.bfloat16, device=source.device)
        self.coefficients, self.before, self.after = (buffers[k][:self.tokens] for k in ("coefficients", "before", "after"))
        self.project = UnquantizedProjection(source, weights["projection"], self.coefficients,
                                             session=session, name=name + ".coefficients")
        self._functional_layout = None

    def functional(self, *, session):
        """Return convolution outputs as local values while retaining weights and its prepared plan."""
        if self._functional_layout is None:
            self._functional_layout = (tuple(self.coefficients.shape), self.coefficients.stride(),
                                       self.coefficients.dtype, tuple(self.before.shape), self.before.stride(),
                                       self.before.dtype)
            self.project.functional(session=session)
            self.coefficients = self.before = self.after = None
        return self

    def convolve(self, source, side, output=None, coefficients=None):
        functional = self._functional_layout is not None
        coefficients = self.coefficients if coefficients is None else coefficients
        if output is None:
            if functional:
                _, _, _, shape, stride, dtype = self._functional_layout
                output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
            else:
                output = self.before if side == 0 else self.after
        tokens = source.shape[0]
        torch.library.wrap_triton(_convolve)[(triton.cdiv(source.numel(), 256),)](
            source, coefficients, self.base, output, tokens, self.hidden, self.group_size,
            self.block_size, self.taps, coefficients.stride(0), side, 256)
        return output

    def prepare(self, source):
        coefficients = self.project(source)
        before = self.convolve(source, 0, coefficients=coefficients)
        return (before, coefficients) if self._functional_layout is not None else before

    def finish(self, source, coefficients=None):
        if self._functional_layout is None:
            return self.convolve(source, 1, self.after)
        return self.convolve(source, 1, coefficients=coefficients)
