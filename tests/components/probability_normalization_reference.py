# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unconditional maximum and threshold arithmetic used as a normalization oracle."""

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _maxima(Probabilities, Maxima, V: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Probabilities + row * V + token, token < V, other=-float("inf"))
    tl.store(Maxima + row * BLOCKS + block, tl.max(value, 0))


@triton.jit
def _filter(Probabilities, Params, Maxima, Masses, Output,
            V: tl.constexpr, WIDTH: tl.constexpr, PARAM_STRIDE: tl.constexpr,
            BLOCKS: tl.constexpr, REDUCE: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    part = tl.arange(0, REDUCE)
    largest = tl.max(tl.load(Maxima + row * BLOCKS + part, part < BLOCKS, other=-float("inf")), 0)
    minimum = tl.load(Params + (row // WIDTH) * PARAM_STRIDE + 2)
    token = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Probabilities + row * V + token, token < V, other=0.)
    filtered = tl.where((value > 0) & (value >= largest * minimum), value, 0.)
    tl.store(Output + row * V + token, filtered, token < V)
    tl.store(Masses + row * BLOCKS + block, tl.sum(filtered, 0))


@triton.jit
def _totals(Masses, Totals, Valid, BLOCKS: tl.constexpr, REDUCE: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, REDUCE)
    mass = tl.sum(tl.load(Masses + row * BLOCKS + part, part < BLOCKS, other=0.), 0)
    valid = (tl.load(Valid + row) != 0) & (mass > 0) & (mass == mass) & (mass < float("inf"))
    tl.store(Valid + row, valid)
    tl.store(Totals + row, tl.maximum(mass, 1.1754943508222875e-38))


@triton.jit
def _apply(Output, Valid, Totals, V: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    valid = tl.load(Valid + row) != 0
    value = tl.load(Output + row * V + token, token < V, other=0.)
    value = tl.math.div_rn(value, tl.load(Totals + row))
    value = tl.where(valid, value, (token == 0).to(tl.float32))
    tl.store(Output + row * V + token, value, token < V)


class UnconditionalProbabilityNormalization:
    """Normalize positive probabilities after the vocabulary-wide min-p threshold."""

    def __init__(self, output, valid):
        batch, self.width, self.vocabulary = output.shape
        self.rows = batch * self.width
        self.blocks = triton.cdiv(self.vocabulary, _BLOCK)
        self.output, self.valid = output, valid
        self.maxima = torch.empty((self.rows, self.blocks), device=output.device, dtype=torch.float32)
        self.masses = torch.empty_like(self.maxima)
        self.totals = torch.empty((self.rows,), device=output.device, dtype=torch.float32)

    def __call__(self, probabilities, params):
        grid = (self.rows, self.blocks)
        reduce = triton.next_power_of_2(self.blocks)
        torch.library.wrap_triton(_maxima)[grid](
            probabilities, self.maxima, self.vocabulary, self.blocks, _BLOCK)
        torch.library.wrap_triton(_filter)[grid](
            probabilities, params, self.maxima, self.masses, self.output,
            self.vocabulary, self.width, params.stride(0), self.blocks, reduce, _BLOCK,
            enable_fp_fusion=False)
        torch.library.wrap_triton(_totals)[(self.rows,)](
            self.masses, self.totals, self.valid, self.blocks, reduce)
        torch.library.wrap_triton(_apply)[grid](
            self.output, self.valid, self.totals, self.vocabulary, _BLOCK)
        return self.output
