# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Parallel min-p normalization with per-row validity."""

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _maxima(Probabilities, Params, Maxima, V: tl.constexpr, WIDTH: tl.constexpr,
            PARAM_STRIDE: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    minimum = tl.load(Params + (row // WIDTH) * PARAM_STRIDE + 2)
    maximum = tl.full((), 0., tl.float32)
    if minimum != 0.:
        value = tl.load(Probabilities + row * V + token, token < V, other=-float("inf"))
        maximum = tl.max(value, 0)
    tl.store(Maxima + row * BLOCKS + block, maximum)


@triton.jit
def _filter(Probabilities, Params, Maxima, Masses, Output,
            V: tl.constexpr, WIDTH: tl.constexpr, PARAM_STRIDE: tl.constexpr,
            BLOCKS: tl.constexpr, REDUCE: tl.constexpr, BLOCK: tl.constexpr,
            IN_PLACE: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    part = tl.arange(0, REDUCE)
    minimum = tl.load(Params + (row // WIDTH) * PARAM_STRIDE + 2)
    token = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Probabilities + row * V + token, token < V, other=0.)
    if minimum == 0.:
        filtered = tl.where(value > 0, value, 0.)
        # The positive predicate excludes NaNs and nonpositive values before reduction.
        mass = tl.sum(filtered, 0)
        if IN_PLACE:
            tl.store(Output + row * V + token, filtered, (token < V) & ~(value > 0))
        else:
            tl.store(Output + row * V + token, filtered, token < V)
    else:
        largest = tl.max(tl.load(Maxima + row * BLOCKS + part,
                                part < BLOCKS, other=-float("inf")), 0)
        filtered = tl.where((value > 0) & (value >= largest * minimum), value, 0.)
        tl.store(Output + row * V + token, filtered, token < V)
        mass = tl.sum(filtered, 0)
    tl.store(Masses + row * BLOCKS + block, mass)


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


class PreparedProbabilityNormalization:
    """Apply min-p and normalize without assigning an entire vocabulary to one CTA."""

    def __init__(self, output, valid):
        batch, self.width, self.vocabulary = output.shape
        self.rows = batch * self.width
        self.blocks = triton.cdiv(self.vocabulary, _BLOCK)
        self.output, self.valid = output, valid
        self.maxima = torch.empty((self.rows, self.blocks), device=output.device, dtype=torch.float32)
        self.masses = torch.empty_like(self.maxima)
        self.totals = torch.empty((self.rows,), device=output.device, dtype=torch.float32)

    def functional(self):
        self.output = self.valid = None
        self.maxima = self.masses = self.totals = None
        return self

    def __call__(self, probabilities, params):
        return self.run(probabilities, params, self.output, self.valid)

    def run(self, probabilities, params, output, valid):
        maxima, masses, totals = self.maxima, self.masses, self.totals
        if maxima is None:
            maxima = torch.empty((self.rows, self.blocks), dtype=torch.float32, device=probabilities.device)
            masses = torch.empty_like(maxima)
            totals = torch.empty((self.rows,), dtype=torch.float32, device=probabilities.device)
        grid = (self.rows, self.blocks)
        reduce = triton.next_power_of_2(self.blocks)
        torch.library.wrap_triton(_maxima)[grid](
            probabilities, params, maxima, self.vocabulary, self.width,
            params.stride(0), self.blocks, _BLOCK)
        torch.library.wrap_triton(_filter)[grid](
            probabilities, params, maxima, masses, output,
            self.vocabulary, self.width, params.stride(0), self.blocks, reduce, _BLOCK,
            probabilities is output,
            enable_fp_fusion=False)
        torch.library.wrap_triton(_totals)[(self.rows,)](
            masses, totals, valid, self.blocks, reduce)
        torch.library.wrap_triton(_apply)[grid](
            output, valid, totals, self.vocabulary, _BLOCK)
        return output
