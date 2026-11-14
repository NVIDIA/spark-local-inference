# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""mHC post and pre for prefill-sized calls on SM103 through FlashInfer's mHC kernels.

b12x's fused post/pre runs a 4096-row GLM call in about 250 us on GB300, far
from bandwidth. FlashInfer's residual mixing and pre-map big fuse run near it;
between them, a Triton kernel projects the flattened residual onto the 24
mixing logits and sums its squares in four K splits, which the big fuse
reduces. The FP32 projection weights split into BF16 high and low parts, which
reproduce them more closely than TF32 does; the parts are stacked as 64 weight rows
so that each K tile issues one BF16 MMA.
The collapsed branch input then takes the weighted RMSNorm. Arithmetic follows b12x's references:
mixes scaled by the RMS of the flattened residual, pre = sigmoid + hc_eps,
post = 2 sigmoid, comb by Sinkhorn with hc_eps, and the mixing
new[j] = post[j] x + sum_i comb[i, j] residual[i].
"""

from __future__ import annotations

import itertools
import weakref

import torch
import triton
import triton.language as tl


SPLITS = 4
_BLOCK_T = 128
_BLOCK_K = 64
_CONFIG = triton.Config({}, num_warps=8, num_stages=3)
# Calls above this row count compose FlashInfer's kernels; decode keeps b12x's fused launches.
MIN_ROWS = 256
_SPLIT_WEIGHTS: dict[int, torch.Tensor] = {}
_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


# FlashInfer's mHC launches build on first use, so compiled forwards see the composition as one operator
# that reads its inputs and writes the residual, mixes and normalized input.
@torch.library.custom_op("flare::sm103_composed_mhc", mutates_args=("residual", "post", "comb", "normalized"))
def _composed(inputs: list[torch.Tensor], residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor,
              normalized: torch.Tensor, dot: torch.Tensor, squares: torch.Tensor,
              collapsed: torch.Tensor, handle: int) -> None:
    _OWNERS[handle].run(inputs, residual, post, comb, normalized, dot, squares, collapsed)


@_composed.register_fake
def _fake_composed(inputs, residual, post, comb, normalized, dot, squares, collapsed, handle) -> None:
    pass


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _project(Residual, Fn, Dot, Sqrsum, rows, K: tl.constexpr, SPLITS: tl.constexpr,
             BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr):
    # Invariant: one MMA per accumulator per pipelined loop iteration. Chained tl.dot calls on one
    # accumulator lose the second MMA's result under Triton's SM10x pipeliner at num_stages > 1: the
    # tmem_load that reads it is not waited on, so some K tiles drop that contribution and replays
    # of identical inputs differ. The weight tile therefore stacks the BF16 high part (rows 0..31)
    # and low part (rows 32..63) so one dot produces both, and the halves are summed after the loop.
    tile, split = tl.program_id(0), tl.program_id(1)
    tokens = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    live = tokens < rows
    mixes = tl.arange(0, 32)
    parts = tl.arange(0, 64)
    acc = tl.zeros((BLOCK_T, 64), tl.float32)
    squares = tl.zeros((BLOCK_T,), tl.float32)
    span = K // SPLITS
    for start in range(split * span, (split + 1) * span, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        values = tl.load(Residual + tokens.to(tl.int64)[:, None] * K + k[None, :], mask=live[:, None], other=0.0)
        acc += tl.dot(values, tl.trans(tl.load(Fn + parts[:, None] * K + k[None, :])))
        wide = values.to(tl.float32)
        squares += tl.sum(wide * wide, 1)
    dot = tl.sum(tl.reshape(acc, (BLOCK_T, 2, 32)), 1)
    tl.store(Dot + (split * rows + tokens.to(tl.int64))[:, None] * 24 + mixes[None, :], dot,
             mask=live[:, None] & (mixes[None, :] < 24))
    tl.store(Sqrsum + split * rows + tokens, squares, mask=live)


def _split_weights(fn):
    """BF16 high (rows 0..31) and low (rows 32..63) parts of the FP32 projection, padded to 32 mixes;
    one tensor shared by every call on ``fn``."""
    key = fn.data_ptr()
    if key not in _SPLIT_WEIGHTS:
        padded = torch.zeros((32, fn.shape[1]), dtype=torch.float32, device=fn.device)
        padded[:24] = fn
        high = padded.to(torch.bfloat16)
        _SPLIT_WEIGHTS[key] = torch.cat((high, (padded - high.float()).to(torch.bfloat16)), 0).contiguous()
    return _SPLIT_WEIGHTS[key]


class ComposedMHC:
    """Post/pre or pre with invocation-local outputs and work tensors."""

    def __init__(self, operation, weights, scratch, *, rms_eps, hc_eps, sinkhorn_iters, norm_eps):
        from flashinfer.mhc import _mhc_post_impl, _mhc_pre_big_fuse_impl
        from flashinfer.norm import rmsnorm

        self.mix, self.fuse, self.norm = _mhc_post_impl, _mhc_pre_big_fuse_impl, rmsnorm

        rows, _, hidden = scratch.residual.shape
        if operation not in ("pre", "post_pre") or weights.fn.shape != (24, 4 * hidden):
            raise ValueError("composed mHC serves pre and post/pre with 24 mixes")
        self.operation, self.weights = operation, weights
        self.rows, self.hidden = rows, hidden
        self.eps = (float(rms_eps), float(hc_eps), int(sinkhorn_iters), float(norm_eps))
        self.fn = _split_weights(weights.fn)
        device = scratch.residual.device
        self.norm_program = None
        if weights.norm.dtype == torch.float32:
            from flashinfer.triton.kernels.norm import rms_norm_kernel
            from ignition.preparation.triton import prepare_triton

            # FlashInfer's CUDA RMSNorm binds weight pointers with the activation dtype.
            self.norm_program = prepare_triton(rms_norm_kernel,
                (hidden, rows, torch.empty((rows, hidden), dtype=torch.bfloat16, device=device),
                 hidden, None, None, 0, weights.norm,
                 torch.empty((rows, hidden), dtype=torch.bfloat16, device=device), hidden, None), (rows, 1, 1),
                EPS=norm_eps, BLOCK_SIZE=triton.next_power_of_2(hidden), HAS_IN_SCALE=False,
                HAS_OUT_SCALE=False, HAS_OUTPUT=True, HAS_RESIDUAL=False, ROUND_BEFORE_WEIGHT=False,
                num_warps=max(8, min(32, triton.next_power_of_2(hidden) // 256)))
        flat = scratch.residual.view(rows, -1)
        dot = torch.empty((SPLITS, rows, 24), dtype=torch.float32, device=device)
        squares = torch.empty((SPLITS, rows), dtype=torch.float32, device=device)
        self.args = (flat, self.fn, dot, squares, rows)
        self.options = {"K": 4 * hidden, "SPLITS": SPLITS, "BLOCK_T": _BLOCK_T, "BLOCK_K": _BLOCK_K}
        self.grid = (triton.cdiv(rows, _BLOCK_T), SPLITS, 1)
        self.compiled = _project.fn.warmup(*self.args, grid=self.grid, num_warps=_CONFIG.num_warps,
                                           num_stages=_CONFIG.num_stages, **self.options)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, *inputs):
        residual = inputs[0] if self.operation == "pre" else inputs[1]
        outputs = (torch.empty_like(residual),
                   torch.empty((residual.shape[0], 4), dtype=torch.float32, device=residual.device),
                   torch.empty((residual.shape[0], 4, 4), dtype=torch.float32, device=residual.device),
                   torch.empty((residual.shape[0], self.hidden), dtype=residual.dtype, device=residual.device))
        dot = torch.empty((SPLITS, self.rows, 24), dtype=torch.float32, device=residual.device)
        squares = torch.empty((SPLITS, self.rows), dtype=torch.float32, device=residual.device)
        collapsed = torch.empty((self.rows, self.hidden), dtype=torch.bfloat16, device=residual.device)
        _composed(list(inputs), *outputs, dot, squares, collapsed, self.handle)
        return outputs

    def functional(self):
        self.args = None
        if self.norm_program is not None:
            self.norm_program.args = None
        return self

    def run(self, inputs, residual, post, comb, normalized, dot, squares, collapsed):
        w = self.weights
        rms_eps, hc_eps, iterations, norm_eps = self.eps
        if self.operation == "post_pre":
            source, source_residual, source_post, source_comb = inputs
            self.mix(residual, source, source_residual, source_post, source_comb)
        else:
            residual.copy_(inputs[0])
        flat = residual.view(self.rows, -1)
        self.compiled[self.grid](flat, self.fn, dot, squares, self.rows)
        self.fuse(post, comb, collapsed, dot, squares, residual, w.scale, w.base,
                               residual.shape[1] * residual.shape[2], rms_eps, hc_eps, hc_eps, 2.0, iterations,
                               SPLITS, 0)
        if self.norm_program is None:
            self.norm(collapsed, w.norm, norm_eps, out=normalized)
        else:
            from ignition.preparation.triton import launch_triton

            args = (self.hidden, self.rows, collapsed, self.hidden, None, None, 0,
                    w.norm, normalized, self.hidden, None)
            launch_triton(self.norm_program.kernel, self.norm_program.compiled, args,
                          self.norm_program.grid, **self.norm_program.options)
