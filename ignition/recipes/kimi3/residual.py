# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 attention residuals fused with the residual update and the following RMSNorm.

One program per row updates the running block sum, optionally commits it as a residual block,
mixes the committed blocks and the running sum by their softmax scores, and normalizes the
mixture for the next projection. The arithmetic follows the canonical composition: BF16 residual
additions, FP32 scores and mixing rounded to BF16, then RMSNorm with its BF16 rounding.
"""

import torch
import triton
import triton.language as tl

KEEP, ADD, REPLACE = 0, 1, 2
# Source tile and warps by committed block count, measured for one decode row.
TILES = ((2, 4), (2, 4), (4, 8), (4, 8), (8, 8), (8, 8), (8, 8), (8, 8), (4, 8))


@triton.jit
def _mix(Prefix, Delta, Blocks, Score, Norm, Output, Mixture, prefix_stride, delta_stride, block_row_stride,
         block_stride, output_stride, mixture_stride, eps, output_eps, HIDDEN: tl.constexpr, SOURCES: tl.constexpr,
         WRITE: tl.constexpr, DELTA: tl.constexpr, MIXTURE: tl.constexpr, BLOCK: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    valid = columns < HIDDEN
    current = tl.load(Prefix + row * prefix_stride + columns, valid, other=0.)
    if DELTA == 1:
        delta = tl.load(Delta + row * delta_stride + columns, valid, other=0.)
        current = (current.to(tl.float32) + delta.to(tl.float32)).to(Prefix.dtype.element_ty)
    elif DELTA == 2:
        current = tl.load(Delta + row * delta_stride + columns, valid, other=0.)
    if DELTA != 0:
        tl.store(Prefix + row * prefix_stride + columns, current, valid)
    if WRITE >= 0:
        tl.store(Blocks + row * block_row_stride + WRITE * block_stride + columns, current, valid)
    if SOURCES == 0:
        mixed = current.to(tl.float32)
    else:
        # The running sum is the last source; it is reread after its store is visible to the whole program.
        tl.debug_barrier()
        weight = tl.load(Score + columns, valid, other=0.)
        maximum = tl.full((), float("-inf"), tl.float32)
        total = tl.zeros((), tl.float32)
        accumulated = tl.zeros((BLOCK,), tl.float32)
        for tile in tl.static_range((SOURCES + TILE) // TILE):
            sources = tile * TILE + tl.arange(0, TILE)
            present = sources <= SOURCES
            pointers = tl.where((sources == SOURCES)[:, None], Prefix + row * prefix_stride + columns[None, :],
                                Blocks + row * block_row_stride + sources[:, None] * block_stride + columns[None, :])
            values = tl.load(pointers, present[:, None] & valid[None, :], other=0.).to(tl.float32)
            scale = tl.rsqrt(tl.sum(values * values, axis=1) / HIDDEN + eps)
            scores = tl.where(present, tl.sum(values * weight[None, :], axis=1) * scale, float("-inf"))
            peak = tl.maximum(maximum, tl.max(scores, axis=0))
            rescale = tl.exp(maximum - peak)
            exponentials = tl.exp(scores - peak)
            total = total * rescale + tl.sum(exponentials, axis=0)
            accumulated = accumulated * rescale + tl.sum(exponentials[:, None] * values, axis=0)
            maximum = peak
        mixed = (accumulated / total).to(Prefix.dtype.element_ty).to(tl.float32)
    if MIXTURE:
        tl.store(Mixture + row * mixture_stride + columns, mixed.to(Mixture.dtype.element_ty), valid)
    normalized = (mixed * tl.rsqrt(tl.sum(mixed * mixed) / HIDDEN + output_eps)).to(Prefix.dtype.element_ty)
    output = tl.load(Norm + columns, valid, other=0.).to(tl.float32) * normalized.to(tl.float32)
    tl.store(Output + row * output_stride + columns, output.to(Output.dtype.element_ty), valid)


def score_weight(norm_weight, projection):
    """Fold the residual score's RMSNorm weight into its projection, as the canonical score does in FP32."""
    return (norm_weight.float() * projection.float().flatten()).contiguous()


class ResidualStream:
    """Running block sum, committed residual blocks, and the normalized input of the next projection."""

    def __init__(self, prefix, blocks, output):
        tokens, hidden = prefix.shape
        if (blocks.shape[0], blocks.shape[2]) != (tokens, hidden) or output.shape != prefix.shape:
            raise ValueError("residual blocks and output must match the running sum's rows and width")
        if any(t.stride(-1) != 1 or t.dtype != torch.bfloat16 for t in (prefix, blocks, output)):
            raise ValueError("residual stream tensors must be BF16 with contiguous rows")
        self.prefix, self.blocks, self.output = prefix, blocks, output

    def mix(self, delta, mode, *, sources, write, score, norm, eps, output_eps, mixture=None, output=None):
        """Apply ``mode`` with ``delta``, commit block ``write`` (or none for -1), and normalize the mixture
        of the first ``sources`` committed blocks and the running sum into ``output`` (by default the stream's);
        ``mixture`` also receives it unnormalized."""
        if not 0 <= sources <= self.blocks.shape[1] or not -1 <= write < self.blocks.shape[1] or write >= 0 and write < sources:
            raise ValueError("residual mixing reads only committed blocks and never overwrites one")
        if delta is not None and (delta.shape != self.prefix.shape or delta.stride(-1) != 1):
            raise ValueError("residual delta must match the running sum")
        if mixture is not None and (mixture.shape != self.prefix.shape or mixture.stride(-1) != 1):
            raise ValueError("residual mixture output must match the running sum")
        output = self.output if output is None else output
        if output.shape != self.prefix.shape or output.stride(-1) != 1 or output.dtype != torch.bfloat16:
            raise ValueError("residual output must be BF16 rows matching the running sum")
        rows, hidden = self.prefix.shape
        delta = self.prefix if delta is None else delta
        tile, warps = TILES[min(sources, len(TILES) - 1)]
        torch.library.wrap_triton(_mix)[(rows,)](
            self.prefix, delta, self.blocks, score, norm, output, output if mixture is None else mixture,
            self.prefix.stride(0), delta.stride(0), self.blocks.stride(0), self.blocks.stride(1),
            output.stride(0), (output if mixture is None else mixture).stride(0), float(eps),
            float(output_eps), hidden, sources, write, mode, mixture is not None, triton.next_power_of_2(hidden),
            tile, num_warps=warps)
        return output
