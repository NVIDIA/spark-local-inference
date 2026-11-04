# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared route binning and BF16 GeGLU/SwiGLU expert stages."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import prepare_triton


@triton.jit
def _init_bins_kernel(
    expert_counts,
    tile_expert_ids,
    sorted_pair_ids,
    NUM_EXPERTS: tl.constexpr,
    MAX_TILES: tl.constexpr,
    MAX_PADDED_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(sorted_pair_ids + offsets, -1, mask=offsets < MAX_PADDED_ROWS)
    tl.store(tile_expert_ids + offsets, -1, mask=offsets < MAX_TILES)
    tl.store(expert_counts + offsets, 0, mask=offsets < NUM_EXPERTS)


@triton.jit
def _count_routes_kernel(
    top_k_index,
    num_valid_tokens,
    expert_counts,
    pair_ranks,
    TOP_K: tl.constexpr,
    TOTAL_PAIRS: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr = 2147483647,
):
    pid = tl.program_id(0)
    pair_ids = pid * BLOCK + tl.arange(0, BLOCK)
    valid_tokens = tl.load(num_valid_tokens).to(tl.int32)
    valid_pairs = valid_tokens * TOP_K
    mask = (pair_ids < TOTAL_PAIRS) & (pair_ids < valid_pairs)
    expert = tl.load(top_k_index + pair_ids, mask=mask, other=0).to(tl.int32)
    mask = mask & (expert >= 0) & (expert < NUM_EXPERTS)
    ranks = tl.atomic_add(expert_counts + expert, 1, mask=mask)
    tl.store(pair_ranks + pair_ids, ranks, mask=mask)


@triton.jit
def _scan_experts_kernel(
    expert_counts,
    expert_tile_starts,
    expert_tile_counts,
    NUM_EXPERTS: tl.constexpr,
    TILE_M: tl.constexpr,
):
    experts = tl.arange(0, NUM_EXPERTS)
    counts = tl.load(expert_counts + experts).to(tl.int32)
    tiles = (counts + TILE_M - 1) // TILE_M
    starts = tl.cumsum(tiles, axis=0) - tiles
    tl.store(expert_tile_starts + experts, starts)
    tl.store(expert_tile_counts + experts, tiles)


@triton.jit
def _scatter_routes_kernel(
    top_k_index,
    num_valid_tokens,
    pair_ranks,
    expert_tile_starts,
    sorted_pair_ids,
    tile_expert_ids,
    TOP_K: tl.constexpr,
    TOTAL_PAIRS: tl.constexpr,
    TILE_M: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr = 2147483647,
):
    pid = tl.program_id(0)
    pair_ids = pid * BLOCK + tl.arange(0, BLOCK)
    valid_tokens = tl.load(num_valid_tokens).to(tl.int32)
    valid_pairs = valid_tokens * TOP_K
    mask = (pair_ids < TOTAL_PAIRS) & (pair_ids < valid_pairs)
    expert = tl.load(top_k_index + pair_ids, mask=mask, other=0).to(tl.int32)
    mask = mask & (expert >= 0) & (expert < NUM_EXPERTS)
    rank = tl.load(pair_ranks + pair_ids, mask=mask, other=0).to(tl.int32)
    tile_start = tl.load(expert_tile_starts + expert, mask=mask, other=0).to(tl.int32)
    dest = tile_start * TILE_M + rank
    tl.store(sorted_pair_ids + dest, pair_ids, mask=mask)
    tile_idx = tile_start + (rank // TILE_M)
    tl.store(tile_expert_ids + tile_idx, expert, mask=mask)


@triton.jit
def _gelu_tanh(x):
    c = 0.7978845608028654
    k = 0.044715
    inner = c * (x + k * x * x * x)
    return 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))


@triton.jit
def _moe_bf16_stage1_kernel(
    source,
    gate_up_proj,
    sorted_pair_ids,
    tile_expert_ids,
    intermediate,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    INTER: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    SILU: tl.constexpr = False,
    ROUND_SILU: tl.constexpr = True,
):
    tile_id = tl.program_id(0)
    n_id = tl.program_id(1)
    expert = tl.load(tile_expert_ids + tile_id).to(tl.int64)
    if expert < 0:
        return
    m_offs = tl.arange(0, TILE_M)
    n_offs = n_id * TILE_N + tl.arange(0, TILE_N)
    n_mask = n_offs < INTER
    pairs = tl.load(sorted_pair_ids + tile_id * TILE_M + m_offs).to(tl.int64)
    m_mask = pairs >= 0
    tokens = tl.where(m_mask, pairs // TOP_K, 0)

    gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    up_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    expert_base = gate_up_proj + expert * (2 * INTER) * HIDDEN

    for k_start in range(0, HIDDEN, TILE_K):
        k_offs = k_start + tl.arange(0, TILE_K)
        k_mask = k_offs < HIDDEN
        x_tile = tl.load(
            source + tokens[:, None] * HIDDEN + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w_gate = tl.load(
            expert_base + n_offs[None, :] * HIDDEN + k_offs[:, None],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        w_up = tl.load(
            expert_base + (INTER + n_offs)[None, :] * HIDDEN + k_offs[:, None],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        gate_acc += tl.dot(x_tile, w_gate)
        up_acc += tl.dot(x_tile, w_up)

    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    if SILU:
        gate_val = gate * tl.sigmoid(gate)
        if ROUND_SILU:
            gate_val = gate_val.to(tl.bfloat16).to(tl.float32)
    else:
        gate_val = _gelu_tanh(gate).to(tl.bfloat16).to(tl.float32)
    up_val = up_acc.to(tl.bfloat16).to(tl.float32)
    out = (gate_val * up_val).to(tl.bfloat16)
    safe_pairs = tl.where(m_mask, pairs, 0)
    tl.store(
        intermediate + safe_pairs[:, None] * INTER + n_offs[None, :],
        out,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _moe_bf16_stage2_kernel(
    intermediate,
    down_proj,
    top_k_weights,
    sorted_pair_ids,
    tile_expert_ids,
    routed_out,
    HIDDEN: tl.constexpr,
    INTER: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    RAW_OUTPUT: tl.constexpr = False,
):
    tile_id = tl.program_id(0)
    n_id = tl.program_id(1)
    expert = tl.load(tile_expert_ids + tile_id).to(tl.int64)
    if expert < 0:
        return
    m_offs = tl.arange(0, TILE_M)
    n_offs = n_id * TILE_N + tl.arange(0, TILE_N)
    n_mask = n_offs < HIDDEN
    pairs = tl.load(sorted_pair_ids + tile_id * TILE_M + m_offs).to(tl.int64)
    m_mask = pairs >= 0
    safe_pairs = tl.where(m_mask, pairs, 0)

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    expert_base = down_proj + expert * HIDDEN * INTER

    for k_start in range(0, INTER, TILE_K):
        k_offs = k_start + tl.arange(0, TILE_K)
        k_mask = k_offs < INTER
        x_tile = tl.load(
            intermediate + safe_pairs[:, None] * INTER + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w_down = tl.load(
            expert_base + n_offs[None, :] * INTER + k_offs[:, None],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x_tile, w_down)

    if RAW_OUTPUT:
        out = acc.to(tl.bfloat16)
    else:
        weights = tl.load(top_k_weights + safe_pairs, mask=m_mask, other=0.0).to(tl.float32)
        out = (acc.to(tl.bfloat16).to(tl.float32) * weights[:, None]).to(tl.bfloat16)
    tl.store(
        routed_out + safe_pairs[:, None] * HIDDEN + n_offs[None, :],
        out,
        mask=m_mask[:, None] & n_mask[None, :],
    )



@dataclass(frozen=True)
class BF16ExpertScratch:
    intermediate: torch.Tensor
    counts: torch.Tensor
    starts: torch.Tensor
    tile_counts: torch.Tensor
    ranks: torch.Tensor
    sorted_ids: torch.Tensor
    tile_experts: torch.Tensor

    @classmethod
    def allocate(cls, tokens, top_k, experts, intermediate, *, buffers):
        pairs = tokens * top_k
        tiles = min(pairs, triton.cdiv(pairs, 16) + experts)
        def zeros(shape, dtype=torch.int32):
            return buffers.zeros(shape, dtype)
        return cls(zeros((pairs, intermediate), torch.bfloat16), zeros((experts,)), zeros((experts,)),
                   zeros((experts,)), zeros((pairs,)), zeros((tiles * 16,)), zeros((tiles,)))

    def rows(self, tokens, top_k):
        pairs = tokens * top_k
        tiles = min(pairs, triton.cdiv(pairs, 16) + self.counts.numel())
        return type(self)(self.intermediate[:pairs], self.counts, self.starts, self.tile_counts,
                          self.ranks[:pairs], self.sorted_ids[:tiles * 16], self.tile_experts[:tiles])


class PreparedBF16Experts:
    """Execute intermediate-sharded BF16 experts into unweighted route rows."""

    def __init__(self, source, gate_up, down, ids, weights, output, valid_tokens, *,
                 buffers=None, scratch=None, round_silu=True):
        tokens, hidden = source.shape
        count, twice_inter, width = gate_up.shape
        inter = twice_inter // 2
        top_k = ids.shape[1]
        if (source.dtype != torch.bfloat16 or gate_up.dtype != source.dtype or down.dtype != source.dtype
                or width != hidden or down.shape != (count, hidden, inter)
                or ids.shape != weights.shape or ids.shape[0] != tokens
                or output.shape != (tokens, top_k, hidden) or output.dtype != source.dtype
                or ids.dtype != torch.int32 or weights.dtype != torch.float32
                or valid_tokens.dtype != torch.int32 or valid_tokens.numel() != 1
                or count <= 0 or count & (count - 1) or twice_inter % 2 or type(round_silu) is not bool
                or any(t.device != source.device or not t.is_contiguous()
                       for t in (source, gate_up, down, ids, weights, output, valid_tokens))):
            raise ValueError("BF16 experts require matched source, weights, routes, and raw output geometry")
        self.output = output
        if scratch is None:
            scratch = BF16ExpertScratch.allocate(tokens, top_k, count, inter, buffers=buffers)
        self.intermediate = scratch.intermediate
        pairs = tokens * top_k
        tile_m, tile_n, tile_k = 16, 64, 64
        tiles = min(pairs, triton.cdiv(pairs, tile_m) + count)
        padded = tiles * tile_m
        counts, starts, tile_counts = scratch.counts, scratch.starts, scratch.tile_counts
        ranks, sorted_ids, tile_experts = scratch.ranks, scratch.sorted_ids, scratch.tile_experts
        self.bins = (
            prepare_triton(_init_bins_kernel, (counts, tile_experts, sorted_ids, count, tiles, padded, 256),
                           (triton.cdiv(max(padded, tiles, count), 256), 1, 1)),
            prepare_triton(_count_routes_kernel, (ids, valid_tokens, counts, ranks, top_k, pairs, 256, count),
                           (triton.cdiv(pairs, 256), 1, 1)),
            prepare_triton(_scan_experts_kernel, (counts, starts, tile_counts, count, tile_m), (1, 1, 1)),
            prepare_triton(_scatter_routes_kernel,
                           (ids, valid_tokens, ranks, starts, sorted_ids, tile_experts, top_k, pairs, tile_m, 256, count),
                           (triton.cdiv(pairs, 256), 1, 1)),
        )
        self.first = prepare_triton(_moe_bf16_stage1_kernel,
            (source, gate_up, sorted_ids, tile_experts, self.intermediate,
             top_k, hidden, inter, tile_m, tile_n, tile_k, True, round_silu),
            (tiles, triton.cdiv(inter, tile_n), 1), num_warps=4)
        self.second = prepare_triton(_moe_bf16_stage2_kernel,
            (self.intermediate, down, weights, sorted_ids, tile_experts, output,
             hidden, inter, tile_m, tile_n, tile_k, True),
            (tiles, triton.cdiv(hidden, tile_n), 1), num_warps=4)

    def __call__(self):
        self.output.zero_()
        for kernel in self.bins:
            kernel()
        self.first()
        self.second()
        return self.output
