# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sparse paged GQA over QSA-selected positions for prefill-sized row counts on SM103.

QSA selection leaves each query row's original-token positions in a padded
row: the valid positions first, then -1. Each program attends one row and one
KV head over those positions, causally masked by the row's query position, with
an online FP32 softmax. On GB300 this Triton schedule (32-position tiles, one
warp per row and head, no per-tile branches so loads stay pipelined) attends a
4096-row Qwen prefill chunk at 32K context in 1.67 ms versus 2.70 ms for b12x's
CuTe kernel; on SM120 the CuTe kernel stays ahead, so only SM103 plans use it.
The design follows vLLM's QSA sparse GQA kernel.

Decode-sized row counts leave too few programs for one pass, so up to 128 rows
split each row's selection across programs, as vLLM's kernel does, and a second
kernel merges the partial softmaxes. At 8K context on GB300 that takes 101 us for
128 rows (b12x's CuTe kernel, unsplit there: 146 us), 32 us for 32 rows (41 us)
and 12 us for 4 rows (14 us).
"""

import math

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import launch_triton, warmup_triton

BLOCK_N = 32
# One warp per row and KV head. The launch configuration rides on the kernel so
# whole-forward compilation, which drops launch keywords, keeps it.
_CONFIG = triton.Config({}, num_warps=1, num_stages=2)
_MERGE_CONFIG = triton.Config({}, num_warps=2, num_stages=1)
# Rows that split their selection, and partial rows (splits x rows) any split launch needs.
SPLIT_ROWS = 128
PARTIAL_ROWS = 256


def splits_for(rows: int) -> int:
    """Programs per row and KV head: 64 at four rows or fewer, halving as rows double."""
    if not 0 < rows <= SPLIT_ROWS:
        return 1
    return max(2, 256 // triton.next_power_of_2(max(rows, 4)))


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _sparse_paged_gqa(Query, KeyCache, ValueCache, Selected, BlockTable, Requests, Positions, Output,
                      query_row_stride, key_page_stride, key_token_stride, value_page_stride, value_token_stride,
                      selected_row_stride, table_row_stride, output_row_stride,
                      SCALE_LOG2: tl.constexpr, KV_HEAD_STRIDE: tl.constexpr, WIDTH: tl.constexpr, PAGE_SIZE: tl.constexpr,
                      TABLE_WIDTH: tl.constexpr, GROUP: tl.constexpr, HEAD_DIM: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.load(Requests + row).to(tl.int64)
    query_position = tl.load(Positions + row)
    heads = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    columns = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP
    query = tl.load(Query + row.to(tl.int64) * query_row_stride + (first_head + heads[:, None]) * HEAD_DIM
                    + dims[None, :], mask=heads[:, None] < GROUP, other=0.0)
    maximum = tl.full((BLOCK_M,), -1.0e20, tl.float32)
    normalizer = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    for tile in range(0, tl.cdiv(WIDTH, BLOCK_N)):
        column = tile * BLOCK_N + columns
        position = tl.load(Selected + row.to(tl.int64) * selected_row_stride + column,
                           mask=(column < WIDTH) & (request >= 0), other=-1)
        safe = tl.maximum(position, 0)
        logical_page = safe // PAGE_SIZE
        valid = (position >= 0) & (position <= query_position) & (logical_page < TABLE_WIDTH)
        page = tl.load(BlockTable + request * table_row_stride + tl.minimum(logical_page, TABLE_WIDTH - 1),
                       mask=valid, other=-1).to(tl.int64)
        valid &= page >= 0
        page = tl.maximum(page, 0)
        offset = (safe % PAGE_SIZE).to(tl.int64)
        keys = tl.load(KeyCache + page[None, :] * key_page_stride + offset[None, :] * key_token_stride
                       + kv_head * KV_HEAD_STRIDE + dims[:, None], mask=valid[None, :], other=0.0).to(query.dtype)
        values = tl.load(ValueCache + page[:, None] * value_page_stride + offset[:, None] * value_token_stride
                         + kv_head * KV_HEAD_STRIDE + dims[None, :], mask=valid[:, None], other=0.0).to(query.dtype)
        scores = tl.dot(query, keys) * SCALE_LOG2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        following = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.math.exp2(maximum - following)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - following[:, None]), 0.0)
        accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(values.dtype), values)
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        maximum = following
    output = tl.where(normalizer[:, None] > 0, accumulator / tl.maximum(normalizer, 1.0e-20)[:, None], 0.0)
    tl.store(Output + row.to(tl.int64) * output_row_stride + (first_head + heads[:, None]) * HEAD_DIM
             + dims[None, :], output.to(Output.dtype.element_ty), mask=heads[:, None] < GROUP)


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _sparse_paged_gqa_split(Query, KeyCache, ValueCache, Selected, BlockTable, Requests, Positions, PartialOut,
                            PartialLse, query_row_stride, key_page_stride, key_token_stride, value_page_stride,
                            value_token_stride, selected_row_stride, table_row_stride,
                            SCALE_LOG2: tl.constexpr, KV_HEAD_STRIDE: tl.constexpr, WIDTH: tl.constexpr,
                            PAGE_SIZE: tl.constexpr, TABLE_WIDTH: tl.constexpr, GROUP: tl.constexpr,
                            HEAD_DIM: tl.constexpr, Q_HEADS: tl.constexpr, ROWS: tl.constexpr,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    request = tl.load(Requests + row).to(tl.int64)
    query_position = tl.load(Positions + row)
    heads = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    columns = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP
    query = tl.load(Query + row.to(tl.int64) * query_row_stride + (first_head + heads[:, None]) * HEAD_DIM
                    + dims[None, :], mask=heads[:, None] < GROUP, other=0.0)
    maximum = tl.full((BLOCK_M,), -1.0e20, tl.float32)
    normalizer = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    for tile in range(split, tl.cdiv(WIDTH, BLOCK_N), SPLITS):
        column = tile * BLOCK_N + columns
        position = tl.load(Selected + row.to(tl.int64) * selected_row_stride + column,
                           mask=(column < WIDTH) & (request >= 0), other=-1)
        safe = tl.maximum(position, 0)
        logical_page = safe // PAGE_SIZE
        valid = (position >= 0) & (position <= query_position) & (logical_page < TABLE_WIDTH)
        page = tl.load(BlockTable + request * table_row_stride + tl.minimum(logical_page, TABLE_WIDTH - 1),
                       mask=valid, other=-1).to(tl.int64)
        valid &= page >= 0
        page = tl.maximum(page, 0)
        offset = (safe % PAGE_SIZE).to(tl.int64)
        keys = tl.load(KeyCache + page[None, :] * key_page_stride + offset[None, :] * key_token_stride
                       + kv_head * KV_HEAD_STRIDE + dims[:, None], mask=valid[None, :], other=0.0).to(query.dtype)
        values = tl.load(ValueCache + page[:, None] * value_page_stride + offset[:, None] * value_token_stride
                         + kv_head * KV_HEAD_STRIDE + dims[None, :], mask=valid[:, None], other=0.0).to(query.dtype)
        scores = tl.dot(query, keys) * SCALE_LOG2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        following = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.math.exp2(maximum - following)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - following[:, None]), 0.0)
        accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(values.dtype), values)
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        maximum = following
    attended = normalizer > 0
    lse = tl.where(attended, maximum + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
    output = tl.where(attended[:, None], accumulator / tl.maximum(normalizer, 1.0e-20)[:, None], 0.0)
    partial = (split * ROWS + row).to(tl.int64)
    tl.store(PartialOut + (partial * Q_HEADS + first_head + heads[:, None]) * HEAD_DIM + dims[None, :], output,
             mask=heads[:, None] < GROUP)
    tl.store(PartialLse + partial * Q_HEADS + first_head + heads, lse, mask=heads < GROUP)


@triton.autotune(configs=[_MERGE_CONFIG], key=[])
@triton.jit
def _merge_splits(PartialOut, PartialLse, Output, output_row_stride, HEAD_DIM: tl.constexpr, Q_HEADS: tl.constexpr,
                  ROWS: tl.constexpr, SPLITS: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, HEAD_DIM)
    partials = (splits * ROWS + row).to(tl.int64)
    lse = tl.load(PartialLse + partials * Q_HEADS + head)
    top = tl.max(lse, axis=0)
    weights = tl.where(top > -float("inf"), tl.math.exp2(lse - top), 0.0)
    denominator = tl.sum(weights, axis=0)
    values = tl.load(PartialOut + (partials[:, None] * Q_HEADS + head) * HEAD_DIM + dims[None, :])
    merged = tl.sum(values * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(Output + row.to(tl.int64) * output_row_stride + head * HEAD_DIM + dims,
             merged.to(Output.dtype.element_ty))


class PreparedSparseAttention:
    """Attend BF16 queries over QSA-selected positions of BF16 or unit-scaled FP8 K/V.

    Positions after a row's query position are ignored. Rows without a request,
    or without a valid selected position, are written as zeros. Up to
    ``SPLIT_ROWS`` rows split their selection into FP32 ``partial`` and
    ``partial_lse`` rows, which layers can share.
    """

    def __init__(self, query, key_cache, value_cache, selected, block_table, requests, positions, output, *,
                 partial=None, partial_lse=None):
        rows, q_heads, dim = query.shape
        _, page_size, kv_heads, cache_dim = key_cache.shape
        if (cache_dim != dim or value_cache.shape != key_cache.shape or q_heads % kv_heads
                or output.shape != query.shape or selected.shape[0] != rows or requests.shape != (rows,)
                or positions.shape != (rows,)
                or any(t.dtype != torch.bfloat16 for t in (query, output))
                or key_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
                or value_cache.dtype != key_cache.dtype
                or selected.dtype != torch.int32 or block_table.dtype != torch.int32
                or query.stride()[1:] != (dim, 1) or output.stride()[1:] != (dim, 1)
                or key_cache.stride()[2:] != (dim, 1) or value_cache.stride()[2:] != (dim, 1)
                or selected.stride(1) != 1 or block_table.stride(1) != 1):
            raise ValueError("sparse attention requires BF16 [rows,heads,dim] queries, paged BF16 or FP8 K/V with "
                             "contiguous heads, and int32 selections and block table")
        self.query, self.selected, self.block_table = query, selected, block_table
        self.requests, self.positions, self.output = requests, positions, output
        self.group = q_heads // kv_heads
        self.grid = (rows, kv_heads, 1)
        self.options = {"KV_HEAD_STRIDE": dim, "WIDTH": selected.shape[1], "PAGE_SIZE": page_size,
                        "TABLE_WIDTH": block_table.shape[1], "GROUP": self.group, "HEAD_DIM": dim,
                        "BLOCK_M": max(16, triton.next_power_of_2(self.group)), "BLOCK_N": BLOCK_N,
                        "SCALE_LOG2": (1.0 / math.sqrt(dim)) * 1.4426950408889634}
        self.splits = splits_for(rows)
        if self.splits > 1:
            if (partial is None or partial_lse is None or partial.dtype != torch.float32
                    or partial_lse.dtype != torch.float32 or partial.shape[1:] != (q_heads, dim)
                    or partial_lse.shape[1:] != (q_heads,) or partial.shape[0] < self.splits * rows
                    or partial_lse.shape[0] < self.splits * rows
                    or not partial.is_contiguous() or not partial_lse.is_contiguous()):
                raise ValueError("split sparse attention requires contiguous FP32 partial rows for every split")
            self.partial, self.partial_lse = partial, partial_lse
            self.options.update(Q_HEADS=q_heads, ROWS=rows, SPLITS=self.splits)
            self.grid = (rows, kv_heads, self.splits)
            self.merge_args = (partial, partial_lse, output, output.stride(0))
            self.merge_options = {"HEAD_DIM": dim, "Q_HEADS": q_heads, "ROWS": rows, "SPLITS": self.splits}
            self.merge_grid = (rows, q_heads, 1)
            self.merge = warmup_triton(_merge_splits.fn, self.merge_args, self.merge_grid,
                                       num_warps=_MERGE_CONFIG.num_warps,
                                       num_stages=_MERGE_CONFIG.num_stages, **self.merge_options)
        self.compiled = warmup_triton(self._kernel().fn, self._args(key_cache, value_cache), self.grid,
                                      num_warps=_CONFIG.num_warps, num_stages=_CONFIG.num_stages,
                                      **self.options)

    def _kernel(self):
        # Looked up by name: compile workers receive this object pickled, and Triton
        # kernels pickle only through their module attributes.
        return _sparse_paged_gqa if self.splits == 1 else _sparse_paged_gqa_split

    def _args(self, key_cache, value_cache):
        common = (self.query.stride(0), key_cache.stride(0), key_cache.stride(1), value_cache.stride(0),
                  value_cache.stride(1), self.selected.stride(0), self.block_table.stride(0))
        operands = (self.query, key_cache, value_cache, self.selected, self.block_table, self.requests, self.positions)
        if self.splits == 1:
            return (*operands, self.output, *common, self.output.stride(0))
        return (*operands, self.partial, self.partial_lse, *common)

    def functional(self, *, session):
        self.output_layout = tuple(self.output.shape), tuple(self.output.stride()), self.output.dtype
        self.partial_shape = None if self.splits == 1 else (self.splits * self.output.shape[0], *self.output.shape[1:])
        self.query = self.selected = self.block_table = self.requests = self.positions = self.output = None
        if self.splits > 1:
            self.partial = self.partial_lse = self.merge_args = None
        return self

    def __call__(self, key_cache, value_cache, *, query=None, selected=None, block_table=None,
                 requests=None, positions=None):
        if self.output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=query.device).zero_()
            common = (query.stride(0), key_cache.stride(0), key_cache.stride(1), value_cache.stride(0),
                      value_cache.stride(1), selected.stride(0), block_table.stride(0))
            operands = (query, key_cache, value_cache, selected, block_table, requests, positions)
            if self.splits == 1:
                args = (*operands, output, *common, output.stride(0))
            else:
                partial = torch.zeros(self.partial_shape, dtype=torch.float32, device=query.device)
                lse = torch.zeros(self.partial_shape[:-1], dtype=torch.float32, device=query.device)
                args = (*operands, partial, lse, *common)
            launch_triton(self._kernel(), self.compiled, args, self.grid, **self.options)
            if self.splits > 1:
                launch_triton(_merge_splits, self.merge, (partial, lse, output, output.stride(0)),
                              self.merge_grid, **self.merge_options)
            return output
        launch_triton(self._kernel(), self.compiled, self._args(key_cache, value_cache), self.grid, **self.options)
        if self.splits > 1:
            launch_triton(_merge_splits, self.merge, self.merge_args, self.merge_grid, **self.merge_options)
        return self.output
