# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-scaled route normalization and BF16 weighted expert accumulation."""

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import launch_triton, prepare_triton


@triton.jit
def _normalize_weight_sum(routes, gain, weights, ids, valid_tokens, output,
                          HIDDEN: tl.constexpr, TOP_K: tl.constexpr, EXPERTS: tl.constexpr, EPS: tl.constexpr,
                          BLOCK: tl.constexpr):
    token = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    slots = tl.arange(0, TOP_K)
    valid = token < tl.load(valid_tokens)
    expert_ids = tl.load(ids + token * TOP_K + slots)
    route_valid = valid & (expert_ids >= 0) & (expert_ids < EXPERTS)
    x = tl.load(routes + (token * TOP_K + slots[:, None]) * HIDDEN + columns[None, :],
                mask=route_valid[:, None] & (columns[None, :] < HIDDEN), other=0).to(tl.float32)
    scale = tl.load(gain + columns, mask=columns < HIDDEN, other=0).to(tl.float32)
    # The centered gain addition and the scaled input each round to BF16.
    scale = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
    x = (x * scale[None, :]).to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(x * x, axis=1) / HIDDEN
    normalized = (x * tl.rsqrt(variance[:, None] + EPS)).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(weights + token * TOP_K + slots)
    weighted = (normalized * weight[:, None]).to(tl.bfloat16).to(tl.float32)
    remaining = tl.where(route_valid, expert_ids, 2147483647)
    total = tl.full((BLOCK,), 0, tl.float32)
    # Canonical expert traversal accumulates one BF16 contribution per expert ID.
    for _ in range(TOP_K):
        first = tl.min(remaining, axis=0)
        selected = (remaining == first) & (remaining < EXPERTS) & route_valid
        contribution = tl.sum(tl.where(selected[:, None], weighted, 0.), axis=0)
        total = (total + contribution).to(tl.bfloat16).to(tl.float32)
        remaining = tl.where(selected, 2147483647, remaining)
    tl.store(output + token * HIDDEN + columns, total, mask=columns < HIDDEN)


class PreparedRouteFinalizer:
    """Normalize replicated raw routes before applying FP32 routing weights."""

    def __init__(self, routes, gain, weights, ids, valid_tokens, output, *, eps, num_experts):
        if (routes.ndim != 3 or routes.dtype != torch.bfloat16 or gain.dtype != torch.bfloat16
                or weights.dtype != torch.float32 or weights.shape != routes.shape[:2]
                or ids.shape != weights.shape or gain.shape != routes.shape[2:]
                or output.shape != (routes.shape[0], routes.shape[2]) or output.dtype != routes.dtype
                or ids.dtype != torch.int32 or valid_tokens.dtype != torch.int32 or valid_tokens.numel() != 1
                or type(num_experts) is not int or num_experts <= 0
                or not routes.shape[1] or routes.shape[1] & (routes.shape[1] - 1)
                or any(t.device != routes.device or not t.is_contiguous()
                       for t in (routes, gain, weights, ids, valid_tokens, output))):
            raise ValueError("route finalization requires BF16 routes/gain/output and FP32 routing weights")
        self.output = output
        self.kernel = prepare_triton(_normalize_weight_sum,
            (routes, gain, weights, ids, valid_tokens, output, routes.shape[2], routes.shape[1],
             num_experts, eps, triton.next_power_of_2(routes.shape[2])), (routes.shape[0], 1, 1), num_warps=8,
            enable_fp_fusion=False)

    def functional(self):
        self.output_shape = tuple(self.output.shape)
        self.output = None
        self.kernel.args = tuple(None if index in (0, 2, 3, 4, 5) else value
                                 for index, value in enumerate(self.kernel.args))
        return self

    def run(self, routes, weights, ids, num_valid_tokens):
        output = torch.empty(self.output_shape, dtype=routes.dtype, device=routes.device)
        args = list(self.kernel.args)
        args[0], args[2], args[3], args[4], args[5] = routes, weights, ids, num_valid_tokens, output
        operation = self.kernel
        launch_triton(operation.kernel, operation.compiled, tuple(args), operation.grid, **operation.options)
        return output

    def __call__(self):
        self.kernel()
        return self.output
