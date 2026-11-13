# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eight-token index compression with pending continuation rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _compress(K, Z, HistoryK, HistoryZ, Requests, Positions, Starts, Begins, Rows, Pages, Output,
              KS: tl.constexpr, ZS: tl.constexpr, HS: tl.constexpr, PS: tl.constexpr):
    token = tl.program_id(0)
    position = tl.load(Positions + token)
    if position >= 0 and position % 8 == 7:
        request = tl.load(Requests + token)
        start, begin = tl.load(Starts + request), tl.load(Begins + request)
        row = tl.load(Rows + request)
        t, d = tl.arange(0, 8), tl.arange(0, 256)
        absolute = position - 7 + t
        current = absolute >= start
        source = begin + absolute - start
        k = tl.load(K + source[:, None] * KS + d[None], current[:, None], 0).to(tl.float32)
        k += tl.load(HistoryK + row * HS + (absolute[:, None] % 8) * 256 + d[None], ~current[:, None], 0).to(tl.float32)
        z = tl.load(Z + source[:, None] * ZS + d[None], current[:, None], 0).to(tl.float32)
        z += tl.load(HistoryZ + row * HS + (absolute[:, None] % 8) * 256 + d[None], ~current[:, None], 0).to(tl.float32)
        # Compression uses one maximum across tokens and channels, then a per-channel denominator.
        maximum = tl.max(tl.max(z, 0), 0)
        probability = tl.exp(z - maximum)
        summary = (tl.sum(k * probability, 0) / tl.sum(probability, 0)).to(tl.bfloat16)
        page = tl.load(Pages + request * PS + position // 128)
        tl.store(Output + (page.to(tl.int64) * 16 + position % 128 // 8) * 256 + d, summary)


def compress(key, gate, history_key, history_gate, metadata, output):
    torch.library.wrap_triton(_compress)[(key.shape[0],)](
        key, gate, history_key, history_gate, metadata.requests, metadata.positions, metadata.starts,
        metadata.query_start_loc, metadata.state_rows, metadata.page_table, output,
        key.stride(0), gate.stride(0), history_key.stride(0), metadata.page_table.stride(0),
        num_warps=8, enable_fp_fusion=False)
