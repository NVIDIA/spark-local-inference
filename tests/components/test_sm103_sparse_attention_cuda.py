# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sparse paged GQA over QSA-selected positions against a dense FP32 computation."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 6, 24, 96, 160])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_attends_selected_causal_positions_and_zeroes_empty_rows(rows, cache_dtype):
    """Up to 128 rows split their selection across programs; more rows attend in one pass."""
    from ignition.components.sm103_sparse_attention import PARTIAL_ROWS, PreparedSparseAttention, splits_for

    generator = torch.Generator(device="cuda").manual_seed(2051)
    pages, width, context = 40, 2051, 2048
    query = torch.randn((rows, 24, 256), generator=generator, device="cuda").bfloat16()
    keys, values = (torch.randn((pages, 64, 2, 256), generator=generator, device="cuda").to(cache_dtype) for _ in range(2))
    table = torch.randperm(pages, generator=generator, device="cuda")[:context // 64].int().view(1, -1).repeat(2, 1)
    table[1, 3] = -1
    requests = torch.zeros(rows, dtype=torch.int64, device="cuda")
    requests[rows * 5 // 12:rows * 5 // 6] = 1
    idle = rows // 24
    if idle:
        requests[-idle:] = -1
    positions = torch.arange(context - rows, context, device="cuda")
    selected = torch.full((rows, width), -1, dtype=torch.int32, device="cuda")
    for row in range(rows - idle):
        # Unordered groups of four, some past the query position, then a causal tail.
        count = 64 if row % 3 else 600
        chosen = torch.randperm(context // 4, generator=generator, device="cuda")[:count] * 4
        tokens = (chosen[:, None] + torch.arange(4, device="cuda")).flatten()
        tokens = torch.cat((tokens, positions[row] - torch.arange(3, device="cuda")))
        selected[row, :tokens.numel()] = tokens.int()
    if rows > 10:
        selected[10] = -1
    output = torch.full_like(query, float("nan"))
    partial = torch.full((PARTIAL_ROWS, 24, 256), float("nan"), device="cuda")
    partial_lse = torch.full((PARTIAL_ROWS, 24), float("nan"), device="cuda")
    attention = PreparedSparseAttention(query, keys, values, selected, table, requests, positions, output,
                                        partial=partial, partial_lse=partial_lse)
    assert attention.splits == splits_for(rows) and (attention.splits > 1) == (rows <= 128)
    attention(keys, values)

    for row in range(rows):
        request, chosen = int(requests[row]), selected[row]
        chosen = chosen[(chosen >= 0) & (chosen <= positions[row])].long()
        pages_of = table[max(request, 0)][chosen // 64].long()
        chosen, pages_of = chosen[pages_of >= 0], pages_of[pages_of >= 0]
        if request < 0 or chosen.numel() == 0:
            assert not output[row].any()
            continue
        k = keys[pages_of, chosen % 64].float().repeat_interleave(12, 1)
        v = values[pages_of, chosen % 64].float().repeat_interleave(12, 1)
        weights = torch.softmax(torch.einsum("hd,nhd->hn", query[row].float(), k) / 16, -1)
        expected = torch.einsum("hn,nhd->hd", weights, v)
        torch.testing.assert_close(output[row].float(), expected, rtol=0, atol=2e-2)
