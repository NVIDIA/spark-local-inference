# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step-5 region selection, paged attention, and partial-block continuation."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="explicit CUDA assignment required")


def metadata(positions, *, starts=None, state_rows=None, pages=None):
    from ignition.components.paged_batch import Metadata

    positions = torch.tensor(positions, device="cuda", dtype=torch.int64)
    rows = 1 if starts is None else len(starts)
    m = Metadata.allocate(positions.numel(), rows, 64, "cuda")
    m.positions.copy_(positions)
    m.starts.copy_(positions[:1] if starts is None else torch.tensor(starts, device="cuda"))
    m.state_rows.copy_(torch.ones(rows, device="cuda") if state_rows is None else torch.tensor(state_rows, device="cuda"))
    if rows == 1:
        m.lengths.fill_(positions.numel())
        m.query_start_loc[1] = positions.numel()
    else:
        m.requests.copy_(torch.arange(rows, device="cuda"))
        m.lengths.copy_((positions >= 0).int())
        m.query_start_loc.copy_(torch.arange(rows + 1, device="cuda"))
    m.page_table.copy_(torch.arange(64, device="cuda").expand(rows, -1) if pages is None else pages)
    return m


def test_region_compression_chunk_boundaries():
    from ignition.components.gated_compression import commit_gated
    from ignition.recipes.step5.compression import compress

    torch.manual_seed(832)
    key = torch.randn(33, 256, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn_like(key)
    hk, hz = (torch.zeros(3, 8, 1, 256, dtype=torch.bfloat16, device="cuda") for _ in range(2))
    output = torch.zeros(64, 16, 256, dtype=torch.float8_e4m3fn, device="cuda")
    start = 0
    for count in (3, 2, 6, 9, 13):
        m = metadata(range(start, start + count))
        compress(key[start:start + count], gate[start:start + count], hk, hz, m, output)
        commit_gated(key[start:start + count], gate[start:start + count], hk, hz, m, m.lengths)
        start += count
        for region in range(start // 8):
            k = key[region * 8:region * 8 + 8].float()
            z = gate[region * 8:region * 8 + 8].float()
            probability = (z - z.max()).exp()
            expected = ((k * probability).sum(0) / probability.sum(0)).bfloat16().to(torch.float8_e4m3fn)
            torch.testing.assert_close(output[0, region].float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(hk[1, 0, 0], key[32], rtol=0, atol=0)
    assert not hk[0].any() and not hk[2].any()
