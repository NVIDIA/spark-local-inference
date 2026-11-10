# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""BRRR prologue residual arithmetic and graph replay on an assigned SM120 GPU.

Zero projection weights isolate residual mixing and normalization. This does not
qualify nonzero projection arithmetic, the TP4 suffix or full-model outputs.
"""

import os

import pytest
import torch

from ignition.recipes.deepseek41.native_brrr.extension import extension

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned SM120 CUDA device")


@pytest.mark.parametrize("rows", [5, 6, 7, 8])
@torch.no_grad()
def test_prologue_identity_mixing_and_zero_projections_replay(rows):
    if torch.cuda.get_device_capability(0) != (12, 0):
        pytest.skip("BRRR compiles for SM120 only")
    device = torch.device("cuda:0")
    ext = extension()
    def zeros(shape, dtype=torch.bfloat16):
        return torch.zeros(shape, device=device, dtype=dtype)
    y, residual = zeros((rows, 5120)), zeros((rows, 4, 5120))
    post = zeros((rows, 4), torch.float32)
    comb = torch.eye(4, device=device).expand(rows, -1, -1).contiguous()
    pre = zeros((rows, 4), torch.float32)
    fn, scale, base = zeros((24, 20480), torch.float32), torch.ones(3, device=device), zeros((24,), torch.float32)
    norm, qnorm, kvnorm = (torch.ones(n, device=device, dtype=torch.bfloat16) for n in (5120, 1280, 512))
    qa, qb = zeros((1792, 5120), torch.uint8), zeros((8192, 1280), torch.uint8)
    sa, sb = (torch.full((size,), 127, device=device, dtype=torch.uint8)
              for size in (1792 * 5120 // 32, 8192 * 1280 // 32))
    out_residual, out_norm = torch.empty_like(residual), torch.empty_like(y)
    qr, kv, q = zeros((rows, 1280)), zeros((rows, 512)), zeros((rows, 8192))
    qr_q, qr_s = zeros((rows, 1280), torch.uint8), zeros((ext.act_scale_bytes(rows),), torch.uint8)
    out_post, out_comb, out_pre = torch.empty_like(post), torch.empty_like(comb), torch.empty_like(pre)
    handle = ext.create_prologue(residual)
    graph = None
    try:
        def forward():
            ext.prologue_out(handle, y, residual, post, comb, pre, fn, norm,
                scale, base, qa, sa, qnorm, kvnorm, qb, sb, out_residual, out_norm,
                qr, kv, qr_q, qr_s, q, out_post, out_comb, out_pre, True)
        for _ in range(3):
            forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        # Changing the incoming basis checks that replay consumes live inputs.
        for basis in (0, 3, 1, 2):
            residual.fill_(.25)
            residual[:, basis].fill_(1.)
            pre.zero_()
            pre[:, basis] = 1
            for tensor in (out_residual, out_norm, qr, kv, q, out_post, out_comb, out_pre):
                tensor.fill_(float("nan"))
            before = torch.cuda.memory_stats()
            for _ in range(16):
                graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for counter in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                assert after[counter] == before[counter]
            assert torch.equal(out_residual, residual)
            assert torch.equal(out_norm, torch.ones_like(out_norm))
            assert all(torch.count_nonzero(tensor).item() == 0 for tensor in (qr, kv, q))
            # Production defers these coefficients to the fused suffix.
            assert all(torch.isnan(tensor).all() for tensor in (out_post, out_comb, out_pre))
    finally:
        torch.cuda.synchronize()
        if graph is not None:
            graph.reset()
        ext.destroy_engine(handle, residual)
