# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Cooperative validation reductions retain disjoint semaphore and reduction storage."""

import os

import pytest
import torch

from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                               reason='requires an explicitly assigned CUDA device')


class PixelValidation:
    def __init__(self):
        self.pixels = torch.full((16384, 1536), .25, device='cuda')
        self.count = torch.full((1,), 256, dtype=torch.int32, device='cuda')
        self.output = torch.empty((256, 1536), dtype=torch.bfloat16, device='cuda')
        self.errors = torch.empty((1,), dtype=torch.int32, device='cuda')

    def __call__(self):
        live = torch.arange(256, device=self.count.device) < self.count
        valid = (self.count > 0) & (self.count <= 256)
        valid &= (~live[:, None] | torch.isfinite(self.pixels[:256])).all()
        self.output.copy_(torch.where(valid & live[:, None], self.pixels[:256], 0))
        finite = torch.isfinite(self.output).all()
        self.errors.copy_((~valid).int() | (~finite).int() * 64)


@torch.no_grad()
def test_combined_pool_cooperative_validation_replays_without_workspace_aliasing():
    forward = PixelValidation()
    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for count, bad_row, expected_error in ((256, None, 0), (3, 0, 1), (3, 20, 0), (0, None, 1)):
        forward.pixels.fill_(.25)
        forward.count.fill_(count)
        if bad_row is not None:
            forward.pixels[bad_row, 0] = float('nan')
        before = torch.cuda.memory_stats()
        graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        assert forward.errors.item() == expected_error
        for key in ('allocation.all.allocated', 'allocation.all.freed',
                    'segment.all.allocated', 'segment.all.freed'):
            assert before[key] == after[key]
        expected = torch.zeros_like(forward.output)
        if expected_error == 0:
            expected[:count].fill_(.25)
        torch.testing.assert_close(forward.output, expected, rtol=0, atol=0)
    graph.reset()
