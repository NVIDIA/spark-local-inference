# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""C4/C128 compression across chunking, rejection, and restored continuations."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.gated_compression import commit_gated, pool_gated
from ignition.components.paged_batch import Metadata

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="CUDA tests are opt-in")


def reference(values, scores, ape, ratio, head, stop):
    result = torch.zeros((stop, head), dtype=torch.bfloat16, device=values.device)
    for end in range(ratio, stop + 1, ratio):
        if ratio == 4:
            previous = max(0, end - 2 * ratio)
            v = torch.cat((values[previous : end - ratio, :head], values[end - ratio : end, head:]))
            s = torch.cat(
                (
                    scores[previous : end - ratio, :head] + ape[: end - ratio - previous, :head],
                    scores[end - ratio : end, head:] + ape[:, head:],
                )
            )
        else:
            v, s = values[end - ratio : end], scores[end - ratio : end] + ape
        result[end - 1] = (v * s.softmax(0)).sum(0)
    return result


@pytest.mark.parametrize("ratio,head", ((4, 128), (4, 512), (128, 512)))
@torch.inference_mode()
def test_chunking_rejection_and_restored_prefix(ratio, head):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    extent, width = (2 * ratio, 2 * head) if ratio == 4 else (ratio, head)
    generator = torch.Generator(device=device).manual_seed(4104)
    values = torch.randn((4096, width), device=device, generator=generator)
    scores = torch.randn_like(values)
    ape = torch.randn((ratio, width), device=device, generator=generator)
    expected = reference(values, scores, ape, ratio, head, 4096)
    # Padded row strides match fields embedded in a larger continuation record.
    state = torch.full((4, extent * width * 2 + 64), 99.0, device=device)
    sv = state[:, : extent * width].view(4, extent, width)
    ss = state[:, extent * width : 2 * extent * width].view(4, extent, width)

    def run(start, size, accepted, *, row=1, inactive=False, capture=False):
        m = Metadata.allocate(size, 1, 1, device)
        m.starts.fill_(start)
        m.state_rows.fill_(0 if inactive else row)
        m.query_start_loc[1] = 0 if inactive else size
        m.positions.copy_(
            torch.arange(start, start + size, device=device)
            if not inactive
            else -torch.ones(size, device=device)
        )
        out = torch.empty((size, head), dtype=torch.bfloat16, device=device)
        counts = torch.tensor([0 if inactive else accepted], dtype=torch.int32, device=device)
        before = state.clone()

        def pool():
            pool_gated(
                values[start : start + size], scores[start : start + size], ape, sv, ss, m, out, ratio=ratio
            )

        pool()
        torch.testing.assert_close(state, before, atol=0, rtol=0)
        torch.testing.assert_close(
            out, torch.zeros_like(out) if inactive else expected[start : start + size], atol=0.003, rtol=0.01
        )

        def commit():
            commit_gated(values[start : start + size], scores[start : start + size], sv, ss, m, counts)

        if capture:
            commit()
            state.copy_(before)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                pool()
                commit()
            graph.replay()
            graph.reset()
        else:
            commit()
        actual = before.clone()
        for p in range(start, start + (0 if inactive else accepted)):
            actual[row, (p % extent) * width : (p % extent + 1) * width] = values[p]
            offset = extent * width + p % extent * width
            actual[row, offset : offset + width] = scores[p]
        torch.testing.assert_close(state, actual, atol=0, rtol=0)

    run(0, ratio - 1, ratio - 1)
    checkpoint = state[1].clone()
    run(ratio - 1, 8, 1, capture=True)
    run(ratio, 9, 3)
    run(ratio + 3, 6, 0)
    run(ratio + 3, 6, 2, inactive=True)
    # Restore into a distinct row and cross the same compression boundary.
    state[2].copy_(checkpoint)
    run(ratio - 1, 2 * ratio + 3, 2 * ratio + 3, row=2)
    state[1].copy_(checkpoint)
    run(ratio - 1, 4096 - ratio + 1, 4096 - ratio + 1)


@pytest.mark.parametrize("ratio", (4, 128))
@torch.inference_mode()
def test_packed_requests_commit_independently(ratio):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    head, extent = 128, ratio * (2 if ratio == 4 else 1)
    width = head * (2 if ratio == 4 else 1)
    values = torch.randn((16, width), device=device)
    scores = torch.randn_like(values)
    sv = torch.randn((5, extent, width), device=device)
    ss = torch.randn_like(sv)
    before_v, before_s = sv.clone(), ss.clone()
    m = SimpleNamespace(
        starts=torch.tensor([3, 127, 16], dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 5, 11, 16], dtype=torch.int32, device=device),
        state_rows=torch.tensor([3, 1, 0], dtype=torch.int64, device=device),
    )
    counts = torch.tensor([2, 4, 0], dtype=torch.int32, device=device)
    commit_gated(values, scores, sv, ss, m, counts)
    for request, count in enumerate((2, 4, 0)):
        for offset in range(count):
            ring = (int(m.starts[request]) + offset) % extent
            source = int(m.query_start_loc[request]) + offset
            row = int(m.state_rows[request])
            before_v[row, ring] = values[source]
            before_s[row, ring] = scores[source]
    torch.testing.assert_close(sv, before_v, atol=0, rtol=0)
    torch.testing.assert_close(ss, before_s, atol=0, rtol=0)
