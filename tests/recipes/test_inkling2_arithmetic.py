# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 serving arithmetic over packed requests against the canonical per-sequence arithmetic."""

import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.inkling2.arithmetic import (commit_history, dynamic_short_convolution, l2_clip, log_scaling,
                                                  residual_mix, short_convolution, stored_history)
from ignition.reference import inkling2 as ref

BF16 = torch.bfloat16


def packed(starts, lengths, rows, capacity):
    """CPU metadata of requests at `starts` with `lengths` query rows in state `rows`."""
    count = len(starts)
    metadata = Metadata.allocate(capacity, count, 1, "cpu")
    metadata.set(torch.tensor(starts), torch.tensor(lengths, dtype=torch.int32), torch.tensor(rows),
                 torch.zeros(count, 1, dtype=torch.int32), torch.zeros(count, max(max(lengths), 1), dtype=torch.long),
                 torch.tensor([count]))
    return metadata


def test_length_scale_follows_clipping_after_rounding_beyond_the_floor():
    generator = torch.Generator().manual_seed(1)
    positions = torch.tensor([0, 127_998, 127_999, 128_000, 200_000, 1_000_000])
    rows = (torch.randn(6, 3, 16, generator=generator) * 3).to(BF16)
    scale = (1 + 0.1 * torch.randn(3, 16, generator=generator)).to(BF16)
    tau = log_scaling(positions, 128_000, 0.1)
    assert torch.equal(tau, ref.log_scaling(positions, 128_000, 0.1))
    assert torch.equal(tau[:3], torch.ones(3)) and bool((tau[3:] > 1).all())
    assert torch.equal(l2_clip(rows, 8.0, scale=scale, tau=tau), ref.l2_clip(rows, 8.0, scale=scale, tau=tau))


def test_packed_convolutions_read_and_commit_each_requests_history():
    """Two live requests, an empty one and padding rows, with static and dynamic taps."""
    generator = torch.Generator().manual_seed(2)
    channels, width, heads = 32, 4, 2
    lengths, starts, rows = [5, 2, 0], [7, 1, 0], [2, 1, 0]
    metadata = packed(starts, lengths, rows, 9)
    x = torch.randn(9, channels, generator=generator).to(BF16)
    weight = (0.3 * torch.randn(channels, 1, width, generator=generator)).to(BF16)
    static = (0.3 * torch.randn(width, channels, generator=generator)).to(BF16)
    dynamic = (0.3 * torch.randn(9, heads, width, generator=generator)).to(BF16)
    history = torch.randn(3, width - 1, channels, generator=generator).to(BF16)
    original = history.clone()
    previous = stored_history(history, metadata)
    output = short_convolution(x, weight, previous, metadata)
    lagged = dynamic_short_convolution(x, static, dynamic, previous, metadata)
    commit_history(history, previous, x, metadata)
    offset = 0
    for start, length, row in zip(starts, lengths, rows, strict=True):
        if not length:
            continue
        # A request at position p < W - 1 has zeros before its first token.
        previous = original[row].clone()
        previous[:max(width - 1 - start, 0)] = 0
        chunk = x[offset:offset + length]
        expected, after = ref.short_convolution(chunk, weight, previous)
        assert torch.equal(output[offset:offset + length], expected)
        expected, after_dynamic = ref.dynamic_short_convolution(chunk, static, dynamic[offset:offset + length],
                                                                previous)
        assert torch.equal(lagged[offset:offset + length], expected)
        assert torch.equal(history[row], after) and torch.equal(after, after_dynamic)
        offset += length
    assert not output[offset:].count_nonzero() and not lagged[offset:].count_nonzero()
    assert torch.equal(history[0], original[0])


def test_residual_mixture_matches_the_canonical_mixture_in_fp32():
    generator = torch.Generator().manual_seed(3)
    hidden = 64
    sources = [(torch.randn(5, hidden, generator=generator) * scale).to(BF16) for scale in (1.0, 8.0, 4.0)]
    queries = 0.5 * torch.randn(3, hidden, generator=generator)
    addend = torch.randn(5, hidden, generator=generator).to(BF16)
    expected = ref.residual_mix(sources, queries, addend=addend)
    actual = residual_mix(sources, queries.float() * hidden ** -0.5, addend=addend)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=2e-6, atol=2e-6)
