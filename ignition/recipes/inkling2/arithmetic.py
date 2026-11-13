# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Serving arithmetic of the Inkling 2 decoder outside its prepared kernels.

These functions run inside compiled forwards over packed request rows. They
follow the operation order and rounding points of the canonical arithmetic in
`ignition.reference.inkling2`, with FP32 accumulation in place of its FP64
accumulation:

- RMS normalization in FP32 with one BF16 rounding.
- Attention-residual mixing: per token, softmax scores of each FP32 query
  (prescaled by `hidden ** -0.5`) against the gainless RMS-normalized sources,
  and the FP32 mixture of the normalized sources.
- L2 clipping `y * min(1, threshold / (|y| + 1e-6))` after an optional
  per-head scale, with the global-layer length scale applied after rounding.
- Causal depthwise short convolutions with a residual term over packed
  requests. Each row reads its request's earlier rows in the batch and, before
  the batch, the request's `W - 1` most recent raw inputs from its history row.
"""

from __future__ import annotations

import torch

BF16 = torch.bfloat16


def rms_norm(x, weight=None, eps=1e-6, dtype=BF16):
    values = x.float()
    y = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(dtype)


def residual_mix(sources, queries, *, addend=None, eps=1e-6):
    """Mix BF16 `[tokens, hidden]` sources per token by each prescaled FP32 query row.

    `queries` is `[streams, hidden]`, already multiplied by `hidden ** -0.5`;
    `addend` joins the last source in FP32. Returns one FP32 mixture per query row.
    """
    values = [source.float() for source in sources]
    if addend is not None:
        values[-1] = values[-1] + addend.float()
    hidden = values[0].shape[-1]
    inverse = [torch.rsqrt(value.square().sum(-1) * (1.0 / hidden) + eps) for value in values]
    mixtures = []
    for query in queries.float().reshape(-1, hidden):
        scores = torch.stack([(value @ query) * scale for value, scale in zip(values, inverse, strict=True)])
        probabilities = torch.softmax(scores, dim=0)
        mixed = None
        for index, value in enumerate(values):
            term = value * (probabilities[index] * inverse[index])[:, None]
            mixed = term if mixed is None else mixed + term
        mixtures.append(mixed)
    return tuple(mixtures)


def log_scaling(positions, floor, alpha):
    """Per-token logit scale of global layers, one through position `floor - 1`."""
    return 1.0 + alpha * torch.log(torch.clamp((positions.clamp_min(0) + 1).float() / floor, min=1.0))


def l2_clip(x, threshold, *, scale=None, tau=None, eps=1e-6):
    """Clip `[..., width]` rows to L2 norm `threshold` after an optional scale; `tau` follows the BF16 rounding."""
    values = x.float()
    if scale is not None:
        values = values * scale.float()
    norm = values.square().sum(-1, keepdim=True).sqrt()
    y = values * torch.clamp(threshold / (norm + eps), max=1.0)
    if tau is not None:
        y = y.to(x.dtype).float() * tau.reshape(-1, *([1] * (x.ndim - 1)))
    return y.to(x.dtype)


def _lagged(x, history, metadata, lag):
    """Each row's input `lag` positions earlier in its request, zero before the sequence start."""
    m = metadata
    tokens = x.shape[0]
    rows = torch.arange(tokens, device=x.device)
    offset = rows - m.query_start_loc[m.requests].long()
    earlier = offset - lag
    width = history.shape[1]
    current = x[(rows - lag).clamp_min(0)]
    previous = history[m.state_rows[m.requests], (width + earlier).clamp(0, width - 1)].to(x.dtype)
    value = torch.where((earlier >= 0)[:, None], current, previous)
    return torch.where(((m.positions - lag >= 0) & (m.positions >= 0))[:, None], value, 0)


def short_convolution(x, weight, history, metadata):
    """`(sum_j w[c, j] x[t - W + 1 + j, c] + x[t, c]) / sqrt(W + 1)` with `weight` `[channels, 1, W]`.

    `history` is `[state rows, W - 1, channels]`, oldest first. Rows without a
    position (padding) produce zero.
    """
    taps = weight.reshape(weight.shape[0], -1).float()
    width = taps.shape[1]
    total = None
    for tap in range(width):
        term = _lagged(x, history, metadata, width - 1 - tap).float() * taps[:, tap]
        total = term if total is None else total + term
    output = (total + x.float()) * (width + 1) ** -0.5
    return torch.where((metadata.positions >= 0)[:, None], output, 0).to(x.dtype)


def dynamic_short_convolution(x, static, dynamic, history, metadata):
    """Global layers' K/V convolution with per-token lag weights.

    `static` is `[W, channels]` with lag 0 first; `dynamic` is BF16 `[tokens, heads, W]`.
    The output is `(sum_lag (static[lag] + dynamic[t, head, lag]) x[t - lag] + x[t]) / sqrt(2W + 1)`.
    """
    width, channels = static.shape
    heads = dynamic.shape[1]
    lags = dynamic.float().repeat_interleave(channels // heads, dim=1)
    total = None
    for lag in range(width):
        term = _lagged(x, history, metadata, lag).float() * (static[lag].float() + lags[:, :, lag])
        total = term if total is None else total + term
    output = (total + x.float()) * (2 * width + 1) ** -0.5
    return torch.where((metadata.positions >= 0)[:, None], output, 0).to(x.dtype)


def commit_history(history, x, metadata):
    """Store each live request's `W - 1` most recent raw inputs, oldest first, in its history row."""
    m = metadata
    width = history.shape[1]
    offsets = torch.arange(width, device=x.device)
    end = (m.starts + m.lengths)[:, None]
    positions = end - width + offsets
    starts = m.starts[:, None]
    in_batch = positions >= starts
    rows = (m.query_start_loc[:-1, None].long() + positions - starts).clamp(0, x.shape[0] - 1)
    old = (positions - starts + width).clamp(0, width - 1)
    state = m.state_rows
    previous = history[state[:, None], old]
    updated = torch.where(in_batch[:, :, None], x[rows].to(history.dtype), previous)
    updated = torch.where((positions >= 0)[:, :, None], updated, 0)
    live = (m.lengths > 0)[:, None, None]
    history[state] = torch.where(live, updated, history[state])
