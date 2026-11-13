# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Elementwise error bounds between prepared Inkling 2 attention and the canonical arithmetic.

The canonical attention (`ignition.reference.inkling2`) accumulates in FP64,
keeps BF16 projections, keys and values and FP32 probabilities. The prepared
path accumulates projections in FP32, stores keys and values as E4M3, and in
prefill also rounds queries and probabilities to E4M3. The bounds below
propagate each rounding as an absolute error through the reference's own
intermediates, all evaluated in FP64:

- A BF16 value lies within `2^-8 |x|` of the exact result it rounds, so two
  roundings of nearby values differ by at most `2^-7 |x|` plus the difference
  of the values rounded. A K-term FP32 dot product differs from the exact sum
  by at most `K 2^-24 sum |x w|` in any order.
- E4M3 rounding moves a value by at most `2^-4 |x|`, or `2^-10` below the
  normal range.
- A query-key logit with element errors `E_q`, `E_k` moves by at most
  `(E_q . |k| + |q| . E_k + E_q . E_k) / sqrt(d)`; the relative bias by the
  propagated relative-query error plus one BF16 step.
- Given per-logit bounds `D_j` for a row with probabilities `p_j` (the sink is
  one more logit with a zero value), each perturbed probability lies within
  `p_s / (p_s + e^{+-D_s} (sum_j p_j e^{+-D_j} - p_s e^{+-D_s}))`, the extremes of
  the softmax over the admissible logits.
- The kernels' own probability and output roundings follow the bounds of
  their FlashInfer tests: E4M3 probabilities move by `2^-4` of the probability
  mass `sum p |v|` plus `keys 2^-18 max|v| p_max`, BF16 probabilities by
  `2^-8` of the mass, with `2^-8 |out|` for the output rounding and `2^-11` of
  the mass for FP32 arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ignition.reference import inkling2 as ref

U_BF16 = 2.0 ** -8
U_E4M3 = 2.0 ** -4
E4M3_FLOOR = 2.0 ** -10
FP32 = 2.0 ** -24


def _bf16_pair(value):
    """Two BF16 roundings of results within `value` of each other: `2^-7 |value|`."""
    return 2 * U_BF16 * value.abs()


def projection_error(x, weight, y):
    """Bound of |FP32-accumulated BF16 projection - canonical BF16 projection| for `y = x . weight^T`."""
    magnitude = x.double().abs() @ weight.double().abs().t()
    accumulated = x.shape[-1] * FP32 * magnitude
    return _bf16_pair(y.double()) + 2 * accumulated


def scaled_norm_error(x, error, factor, output, *, extra=None):
    """Bound of `bf16(x * c(x))` for a row factor `c` (clip or RMS normalization) under input error `error`.

    `factor` is the reference's `c` per row. The relative change of a row norm is
    at most `|error| / |x|`, which bounds the relative change of `c`.
    """
    norm = x.double().norm(dim=-1, keepdim=True)
    moved = error.norm(dim=-1, keepdim=True)
    relative = moved / (norm - moved).clamp_min(1e-30)
    bound = factor * error + x.double().abs() * factor * relative + _bf16_pair(output.double())
    return bound if extra is None else bound + extra


@dataclass(frozen=True)
class AttentionParts:
    """Canonical intermediates of one rank's attention and the propagated bounds of the prepared ones."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    relative: torch.Tensor
    bias: torch.Tensor
    sink: torch.Tensor
    gate: torch.Tensor
    gated: torch.Tensor
    partial: torch.Tensor
    errors: dict


def reference_parts(weights: ref.LayerWeights, config, streams, positions):
    """The canonical attention of `weights` over a whole sequence, with intermediate error bounds."""
    dim, tokens = config.head_dim, positions.shape[0]
    x_q, x_k, x_v = streams if len(streams) == 3 else streams * 3
    heads, kv_heads = weights.query.shape[0] // dim, weights.key.shape[0] // dim
    errors = {}

    def project(name, source, weight):
        value = ref.matmul(source, weight).to(torch.bfloat16)
        errors[name] = projection_error(source, weight, value)
        return value

    query = project("query", x_q, weights.query)
    relative = project("relative", x_q, weights.relative_query)
    gate = project("gate", x_q, weights.output_gate)
    sink = project("sink", x_q, weights.sink)
    key = project("key_raw", x_k, weights.key)
    value = project("value_raw", x_v, weights.value)
    history = torch.zeros(config.convolution_width - 1, key.shape[1], dtype=torch.bfloat16, device=key.device)

    def lagged(values, lag):
        return torch.cat((torch.zeros_like(values[:lag]), values))[:tokens] if lag else values

    if weights.local:
        key_c, _ = ref.short_convolution(key, weights.key_convolution, history)
        value_c, _ = ref.short_convolution(value, weights.value_convolution, history)
        width = weights.key_convolution.shape[-1]
        scale = (width + 1) ** -0.5
        for name, raw, taps, out in (("key", key, weights.key_convolution, key_c),
                                     ("value", value, weights.value_convolution, value_c)):
            taps = taps.reshape(taps.shape[0], width).double().abs()
            error, magnitude = errors[name + "_raw"].clone(), raw.double().abs()
            total_error, total_magnitude = error.clone(), magnitude.clone()
            for tap in range(width):
                lag = width - 1 - tap
                total_error += lagged(errors[name + "_raw"], lag) * taps[:, tap]
                total_magnitude += lagged(magnitude, lag) * taps[:, tap]
            errors[name + "_convolved"] = (scale * total_error + _bf16_pair(out.double())
                                          + 8 * FP32 * scale * total_magnitude)
        tau = None
    else:
        width = weights.key_convolution_static.shape[0]
        lags = {}
        for name, source, projection in (("key", x_k, weights.key_convolution_dynamic),
                                         ("value", x_v, weights.value_convolution_dynamic)):
            lags[name] = project(name + "_lags", source, projection)
        key_c, _ = ref.dynamic_short_convolution(key, weights.key_convolution_static,
                                                 lags["key"].reshape(tokens, kv_heads, width), history)
        value_c, _ = ref.dynamic_short_convolution(value, weights.value_convolution_static,
                                                   lags["value"].reshape(tokens, kv_heads, width), history)
        scale = (2 * width + 1) ** -0.5
        for name, raw, static, out in (("key", key, weights.key_convolution_static, key_c),
                                       ("value", value, weights.value_convolution_static, value_c)):
            dynamic = lags[name].double().reshape(tokens, kv_heads, width).repeat_interleave(dim, 1)
            dynamic_error = errors[name + "_lags"].reshape(tokens, kv_heads, width).repeat_interleave(dim, 1)
            error, magnitude = errors[name + "_raw"].clone(), raw.double().abs()
            total_error, total_magnitude = error.clone(), magnitude.clone()
            for lag in range(width):
                coefficient = (static[lag].double() + dynamic[:, :, lag]).abs()
                shifted = lagged(raw.double().abs(), lag)
                total_error += lagged(errors[name + "_raw"], lag) * coefficient + shifted * dynamic_error[:, :, lag]
                total_magnitude += shifted * coefficient
            errors[name + "_convolved"] = (scale * total_error + _bf16_pair(out.double())
                                          + 8 * FP32 * scale * total_magnitude)
        tau = ref.log_scaling(positions, config.log_scaling_floor, config.log_scaling_alpha)

    def clip(name, values, error, threshold, scale_=None, tau_=None):
        rows = values.float() if scale_ is None else values.float() * scale_.float()
        rows_error = error if scale_ is None else error * scale_.double().abs()
        norm = rows.double().norm(dim=-1, keepdim=True)
        factor = torch.clamp(threshold / (norm + 1e-6), max=1.0)
        output = ref.l2_clip(values, threshold, scale=scale_, tau=tau_)
        extra = None if tau_ is None else _bf16_pair(output.double())
        bound = scaled_norm_error(rows, rows_error, factor, output, extra=extra)
        if tau_ is not None:
            bound = bound * tau_.double().reshape(-1, *([1] * (values.ndim - 1)))
        errors[name] = bound
        return output

    q = clip("query", query.reshape(tokens, heads, dim), errors["query"].reshape(tokens, heads, dim),
             config.qk_clip, weights.query_clip_scale, tau)
    k = clip("key", key_c.reshape(tokens, kv_heads, dim), errors["key_convolved"].reshape(tokens, kv_heads, dim),
             config.qk_clip)
    v_rows = value_c.reshape(tokens, kv_heads, dim)
    v = ref.rms_norm(v_rows, None, config.eps)
    v_factor = torch.rsqrt(v_rows.double().square().mean(-1, keepdim=True) + config.eps)
    errors["value"] = scaled_norm_error(v_rows, errors["value_convolved"].reshape(tokens, kv_heads, dim), v_factor, v)
    r = clip("relative", relative.reshape(tokens, heads, config.relative_width),
             errors["relative"].reshape(tokens, heads, config.relative_width), config.relative_clip,
             weights.relative_clip_scale, tau)
    bias = ref.relative_bias(r, weights.relative_projection)
    errors["bias"] = (torch.einsum("thd,de->the", errors["relative"], weights.relative_projection.double().abs())
                      + _bf16_pair(bias.double()) + 16 * FP32 * torch.einsum(
                          "thd,de->the", r.double().abs(), weights.relative_projection.double().abs()))
    output, lse = ref.attention(q, k, v, bias, positions, positions,
                                window=config.window - 1 if weights.local else None)
    gated = ref.gate_sink(output, lse, gate, sink)
    partial = ref.matmul(gated, weights.output).to(torch.bfloat16)
    return AttentionParts(q, k, v, r, bias, sink, gate, gated, partial, errors)


def attention_output_bound(parts: AttentionParts, config, rows, *, local, fp8_query, e4m3_probabilities):
    """Elementwise bound of |prepared gated attention - canonical gated attention| at query `rows`.

    Returns the bound on the gated output `[rows, heads * dim]`.
    """
    q, k, v = parts.query[rows].double(), parts.key.double(), parts.value.double()
    tokens, heads, dim = parts.query.shape
    group = heads // k.shape[1]
    e = parts.errors
    eq = e["query"][rows]
    if fp8_query:
        eq = eq + U_E4M3 * (q.abs() + eq) + E4M3_FLOOR
    ek = e["key"] + U_E4M3 * (k.abs() + e["key"]) + E4M3_FLOOR
    ev = e["value"] + U_E4M3 * (v.abs() + e["value"]) + E4M3_FLOOR
    k, ek, ev, v = (t.repeat_interleave(group, dim=1) for t in (k, ek, ev, v))
    scale = dim ** -0.5
    logit_error = (torch.einsum("thd,shd->ths", eq, k.abs()) + torch.einsum("thd,shd->ths", q.abs(), ek)
                   + torch.einsum("thd,shd->ths", eq, ek)) * scale
    positions = torch.arange(tokens, device=q.device)
    distance = rows[:, None] - positions[None, :]
    extent = parts.bias.shape[-1]
    inside = (distance >= 0) & (distance < extent)
    index = distance.clamp(0, extent - 1)[:, None, :].expand(-1, heads, -1)
    logit_error = logit_error + torch.where(inside[:, None, :], e["bias"][rows].gather(-1, index), 0.0)
    scores = torch.einsum("thd,shd->ths", q, k) * scale + torch.where(
        inside[:, None, :], parts.bias[rows].double().gather(-1, index), 0.0)
    visible = distance >= 0
    if local:
        visible = visible & (distance <= config.window - 1)
    scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
    logit_error = logit_error.masked_fill(~visible[:, None, :], 0.0)
    sink = parts.sink[rows].double()
    scores = torch.cat((scores, sink[:, :, None]), -1)
    logit_error = torch.cat((logit_error, e["sink"][rows][:, :, None]), -1)
    p = torch.softmax(scores, -1)
    upper_sum = (p * logit_error.exp()).sum(-1, keepdim=True)
    lower_sum = (p * (-logit_error).exp()).sum(-1, keepdim=True)
    # A probability is largest when its logit rises and every other falls, and smallest in reverse.
    high = p / (p + (-logit_error).exp() * (lower_sum - p * (-logit_error).exp()).clamp_min(0)).clamp_min(1e-300)
    low = p / (p + logit_error.exp() * (upper_sum - p * logit_error.exp()).clamp_min(0)).clamp_min(1e-300)
    high, low = (torch.where(p > 0, value, 0.0) for value in (high, low))
    moved = torch.maximum(high - p, p - low)[..., :-1]
    p_high = torch.maximum(high, p)[..., :-1]
    v_rows = v
    output = torch.einsum("ths,shd->thd", p[..., :-1], v_rows)
    mass = torch.einsum("ths,shd->thd", p_high, v_rows.abs() + ev)
    bound = torch.einsum("ths,shd->thd", moved, v_rows.abs()) + torch.einsum("ths,shd->thd", p_high, ev)
    if e4m3_probabilities:
        keys = visible.sum(-1).double()[:, None, None]
        bound = bound + U_E4M3 * mass + keys * 2.0 ** -18 * (v_rows.abs() + ev).amax() * p_high.amax(-1, keepdim=True)
    else:
        bound = bound + U_BF16 * mass
    bound = bound + 2.0 ** -11 * mass + U_BF16 * (output.abs() + bound)
    gate = parts.gate[rows].double().reshape(-1, heads, dim)
    sigmoid = torch.sigmoid(gate)
    gate_error = e["gate"][rows].reshape(-1, heads, dim)
    gated = (output * sigmoid).abs()
    gated_bound = (bound * (sigmoid + 0.25 * gate_error) + output.abs() * 0.25 * gate_error
                   + _bf16_pair(gated) + 2.0 ** -20 * output.abs())
    return gated_bound.flatten(1)


def partial_bound(gated_bound, gated, weight, partial):
    """Bound of the output projection given its input bound: propagated error plus projection rounding."""
    propagated = gated_bound @ weight.double().abs().t()
    return propagated + projection_error(gated, weight, partial) + U_BF16 * propagated
