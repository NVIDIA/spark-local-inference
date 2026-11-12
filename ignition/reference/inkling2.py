# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical PyTorch arithmetic of the Inkling 2 text decoder, whole and at four ranks.

The operation order, rounding points and constants follow the public SGLang
implementation of `Inkling2ForConditionalGeneration` (PR 50186, commit 237ba6b):

- Attention residuals replace the additive stream. Each sublayer input is a
  per-token softmax mixture of block summaries; values enter the mixture RMS
  normalized without a gain, and the scores use FP32 queries scaled by
  `hidden ** -0.5`. Global layers mix three streams for their Q, K and V inputs.
- Q, relative-query and K rows are L2 clipped (`y * min(1, threshold / (|y| + 1e-6))`),
  Q and the relative query after a learned per-head scale; on global layers the
  clipped rows, rounded to BF16, are multiplied by `1 + alpha * ln(max((pos + 1) / floor, 1))`.
  Values are RMS normalized per head without a gain.
- K and V pass through causal depthwise short convolutions with a residual
  term, scaled by `(width + 1) ** -0.5` with static taps and by
  `(2 * width + 1) ** -0.5` on global layers, whose taps add per-token lag weights
  projected from the normalized stream.
- The relative bias `r . proj` covers distances `0 <= t - s < extent` and is zero
  beyond; scores are `q . k / sqrt(head_dim) + bias`. The output is multiplied
  by `sigmoid(gate)` and by `sigmoid(lse - sink)`, a per-token sink logit.
- The attention and expert outputs pass through static short convolutions over
  the hidden width before they join the residual state.
- The router selects the top eight of `sigmoid(logits) + bias`; weights
  renormalize the eight sigmoids and scale by `route_scale`. The routing weight
  multiplies the capped expert activation `4 tanh(g / 4) sigmoid(g) 25 tanh(u / 25)`
  before the down projection's activation quantization; routes sum with unit weights.
- NVFP4 experts quantize activations per row with the four-over-six rule (outer
  scale `amax / 1536`, each 16-value block choosing a `max / 6` or `max / 4`
  E4M3 scale by L1 reconstruction error); MXFP8 experts use per-32 UE8M0 scales.

Projections accumulate in FP64 and round to FP32, so a projection's rows do not
depend on which other rows are computed with them; tensor-core GEMMs differ in
FP32 accumulation order. Exact `torch.tanh`, `torch.sigmoid` and `torch.exp`
replace the PTX approximations SGLang uses in the expert activation and the gate.
Attention keeps FP32 probabilities, where flash attention rounds them to BF16
before the value product.

The four-rank functions shard heads and expert intermediates as the recipe's
weight layout places them. Each rank convolves its own partial output sums with
its own partial-sum history before the cross-rank reduction, and the NVFP4 down
projection quantizes each rank's activation slice with the maximum of the ranks'
row maxima, so only the BF16 rounding of partial sums separates them from the
single-rank functions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import torch

from ignition.reference.qwen38_quant import decode_mxfp8

BF16 = torch.bfloat16
_FLOAT_MAX = torch.finfo(torch.float32).max
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
# Midpoints between adjacent E2M1 magnitudes. Ties round to the even code: down at
# 0.25, 1.25, 2.5 and 5, up at 0.75, 1.75 and 3.5.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
NVFP4_OUTER = 1536.0


def matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`x @ weight.T` accumulated in FP64 and rounded once to FP32."""
    return (x.double() @ weight.double().transpose(-1, -2)).float()


def _fma(a, b, c):
    """`a * b + c` with the single FP32 rounding of a fused multiply-add."""
    return (a.double() * b.double() + c.double()).float()


def rms_norm(x, weight=None, eps=1e-6, dtype=BF16):
    """RMSNorm in FP32, `x * rsqrt(mean(x^2) + eps) * weight`, rounded to `dtype`."""
    values = x.float()
    y = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(dtype)


def residual_mix(sources: Sequence[torch.Tensor], queries: torch.Tensor, *, addend=None, eps=1e-6):
    """Mix `sources` per token by each FP32 query's softmax scores over RMS-normalized values.

    `addend` joins the last source in FP32. Returns one FP32 `[tokens, hidden]` tensor per query row.
    """
    values = [source.float() for source in sources]
    if addend is not None:
        values[-1] = values[-1] + addend.float()
    stacked = torch.stack(values)
    hidden = stacked.shape[-1]
    scaled = queries.float().reshape(-1, hidden) * hidden ** -0.5
    inverse = torch.rsqrt(stacked.square().sum(-1) * (1.0 / hidden) + eps)
    scores = torch.einsum("ntd,qd->qnt", stacked.double(), scaled.double()).float() * inverse
    probabilities = torch.softmax(scores, dim=1) * inverse
    mixed = torch.einsum("qnt,ntd->qtd", probabilities.double(), stacked.double()).float()
    return tuple(mixed.unbind(0))


def log_scaling(positions, floor, alpha):
    """Per-token logit scale of global layers; one until position `floor - 1`."""
    return 1.0 + alpha * torch.log(torch.clamp((positions + 1).float() / floor, min=1.0))


def l2_clip(x, threshold, *, scale=None, tau=None, eps=1e-6):
    """Clip each `[..., width]` row to an L2 norm of `threshold`, after an optional per-head scale.

    `tau`, one FP32 value per token, multiplies the clipped rows after rounding to `x.dtype`.
    """
    values = x.float()
    if scale is not None:
        values = values * scale.float()
    norm = values.square().sum(-1, keepdim=True).sqrt()
    y = values * torch.clamp(threshold / (norm + eps), max=1.0)
    if tau is not None:
        y = y.to(x.dtype).float() * tau.reshape(-1, *([1] * (x.ndim - 1)))
    return y.to(x.dtype)


def short_convolution(x, weight, history=None):
    """Causal depthwise convolution with a residual term.

    The output is `(sum_j w[c, j] x[t - W + 1 + j, c] + x[t, c]) / sqrt(W + 1)`, where
    `weight` is `[channels, 1, W]` with the current token's tap last. `history` holds the
    `W - 1` raw inputs before `x`, oldest first (zeros at a sequence start). Returns the
    output in `x.dtype` and the history after `x`.
    """
    tokens, channels = x.shape
    width = weight.shape[-1]
    history = x.new_zeros(width - 1, channels) if history is None else history.to(x.dtype)
    padded = torch.cat((history, x)).float()
    taps = weight.reshape(channels, width).to(x.dtype).float()
    total = torch.zeros(tokens, channels, dtype=torch.float32, device=x.device)
    for tap in range(width):
        total = _fma(padded[tap:tap + tokens], taps[:, tap], total)
    output = (total + x.float()) * (width + 1) ** -0.5
    return output.to(x.dtype), padded[tokens:].to(x.dtype)


def dynamic_short_convolution(x, static, dynamic, history=None):
    """Global layers' K/V convolution with per-token, per-head lag weights.

    `static` is `[W, channels]` with lag 0 (the current token) first; `dynamic` is BF16
    `[tokens, heads, W]`. The output is
    `(sum_lag (static[lag] + dynamic[t, head, lag]) x[t - lag] + x[t]) / sqrt(2W + 1)`.
    """
    tokens, channels = x.shape
    width, heads = static.shape[0], dynamic.shape[1]
    history = x.new_zeros(width - 1, channels) if history is None else history.to(x.dtype)
    padded = torch.cat((history, x)).float()
    lags = dynamic.float().repeat_interleave(channels // heads, dim=1)
    total = None
    for lag in range(width):
        value = padded[width - 1 - lag:width - 1 - lag + tokens]
        weight = static[lag].float() + lags[:, :, lag]
        total = value * weight if total is None else _fma(value, weight, total)
    output = (total + x.float()) * (2 * width + 1) ** -0.5
    return output.to(x.dtype), padded[tokens:].to(x.dtype)


def relative_bias(relative, projection):
    """BF16 relative logits `[tokens, heads, extent]` from clipped `[tokens, heads, width]` queries."""
    return torch.einsum("thd,de->the", relative.double(), projection.double()).float().to(BF16)


def attention(query, key, value, bias, query_positions, key_positions, *, window=None):
    """FP32 causal attention with an additive relative bias and an optional left window.

    `query` is `[tokens, heads, dim]`, `key` and `value` are `[keys, kv_heads, dim]`, and
    `bias[t, h, d]` applies at distance `d = query_position - key_position` for `d < extent`.
    Keys farther than `window` are masked. Returns the FP32 output and log-sum-exp.
    """
    tokens, heads, dim = query.shape
    group = heads // key.shape[1]
    keys = key.double().repeat_interleave(group, dim=1)
    values = value.double().repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->ths", query.double(), keys).float() * dim ** -0.5
    distance = query_positions[:, None] - key_positions[None, :]
    extent = bias.shape[-1]
    inside = (distance >= 0) & (distance < extent)
    index = distance.clamp(0, extent - 1)[:, None, :].expand(tokens, heads, -1)
    scores = scores + torch.where(inside[:, None, :], bias.float().gather(-1, index), 0.0)
    masked = distance < 0
    if window is not None:
        masked = masked | (distance > window)
    scores = scores.masked_fill(masked[:, None, :], float("-inf"))
    output = torch.einsum("ths,shd->thd", torch.softmax(scores, -1).double(), values).float()
    return output, torch.logsumexp(scores, -1)


def gate_sink(output, lse, gate, sink, dtype=BF16):
    """`output * sigmoid(gate) * sigmoid(lse - sink)` per head, flattened and rounded to `dtype`."""
    scale = torch.sigmoid(lse.float() - sink.float())
    gated = output.float() * torch.sigmoid(gate.float().reshape(output.shape))
    return (gated * scale.unsqueeze(-1)).flatten(1).to(dtype)


def route(x, weight, bias, *, top_k, route_scale):
    """Expert ids ordered by descending `sigmoid(logits) + bias` (ties to the lower id) and FP32 weights."""
    logits = matmul(x, weight)
    selection = 1.0 / (1.0 + torch.exp(-logits)) + bias.float()
    ids = torch.sort(selection, dim=-1, descending=True, stable=True).indices[:, :top_k]
    chosen = torch.nn.functional.logsigmoid(logits.gather(-1, ids))
    weights = torch.exp(chosen - torch.logsumexp(chosen, dim=-1, keepdim=True)) * route_scale
    return ids, weights


def situ(gate_up, gamma, gate_cap, up_cap):
    """Capped activation of interleaved gate/up rows, multiplied by each row's routing weight."""
    gate, up = gate_up[:, 0::2].float(), gate_up[:, 1::2].float()
    sigmoid = 1.0 / (1.0 + torch.exp(-gate))
    capped_gate = gate_cap * torch.tanh(gate / gate_cap)
    capped_up = up_cap * torch.tanh(up / up_cap)
    return (((capped_gate * sigmoid) * capped_up) * gamma.float()[:, None]).to(BF16)


@dataclass(frozen=True)
class Nvfp4Rows:
    """Row-quantized NVFP4 activations: E2M1 values, E4M3 block scales and each row's amax."""

    values: torch.Tensor
    scales: torch.Tensor
    amax: torch.Tensor

    def blocks(self):
        """Exact `e2m1 * block_scale` products in FP32, excluding the outer scale `amax / 1536`."""
        rows, width = self.values.shape
        return (self.values.reshape(rows, -1, 16) * self.scales.float()[..., None]).reshape(rows, width)


def round_e2m1(x):
    """Round to the nearest E2M1 value, ties to even, saturating at 6 like `cvt.rn.satfinite.e2m1x2.f32`."""
    magnitude = x.abs()
    code = torch.zeros_like(magnitude, dtype=torch.long)
    for index, midpoint in enumerate(_E2M1_MIDPOINTS):
        code += (magnitude >= midpoint if index % 2 else magnitude > midpoint).long()
    value = torch.tensor(_E2M1[:8], dtype=torch.float32, device=x.device)[code]
    return torch.where(x < 0, -value, value)


def _pairwise(error):
    while error.shape[-1] > 1:
        error = error[..., 0::2] + error[..., 1::2]
    return error[..., 0]


def quantize_nvfp4(x, amax=None):
    """Four-over-six NVFP4 quantization of `[rows, width]` rows.

    `amax` defaults to each row's maximum magnitude; a supplied amax may cover more
    elements than `x` holds. Each block scale is the E4M3 rounding of
    `block_max * (1536 / amax) / 6` or `/ 4`, whichever reconstructs the block with
    the smaller L1 error (ties keep `/ 6`).
    """
    values = x.float()
    rows, width = values.shape
    if width % 16:
        raise ValueError("NVFP4 rows require a width divisible by 16")
    amax = values.abs().amax(-1) if amax is None else amax.float()
    encoder = torch.clamp(1536.0 / amax, max=_FLOAT_MAX)[:, None]
    decoder = 1.0 / encoder
    outer = (amax * (1.0 / 1536.0))[:, None, None]
    blocks = values.reshape(rows, -1, 16)
    peak = blocks.abs().amax(-1)
    candidates = []
    for factor in (1.0 / 6.0, 0.25):
        scale = torch.clamp(peak * (encoder * factor), max=448.0).to(torch.float8_e4m3fn)
        inverse = torch.clamp(1.0 / (scale.float() * decoder), max=_FLOAT_MAX)
        rounded = round_e2m1(blocks * inverse[..., None])
        error = _pairwise((rounded * scale.float()[..., None] * outer - blocks).abs())
        candidates.append((rounded, scale, error))
    (six, six_scale, six_error), (four, four_scale, four_error) = candidates
    choose = four_error < six_error
    return Nvfp4Rows(torch.where(choose[..., None], four, six).reshape(rows, width),
                     torch.where(choose, four_scale.float(), six_scale.float()).to(torch.float8_e4m3fn), amax)


@dataclass(frozen=True)
class Mxfp8Rows:
    values: torch.Tensor
    exponents: torch.Tensor

    def dequantized(self):
        rows, width = self.values.shape
        scale = torch.ldexp(torch.ones_like(self.exponents, dtype=torch.float32), self.exponents.int() - 127)
        return (self.values.float().reshape(rows, -1, 32) * scale[..., None]).reshape(rows, width)


def quantize_mxfp8(x):
    """MXFP8 rows: per 32 values a UE8M0 power of two rounded up from `max / 448`, values saturated to E4M3."""
    values = x.float()
    rows, width = values.shape
    if width % 32:
        raise ValueError("MXFP8 rows require a width divisible by 32")
    groups = values.reshape(rows, -1, 32)
    raw = groups.abs().amax(-1) * (1.0 / 448.0)
    exponent = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2.0 ** -127, 0, torch.clamp(exponent, max=254))
    exponent = torch.where(raw != raw, 255, exponent)
    scale = torch.clamp((exponent << 23).view(torch.float32), min=2.0 ** -127)
    inverse = torch.where(exponent == 0, 2.0 ** 127, torch.where(exponent == 255, 0.0, 1.0 / scale))
    quantized = torch.clamp(groups * inverse[..., None], -448.0, 448.0).to(torch.float8_e4m3fn)
    return Mxfp8Rows(quantized.reshape(rows, width), exponent.to(torch.uint8))


def decode_nvfp4_blocks(packed, scales):
    """Exact `e2m1 * block_scale` FP32 values of row-major NVFP4 weights (low nibble first)."""
    table = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    rows = packed.shape[0]
    return (table[codes].reshape(rows, -1, 16) * scales.float()[..., None]).reshape(rows, -1)


def nvfp4_projection(x: Nvfp4Rows, packed, scales, amax, denominator):
    """Block-scaled NVFP4 GEMM with the reference epilogue `acc * ((x_amax * w_amax) / (1536 * denominator))`."""
    accumulator = matmul(x.blocks(), decode_nvfp4_blocks(packed, scales))
    gain = (x.amax * amax.float()) * (1.0 / (NVFP4_OUTER * denominator.float()))
    return (accumulator * gain[:, None]).to(BF16)


def mxfp8_projection(x: Mxfp8Rows, values, scales):
    return matmul(x.dequantized(), decode_mxfp8(values, scales)).to(BF16)


@dataclass(frozen=True, kw_only=True)
class LayerWeights:
    """One decoder layer's checkpoint tensors, whole or one rank's placed extent.

    Field names are the roles of `ignition.recipes.inkling2.weights.checkpoint_layout`.
    Local layers carry `attention_norm` and static K/V convolutions; global layers
    carry the three stream norms and the static and dynamic K/V convolution tensors.
    """

    query: torch.Tensor
    output_gate: torch.Tensor
    relative_query: torch.Tensor
    sink: torch.Tensor
    query_clip_scale: torch.Tensor
    relative_clip_scale: torch.Tensor
    relative_projection: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    mlp_query: torch.Tensor
    mlp_norm: torch.Tensor
    attention_convolution: torch.Tensor
    mlp_convolution: torch.Tensor
    router: torch.Tensor
    router_bias: torch.Tensor
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    attention_query: torch.Tensor | None = None
    attention_norm: torch.Tensor | None = None
    query_norm: torch.Tensor | None = None
    key_norm: torch.Tensor | None = None
    value_norm: torch.Tensor | None = None
    key_convolution: torch.Tensor | None = None
    value_convolution: torch.Tensor | None = None
    key_convolution_static: torch.Tensor | None = None
    key_convolution_dynamic: torch.Tensor | None = None
    value_convolution_static: torch.Tensor | None = None
    value_convolution_dynamic: torch.Tensor | None = None
    w13_amax: torch.Tensor | None = None
    w13_denominator: torch.Tensor | None = None
    w2_amax: torch.Tensor | None = None
    w2_denominator: torch.Tensor | None = None

    @property
    def local(self) -> bool:
        return self.attention_norm is not None

    @property
    def expert_format(self) -> str:
        return "nvfp4" if self.w13.dtype == torch.uint8 else "mxfp8"


@dataclass(frozen=True)
class LayerCache:
    """One sequence's attention history and short-convolution inputs for one layer (or one rank of it).

    `key` and `value` hold the cached post-convolution, clipped and normalized rows at
    `positions`. The convolution histories hold the last `W - 1` raw inputs; the
    attention and MLP output histories hold a rank's own partial sums.
    """

    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor
    key_history: torch.Tensor
    value_history: torch.Tensor
    attention_history: torch.Tensor
    mlp_history: torch.Tensor

    @classmethod
    def empty(cls, weights: LayerWeights, head_dim, device=None):
        device = weights.key.device if device is None else device
        kv_channels, hidden = weights.key.shape[0], weights.key.shape[1]
        width = weights.attention_convolution.shape[-1]
        zeros = lambda *shape: torch.zeros(*shape, dtype=BF16, device=device)  # noqa: E731
        return cls(zeros(0, kv_channels // head_dim, head_dim), zeros(0, kv_channels // head_dim, head_dim),
                   torch.zeros(0, dtype=torch.long, device=device), zeros(width - 1, kv_channels),
                   zeros(width - 1, kv_channels), zeros(width - 1, hidden), zeros(width - 1, hidden))


@dataclass(frozen=True)
class ResidualState:
    """Attention-residual block summaries and the running sum of the current block (BF16)."""

    blocks: tuple[torch.Tensor, ...]
    partial: torch.Tensor

    def attention_input(self, weights: LayerWeights) -> tuple[torch.Tensor, ...]:
        if not self.blocks:
            return (self.partial,) * (1 if weights.local else 3)
        return residual_mix((*self.blocks, self.partial), weights.attention_query)

    def mlp_input(self, layer, attention, weights: LayerWeights, block):
        if layer % block == 0:
            return residual_mix((*self.blocks, self.partial, attention), weights.mlp_query)[0]
        return residual_mix((*self.blocks, self.partial), weights.mlp_query, addend=attention)[0]

    def update(self, layer, attention, mlp, block) -> ResidualState:
        if layer % block == 0:
            return ResidualState((*self.blocks, self.partial), attention + mlp)
        total = self.partial.float() + attention.float() + mlp.float()
        return ResidualState(self.blocks, total.to(self.partial.dtype))

    def output(self, query):
        return residual_mix((*self.blocks, self.partial), query)[0]


@dataclass(frozen=True)
class AttentionRecord:
    gated: torch.Tensor
    partial: torch.Tensor
    cache: LayerCache


def attention_block(weights: LayerWeights, config, streams, positions, cache: LayerCache):
    """Normalized streams to the BF16 output projection `[tokens, hidden]`, for the heads `weights` holds.

    `streams` holds one normalized BF16 stream on local layers and the Q, K and V
    streams on global layers. The output convolution is not applied.
    """
    dim, tokens = config.head_dim, positions.shape[0]
    x_q, x_k, x_v = streams if len(streams) == 3 else streams * 3
    heads, kv_heads = weights.query.shape[0] // dim, weights.key.shape[0] // dim
    query = matmul(x_q, weights.query).to(BF16)
    relative = matmul(x_q, weights.relative_query).to(BF16)
    gate = matmul(x_q, weights.output_gate).to(BF16)
    sink = matmul(x_q, weights.sink).to(BF16)
    key = matmul(x_k, weights.key).to(BF16)
    value = matmul(x_v, weights.value).to(BF16)
    if weights.local:
        key, key_history = short_convolution(key, weights.key_convolution, cache.key_history)
        value, value_history = short_convolution(value, weights.value_convolution, cache.value_history)
        tau = None
    else:
        width = weights.key_convolution_static.shape[0]
        lags = []
        for source, projection in ((x_k, weights.key_convolution_dynamic), (x_v, weights.value_convolution_dynamic)):
            lags.append(matmul(source, projection).to(BF16).reshape(tokens, kv_heads, width))
        key, key_history = dynamic_short_convolution(key, weights.key_convolution_static, lags[0], cache.key_history)
        value, value_history = dynamic_short_convolution(value, weights.value_convolution_static, lags[1],
                                                         cache.value_history)
        tau = log_scaling(positions, config.log_scaling_floor, config.log_scaling_alpha)
    query = l2_clip(query.reshape(tokens, heads, dim), config.qk_clip, scale=weights.query_clip_scale, tau=tau)
    key = l2_clip(key.reshape(tokens, kv_heads, dim), config.qk_clip)
    value = rms_norm(value.reshape(tokens, kv_heads, dim), None, config.eps)
    relative = l2_clip(relative.reshape(tokens, heads, config.relative_width), config.relative_clip,
                       scale=weights.relative_clip_scale, tau=tau)
    bias = relative_bias(relative, weights.relative_projection)
    keys, values = torch.cat((cache.key, key)), torch.cat((cache.value, value))
    key_positions = torch.cat((cache.positions, positions))
    output, lse = attention(query, keys, values, bias, positions, key_positions,
                            window=config.window - 1 if weights.local else None)
    gated = gate_sink(output, lse, gate, sink)
    cache = replace(cache, key=keys, value=values, positions=key_positions, key_history=key_history,
                    value_history=value_history)
    return AttentionRecord(gated, matmul(gated, weights.output).to(BF16), cache)


@dataclass(frozen=True)
class ExpertRecord:
    """Routed-expert intermediates.

    `down_inputs[expert]` holds the token and route-slot indices of the expert's rows, its
    BF16 activation and the quantized down-projection input; `routes` holds each route's
    BF16 output `[tokens, top_k, hidden]`.
    """

    ids: torch.Tensor
    weights: torch.Tensor
    down_inputs: dict = field(default_factory=dict)
    routes: torch.Tensor | None = None


def _quantize(x, expert_format, amax=None):
    return quantize_nvfp4(x, amax) if expert_format == "nvfp4" else quantize_mxfp8(x)


def _rows(quantized, rows):
    if isinstance(quantized, Nvfp4Rows):
        return Nvfp4Rows(quantized.values[rows], quantized.scales[rows], quantized.amax[rows])
    return Mxfp8Rows(quantized.values[rows], quantized.exponents[rows])


def _project(weights: LayerWeights, name, expert, quantized):
    values, scales = getattr(weights, name)[expert], getattr(weights, f"{name}_scale")[expert]
    if isinstance(quantized, Nvfp4Rows):
        return nvfp4_projection(quantized, values, scales, getattr(weights, f"{name}_amax")[expert],
                                getattr(weights, f"{name}_denominator")[0])
    return mxfp8_projection(quantized, values, scales)


def _combine(routes):
    total = torch.zeros(routes.shape[0], routes.shape[2], dtype=torch.float32, device=routes.device)
    for slot in range(routes.shape[1]):
        total = total + routes[:, slot].float()
    return total.to(BF16)


def routed_experts_ranks(rank_weights: Sequence[LayerWeights], config, x):
    """Routed experts with each rank holding an intermediate slice of every expert.

    Every rank routes and quantizes the same input rows. The NVFP4 down projection
    quantizes each rank's slice with the elementwise maximum of all ranks' row maxima.
    Returns each rank's BF16 partial output (route outputs summed with unit weights)
    and its intermediates. One rank holding whole experts is the single-rank computation.
    """
    first = rank_weights[0]
    expert_format = first.expert_format
    ids, gammas = route(x, first.router, first.router_bias, top_k=config.experts_per_token,
                        route_scale=config.route_scale)
    quantized = _quantize(x, expert_format)
    records = [ExpertRecord(ids, gammas, {}, torch.zeros(*ids.shape, x.shape[1], dtype=BF16, device=x.device))
               for _ in rank_weights]
    for expert in ids.unique().tolist():
        tokens, slots = (ids == expert).nonzero(as_tuple=True)
        rows = _rows(quantized, tokens)
        activations = [situ(_project(weights, "w13", expert, rows), gammas[tokens, slots], config.gate_softcap,
                            config.up_softcap) for weights in rank_weights]
        amax = None
        if expert_format == "nvfp4":
            amax = torch.stack([activation.float().abs().amax(-1) for activation in activations]).amax(0)
        for weights, activation, record in zip(rank_weights, activations, records, strict=True):
            down_input = _quantize(activation, expert_format, amax)
            record.down_inputs[expert] = (tokens, slots, activation, down_input)
            record.routes[tokens, slots] = _project(weights, "w2", expert, down_input)
    return [_combine(record.routes) for record in records], records


def reduce_ranks(partials: Sequence[torch.Tensor]):
    """Sum per-rank BF16 partials in FP32 in rank order and round once to BF16."""
    total = torch.zeros_like(partials[0], dtype=torch.float32)
    for partial in partials:
        total = total + partial.float()
    return total.to(BF16)


@dataclass(frozen=True)
class LayerRecord:
    """A layer's per-rank intermediates: output-projection and expert partial sums, their
    convolutions, and the reduced attention and expert outputs."""

    attention: tuple[AttentionRecord, ...]
    attention_convolved: tuple[torch.Tensor, ...]
    attention_output: torch.Tensor
    mlp_input: torch.Tensor
    experts: tuple[ExpertRecord, ...]
    expert_partials: tuple[torch.Tensor, ...]
    mlp_convolved: tuple[torch.Tensor, ...]
    mlp_output: torch.Tensor


def normalized_streams(weights: LayerWeights, config, state: ResidualState):
    streams = state.attention_input(weights)
    if weights.local:
        return (rms_norm(streams[0], weights.attention_norm, config.eps),)
    norms = (weights.query_norm, weights.key_norm, weights.value_norm)
    return tuple(rms_norm(stream, norm, config.eps) for stream, norm in zip(streams, norms, strict=True))


def decoder_layer_ranks(rank_weights: Sequence[LayerWeights], config, layer, state: ResidualState, positions,
                        caches: Sequence[LayerCache]):
    """One decoder layer over ranks that each hold a head and expert-intermediate share.

    Replicated computations (residual mixing, norms, routing, activation
    quantization) read rank 0's replicated tensors. Each rank convolves its own
    attention and expert partial sums with its own partial-sum history; the
    reductions follow the convolutions. Returns the updated residual state, each
    rank's cache, and the layer's intermediates.
    """
    first = rank_weights[0]
    streams = normalized_streams(first, config, state)
    attention = tuple(attention_block(weights, config, streams, positions, cache)
                      for weights, cache in zip(rank_weights, caches, strict=True))
    convolved = [short_convolution(record.partial, first.attention_convolution, cache.attention_history)
                 for record, cache in zip(attention, caches, strict=True)]
    attention_output = reduce_ranks([output for output, _ in convolved])
    mlp_input = rms_norm(state.mlp_input(layer, attention_output, first, config.residual_block), first.mlp_norm,
                         config.eps)
    expert_partials, experts = routed_experts_ranks(rank_weights, config, mlp_input)
    mlp = [short_convolution(partial, first.mlp_convolution, cache.mlp_history)
           for partial, cache in zip(expert_partials, caches, strict=True)]
    mlp_output = reduce_ranks([output for output, _ in mlp])
    caches = tuple(replace(record.cache, attention_history=attention_history, mlp_history=mlp_history)
                   for record, (_, attention_history), (_, mlp_history) in zip(attention, convolved, mlp, strict=True))
    record = LayerRecord(attention, tuple(output for output, _ in convolved), attention_output, mlp_input,
                         tuple(experts), tuple(expert_partials), tuple(output for output, _ in mlp), mlp_output)
    return state.update(layer, attention_output, mlp_output, config.residual_block), caches, record


def decoder_layer(weights: LayerWeights, config, layer, state: ResidualState, positions, cache: LayerCache):
    """One decoder layer with whole tensors: the canonical single-rank computation."""
    state, caches, record = decoder_layer_ranks((weights,), config, layer, state, positions, (cache,))
    return state, caches[0], record


@dataclass(frozen=True)
class GlobalWeights:
    """Embedding, final normalization, output residual query and unembedding (whole or per rank)."""

    embedding: torch.Tensor
    embedding_norm: torch.Tensor
    final_norm: torch.Tensor
    output_query: torch.Tensor
    unembedding: torch.Tensor


def embed(ids, weights: GlobalWeights, config):
    """`embed_norm(embed[ids])` in BF16: the residual state's first block."""
    return rms_norm(weights.embedding[ids], weights.embedding_norm, config.eps)


def embed_ranks(ids, rank_weights: Sequence[GlobalWeights], config):
    """Vocabulary-parallel lookup: each rank contributes the rows of ids in its range; the sum is exact."""
    rows = []
    for rank, weights in enumerate(rank_weights):
        width = weights.embedding.shape[0]
        local = ids - rank * width
        inside = (local >= 0) & (local < width)
        rows.append(torch.where(inside[:, None], weights.embedding[local.clamp(0, width - 1)], 0))
    return rms_norm(reduce_ranks(rows), rank_weights[0].embedding_norm, config.eps)


def logits(hidden, weights: GlobalWeights, config):
    """FP32 unembedding, softcap `c * tanh(x / c)`, and `-inf` at ids past the valid vocabulary."""
    values = matmul(hidden, weights.unembedding)
    values = config.softcap * torch.tanh(values / config.softcap)
    values[..., config.valid_vocabulary:] = float("-inf")
    return values


def logits_ranks(hidden, rank_weights: Sequence[GlobalWeights], config):
    """Per-rank vocabulary shards, softcapped before the gather, trimmed to the table and masked."""
    shards = [config.softcap * torch.tanh(matmul(hidden, weights.unembedding) / config.softcap)
              for weights in rank_weights]
    values = torch.cat(shards, -1)[..., :config.vocabulary]
    values[..., config.valid_vocabulary:] = float("-inf")
    return values


def text_logits(ids, positions, config, weights: GlobalWeights, layer_weights: Callable[[int], LayerWeights],
                caches: Sequence[LayerCache] | None = None):
    """Stack forward of one sequence: embedding, every decoder layer, output mixture, final norm and logits.

    `layer_weights(i)` supplies layer `i` when it runs, so callers can stream layers.
    Returns FP32 logits, the BF16 final hidden states and each layer's cache.
    """
    state = ResidualState((), embed(ids, weights, config))
    updated = []
    for layer in range(config.layers):
        current = layer_weights(layer)
        cache = LayerCache.empty(current, config.head_dim) if caches is None else caches[layer]
        state, cache, _ = decoder_layer(current, config, layer, state, positions, cache)
        updated.append(cache)
    hidden = rms_norm(state.output(weights.output_query), weights.final_norm, config.eps)
    return logits(hidden, weights, config), hidden, tuple(updated)


def attention_sources(layer: int, block: int) -> int:
    """Sources the attention input of `layer >= 1` mixes: committed blocks plus the running sum."""
    return (layer - 1) // block + 2


def mlp_sources(layer: int, block: int) -> int:
    return layer // block + 2

