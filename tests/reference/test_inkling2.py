# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 reference operators against the SGLang formulas, and four-rank layers against one rank.

The `sglang_*` functions, and the expected values of the gate, tau, routing and
softcap tests, transcribe SGLang PR 50186 (commit 237ba6b): its CPU fallbacks where
they exist (`gate_sink`, `compute_log_scaling_tau`, the router's log-sigmoid
renormalization, the logits softcap) and otherwise the per-element arithmetic of the
Triton or CuTe kernels in `python/sglang/kernels/ops/inkling2/`, evaluated with exact
elementary functions. The four-rank tests build whole and per-rank weights from the
same synthetic stored tensors through the recipe's weight layout.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from ignition.reference import inkling2 as ref
from tests.reference.inkling2_support import (
    attention_bound,
    layer_weights,
    mlp_bound,
    shard,
    synthetic_model,
)

BF16 = torch.bfloat16


def generator(seed=0):
    return torch.Generator().manual_seed(seed)


def sglang_rms_norm(x, weight, eps):
    # norm.py RMSNormFwd: x * rsqrt(sum(x^2) / N + eps) * w in FP32.
    values = x.float()
    return (values * torch.rsqrt(values.square().sum(-1, keepdim=True) / x.shape[-1] + eps) * weight.float()).to(BF16)


def test_rms_norm_matches_kernel():
    x = torch.randn(5, 64, generator=generator()).to(BF16) * 3
    weight = (1 + 0.1 * torch.randn(64, generator=generator(1))).to(BF16)
    torch.testing.assert_close(ref.rms_norm(x, weight, 1e-6), sglang_rms_norm(x, weight, 1e-6), rtol=2 ** -7, atol=0)


def sglang_attn_res(sources, addend, queries, eps=1e-6):
    # attn_res.py: per source sum of squares and query dot products (addend joins the last
    # source in FP32), rsqrt(sumsq / D + eps), softmax of dot * rsqrt, then
    # sum(exp * rsqrt * x) / sum(exp).
    tokens, hidden = sources[0].shape
    queries = queries.float().reshape(-1, hidden) * hidden ** -0.5
    out = torch.empty(queries.shape[0], tokens, hidden)
    for t in range(tokens):
        rows = [source[t].float() for source in sources]
        if addend is not None:
            rows[-1] = rows[-1] + addend[t].float()
        stats = [torch.rsqrt((row * row).sum() / hidden + eps) for row in rows]
        for q, query in enumerate(queries):
            logits = torch.stack([(row * query).sum() * stat for row, stat in zip(rows, stats)])
            exps = torch.exp(logits - logits.max())
            out[q, t] = sum(e * s * row for e, s, row in zip(exps, stats, rows)) / exps.sum()
    return out


@pytest.mark.parametrize("queries,addend", [(1, False), (1, True), (3, False)])
def test_residual_mix_matches_attn_res_kernel(queries, addend):
    g = generator(queries + 2 * addend)
    sources = [(torch.randn(4, 96, generator=g) * scale).to(BF16) for scale in (1, 4, 0.3)]
    query = torch.randn(queries, 96, generator=g)
    extra = torch.randn(4, 96, generator=g).to(BF16) if addend else None
    mixed = torch.stack(ref.residual_mix(sources, query, addend=extra))
    torch.testing.assert_close(mixed, sglang_attn_res(sources, extra, query), rtol=1e-5, atol=1e-6)


def sglang_l2_clip(x, threshold, eps, scale=None, tau=None):
    # qk_l2_clip.py: x * scale, coeff = min(1, threshold / (norm + eps)) with norm = sumsq * rsqrt(sumsq)
    # (zero for a zero row), y = x * coeff, rounded to the input dtype before the tau multiply.
    values = x.float() * (scale.float() if scale is not None else 1)
    sumsq = values.square().sum(-1, keepdim=True)
    norm = torch.where(sumsq > 0, sumsq * torch.rsqrt(sumsq), torch.zeros_like(sumsq))
    y = values * torch.minimum(torch.ones_like(norm), threshold / (norm + eps))
    if tau is not None:
        y = y.to(x.dtype).float() * tau[:, None, None]
    return y.to(x.dtype)


def test_l2_clip_matches_kernel():
    g = generator(3)
    x = (torch.randn(6, 4, 32, generator=g) * torch.tensor([0.1, 1, 10, 0])[None, :, None]).to(BF16)
    scale = (1 + 0.2 * torch.randn(4, 32, generator=g)).to(BF16)
    tau = torch.tensor([1.0, 1.0, 1.05, 1.1, 1.2, 1.3])
    for kwargs in ({}, dict(scale=scale), dict(scale=scale, tau=tau)):
        actual = ref.l2_clip(x, 4.0, **kwargs)
        torch.testing.assert_close(actual, sglang_l2_clip(x, 4.0, 1e-6, **kwargs), rtol=2 ** -7, atol=0)
    clipped = ref.l2_clip(x, 4.0).float().norm(dim=-1)
    assert bool((clipped <= 4.0 * (1 + 2 ** -6)).all()) and bool((clipped[:, 3] == 0).all())


def test_log_scaling_matches_cpu_fallback():
    positions = torch.tensor([0, 127998, 127999, 128000, 200000, 1048575])
    expected = 1.0 + 0.1 * torch.log(torch.clamp((positions + 1).float() / 128000, min=1.0))
    actual = ref.log_scaling(positions, 128000, 0.1)
    assert torch.equal(actual, expected) and bool((actual[:3] == 1).all()) and bool((actual[3:] > 1).all())


def sglang_sconv(x, weight, dynamic, cache, *, dynamic_kernel):
    # sconv.py _sconv_kernel: lag = tap (dynamic) or W - 1 - tap (static); positions before the
    # chunk read the cache; acc = fma(value, weight, acc) (a plain product for the dynamic tap 0);
    # acc = (acc + current) * (W * (2 if dynamic else 1) + 1) ** -0.5.
    tokens, channels = x.shape
    width = weight.shape[0] if dynamic_kernel else weight.shape[-1]
    head_dim = channels // dynamic.shape[1] if dynamic_kernel else channels
    out = torch.empty(tokens, channels)
    for row in range(tokens):
        acc = torch.zeros(channels, dtype=torch.float64)
        for tap in range(width):
            lag = tap if dynamic_kernel else width - 1 - tap
            src = row - lag
            value = x[src].double() if src >= 0 else cache[src + width - 1].double()
            if dynamic_kernel:
                columns = torch.arange(channels)
                w = weight[lag].float() + dynamic[row, columns // head_dim, lag].float()
            else:
                w = weight.reshape(channels, width)[:, tap].float()
            acc = (value * w.double() + acc).float().double()
        out[row] = ((acc.float() + x[row].float()) * (width * (2 if dynamic_kernel else 1) + 1) ** -0.5)
    return out.to(x.dtype)


def test_static_short_convolution_matches_kernel():
    g = generator(4)
    x, history = torch.randn(9, 48, generator=g).to(BF16), torch.randn(3, 48, generator=g).to(BF16)
    weight = (0.4 * torch.randn(48, 1, 4, generator=g)).to(BF16)
    output, after = ref.short_convolution(x, weight, history)
    assert torch.equal(output, sglang_sconv(x, weight, None, history, dynamic_kernel=False))
    assert torch.equal(after, x[-3:])
    first, middle = ref.short_convolution(x[:4], weight, history)
    second, _ = ref.short_convolution(x[4:], weight, middle)
    assert torch.equal(torch.cat((first, second)), output)


def test_dynamic_short_convolution_matches_kernel():
    g = generator(5)
    x, history = torch.randn(7, 64, generator=g).to(BF16), torch.randn(3, 64, generator=g).to(BF16)
    static = (0.4 * torch.randn(4, 64, generator=g)).to(BF16)
    dynamic = torch.randn(7, 2, 4, generator=g).to(BF16)
    output, _ = ref.dynamic_short_convolution(x, static, dynamic, history)
    assert torch.equal(output, sglang_sconv(x, static, dynamic, history, dynamic_kernel=True))


def sglang_attention(q, keys, values, rel, positions, scale, window):
    # test_inkling2_attention_kernels.py _rel_bias_attention_ref: FP32 scores with the compact
    # relative bias (zero past the extent), causal and optional left-window masks.
    tokens, heads, _ = q.shape
    group = heads // keys.shape[1]
    out, lse = torch.empty(q.shape), torch.empty(tokens, heads)
    pos = torch.arange(keys.shape[0])
    for i in range(tokens):
        dist = positions[i] - pos
        for h in range(heads):
            s = (keys[:, h // group].float() @ q[i, h].float()) * scale
            bias = torch.zeros(keys.shape[0])
            inside = (dist >= 0) & (dist < rel.shape[-1])
            bias[inside] = rel[i, h, dist[inside]].float()
            s = (s + bias).masked_fill(dist < 0, float("-inf"))
            if window is not None:
                s = s.masked_fill(dist > window, float("-inf"))
            lse[i, h] = torch.logsumexp(s, 0)
            out[i, h] = torch.softmax(s, 0) @ values[:, h // group].float()
    return out, lse


@pytest.mark.parametrize("window", [None, 5])
def test_attention_matches_fp32_reference(window):
    g = generator(6)
    q = torch.randn(4, 6, 16, generator=g).to(BF16)
    keys, values = (torch.randn(12, 2, 16, generator=g).to(BF16) for _ in range(2))
    rel = torch.randn(4, 6, 8, generator=g).to(BF16)
    positions = torch.arange(8, 12)
    output, lse = ref.attention(q, keys, values, rel, positions, torch.arange(12), window=window)
    expected, expected_lse = sglang_attention(q, keys, values, rel, positions, 16 ** -0.5, window)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-6, atol=1e-6)


def test_gate_sink_matches_cpu_fallback():
    g = generator(7)
    output, lse = torch.randn(5, 3, 8, generator=g), torch.randn(5, 3, generator=g)
    gate, sink = torch.randn(5, 24, generator=g).to(BF16), torch.randn(5, 3, generator=g).to(BF16)
    scale = torch.sigmoid(lse.float() - sink.float())
    gated = output.reshape(5, 24).float() * torch.sigmoid(gate.float())
    expected = (gated.view(5, 3, -1) * scale.unsqueeze(-1)).flatten(1).to(BF16)
    assert torch.equal(ref.gate_sink(output, lse, gate, sink), expected)


def test_sink_is_one_extra_softmax_logit_with_a_zero_value():
    g = generator(15)
    scores, values, sink = torch.randn(9, generator=g).double(), torch.randn(9, 4, generator=g).double(), 0.7
    output, lse = torch.softmax(scores, 0) @ values, torch.logsumexp(scores, 0)
    extended = torch.softmax(torch.cat((scores, torch.tensor([sink], dtype=torch.float64))), 0)
    torch.testing.assert_close(output * torch.sigmoid(lse - sink), extended[:-1] @ values)


def test_route_matches_cpu_renormalization():
    g = generator(8)
    x = torch.randn(6, 64, generator=g).to(BF16)
    weight = (torch.randn(24, 64, generator=g) / 8).to(BF16)
    bias = 0.05 * torch.randn(24, generator=g)
    bias[3] = bias[4] = 10.0
    weight[4] = weight[3]
    ids, weights = ref.route(x, weight, bias, top_k=8, route_scale=8.0)
    logits = (x.double() @ weight.double().T).float()
    selection = 1.0 / (1.0 + torch.exp(-logits)) + bias
    for token in range(6):
        ranked = sorted(range(24), key=lambda e: (-selection[token, e].item(), e))[:8]
        assert ids[token].tolist() == ranked
    assert bool((ids[:, 0] == 3).all() & (ids[:, 1] == 4).all())
    log_probs = torch.nn.functional.logsigmoid(logits.gather(-1, ids))
    expected = torch.exp(log_probs - torch.logsumexp(log_probs, -1, keepdim=True)) * 8.0
    assert torch.equal(weights, expected)
    sig = torch.sigmoid(logits.gather(-1, ids).double())
    torch.testing.assert_close(weights.double(), 8 * sig / sig.sum(-1, keepdim=True), rtol=1e-6, atol=0)


def test_situ_matches_capped_activation():
    g = generator(9)
    gate_up = (torch.randn(5, 64, generator=g) * 6).to(BF16)
    gamma = torch.rand(5, generator=g) * 3
    gate, up = gate_up[:, 0::2].float(), gate_up[:, 1::2].float()
    # moe.py _capped_swiglu: ((4 tanh(g / 4) * sigmoid(g)) * 25 tanh(u / 25)) * gamma, sigmoid of the uncapped gate.
    expected = (((4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(up / 25))) * gamma[:, None])
    torch.testing.assert_close(ref.situ(gate_up, gamma, 4.0, 25.0).float(), expected.to(BF16).float(),
                               rtol=2 ** -7, atol=0)


def sglang_e2m1_round(x):
    # experts_sm90.py _e2m1_round.
    a = x.abs()
    r = torch.where(a <= 0.25, 0.0, torch.where(a < 0.75, 0.5, torch.where(a <= 1.25, 1.0, torch.where(
        a < 1.75, 1.5, torch.where(a <= 2.5, 2.0, torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0, 6.0)))))))
    return torch.where(x < 0, -r, r)


def test_e2m1_rounding_matches_emulation():
    grid = torch.linspace(-8, 8, 8193)
    special = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 7.0, float("inf")])
    values = torch.cat((grid, special, -special))
    assert torch.equal(ref.round_e2m1(values), sglang_e2m1_round(values))


def sglang_nvfp4_values(x, amax):
    # experts_sm90.py _nvfp4_values: E2M1 x E4M3 block values (no outer scale) of four-over-six blocks.
    f32max = torch.finfo(torch.float32).max
    blocks = x.float().reshape(x.shape[0], -1, 16)
    encoder = torch.minimum(1536.0 / amax, torch.tensor(f32max))[:, None]
    decoder = 1.0 / encoder
    outer = (amax * (1.0 / 1536.0))[:, None, None]
    block_max = blocks.abs().amax(-1)
    s6 = torch.minimum(block_max * (encoder * (1.0 / 6.0)), torch.tensor(448.0)).to(torch.float8_e4m3fn).float()
    s4 = torch.minimum(block_max * (encoder * 0.25), torch.tensor(448.0)).to(torch.float8_e4m3fn).float()
    i6 = torch.minimum(1.0 / (s6 * decoder), torch.tensor(f32max))
    i4 = torch.minimum(1.0 / (s4 * decoder), torch.tensor(f32max))
    r6 = sglang_e2m1_round(blocks * i6[..., None]) * s6[..., None]
    r4 = sglang_e2m1_round(blocks * i4[..., None]) * s4[..., None]

    def block_error(reconstructed):
        error = (reconstructed * outer - blocks).abs()
        for width in (3, 2, 1, 0):
            pairs = error.reshape(*error.shape[:-1], 1 << width, 2)
            error = pairs[..., 0] + pairs[..., 1]
        return error[..., 0]

    return torch.where((block_error(r4) < block_error(r6))[..., None], r4, r6).reshape(x.shape).to(BF16)


def test_nvfp4_quantization_matches_fake_quantizer():
    g = generator(10)
    x = (torch.randn(32, 256, generator=g) * torch.logspace(-3, 1, 32)[:, None]).to(BF16)
    x[3] = 0
    x[5, :16] = 0
    quantized = ref.quantize_nvfp4(x)
    amax = x.float().abs().amax(-1)
    assert torch.equal(quantized.amax, amax)
    assert torch.equal(quantized.blocks().to(BF16), sglang_nvfp4_values(x, amax))
    peak = x.float().reshape(32, -1, 16).abs().amax(-1)
    encoder = 1536.0 / amax[:, None]
    six, four = (torch.clamp(peak * (encoder * factor), max=448.0).to(torch.float8_e4m3fn).float()
                 for factor in (1.0 / 6.0, 0.25))
    chose_four = (quantized.scales.float() == four) & (four != six)
    assert 0.05 < float(chose_four.float().mean()) < 0.95
    dequantized = quantized.blocks() * (amax / 1536)[:, None]
    assert float((dequantized - x.float()).abs().max()) <= float(amax.max()) / 6
    # A supplied amax covering a wider row changes the outer scale and therefore the codes.
    wider = ref.quantize_nvfp4(x, amax * 2)
    assert torch.equal(wider.blocks().to(BF16), sglang_nvfp4_values(x, amax * 2))
    assert not torch.equal(wider.blocks(), quantized.blocks())


def sglang_mxfp8_values(x):
    # moe.py _mxfp8_block and experts_sm90.py _ue8m0.
    groups = x.float().reshape(x.shape[0], -1, 32)
    raw = groups.abs().amax(-1) * (1.0 / 448.0)
    exponent = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2.0 ** -127, 0, torch.minimum(exponent, torch.tensor(254)))
    scale = torch.maximum((exponent << 23).view(torch.float32), torch.tensor(2.0 ** -127))
    inverse = torch.where(exponent == 0, 2.0 ** 127, 1.0 / scale)
    values = torch.clamp(groups * inverse[..., None], -448, 448).to(torch.float8_e4m3fn).float()
    unit = torch.where(exponent == 0, 2.0 ** -127, (exponent << 23).view(torch.float32))
    return (values * unit[..., None]).reshape(x.shape).to(BF16), exponent


def test_mxfp8_quantization_matches_fake_quantizer():
    g = generator(11)
    x = (torch.randn(16, 128, generator=g) * torch.logspace(-30, 3, 16)[:, None]).to(BF16)
    x[2] = 0
    quantized = ref.quantize_mxfp8(x)
    expected, exponent = sglang_mxfp8_values(x)
    assert torch.equal(quantized.exponents.int(), exponent)
    assert torch.equal(quantized.dequantized().to(BF16), expected)


def test_nvfp4_projection_epilogue():
    g = generator(12)
    x = torch.randn(3, 64, generator=g).to(BF16)
    packed = torch.randint(0, 256, (5, 32), dtype=torch.uint8, generator=g)
    scales = (1 + 400 * torch.rand(5, 4, generator=g)).to(torch.float8_e4m3fn)
    amax, denominator = torch.tensor(0.3), torch.tensor([2688.0])
    quantized = ref.quantize_nvfp4(x)
    actual = ref.nvfp4_projection(quantized, packed, scales, amax, denominator)
    table = torch.tensor(ref._E2M1)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    weight = table[codes] * scales.float().repeat_interleave(16, dim=1)
    accumulator = (quantized.blocks().double() @ weight.double().T).float()
    # nvfp4.py _scale_output: acc * ((x_amax * w_amax) * rcp(1536 * w_denominator)).
    gain = (quantized.amax * amax) * (1.0 / (1536.0 * denominator))
    assert torch.equal(actual, (accumulator * gain[:, None]).to(BF16))


def test_logits_softcap_and_vocabulary_mask():
    config, layout, tensors = synthetic_model()
    hidden = torch.randn(3, config.text.hidden, generator=generator(13)).to(BF16)
    weights = ref.GlobalWeights(*(tensors[f"model.llm.{name}"] for name in (
        "embed.weight", "embed_norm.weight", "norm.weight", "attn_res.output_query", "unembed.weight")))
    values = ref.logits(hidden * 40, weights, config.text)
    raw = (hidden * 40).double() @ weights.unembedding.double().T
    expected = 30.0 * torch.tanh(raw.float() / 30.0)
    expected[..., config.text.valid_vocabulary:] = float("-inf")
    assert torch.equal(values, expected) and float(values[:, :990].abs().max()) < 30


def test_vocabulary_parallel_embedding_and_logits_match_whole_tables():
    config, layout, tensors = synthetic_model()
    names = ("embed.weight", "embed_norm.weight", "norm.weight", "attn_res.output_query", "unembed.weight")
    whole = ref.GlobalWeights(*(tensors[f"model.llm.{name}"] for name in names))
    ranks = [ref.GlobalWeights(*(shard(tensors[f"model.llm.{name}"], layout.model[f"model.llm.{name}"], rank)
                                 for name in names)) for rank in range(4)]
    assert ranks[3].embedding.shape[0] == 256 and bool((ranks[3].embedding[-24:] == 0).all())
    ids = torch.tensor([0, 255, 256, 767, 768, 999, 989, 990])
    assert torch.equal(ref.embed_ranks(ids, ranks, config.text), ref.embed(ids, whole, config.text))
    hidden = torch.randn(4, config.text.hidden, generator=generator(14)).to(BF16)
    assert torch.equal(ref.logits_ranks(hidden, ranks, config.text), ref.logits(hidden, whole, config.text))


def residual_state(config, layer, tokens, seed):
    """A residual state as layer `layer` receives it: the embedding block, committed blocks and a running sum."""
    g = generator(seed)
    blocks = tuple((torch.randn(tokens, config.hidden, generator=g) * scale).to(BF16)
                   for scale in [1.0] + [4.0] * (layer // config.residual_block))
    if not layer:
        return ref.ResidualState((), blocks[0])
    return ref.ResidualState(blocks, (torch.randn(tokens, config.hidden, generator=g) * 3).to(BF16))


def stage_records(config, layout, tensors, layer, tokens=24, start=0, seed=20):
    whole = layer_weights(layout, tensors, layer)
    ranks = [layer_weights(layout, tensors, layer, rank) for rank in range(4)]
    state = residual_state(config.text, layer, tokens, seed)
    positions = torch.arange(start, start + tokens)
    single = ref.decoder_layer(whole, config.text, layer, state, positions,
                               ref.LayerCache.empty(whole, config.text.head_dim))
    parallel = ref.decoder_layer_ranks(ranks, config.text, layer, state, positions,
                                       [ref.LayerCache.empty(weights, config.text.head_dim) for weights in ranks])
    return whole, ranks, state, positions, single, parallel


@pytest.mark.parametrize("layer,start", [(1, 0), (5, 0), (5, 30), (6, 0)])
def test_four_rank_attention_matches_one_rank_within_partial_sum_rounding(layer, start):
    config, layout, tensors = synthetic_model()
    whole, ranks, _, _, single, parallel = stage_records(config, layout, tensors, layer, start=start)
    _, _, single_record = single
    _, _, parallel_record = parallel
    heads = config.text.heads // 4 * config.text.head_dim
    for rank, record in enumerate(parallel_record.attention):
        assert torch.equal(record.gated, single_record.attention[0].gated[:, rank * heads:(rank + 1) * heads])
    difference = (parallel_record.attention_output.float() - single_record.attention_output.float()).abs()
    bound = attention_bound(single_record, parallel_record, whole)
    assert bool((difference <= bound).all()), float((difference - bound).max())
    assert float(difference.max()) > 0


@pytest.mark.parametrize("layer", [2, 5, 6])
def test_four_rank_experts_match_one_rank_within_partial_sum_rounding(layer):
    config, layout, tensors = synthetic_model()
    whole = layer_weights(layout, tensors, layer)
    ranks = [layer_weights(layout, tensors, layer, rank) for rank in range(4)]
    x = ref.rms_norm(torch.randn(24, config.text.hidden, generator=generator(layer)) * 2, whole.mlp_norm)
    single_outputs, single_records = ref.routed_experts_ranks((whole,), config.text, x)
    outputs, records = ref.routed_experts_ranks(ranks, config.text, x)
    width = config.text.intermediate // 4
    for expert, (_, _, activation, down) in single_records[0].down_inputs.items():
        for rank, record in enumerate(records):
            assert torch.equal(record.down_inputs[expert][2], activation[:, rank * width:(rank + 1) * width])
            part = record.down_inputs[expert][3]
            columns = slice(rank * width, (rank + 1) * width)
            if whole.expert_format == "nvfp4":
                assert torch.equal(part.values, down.values[:, columns]) and torch.equal(part.amax, down.amax)
                blocks = slice(rank * width // 16, (rank + 1) * width // 16)
                assert torch.equal(part.scales.float(), down.scales.float()[:, blocks])
            else:
                assert torch.equal(part.values.float(), down.values.float()[:, columns])
                assert torch.equal(part.exponents, down.exponents[:, rank * width // 32:(rank + 1) * width // 32])
    if whole.expert_format == "nvfp4":
        # Quantizing a slice with its own row maximum changes the down-projection input.
        assert any(not torch.equal(ref.quantize_nvfp4(activation).blocks(), down.blocks())
                   for record in records for _, _, activation, down in record.down_inputs.values())
    single_m = ref.short_convolution(single_outputs[0], whole.mlp_convolution)[0]
    convolved = [ref.short_convolution(output, whole.mlp_convolution)[0] for output in outputs]
    parallel_m = ref.reduce_ranks(convolved)
    single_record = SimpleNamespace(expert_partials=single_outputs, mlp_convolved=[single_m], mlp_output=single_m,
                                    experts=single_records)
    parallel_record = SimpleNamespace(expert_partials=outputs, mlp_convolved=convolved, mlp_output=parallel_m,
                                      experts=records)
    difference = (parallel_m.float() - single_m.float()).abs()
    bound = mlp_bound(single_record, parallel_record, whole)
    assert bool((difference <= bound).all()), float((difference - bound).max())
    assert float(difference.max()) > 0


@pytest.mark.parametrize("layer", [0, 5, 6])
def test_four_rank_layer_composes_its_stages(layer):
    config, layout, tensors = synthetic_model()
    whole, ranks, state, positions, single, parallel = stage_records(config, layout, tensors, layer)
    single_state, _, single_record = single
    parallel_state, caches, parallel_record = parallel
    difference = (parallel_record.attention_output.float() - single_record.attention_output.float()).abs()
    assert bool((difference <= attention_bound(single_record, parallel_record, whole)).all())
    expected_input = ref.rms_norm(state.mlp_input(layer, parallel_record.attention_output, whole,
                                                  config.text.residual_block), whole.mlp_norm, config.text.eps)
    assert torch.equal(parallel_record.mlp_input, expected_input)
    outputs, records = ref.routed_experts_ranks((whole,), config.text, parallel_record.mlp_input)
    reference_m = ref.short_convolution(outputs[0], whole.mlp_convolution)[0]
    record = SimpleNamespace(expert_partials=outputs, mlp_convolved=[reference_m], mlp_output=reference_m,
                             experts=records)
    difference = (parallel_record.mlp_output.float() - reference_m.float()).abs()
    assert bool((difference <= mlp_bound(record, parallel_record, whole)).all())
    expected = state.update(layer, parallel_record.attention_output, parallel_record.mlp_output,
                            config.text.residual_block)
    assert torch.equal(parallel_state.partial, expected.partial) and len(parallel_state.blocks) == len(expected.blocks)
    for rank, cache in enumerate(caches):
        assert cache.key.shape == (24, 1, config.text.head_dim)
        assert torch.equal(cache.attention_history, parallel_record.attention[rank].partial[-3:])


@pytest.mark.parametrize("layer", [1, 5])
def test_chunked_layer_matches_one_pass(layer):
    config, layout, tensors = synthetic_model()
    whole = layer_weights(layout, tensors, layer)
    state = residual_state(config.text, layer, 24, 30)
    positions = torch.arange(10, 34)
    cache = ref.LayerCache.empty(whole, config.text.head_dim)
    one, _, _ = ref.decoder_layer(whole, config.text, layer, state, positions, cache)
    split = 17
    head = ref.ResidualState(tuple(b[:split] for b in state.blocks), state.partial[:split])
    tail = ref.ResidualState(tuple(b[split:] for b in state.blocks), state.partial[split:])
    first, cache, _ = ref.decoder_layer(whole, config.text, layer, head, positions[:split], cache)
    second, cache, _ = ref.decoder_layer(whole, config.text, layer, tail, positions[split:], cache)
    torch.testing.assert_close(torch.cat((first.partial, second.partial)), one.partial, rtol=2 ** -7, atol=1e-3)
    assert cache.key.shape[0] == 24 and torch.equal(cache.positions, positions)


def test_stack_mixes_the_declared_residual_sources():
    config, layout, tensors = synthetic_model()
    text = config.text
    names = ("embed.weight", "embed_norm.weight", "norm.weight", "attn_res.output_query", "unembed.weight")
    weights = ref.GlobalWeights(*(tensors[f"model.llm.{name}"] for name in names))
    ids = torch.tensor([5, 17, 990, 3, 999, 42, 7, 7])
    positions = torch.arange(8)
    seen = []

    def layer(index):
        current = layer_weights(layout, tensors, index)
        seen.append(index)
        return current

    values, hidden, caches = ref.text_logits(ids, positions, text, weights, layer)
    assert seen == list(range(text.layers)) and len(caches) == text.layers
    assert values.shape == (8, text.vocabulary) and hidden.dtype == BF16
    assert bool(torch.isfinite(values[:, :text.valid_vocabulary]).all())
    assert bool(torch.isneginf(values[:, text.valid_vocabulary:]).all())
    state = ref.ResidualState((), ref.embed(ids, weights, text))
    for index in range(text.layers):
        if index:
            assert len(state.blocks) + 1 == ref.attention_sources(index, text.residual_block)
        mlp = len(state.blocks) + 1 + (index % text.residual_block == 0)
        assert mlp == ref.mlp_sources(index, text.residual_block)
        current = layer_weights(layout, tensors, index)
        state, _, _ = ref.decoder_layer(current, text, index, state, positions,
                                        ref.LayerCache.empty(current, text.head_dim))
    assert len(state.blocks) == 2
    final = ref.rms_norm(state.output(weights.output_query), weights.final_norm, text.eps)
    assert torch.equal(final, hidden)
    assert math.isclose(float(ref.log_scaling(torch.tensor([39]), text.log_scaling_floor, 0.1)), 1 + 0.1 * math.log(2))
