# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP16 DSpark primitive arithmetic, independent composition, and captured state effects.

Primitive inputs and outputs satisfy independent checkpoint arithmetic
intervals for FP32 accumulation, BF16 rounding, and SFU instructions. Eager
captures match separately sequenced native primitives; compiled captures
match the retained compiled callable. Rotary intervals use canonical FP32
angles and trigonometric values with FP64 multiply/add bounds.
"""

from dataclasses import fields, is_dataclass, replace
from datetime import timedelta
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

WORLD, BLOCK, PAGE, RECORD = 16, 7, 128, 576
CACHE_DTYPE = torch.bfloat16
U32, U16 = 2. ** -24, 2. ** -8
BF16_HALF_SUBNORMAL = 2. ** -134
SFU_REL = 2. ** -22
TARGET = Path(os.environ.get("FLARE_KIMI31_CHECKPOINT", str(Path.home() /
    ".cache/huggingface/hub/models--moonshotai--Kimi-K3.1-Pro/snapshots/"
    "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e")))
DRAFT = Path(os.environ.get("FLARE_KIMI31_DSPARK_CHECKPOINT", str(Path.home() /
    ".cache/huggingface/hub/models--Inferact--Kimi-K3.1-Pro-DSpark-preview/snapshots/"
    "6a2ef20a05efe88aabbf9ecaa12d020c3b0edf94")))

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires sixteen explicitly assigned CUDA devices")


class _ContextForward:
    def __init__(self, operation, auxiliary, counts):
        self.operation, self.auxiliary, self.counts = operation, auxiliary, counts

    def __call__(self):
        self.operation(self.auxiliary, self.counts)


class _RotaryForward:
    def __init__(self, operation, source, positions, query, key):
        self.operation, self.source, self.positions = operation, source, positions
        self.query, self.key = query, key

    def __call__(self):
        self.query.copy_(self.source)
        self.key.copy_(self.source[:, 0])
        self.operation(self.query, self.key, self.positions)


def _projected_values(projected):
    return {name: getattr(projected, name).clone() for name in (
        "local", "input", "query_norm", "query", "latent", "absorbed")}


class _ObservedInvocation:
    def __init__(self, invocation, probe):
        self.invocation, self.probe = invocation, probe

    def run(self, reads, writes, integers=()):
        operands = []
        for _, mutable, index, view in self.invocation.operands:
            value = writes[index] if mutable else reads[index]
            if view is not None:
                dtype, shape, stride, offset = view
                value = value.view(dtype)
                value = value.as_strided(shape, stride, value.storage_offset() + offset)
            operands.append((mutable, value))
        result = self.probe.before(self.invocation.target, operands)
        self.invocation.run(reads, writes, integers)
        self.probe.save(**result)


class _DraftBoundaries:
    def __init__(self, prepared):
        self.prepared = prepared
        self.attention = prepared.layers[0].self_attn
        self.projected = self.attention.projected
        self.values = {"layers": []}

    def save(self, **values):
        if values:
            self.values["layers"][-1].update({name: value.clone() for name, value in values.items()})

    def before(self, target, operands):
        def writes(target):
            return any(mutable and value.dtype == target.dtype
                       and value.data_ptr() == target.data_ptr() and value.numel() == target.numel()
                       for mutable, value in operands)

        scratch, attention = self.prepared.scratch, self.attention
        if writes(self.projected.local):
            if self.values["layers"]:
                self.save(mlp_output=scratch.output)
            else:
                self.values.update(local_embedding=scratch.local_embedding.clone(), embedding=scratch.embedding.clone())
            self.values["layers"].append({})
            source = next(value for mutable, value in operands if not mutable and value.shape == scratch.embedding.shape)
            self.save(input_norm=source)
            return {"local": self.projected.local}
        if writes(attention.block):
            self.values["layers"][-1].update(_projected_values(self.projected))
            self.save(block_before=attention.block, slots=attention.slots)
            return {"block": attention.block}
        if target is torch.ops.flare.dense_mla.default:
            if writes(attention.history):
                self.save(gathered=attention.gathered, history_pages=attention.history_pages,
                    current_pages=attention.current_pages, history_lengths=attention.history_lengths,
                    current_lengths=attention.current_lengths)
                return {"history": attention.history, "history_lse": attention.history_lse}
            if writes(attention.current):
                return {"current": attention.current, "current_lse": attention.current_lse}
        if target is torch.ops.flare.pcie_lse_reduce_scatter.default:
            self.save(merged=attention.history, merged_lse=attention.history_lse)
        if writes(self.projected.projections["output"].output):
            self.save(attended=self.projected.attended, expanded=self.projected.expanded)
            return {"attention_partial": self.projected.projections["output"].output}
        if writes(scratch.mlp.gate_up):
            source = next(value for mutable, value in operands if not mutable and value.shape == scratch.embedding.shape)
            self.save(post_norm=source, attention_output=scratch.output)
            return {"gate_up": scratch.mlp.gate_up}
        if writes(scratch.mlp.output):
            self.save(intermediate=scratch.mlp.intermediate)
            return {"mlp_partial": scratch.mlp.output}
        return {}

    def finish(self):
        assert len(self.values["layers"]) == len(self.prepared.layers), "compiled draft layer boundaries"
        self.save(mlp_output=self.prepared.scratch.output)
        for name in ("hidden", "local_logits", "logits"):
            self.values[name] = getattr(self.prepared.scratch, name).clone()
        return self.values


def _gather_columns(value):
    import torch.distributed as dist

    source = value.contiguous().cpu()
    peers = [torch.empty_like(source) for _ in range(WORLD)]
    dist.all_gather(peers, source)
    return torch.cat(peers, -1).to(value.device)


def _matrix(weight):
    if isinstance(weight, torch.Tensor):
        return weight.float()
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    assert isinstance(weight, MXFP8LinearWeight)
    packed = weight.weight
    scales = packed.scale_rows.view(torch.float8_e8m0fnu).float()[0]
    expanded = packed.values.float().view(weight.out_features, -1, 32) * scales[..., None]
    return expanded.flatten(1)[:, :weight.in_features].contiguous()


def _gamma(operations, *, rounding=U32):
    product = operations * rounding
    assert product < 1
    return product / (1 - product)


def _interval(actual, lower, upper, label):
    import torch.distributed as dist

    value = actual.double()
    lower, upper = torch.broadcast_tensors(lower.double(), upper.double())
    assert value.shape == lower.shape, (label, value.shape, lower.shape)
    violation = torch.maximum(lower - value, value - upper).clamp_min(0)
    finite = torch.isfinite(value) & torch.isfinite(lower) & torch.isfinite(upper)
    metrics = torch.tensor([violation.max().item(), (~finite).sum().item()], dtype=torch.float64)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    assert not metrics.count_nonzero(), (label, "outside arithmetic interval", metrics.tolist())


def _bf16_interval(lower, upper):
    return lower.bfloat16().double(), upper.bfloat16().double()


def _dot_interval(source, weight, *, output_dtype=torch.bfloat16):
    """BF16 products are exact in FP32; MMA's unspecified rounding permits one full ULP.

    https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
    """
    source, weight = source.double(), weight.double()
    center = source @ weight.transpose(-1, -2)
    absolute = source.abs() @ weight.abs().transpose(-1, -2)
    k = source.shape[-1]
    gamma64 = k * 2. ** -53 / (1 - k * 2. ** -53)
    error = (_gamma(k, rounding=2 * U32) + gamma64) * absolute
    if output_dtype == torch.bfloat16:
        return _bf16_interval(center - error, center + error)
    assert output_dtype == torch.float32
    return center - error, center + error


def _linear_check(source, weight, actual, label):
    lower, upper = _dot_interval(source, weight, output_dtype=actual.dtype)
    _interval(actual, lower, upper, label)
    return lower, upper


def _norm_interval(source, weight, eps):
    """FP32 variance and rsqrt, then BF16 normalized value and BF16 weighted output.

    PTX rsqrt.approx.f32 has relative error at most 2^-22.9. The 2^-22
    allowance also covers the documented two-ULP exp2 bound used below.
    https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt
    """
    source, weight = source.double(), weight.double()
    variance = source.square().mean(-1, keepdim=True)
    error = _gamma(source.shape[-1] + 2) * variance + U32 * eps
    low_denominator, high_denominator = variance + eps - error, variance + eps + error
    assert (low_denominator > 0).all()
    inv_low = high_denominator.rsqrt() * (1 - SFU_REL)
    inv_high = low_denominator.rsqrt() * (1 + SFU_REL)
    lower = torch.minimum(source * inv_low, source * inv_high)
    upper = torch.maximum(source * inv_low, source * inv_high)
    multiplication = U32 * torch.maximum(lower.abs(), upper.abs())
    lower, upper = _bf16_interval(lower - multiplication, upper + multiplication)
    first, second = lower * weight, upper * weight
    return _bf16_interval(torch.minimum(first, second), torch.maximum(first, second))


def _norm_check(source, weight, actual, eps, label):
    lower, upper = _norm_interval(source, weight, eps)
    _interval(actual, lower, upper, label)


def _rotate_interval(lower, upper, positions, inverse):
    angle = positions.clamp_min(0).float()[:, None] * inverse
    shape = (len(positions),) + (1,) * (lower.ndim - 2) + (inverse.numel(),)
    cosine, sine = angle.cos().view(shape).double(), angle.sin().view(shape).double()
    lo0, lo1 = lower[..., ::2], lower[..., 1::2]
    hi0, hi1 = upper[..., ::2], upper[..., 1::2]
    bounds = []
    for a0, a1, b0, b1, c, s in ((lo0, hi0, lo1, hi1, cosine, -sine),
                                 (lo1, hi1, lo0, hi0, cosine, sine)):
        first, second = a0 * c, a1 * c
        third, fourth = b0 * s, b1 * s
        low = torch.minimum(first, second) + torch.minimum(third, fourth)
        high = torch.maximum(first, second) + torch.maximum(third, fourth)
        error = _gamma(3) * (torch.maximum(first.abs(), second.abs())
                            + torch.maximum(third.abs(), fourth.abs()))
        bounds.append(_bf16_interval(low - error, high + error))
    return (torch.stack((bounds[0][0], bounds[1][0]), -1).flatten(-2),
            torch.stack((bounds[0][1], bounds[1][1]), -1).flatten(-2))


def _sum_check(partial, actual, label, *, plan):
    import torch.distributed as dist

    assert plan.query.surface in ("PCIeHierarchicalAllReduce.all_reduce", "PCIeTwoShotBF16.all_reduce")
    # Hierarchy stores island partials in FP32; both transports round only the final BF16 output.
    total, absolute = partial.double().cpu(), partial.double().abs().cpu()
    dist.all_reduce(total)
    dist.all_reduce(absolute)
    error = _gamma(WORLD) * absolute
    lower, upper = _bf16_interval((total - error).to(actual.device), (total + error).to(actual.device))
    _interval(actual, lower, upper, label)


def _swiglu_check(gate_up, actual, label):
    gate, up = gate_up.double().chunk(2, -1)
    silu = F.silu(gate)
    error = (SFU_REL + _gamma(6) * (1 + gate.abs())) * silu.abs() + torch.finfo(torch.float32).tiny
    lower, upper = _bf16_interval(silu - error, silu + error)
    first, second = lower * up, upper * up
    _interval(actual, *_bf16_interval(torch.minimum(first, second), torch.maximum(first, second)), label)


def _attention_check(query, records, actual, label, *, output_roundings=1, lse=None):
    """Bound QK accumulation, online FP32 softmax, BF16 probabilities, and output rounds."""
    if not len(records):
        # Empty native partitions have unspecified scratch; merge masking must exclude it.
        # Keep collective participation identical on empty and nonempty context ranks.
        zero = torch.zeros_like(actual, dtype=torch.float64)
        _interval(zero, zero, zero, label)
        if lse is not None:
            zero = torch.zeros_like(lse, dtype=torch.float64)
            _interval(zero, zero, zero, label + " LSE")
        return
    query, records = query.double(), records.double()
    scale = 192 ** -.5
    scores = query @ records.T * scale
    absolute = query.abs() @ records.abs().T * scale
    score_error = _gamma(query.shape[-1], rounding=2 * U32) * absolute
    score_error += _gamma(5) * scores.abs().amax(-1, keepdim=True)
    maximum = score_error.amax(-1, keepdim=True)
    count, chunks = len(records), (len(records) + 63) // 64
    # Numerator and denominator each traverse at most chunks + 3 exponentials.
    # Eight scalar roundings per record cover sums, rescaling, and normalization.
    softmax_error = _gamma(8 * count + 16) + (2 * chunks + 6) * SFU_REL
    coefficient = (2 * maximum).exp() * (1 + softmax_error) / (1 - softmax_error)
    coefficient *= (1 + U16) ** (output_roundings + 1)
    coefficient *= 1 + _gamma(count + 2 * chunks, rounding=2 * U32)
    probabilities = scores.softmax(-1)
    center = probabilities @ records[:, :512]
    absolute_output = probabilities @ records[:, :512].abs()
    error = (coefficient - 1) * absolute_output
    error += count * torch.finfo(torch.bfloat16).tiny * records[:, :512].abs().amax(0)
    error += BF16_HALF_SUBNORMAL
    _interval(actual, center - error, center + error, label)
    if lse is not None:
        expected = scores.logsumexp(-1)
        allowance = maximum[:, 0] - math.log(1 - softmax_error)
        allowance += SFU_REL * (1 + expected.abs()) + U32 * expected.abs()
        _interval(lse, expected - allowance, expected + allowance, label + " LSE")


def _inverse(config, device):
    text, rope = config.draft.text, config.draft.text["rope_parameters"]
    width = text["qk_rope_head_dim"]
    correction = lambda rotations: width * math.log(
        rope["original_max_position_embeddings"] / (rotations * 2 * math.pi)) / (2 * math.log(rope["rope_theta"]))
    low = max(math.floor(correction(rope["beta_fast"])), 0)
    high = min(math.ceil(correction(rope["beta_slow"])), width - 1)
    interpolation = ((torch.arange(width // 2, dtype=torch.float32, device=device) - low)
                     / (high - low)).clamp(0, 1)
    extrapolation = 1 - interpolation
    wavelengths = rope["rope_theta"] ** (torch.arange(0, width, 2, dtype=torch.float32, device=device) / width)
    return ((1 / (wavelengths * rope["factor"])) * (1 - extrapolation)
            + (1 / wavelengths) * extrapolation)


def _rotate(source, positions, inverse):
    angle = positions.clamp_min(0).float()[:, None] * inverse
    shape = (len(positions),) + (1,) * (source.ndim - 2) + (inverse.numel(),)
    cosine, sine = angle.cos().view(shape), angle.sin().view(shape)
    pairs = source.float().reshape(*source.shape[:-1], -1, 2)
    output = torch.empty_like(pairs)
    output[..., 0] = pairs[..., 0] * cosine - pairs[..., 1] * sine
    output[..., 1] = pairs[..., 0] * sine + pairs[..., 1] * cosine
    return output.flatten(-2).to(source.dtype)


def _accuracy(actual, expected, label, *, rms=1e-5, peak=1e-4):
    import torch.distributed as dist

    actual, expected = actual.float(), expected.float()
    assert actual.shape == expected.shape, label
    scale = expected.square().mean(-1).sqrt().clamp_min(1e-5)
    error = actual - expected
    relative = error.square().mean(-1).sqrt() / scale
    maximum = error.abs().amax(-1) / scale
    metrics = torch.tensor([relative.max().item(), maximum.max().item()], dtype=torch.float64)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    assert torch.isfinite(metrics).all() and metrics[0] <= rms and metrics[1] <= peak, (
        label, "relative_rms", metrics[0].item(), "peak_over_rms", metrics[1].item(), rms, peak)
    return metrics[0].item()


def _metadata(metadata, starts, lengths, pages, *, block=None, tokens=None):
    device = metadata.positions.device
    if block is not None:
        ids = torch.zeros((len(starts), block), dtype=torch.int64, device=device)
        if tokens is not None:
            ids.copy_(torch.tensor(tokens, device=device).view_as(ids))
        metadata.set(torch.tensor(starts, device=device), torch.tensor(lengths, device=device, dtype=torch.int32),
            torch.arange(1, len(starts) + 1, device=device), torch.tensor(pages, device=device, dtype=torch.int32),
            ids, torch.tensor([sum(count > 0 for count in lengths)], device=device))
        return
    spans = lengths if block is None else [block] * len(starts)
    assert sum(spans) == metadata.positions.numel()
    requests, positions = [], []
    for request, (start, count, span) in enumerate(zip(starts, lengths, spans, strict=True)):
        requests.extend([request] * span)
        positions.extend(start + offset if offset < count else -1 for offset in range(span))
    ends = [0]
    for count in lengths:
        ends.append(ends[-1] + count)
    metadata.positions.copy_(torch.tensor(positions, device=device))
    metadata.requests.copy_(torch.tensor(requests, device=device))
    metadata.starts.copy_(torch.tensor(starts, device=device))
    metadata.lengths.copy_(torch.tensor(lengths, device=device))
    metadata.state_rows.copy_(torch.arange(1, len(starts) + 1, device=device))
    metadata.query_start_loc.copy_(torch.tensor(ends, device=device))
    metadata.page_table.copy_(torch.tensor(pages, device=device))
    if len(starts) > 1:
        metadata.token_page_table.copy_(metadata.page_table[metadata.requests])
    metadata.sample_indices.copy_(torch.arange(metadata.sample_indices.numel(), device=device))
    metadata.live_counts.copy_(torch.tensor([sum(lengths), sum(count > 0 for count in lengths)], device=device))
    metadata.token_ids.zero_()
    if tokens is not None:
        metadata.token_ids.copy_(torch.tensor(tokens, device=device))


def _persistent(pool, *, draft=True):
    return {**{name: value for name, value in pool.pages.items() if draft or not name.startswith("draft")},
            "continuations": pool.backing}


def _snapshot(pool, *, draft=True):
    return {name: value.clone() for name, value in _persistent(pool, draft=draft).items()}


def _restore(pool, snapshot):
    for name, value in _persistent(pool).items():
        value.copy_(snapshot[name])


def _unchanged(pool, snapshot):
    for name, saved in snapshot.items():
        value = _persistent(pool)[name]
        torch.testing.assert_close(value.view(torch.uint8), saved.view(torch.uint8), rtol=0, atol=0, msg=name)


def _poison(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            value.fill_(torch.nan)
        else:
            value.fill_(-1)
    elif is_dataclass(value):
        for field in fields(value):
            _poison(getattr(value, field.name))
    elif isinstance(value, dict):
        for child in value.values():
            _poison(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _poison(child)


def _replay(graph, arena, device):
    import torch.distributed as dist

    arena.tensor.fill_(255)
    torch.cuda.synchronize(device)
    dist.barrier()
    before = torch.cuda.memory_stats(device)
    graph.replay()
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_stats(device)
    for field in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
        assert before[field] == after[field], field


def _oracle_weights(draft):
    return dict(projection=_matrix(draft.projection), context_kv=draft.context_kv.float(),
        layers=[dict(attention={key: _matrix(layer.attention[key]) for key in ("latent_gate", "q_b", "output")},
                     gate_up=_matrix(layer.mlp.gate_up), down=_matrix(layer.mlp.down)) for layer in draft.layers])


def _context_check(operator, auxiliary, counts, weights, oracle, before, rank):
    config, draft, metadata = weights.config, weights.draft, operator.metadata
    eps = config.draft.text["rms_norm_eps"]
    _linear_check(auxiliary, oracle["projection"], operator.local, "context projection")
    torch.testing.assert_close(operator.projected, _gather_columns(operator.local), rtol=0, atol=0)
    _norm_check(operator.projected, draft.hidden_norm, operator.normalized, eps, "context normalization")
    _linear_check(operator.normalized, oracle["context_kv"], operator.kv, "context KV projection")
    requests, positions = metadata.requests.tolist(), metadata.positions.tolist()
    starts, accepted, tables = metadata.starts.tolist(), counts.tolist(), metadata.token_page_table.tolist()
    inverse = _inverse(config, auxiliary.device)
    for index, layer in enumerate(draft.layers):
        cache = operator.stages[index][1]
        saved = before[f"draft{index}.latent"].view(CACHE_DTYPE).view_as(cache)
        allowed = torch.zeros(cache.shape[:2], dtype=torch.bool, device=cache.device)
        values = operator.kv.view(len(auxiliary), len(draft.layers), RECORD)[:, index]
        lower, upper = _norm_interval(values[:, :512], layer.attention["kv_norm"], eps)
        key_low, key_high = _rotate_interval(values[:, 512:].double(), values[:, 512:].double(),
                                            metadata.positions, inverse)
        lower, upper = torch.cat((lower, key_low), -1), torch.cat((upper, key_high), -1)
        expected_low, expected_high, observed = [], [], []
        for token, (request, position) in enumerate(zip(requests, positions, strict=True)):
            if position < 0 or not 0 <= position - starts[request] < accepted[request] or position % WORLD != rank:
                continue
            local = position // WORLD
            page = tables[token][local // PAGE]
            assert page >= 0
            allowed[page, local % PAGE] = True
            expected_low.append(lower[token])
            expected_high.append(upper[token])
            observed.append(cache[page, local % PAGE])
        torch.testing.assert_close(cache.view(torch.uint8)[~allowed], saved.view(torch.uint8)[~allowed],
                                   rtol=0, atol=0, msg=f"draft{index} unaccepted or foreign records")
        if observed:
            _interval(torch.stack(observed), torch.stack(expected_low), torch.stack(expected_high),
                      f"draft{index} accepted context")
        else:
            _interval(cache[:1, :1], saved[:1, :1].double(), saved[:1, :1].double(),
                      f"draft{index} no accepted local context")


def _history(cache, starts, tables, rank):
    import torch.distributed as dist

    result = []
    for length, pages in zip(starts, tables, strict=True):
        extent = (length + WORLD - 1) // WORLD
        local = torch.zeros((extent, RECORD * CACHE_DTYPE.itemsize), dtype=torch.uint8)
        count = length // WORLD + (rank < length % WORLD)
        if count:
            offsets = torch.arange(count, device=cache.device)
            page = torch.tensor(pages, device=cache.device)[offsets // PAGE]
            assert (page >= 0).all()
            local[:count] = cache[page, offsets % PAGE].view(torch.uint8).cpu()
        peers = [torch.empty_like(local) for _ in range(WORLD)]
        dist.all_gather(peers, local)
        result.append(torch.stack(peers, 1).flatten(0, 1)[:length].to(cache.device).view(CACHE_DTYPE).float())
    return result


def _proposal_state(metadata, rank):
    starts, requests, positions = metadata.starts.tolist(), metadata.requests.tolist(), metadata.positions.tolist()
    slots, history, current = [], [], []
    for request, position in zip(requests, positions, strict=True):
        prefix = starts[request] // WORLD + (rank < starts[request] % WORLD)
        count = ((position + 1) // WORLD + (rank < (position + 1) % WORLD) - prefix
                 if position >= 0 else 0)
        slots.append(request * PAGE + position // WORLD - prefix
                     if position >= 0 and position % WORLD == rank else -1)
        history.append(prefix if position >= 0 else 0)
        current.append(count)
    device = metadata.positions.device
    return dict(slots=torch.tensor(slots, device=device),
        history_lengths=torch.tensor(history, dtype=torch.int32, device=device),
        current_lengths=torch.tensor(current, dtype=torch.int32, device=device),
        history_pages=metadata.token_page_table, current_pages=metadata.requests[:, None].int())


def _sequence_draft(prepared, metadata, rank):
    """Sequence native primitives without calling a draft, layer, or attention forward."""
    from ignition.recipes.kimi3.dcp import write_records
    from ignition.recipes.kimi31.dspark import merge_attention

    local = prepared.embedding.lookup()
    hidden = prepared.embedding.gather(local).clone()
    values = dict(local_embedding=local.clone(), embedding=hidden.clone(), layers=[])
    state, live = _proposal_state(metadata, rank), metadata.positions >= 0
    for module in prepared.layers:
        operation, projection = module.self_attn, module.self_attn.projected
        normalized = module.input_layernorm(hidden)
        stage = dict(input_norm=normalized.clone())
        projection.project(normalized)
        stage.update(_projected_values(projection))
        for name, value in state.items():
            getattr(operation, name).copy_(value)
            stage[name] = value.clone()
        stage["block_before"] = operation.block.clone()
        write_records(projection.latent, operation.slots, operation.block)
        stage["block"] = operation.block.clone()
        operation.query.copy_(projection.absorbed)
        operation.gather(operation.query.flatten(1))
        stage["gathered"] = operation.gathered.clone()
        operation.history_attention()
        operation.current_attention()
        for name in ("history", "history_lse", "current", "current_lse"):
            stage[name] = getattr(operation, name).clone()
        merge_attention(operation.history, operation.history_lse, operation.current, operation.current_lse,
                        operation.history_lengths, operation.current_lengths, operation.history, operation.history_lse)
        stage.update(merged=operation.history.clone(), merged_lse=operation.history_lse.clone())
        operation.combine(operation.history, operation.history_lse)
        projection.attended.copy_(torch.where(live[:, None, None], projection.attended, 0.))
        partial = projection.finish()
        stage.update(attended=projection.attended.clone(), expanded=projection.expanded.clone(),
                     attention_partial=partial.clone())
        attention_output = operation.reduce(partial)
        stage["attention_output"] = attention_output.clone()
        hidden = hidden + attention_output
        normalized = module.post_attention_layernorm(hidden)
        stage["post_norm"] = normalized.clone()
        mlp_output = module.mlp(normalized)
        stage.update(gate_up=prepared.scratch.mlp.gate_up.clone(),
            intermediate=prepared.scratch.mlp.intermediate.clone(), mlp_partial=prepared.scratch.mlp.output.clone(),
            mlp_output=mlp_output.clone())
        hidden = hidden + mlp_output
        values["layers"].append(stage)
    normalized = prepared.norm(hidden[metadata.sample_indices])
    hidden = torch.where((metadata.lengths > 0).repeat_interleave(BLOCK)[:, None], normalized, 0.)
    prepared.head(hidden)
    values.update(hidden=hidden.clone(), local_logits=prepared.scratch.local_logits.clone(),
                  logits=prepared.scratch.logits.clone())
    return values


def _check_draft_boundaries(weights, oracle, metadata, pool, prepared, values):
    """Check actual primitive inputs and outputs against checkpoint math and ownership."""
    config, draft = weights.config, weights.draft
    eps, tokens = config.draft.text["rms_norm_eps"], metadata.token_ids.numel()
    live = metadata.positions >= 0
    local = values["local_embedding"]
    expected_embedding = draft.embedding[metadata.token_ids].clone()
    expected_embedding[~live] = 0
    torch.testing.assert_close(local, expected_embedding, rtol=0, atol=0, msg="checkpoint embedding lookup")
    hidden = values["embedding"]
    torch.testing.assert_close(hidden, _gather_columns(local), rtol=0, atol=0, msg="embedding shard assembly")
    inverse = _inverse(config, hidden.device)
    starts, tables = metadata.starts.tolist(), metadata.page_table.tolist()
    requests, positions = metadata.requests.tolist(), metadata.positions.tolist()
    rank, state = weights.shard.rank, _proposal_state(metadata, weights.shard.rank)
    slots = state["slots"].tolist()
    history_lengths, current_lengths = state["history_lengths"].tolist(), state["current_lengths"].tolist()
    assert len(values["layers"]) == len(draft.layers), "draft layer boundary count"
    for index, (layer, matrices, module, captured) in enumerate(zip(
            draft.layers, oracle["layers"], prepared.layers, values["layers"], strict=True)):
        label, stage = f"layer{index}", SimpleNamespace(**captured)
        _norm_check(hidden, layer.input_norm, stage.input_norm, eps, label + " input norm and residual")
        _linear_check(stage.input_norm, matrices["attention"]["latent_gate"], stage.local,
                      label + " latent projection")
        gathered = _gather_columns(stage.local).view(tokens, WORLD, -1)
        assembled = torch.cat((gathered[:, :, :96].flatten(1), gathered[:, :, 96:132].flatten(1)), -1)
        torch.testing.assert_close(stage.input, assembled, rtol=0, atol=0, msg=label + " latent shard assembly")
        _norm_check(stage.input[:, :1536], layer.attention["query_norm"], stage.query_norm, eps, label + " query norm")
        _norm_check(stage.input[:, 1536:2048], layer.attention["kv_norm"], stage.latent[:, :512], eps, label + " KV norm")
        query_low, query_high = _dot_interval(stage.query_norm, matrices["attention"]["q_b"])
        query_low, query_high = query_low.view(tokens, 6, 192), query_high.view(tokens, 6, 192)
        rope_low, rope_high = _rotate_interval(query_low[:, :, 128:], query_high[:, :, 128:], metadata.positions, inverse)
        query_low, query_high = torch.cat((query_low[:, :, :128], rope_low), -1), torch.cat((query_high[:, :, :128], rope_high), -1)
        _interval(stage.query, query_low, query_high, label + " Q projection and rotary")
        raw_key = stage.input[:, -64:].double()
        _interval(stage.latent[:, 512:], *_rotate_interval(raw_key, raw_key, metadata.positions, inverse),
                  label + " key rotary")
        lower, upper = _dot_interval(stage.query[:, :, :128].transpose(0, 1), layer.attention["key"].transpose(-1, -2))
        _interval(stage.absorbed[:, :, :512], lower.transpose(0, 1), upper.transpose(0, 1), label + " absorbed query")
        torch.testing.assert_close(stage.absorbed[:, :, 512:], stage.query[:, :, 128:], rtol=0, atol=0,
                                   msg=label + " query rotary passthrough")
        replicated = _gather_columns(stage.latent).view(tokens, WORLD, RECORD)
        torch.testing.assert_close(replicated, stage.latent[:, None].expand_as(replicated), rtol=0, atol=0,
                                   msg=label + " replicated latent")
        for name, expected in state.items():
            torch.testing.assert_close(getattr(stage, name), expected, rtol=0, atol=0, msg=label + " " + name)
        touched = torch.zeros(stage.block.shape[:2], dtype=torch.bool, device=hidden.device)
        for token, slot in enumerate(slots):
            if slot >= 0:
                touched[slot // PAGE, slot % PAGE] = True
                torch.testing.assert_close(stage.block[slot // PAGE, slot % PAGE].view(torch.uint8),
                    stage.latent[token].view(torch.uint8), rtol=0, atol=0, msg=label + " private owned record")
        torch.testing.assert_close(stage.block.view(torch.uint8)[~touched], stage.block_before.view(torch.uint8)[~touched],
                                   rtol=0, atol=0, msg=label + " untouched private records")
        torch.testing.assert_close(stage.gathered.flatten(1), _gather_columns(stage.absorbed.flatten(1)),
                                   rtol=0, atol=0, msg=label + " gathered query heads")
        cache = pool.draft_attention(index)
        for row, position in enumerate(positions):
            if position < 0:
                continue
            request, count = requests[row], history_lengths[row]
            local_positions = torch.arange(count, device=hidden.device)
            pages = metadata.page_table[request, local_positions // PAGE]
            history = cache[pages, local_positions % PAGE]
            current = stage.block[request, :current_lengths[row]]
            _attention_check(stage.gathered[row], history, stage.history[row], label + f" history row{row}",
                             lse=stage.history_lse[row])
            _attention_check(stage.gathered[row], current, stage.current[row], label + f" current row{row}",
                             lse=stage.current_lse[row])
            _attention_check(stage.gathered[row], torch.cat((history, current)), stage.merged[row],
                             label + f" merged attention row{row}", output_roundings=2, lse=stage.merged_lse[row])
        h_live, c_live = stage.history_lengths[:, None] > 0, stage.current_lengths[:, None] > 0
        h = torch.where(h_live, stage.history_lse.double(), -torch.inf)
        c = torch.where(c_live, stage.current_lse.double(), -torch.inf)
        combined_lse = torch.logaddexp(h, c)
        h_weight, c_weight = torch.where(h_live, (h - combined_lse).exp(), 0.), torch.where(c_live, (c - combined_lse).exp(), 0.)
        expected = (torch.where(h_live[..., None], stage.history.double(), 0.) * h_weight[..., None]
                    + torch.where(c_live[..., None], stage.current.double(), 0.) * c_weight[..., None])
        absolute = (torch.where(h_live[..., None], stage.history.double().abs(), 0.) * h_weight[..., None]
                    + torch.where(c_live[..., None], stage.current.double().abs(), 0.) * c_weight[..., None])
        merge_error = (_gamma(16) + 4 * SFU_REL) * absolute
        _interval(stage.merged, *_bf16_interval(expected - merge_error, expected + merge_error), label + " partition merge")
        empty = (stage.history_lengths + stage.current_lengths) == 0
        assert torch.isneginf(stage.merged_lse[empty]).all(), label + " empty merged LSE"
        histories = _history(cache, starts, tables, rank)
        for row, position in enumerate(positions):
            if position < 0:
                assert not stage.attended[row].count_nonzero(), label + " inactive attention output"
                continue
            request, first = requests[row], metadata.query_start_loc[requests[row]].item()
            records = torch.cat((histories[request], stage.latent[first:row + 1].float()))
            _attention_check(stage.absorbed[row], records, stage.attended[row],
                             label + f" global attention row{row}", output_roundings=3)
        lower, upper = _dot_interval(stage.attended.transpose(0, 1), layer.attention["value"].transpose(-1, -2))
        _interval(stage.expanded, lower.transpose(0, 1).flatten(1), upper.transpose(0, 1).flatten(1), label + " value expansion")
        _linear_check(stage.expanded, matrices["attention"]["output"], stage.attention_partial, label + " output projection")
        _sum_check(stage.attention_partial, stage.attention_output, label + " attention reduction", plan=module.self_attn.reduce.plan)
        hidden = hidden + stage.attention_output
        _norm_check(hidden, layer.post_attention_norm, stage.post_norm, eps, label + " post-attention norm and residual")
        _linear_check(stage.post_norm, matrices["gate_up"], stage.gate_up, label + " gate/up projection")
        _swiglu_check(stage.gate_up, stage.intermediate, label + " SwiGLU")
        _linear_check(stage.intermediate, matrices["down"], stage.mlp_partial, label + " down projection")
        _sum_check(stage.mlp_partial, stage.mlp_output, label + " MLP reduction", plan=module.mlp.reduce.plan)
        hidden = hidden + stage.mlp_output
    lower, upper = _norm_interval(hidden[metadata.sample_indices], draft.norm, eps)
    active = (metadata.lengths > 0).repeat_interleave(BLOCK)[:, None]
    _interval(values["hidden"], torch.where(active, lower, 0.), torch.where(active, upper, 0.), "final norm and residual")
    _linear_check(values["hidden"], weights.head.float(), values["local_logits"], "vocabulary projection")
    torch.testing.assert_close(values["logits"], _gather_columns(values["local_logits"]), rtol=0, atol=0,
                               msg="vocabulary shard assembly")
    return values["hidden"], values["logits"]


def _compose_and_check(weights, oracle, metadata, pool, *, prepared, bound=None):
    if bound is None:
        values = _sequence_draft(prepared, metadata, weights.shard.rank)
    else:
        boundaries = _DraftBoundaries(prepared)
        backend, original = bound.backend, bound.backend.invocations
        backend.invocations = [_ObservedInvocation(invocation, boundaries) for invocation in original]
        try:
            _poison(prepared.scratch.scratch_tensors())
            bound()
        finally:
            backend.invocations = original
        values = boundaries.finish()
    return _check_draft_boundaries(weights, oracle, metadata, pool, prepared, values)


@torch.no_grad()
def _rank(rank, rendezvous, cache_root, compile_decoder):
    import traceback
    import torch.distributed as dist

    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
        create_all_reduce_runtime, create_head_exchange)
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.config import ModelConfig
    from ignition.recipes.kimi3.dspark import PreparedMarkov
    from ignition.recipes.kimi3.pool import ContinuationPool
    from ignition.recipes.kimi31.dcp import ContextShards
    from ignition.recipes.kimi31.dspark import (DraftScratch, PreparedContext, PreparedDraft, Rotary,
                                               inverse_frequencies, rotary_gain)
    from ignition.recipes.kimi31.dspark_config import DSparkConfig
    from ignition.recipes.kimi31.dspark_weights import DraftWeights

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=20))
    runtimes, graphs, compiled = [], [], {}
    try:
        with CoreStorage(rank=rank, world=WORLD, device=rank) as storage:
            from ignition.recipes.kimi3.dcp import write_records

            records = torch.linspace(-1024, 1024, 3 * RECORD, device=device).view(3, RECORD).bfloat16()
            slots = torch.tensor([1, -1, 0], device=device)
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                cache = torch.zeros((1, 3, RECORD), device=device, dtype=dtype)
                write_records(records, slots, cache)
                wanted = records[[2, 0]]
                if dtype == torch.float8_e4m3fn:
                    wanted = wanted.float().clamp(-448, 448).to(dtype)
                torch.testing.assert_close(cache[0, :2].view(torch.uint8), wanted.view(torch.uint8), rtol=0, atol=0)
                assert not cache[0, 2].view(torch.uint8).count_nonzero()
            shard = TensorParallelShard(rank, WORLD)
            target = ModelConfig.load(TARGET)
            config = replace(target, draft=DSparkConfig.load(DRAFT, target))
            assert config.draft.auxiliary_layers == (76, 80, 84, 88, 92)
            assert config.draft.auxiliary_stream == "prefix" and config.draft.layers == 5
            with DirectWeightLoader(inspect_indexed_checkpoint(TARGET), device,
                    prefixes=("language_model.model.embed_tokens.", "language_model.lm_head.")) as loader:
                embedding = loader.load_tensor("language_model.model.embed_tokens.weight", extent=(
                    slice(None), slice(*shard.partition(config.hidden, alignment=64))))
                head = loader.load_tensor("language_model.lm_head.weight", extent=(
                    slice(*shard.partition(config.vocabulary, alignment=64)), slice(None)))
            draft = DraftWeights.load(DRAFT, config, device, shard, target_embedding=embedding,
                                      target_head=head, online_mxfp8=True)
            assert draft.embedding is embedding and draft.head is None
            weights = SimpleNamespace(config=config, draft=draft, head=head, shard=shard)
            oracle = _oracle_weights(draft)
            assert rotary_gain(config) == pytest.approx(1., abs=2e-16)
            inverse = _inverse(config, device)
            torch.testing.assert_close(inverse_frequencies(config, device), inverse, rtol=0, atol=0)
            positions = torch.tensor([0, 2047, 2048, 32767, 32768, 65535, 1048569, 1048575], device=device)
            rotary_input = torch.linspace(-1, 1, len(positions) * 6 * 64, device=device).view(-1, 6, 64).bfloat16()
            query, key = rotary_input.clone(), rotary_input[:, 0].clone()
            rotary = Rotary(config, device)
            rotary(query, key, positions)
            torch.testing.assert_close(query, _rotate(rotary_input, positions, inverse), rtol=0, atol=0)
            torch.testing.assert_close(key, _rotate(rotary_input[:, 0], positions, inverse), rtol=0, atol=0)
            rotary_forward = _RotaryForward(rotary, rotary_input, positions, query, key)
            pool = ContinuationPool.for_model(config, WORLD, 3, 7, device, speculative_tokens=7)
            for index in range(5):
                pool.draft_attention(index).view(torch.uint8).fill_(65)
            pristine = _snapshot(pool)
            group = dist.group.WORLD
            coordinator = CollectivePreparation(group)

            def runtime(factory, **kwargs):
                value = factory(group, device, ipc=storage.peer_storage, **kwargs)
                runtimes.append(value)
                return value

            reduce = runtime(create_all_reduce_runtime, max_tokens=14, hidden_size=config.hidden)
            vocabulary = runtime(create_all_gather_runtime, max_rows=14, columns=config.vocabulary)
            context_runtime = runtime(create_all_gather_runtime, max_rows=14, columns=config.hidden)
            embedding_runtime = runtime(create_all_gather_runtime, max_rows=14, columns=config.hidden)
            markov_runtime = runtime(create_all_gather_runtime, max_rows=2, columns=2 * config.vocabulary)
            heads = runtime(create_head_exchange, max_rows=14, heads=96, head_dim=512, query_head_dim=RECORD)
            latents = runtime(create_head_exchange, max_rows=14, heads=WORLD, head_dim=136)
            context = SimpleNamespace(shards=ContextShards.on(rank, WORLD, device), heads=heads,
                                      coordinator=coordinator)
            initial = Metadata.allocate(2052, 2, 2, device, samples=2052)
            commit = Metadata.allocate(14, 2, 2, device, samples=14)
            single = Metadata.allocate(2, 2, 2, device, samples=2)
            proposal = Metadata.allocate(14, 2, 2, device, samples=14)
            short = Metadata.allocate(7, 1, 2, device, samples=7)
            pages = [[3, 1], [5, 2]]
            _metadata(initial, [0, 0], [2047, 5], pages)
            _metadata(commit, [2047, 5], [7, 7], pages, block=7)
            _metadata(single, [2047, 5], [1, 1], pages, block=1)
            anchors = [1205, *([config.draft.noise_token] * 6), 921, *([config.draft.noise_token] * 6)]
            _metadata(proposal, [2047, 5], [7, 7], pages, block=7, tokens=anchors)
            _metadata(short, [2047], [7], pages[:1], block=7, tokens=anchors[:7])
            generator = torch.Generator(device=device).manual_seed(3131)
            auxiliary = torch.randn((2052, 5 * config.hidden), generator=generator,
                                    dtype=torch.bfloat16, device=device) * .2
            feedback = torch.randn((14, 5 * config.hidden), generator=generator,
                                   dtype=torch.bfloat16, device=device) * .2
            one_feedback = feedback[[0, 7]].contiguous()
            initial_counts = torch.tensor([2047, 5], dtype=torch.int32, device=device)
            counts = torch.tensor([2, 0], dtype=torch.int32, device=device)
            one_counts = torch.ones(2, dtype=torch.int32, device=device)
            predecessor = torch.tensor([1205, 921], device=device)
            with RecordedPreparationSession(device=device, checkpoint=str(DRAFT),
                    workload_schema="kimi31-dspark-tp16", numerical_policy="bf16-a16-latent",
                    autotune_effort="off", compile_workers=WORLD, coordinator=coordinator) as session:
                arena = ScratchArena(device)
                session.register_scratch(arena)
                buffers = {}
                common = dict(session=session, workspace=arena, runtime=context_runtime,
                              coordinator=coordinator, buffers=buffers, context=context)
                initial_context = PreparedContext(weights, auxiliary, initial, pool, name="initial", **common)
                commit_context = PreparedContext(weights, feedback, commit, pool, name="commit", **common)
                single_context = PreparedContext(weights, one_feedback, single, pool, name="single", **common)
                assert initial_context.projected.data_ptr() == commit_context.projected.data_ptr()
                assert commit_context.kv.data_ptr() == single_context.kv.data_ptr()
                scratch = DraftScratch.allocate(config, shard, 14, BLOCK, device)
                short_scratch = scratch.rows(7, 7)
                assert scratch.embedding.data_ptr() == short_scratch.embedding.data_ptr()
                kwargs = dict(block=BLOCK, runtime=reduce, vocabulary_runtime=vocabulary,
                    embedding_runtime=embedding_runtime, coordinator=coordinator, session=session,
                    workspace=arena, context=context, latents=latents)
                propose = PreparedDraft(weights, scratch, proposal, pool, name="proposal14", **kwargs)
                propose_short = PreparedDraft(weights, short_scratch, short, pool, name="proposal7", **kwargs)
                assert propose.layers[0].self_attn.block.data_ptr() == propose_short.layers[0].self_attn.block.data_ptr()
                assert propose.layers[0].self_attn.history.data_ptr() == propose_short.layers[0].self_attn.history.data_ptr()
                markov = PreparedMarkov(weights, predecessor, runtime=markov_runtime,
                    coordinator=coordinator, session=session, name="markov512")
                session.tune(show_progress=False)
                functions = dict(
                    initial=_ContextForward(initial_context, auxiliary, initial_counts),
                    commit=_ContextForward(commit_context, feedback, counts),
                    single=_ContextForward(single_context, one_feedback, one_counts),
                    proposal14=propose, proposal7=propose_short, markov512=markov, rotary=rotary_forward)
                signatures = dict(
                    initial=GraphSignature("prefill", 2, auxiliary.shape[0]),
                    commit=GraphSignature("speculative", 2, BLOCK),
                    single=GraphSignature("decode", 2, 1),
                    proposal14=GraphSignature("speculative", 2, BLOCK),
                    proposal7=GraphSignature("speculative", 1, BLOCK),
                    markov512=GraphSignature("decode", 2, 1),
                    rotary=GraphSignature("decode", len(positions), 1))
                stream = torch.cuda.Stream(device=device)
                graph_pool = storage.graph_pool.id
                captured, compiled = capture_forwards(functions, signatures,
                    session=session, device=device, stream=stream,
                    pool=graph_pool, compile_decoder=compile_decoder, coordinator=coordinator,
                    cache_root=Path(cache_root),
                    warmup_storage=storage, report_catalog=False)
                graphs.extend(captured.values())
                assert set(compiled) == (set(functions) if compile_decoder else set())
                session.freeze()
                query.fill_(torch.nan)
                key.fill_(torch.nan)
                _replay(graphs[6], arena, device)
                for value, source, label in ((query, rotary_input, "long-position query rotary"),
                                             (key, rotary_input[:, 0], "long-position key rotary")):
                    lower, upper = _rotate_interval(source.double(), source.double(), positions, inverse)
                    _interval(value, lower, upper, label)
                if compile_decoder:
                    rotary_query, rotary_key = query.clone(), key.clone()
                    compiled["rotary"]()
                    _accuracy(query, rotary_query, "compiled rotary query versus graph", rms=1e-5, peak=1e-4)
                    _accuracy(key, rotary_key, "compiled rotary key versus graph", rms=1e-5, peak=1e-4)
                _restore(pool, pristine)
                _replay(graphs[0], arena, device)
                _context_check(initial_context, auxiliary, initial_counts, weights, oracle, pristine, rank)
                _unchanged(pool, {name: value for name, value in pristine.items() if not name.startswith("draft")})
                baseline = _snapshot(pool)
                nondraft = _snapshot(pool, draft=False)
                if compile_decoder:
                    compiled["initial"]()
                    _unchanged(pool, baseline)

                for operation, metadata, output, graph, active in (
                        (propose, proposal, scratch, graphs[3], [7, 7]),
                        (propose_short, short, short_scratch, graphs[4], [7]),
                        (propose, proposal, scratch, graphs[3], [7, 0]),
                        (propose, proposal, scratch, graphs[3], [0, 7]),
                        (propose, proposal, scratch, graphs[3], [0, 0])):
                    starts = [2047, 5][:len(active)]
                    mapped = [[3, -1], [5, -1]][:len(active)]
                    _metadata(metadata, starts, active, mapped, block=7, tokens=anchors[:len(active) * 7])
                    _poison(output.scratch_tensors())
                    _replay(graph, arena, device)
                    actual_hidden, actual_logits = output.hidden.clone(), output.logits.clone()
                    bound = compiled.get("proposal14" if operation is propose else "proposal7")
                    if bound is None:
                        operation()
                        _accuracy(output.hidden, actual_hidden, f"eager versus graph {active} hidden", rms=1e-5, peak=1e-4)
                        _accuracy(output.logits, actual_logits, f"eager versus graph {active} logits", rms=1e-5, peak=1e-4)
                    _unchanged(pool, baseline)
                    checked_hidden, checked_logits = _compose_and_check(weights, oracle, metadata, pool,
                        prepared=operation, bound=bound)
                    _unchanged(pool, baseline)
                    reference = "compiled callable" if compile_decoder else "native composition"
                    _accuracy(actual_hidden, checked_hidden, f"graph versus {reference} {active} hidden", rms=1e-5, peak=1e-4)
                    _accuracy(actual_logits, checked_logits, f"graph versus {reference} {active} logits", rms=1e-5, peak=1e-4)
                    assert torch.isfinite(actual_logits).all()
                    if active == [0, 0]:
                        assert not actual_hidden.count_nonzero() and not actual_logits.count_nonzero()
                    if active == [7, 0]:
                        assert not actual_hidden[7:].count_nonzero() and not actual_logits[7:].count_nonzero()
                    if active == [0, 7]:
                        assert not actual_hidden[:7].count_nonzero() and not actual_logits[:7].count_nonzero()
                    if active == [7, 7]:
                        metadata.token_ids[4:7].copy_(torch.tensor([4283, 8002, 111], device=device))
                        _poison(output.scratch_tensors())
                        _replay(graph, arena, device)
                        torch.testing.assert_close(output.hidden[:4], actual_hidden[:4], rtol=0, atol=0)
                        torch.testing.assert_close(output.logits[:4], actual_logits[:4], rtol=0, atol=0)
                        assert not torch.equal(output.hidden[4:7], actual_hidden[4:7])
                        _unchanged(pool, baseline)
                        suffix_hidden, suffix_logits = output.hidden.clone(), output.logits.clone()
                        wanted_hidden, wanted_logits = _compose_and_check(weights, oracle, metadata, pool,
                            prepared=operation, bound=bound)
                        _unchanged(pool, baseline)
                        _accuracy(suffix_hidden, wanted_hidden, "causal suffix hidden", rms=1e-5, peak=1e-4)
                        _accuracy(suffix_logits, wanted_logits, "causal suffix logits", rms=1e-5, peak=1e-4)

                for accepted in ([2, 0], [0, 3], [7, 0]):
                    _restore(pool, baseline)
                    counts.copy_(torch.tensor(accepted, dtype=torch.int32, device=device))
                    _poison(commit_context.scratch_tensors())
                    _replay(graphs[1], arena, device)
                    _context_check(commit_context, feedback, counts, weights, oracle, baseline, rank)
                    _unchanged(pool, nondraft)
                    committed = _snapshot(pool)
                    if compile_decoder:
                        compiled["commit"]()
                        _unchanged(pool, committed)
                    starts = [2047 + accepted[0], 5 + accepted[1]]
                    _metadata(proposal, starts, [7, 7], pages, block=7, tokens=anchors)
                    _poison(scratch.scratch_tensors())
                    _replay(graphs[3], arena, device)
                    _unchanged(pool, committed)
                    captured_hidden, captured_logits = scratch.hidden.clone(), scratch.logits.clone()
                    expected_hidden, expected_logits = _compose_and_check(weights, oracle, proposal, pool,
                        prepared=propose, bound=compiled.get("proposal14"))
                    _unchanged(pool, committed)
                    _accuracy(captured_hidden, expected_hidden, f"after accepted {accepted} hidden", rms=1e-5, peak=1e-4)
                    _accuracy(captured_logits, expected_logits, f"after accepted {accepted} logits", rms=1e-5, peak=1e-4)

                _restore(pool, baseline)
                counts.fill_(1)
                _replay(graphs[1], arena, device)
                _context_check(commit_context, feedback, counts, weights, oracle, baseline, rank)
                if compile_decoder:
                    committed = _snapshot(pool)
                    compiled["commit"]()
                    _unchanged(pool, committed)
                _restore(pool, baseline)
                _replay(graphs[2], arena, device)
                _context_check(single_context, one_feedback, one_counts, weights, oracle, baseline, rank)
                _unchanged(pool, nondraft)
                resumed_cache = _snapshot(pool)
                if compile_decoder:
                    compiled["single"]()
                    _unchanged(pool, resumed_cache)
                _metadata(short, [2048], [7], pages[:1], block=7, tokens=anchors[:7])
                _poison(short_scratch.scratch_tensors())
                _replay(graphs[4], arena, device)
                _unchanged(pool, resumed_cache)
                captured_hidden, captured_logits = short_scratch.hidden.clone(), short_scratch.logits.clone()
                expected_hidden, expected_logits = _compose_and_check(weights, oracle, short, pool,
                    prepared=propose_short, bound=compiled.get("proposal7"))
                _unchanged(pool, resumed_cache)
                _accuracy(captured_hidden, expected_hidden, "width-one resume hidden", rms=1e-5, peak=1e-4)
                _accuracy(captured_logits, expected_logits, "width-one resume logits", rms=1e-5, peak=1e-4)
                for values in ([1205, 921], [163839, 2048]):
                    predecessor.copy_(torch.tensor(values, device=device))
                    markov.local.fill_(torch.nan)
                    markov.logits.fill_(torch.nan)
                    _replay(graphs[5], arena, device)
                    _linear_check(draft.markov_embedding[predecessor], draft.markov_head.float(),
                                  markov.local, "rank512 Markov projection")
                    torch.testing.assert_close(markov.logits, _gather_columns(markov.local), rtol=0, atol=0)
                    if compile_decoder:
                        markov_local, markov_logits = markov.local.clone(), markov.logits.clone()
                        compiled["markov512"]()
                        _accuracy(markov.local, markov_local, "compiled Markov local versus graph", rms=1e-5, peak=1e-4)
                        _accuracy(markov.logits, markov_logits, "compiled Markov logits versus graph", rms=1e-5, peak=1e-4)
                    _unchanged(pool, resumed_cache)
                for graph in graphs:
                    graph.reset()
                graphs.clear()
                compiled.clear()
            for value in reversed(runtimes):
                value.close()
            runtimes.clear()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    finally:
        for graph in graphs:
            graph.reset()
        compiled.clear()
        for value in reversed(runtimes):
            value.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("compile_decoder", [False, True], ids=["eager", "compiled"])
def test_checkpoint_dspark_compositional_arithmetic_and_replay_on_tp16(tmp_path, compile_decoder):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == WORLD, "assign exactly sixteen CUDA devices"
    if not (TARGET / "config.json").is_file() or not (DRAFT / "model.safetensors").is_file():
        pytest.skip("the Kimi K3.1 target and Inferact DSpark checkpoints are required")
    mp.spawn(_rank, args=((tmp_path / "dspark-tp16").as_uri(),
                         str(tmp_path / "forwards"), compile_decoder), nprocs=WORLD, join=True)
