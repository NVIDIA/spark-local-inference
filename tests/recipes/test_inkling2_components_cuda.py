# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Inkling 2 components of one rank on checkpoint layers 0, 5 and 40.

Requires `FLARE_RUN_CUDA_TESTS=1` and the checkpoint (`tests/recipes/inkling2_checkpoint.py`).
Inputs are the canonical decoder's activations on a 1408-token prompt.
"""

import os

import pytest
import torch

from tests.recipes.inkling2_checkpoint import Checkpoint, reference_activations

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an assigned CUDA device")

LAYERS = (0, 5, 40)


@pytest.fixture(scope="module")
def checkpoint():
    return Checkpoint(torch.device("cuda", torch.cuda.current_device()))


@pytest.fixture(scope="module")
def activations(request, checkpoint):
    return reference_activations(request, checkpoint, LAYERS)


@torch.no_grad()
def test_reference_activations_cover_the_tested_layers(activations, checkpoint):
    tokens = activations["ids"].numel()
    for layer in LAYERS:
        entry = activations[layer]
        assert len(entry["blocks"]) == (layer + checkpoint.config.text.residual_block - 1) // checkpoint.config.text.residual_block if layer else not entry["blocks"]
        for value in (*entry["blocks"], entry["partial"], entry["mlp_input"]):
            assert value.shape == (tokens, checkpoint.config.text.hidden)
            assert bool(torch.isfinite(value.float()).all())


EXPERT_ROLES = ("w13", "w13_scale", "w2", "w2_scale", "w13_amax", "w2_amax", "w13_denominator", "w2_denominator")


def load_rank(checkpoint, layer, rank, *, experts):
    """A rank's placed tensors of `layer` by role; expert roles only when `experts`."""
    placements = checkpoint.layout.layer(layer)
    names = {p.name for role, p in placements.items() if experts or role not in EXPERT_ROLES}
    with checkpoint.loader(names) as loader:
        return {role: loader.load(p.name, rank=rank) for role, p in placements.items() if p.name in names}


def reference_weights(placed):
    """`ref.LayerWeights` of placed tensors; absent expert roles are empty."""
    from ignition.reference import inkling2 as ref

    empty = torch.empty(0)
    values = dict(placed)
    for role in ("w13", "w13_scale", "w2", "w2_scale", "router", "router_bias"):
        values.setdefault(role, empty)
    return ref.LayerWeights(**values)


def relative_l2(actual, expected):
    actual, expected = actual.double(), expected.double()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


def cosine(actual, expected):
    actual, expected = actual.double().flatten(), expected.double().flatten()
    return float(actual @ expected / (actual.norm() * expected.norm()).clamp_min(1e-30))


def report(label, actual, expected, bound=None):
    difference = (actual.double() - expected.double()).abs()
    line = (f"{label}: max {float(difference.max()):.3g}, relative L2 {relative_l2(actual, expected):.3e}, "
            f"cosine {cosine(actual, expected):.7f}, reference RMS {float(expected.double().square().mean().sqrt()):.3g}")
    if bound is not None:
        line += f", largest difference/bound {float((difference / bound.clamp_min(1e-300)).max()):.3f}"
    print(line)
    return difference


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("rank", (0, 3))
@torch.no_grad()
def test_prepared_attention_matches_the_canonical_rank_attention(checkpoint, activations, layer, rank):
    """Chunked prefill, a forked continuation row and batched decode against one canonical prefill.

    Request A prefills 1100 then 305 tokens and decodes three; request B is A's
    continuation row and pages copied after the first chunk, decoding from position
    1100. The decode graph's third row is inactive. Every gated attention row is
    bounded by `inkling2_bounds.attention_output_bound` and every partial output row
    by the propagated output-projection bound.
    """
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.inkling2.attention import LayerCache, PreparedAttention, attention_weights
    from ignition.recipes.inkling2.pool import ContinuationPool, RankGeometry
    from ignition.reference import inkling2 as ref
    from tests.recipes.inkling2_bounds import (attention_output_bound, partial_bound, reference_parts)

    device = checkpoint.device
    text = checkpoint.config.text
    placed = load_rank(checkpoint, layer, rank, experts=False)
    weights = reference_weights(placed)
    entry = activations[layer]
    state = ref.ResidualState(tuple(entry["blocks"]), entry["partial"])
    streams = ref.normalized_streams(weights, text, state)
    tokens = streams[0].shape[0]
    positions = torch.arange(tokens, device=device)
    parts = reference_parts(weights, text, streams, positions)
    local = text.local(layer)
    geometry = RankGeometry.of(text, 4)
    serving = attention_weights(placed, local=local, heads=geometry.heads, kv_heads=geometry.kv_heads,
                                dim=text.head_dim, width=text.convolution_width)
    pool = ContinuationPool.allocate(text, geometry, 4, 32, device)
    prefill_rows, decode_rows, blocks = 1152, 3, 12
    prefill = Metadata.allocate(prefill_rows, 1, blocks, device)
    decode = Metadata.allocate(decode_rows, decode_rows, blocks, device)
    arena = ScratchArena(device)
    inputs = {mode: tuple(torch.zeros(rows, text.hidden, dtype=torch.bfloat16, device=device) for _ in streams)
              for mode, rows in (("prefill", prefill_rows), ("decode", decode_rows))}
    with RecordedPreparationSession(device=device, checkpoint="inkling-2-small", workload_schema="inkling2-components-v1",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        modules = {mode: PreparedAttention(layer, serving, text, LayerCache.of(pool, layer), metadata, mode=mode,
                                           tokens=rows, session=session, workspace=arena, name=f"layer{layer}.{mode}")
                   for mode, metadata, rows in (("prefill", prefill, prefill_rows), ("decode", decode, decode_rows))}
        session.tune()
        arena.release_preparation()
        graphs, outputs = {}, {}
        for mode, metadata in (("prefill", prefill), ("decode", decode)):
            compiled = compile_forward(lambda mode=mode, metadata=metadata: modules[mode].attend(inputs[mode], metadata))
            pool.backing.zero_()
            for value in pool.pages.values():
                value.zero_()
            graphs[mode] = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graphs[mode]):
                outputs[mode] = compiled()
        session.freeze()
        pool.backing.zero_()
        for value in pool.pages.values():
            value.zero_()
        untouched = pool.backing[3].clone()
        null_row = pool.backing[0].clone()

        def pages(*indices):
            table = torch.full((blocks,), -1, dtype=torch.int32, device=device)
            table[:len(indices)] = torch.tensor(indices, dtype=torch.int32, device=device)
            return table

        table_a = pages(*range(12))
        table_b = pages(*range(8), 20, 21, 22, 23)
        gated = torch.zeros(tokens, geometry.heads * text.head_dim, dtype=torch.bfloat16, device=device)
        partial = torch.zeros(tokens, text.hidden, dtype=torch.bfloat16, device=device)
        decoded = {}

        def run_prefill(start, length):
            for source, value in zip(inputs["prefill"], streams, strict=True):
                source.zero_()
                source[:length].copy_(value[start:start + length])
            prefill.set(torch.tensor([start], device=device), torch.tensor([length], dtype=torch.int32, device=device),
                        torch.tensor([1], device=device), table_a.view(1, -1),
                        torch.zeros(1, prefill_rows, dtype=torch.int64, device=device),
                        torch.tensor([1], device=device))
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graphs["prefill"].replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
            gated[start:start + length] = outputs["prefill"][0][:length]
            partial[start:start + length] = outputs["prefill"][1][:length]
            assert not outputs["prefill"][1][length:].count_nonzero()

        run_prefill(0, 1100)
        # Request B continues from a copy of A's continuation row and partial page.
        pool.backing[2].copy_(pool.backing[1])
        for value in pool.pages.values():
            value[20].copy_(value[8])
        run_prefill(1100, 305)
        for step in range(3):
            requests = ((1405 + step, 1, table_a), (1100 + step, 2, table_b), (0, 0, table_a))
            for row, (position, _, _) in enumerate(requests):
                live = row < 2
                for source, value in zip(inputs["decode"], streams, strict=True):
                    source[row].copy_(value[position] if live else torch.zeros_like(value[0]))
            decode.set(torch.tensor([r[0] for r in requests], device=device),
                       torch.tensor([1, 1, 0], dtype=torch.int32, device=device),
                       torch.tensor([r[1] for r in requests], device=device),
                       torch.stack([r[2] for r in requests]),
                       torch.zeros(decode_rows, 1, dtype=torch.int64, device=device),
                       torch.tensor([2], device=device))
            graphs["decode"].replay()
            torch.cuda.synchronize(device)
            for row in range(2):
                decoded[requests[row][0]] = (outputs["decode"][0][row].clone(), outputs["decode"][1][row].clone())
            assert not outputs["decode"][1][2].count_nonzero()
        assert torch.equal(pool.backing[3], untouched) and torch.equal(pool.backing[0], null_row)

    prefill_positions = torch.arange(1405, device=device)
    gated_bound = attention_output_bound(parts, text, prefill_positions, local=local, fp8_query=True,
                                         e4m3_probabilities=True)
    label = f"layer {layer} rank {rank}"
    difference = report(f"{label} prefill gated attention (FP8 Q, P, K/V)", gated[:1405], parts.gated[:1405],
                        gated_bound)
    assert bool((difference <= gated_bound).all())
    partial_limit = partial_bound(gated_bound, parts.gated[:1405], serving.output, parts.partial[:1405])
    difference = report(f"{label} prefill partial output", partial[:1405], parts.partial[:1405], partial_limit)
    assert bool((difference <= partial_limit).all())
    decode_positions = torch.tensor(sorted(decoded), device=device)
    decode_gated = torch.stack([decoded[p][0] for p in decode_positions.tolist()])
    decode_partial = torch.stack([decoded[p][1] for p in decode_positions.tolist()])
    decode_bound = attention_output_bound(parts, text, decode_positions, local=local, fp8_query=False,
                                          e4m3_probabilities=False)
    difference = report(f"{label} decode gated attention (FP8 K/V)", decode_gated, parts.gated[decode_positions],
                        decode_bound)
    assert bool((difference <= decode_bound).all())
    decode_limit = partial_bound(decode_bound, parts.gated[decode_positions], serving.output,
                                 parts.partial[decode_positions])
    difference = report(f"{label} decode partial output", decode_partial, parts.partial[decode_positions],
                        decode_limit)
    assert bool((difference <= decode_limit).all())
    # The same positions through FP8 Q and P (prefill) and BF16 Q and P (decode).
    overlap = torch.tensor([1100, 1101, 1102], device=device)
    report(f"{label} FP8 Q/P prefill against BF16 Q/P decode at positions 1100-1102", gated[overlap],
           torch.stack([decoded[p][0] for p in overlap.tolist()]))
    # Canonical attention with E4M3 keys and values separates their effect from FP8 queries and probabilities.
    key8 = parts.key.to(torch.float8_e4m3fn).to(torch.bfloat16)
    value8 = parts.value.to(torch.float8_e4m3fn).to(torch.bfloat16)
    output8, lse8 = ref.attention(parts.query, key8, value8, parts.bias, positions, positions,
                                  window=text.window - 1 if local else None)
    gated8 = ref.gate_sink(output8, lse8, parts.gate, parts.sink)
    report(f"{label} canonical attention with E4M3 K/V against BF16 K/V", gated8[:1405], parts.gated[:1405])
    report(f"{label} prefill against canonical attention with E4M3 K/V", gated[:1405], gated8[:1405])


def _spacing(value):
    """Distance from |x| to the next BF16 value: at most `2^-7 |x|`."""
    return torch.exp2(torch.floor(torch.log2(value.abs().clamp_min(2.0 ** -126))) - 7)


def situ_error(gate, up, gamma, gate_error, up_error, activation):
    """Bound of |SiTU(g', u') gamma - SiTU(g, u) gamma| rounded to BF16 on both sides.

    `|d/dg 4 tanh(g / 4) sigmoid(g)| <= 2`, `|4 tanh(g / 4) sigmoid(g)| <= min(|g|, 4)`,
    `|d/du 25 tanh(u / 25)| <= 1` and `|25 tanh(u / 25)| <= min(|u|, 25)`. The
    approximate `tanh`, `ex2` and reciprocal instructions of the kernel add at most
    `2^-9` of the value; each BF16 rounding at most one BF16 step.
    """
    g, u, w = gate.double().abs(), up.double().abs(), gamma.double().abs()[:, None]
    propagated = w * (2 * gate_error * torch.clamp(u + up_error, max=25.0)
                      + up_error * torch.clamp(g + gate_error, max=4.0))
    return propagated + 2.0 ** -9 * activation.double().abs() + 2 * _spacing(activation.double().abs() + propagated)


@pytest.mark.parametrize("layer", LAYERS)
@torch.no_grad()
def test_prepared_experts_match_the_canonical_four_rank_experts(checkpoint, activations, layer):
    """Router, records, both phases and the cross-rank amax of four rank slices on one GPU.

    The canonical four-rank experts (`ref.routed_experts_ranks`) run on the layer's
    real MLP input. The prepared ranks run with the canonical routes. Each stage is
    checked on its own inputs: FC1 activations within the propagated projection and
    SiTU bound, FC2 inputs bit-equal to the canonical quantizer applied to the
    prepared activation and the every-rank amax, FC2 route rows within one
    projection's rounding, and token sums equal to the FP32 route-order sum.
    """
    from b12x.moe import fused_moe as moe

    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.inkling2.experts import ExpertWeights, PreparedExperts, prepare_experts
    from ignition.reference import inkling2 as ref
    from tests.recipes.inkling2_bounds import FP32

    device = checkpoint.device
    text = checkpoint.config.text
    expert_format = checkpoint.config.expert_format(layer)
    nvfp4 = expert_format == "nvfp4"
    source = activations[layer]["mlp_input"]
    tokens, hidden = source.shape
    width, top_k = text.intermediate // 4, text.experts_per_token
    placed = [load_rank(checkpoint, layer, rank, experts=True) for rank in range(4)]
    canonical = [reference_weights(values) for values in placed]
    ids_ref, gamma_ref = ref.route(source, canonical[0].router, canonical[0].router_bias, top_k=top_k,
                                   route_scale=text.route_scale)
    partials_ref, records_ref = ref.routed_experts_ranks(canonical, text, source)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="inkling-2-small", workload_schema="inkling2-components-v1",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        modules = []
        for rank, values in enumerate(placed):
            experts, denominators = prepare_experts({role: value.clone() for role, value in values.items()},
                                                    format=expert_format, hidden=hidden, intermediate=width)
            modules.append(PreparedExperts(ExpertWeights(values["router"], values["router_bias"], experts,
                                                         expert_format, denominators),
                                           text, tokens=tokens, session=session, workspace=arena,
                                           name=f"layer{layer}.rank{rank}"))
        session.tune()
        arena.release_preparation()
        valid = torch.tensor([tokens], dtype=torch.int32, device=device)

        # Router: FP32 logits within accumulation error; equal selections away from ties.
        logits = modules[0].router_projection(source)
        logits_ref = ref.matmul(source, canonical[0].router)
        limit = 2 * hidden * FP32 * (source.double().abs() @ canonical[0].router.double().abs().t())
        report(f"layer {layer} router logits", logits, logits_ref, limit)
        assert bool(((logits.double() - logits_ref.double()).abs() <= limit).all())
        ids, gamma = modules[0].route(source, valid)
        selection = torch.sigmoid(logits_ref.double()) + canonical[0].router_bias.double()
        ranked = selection.sort(-1, descending=True).values
        margin = ranked[:, top_k - 1] - ranked[:, top_k]
        same = (ids.long().sort(-1).values == ids_ref.sort(-1).values).all(-1)
        # sigmoid is 1/4-Lipschitz: selections can change only where the margin is within the logit error.
        assert bool((same | (margin <= 0.5 * limit.amax(-1))).all())
        matched = (ids.long() == ids_ref).all(-1)
        print(f"layer {layer} routing: {int(same.sum())}/{tokens} rows select the canonical experts, "
              f"{int(matched.sum())} in the canonical order")
        # |d log w_k| <= |d log sigmoid(l_k)| + |d log sum_j sigmoid(l_j)| <= 2 max_j |d l_j|, plus FP32 rounding.
        relative = (2 * limit.amax(-1, keepdim=True) + 2.0 ** -20)[matched]
        gamma_difference = (gamma[matched].double() - gamma_ref[matched].double()).abs()
        assert bool((gamma_difference <= relative * gamma_ref[matched].double().abs()).all())

        # Records: the owner-rank quantizer is the canonical quantizer, byte for byte.
        records = modules[0].quantize(source)
        values, scales, amax = modules[0].experts.views(records)
        quantized = ref._quantize(source, expert_format)
        if nvfp4:
            codes = torch.stack((values & 15, values >> 4), -1).flatten(-2).long()
            table = torch.tensor(ref._E2M1, dtype=torch.float32, device=device)
            assert torch.equal(table[codes], quantized.values)
            assert torch.equal(scales.view(torch.float8_e4m3fn).float(), quantized.scales.float())
            assert torch.equal(amax, quantized.amax)
        else:
            assert torch.equal(values.view(torch.float8_e4m3fn).float(), quantized.values.float())
            assert torch.equal(scales, quantized.exponents)

        # Both phases of every rank with the canonical routes and the every-rank amax.
        ids32, gamma32 = ids_ref.int().contiguous(), gamma_ref.float().contiguous()
        bindings, outputs = [], []
        global_amax = torch.zeros(tokens * top_k, dtype=torch.float32, device=device) if nvfp4 else None
        for module in modules:
            scratch = torch.empty(module.experts.workspace_nbytes, dtype=torch.uint8, device=device)
            output = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device)
            extra = dict(local_row_amax=torch.empty(tokens * top_k, device=device),
                         intermediate_row_amax=global_amax) if nvfp4 else {}
            bindings.append(moe.bind(module.experts.plan, a=values, input_block_scales=scales,
                                     input_row_scale=amax, topk_ids=ids32, topk_weights=gamma32, output=output,
                                     scratch=scratch, **extra))
            outputs.append(output)
        for binding in bindings:
            moe.run_phase1(binding=binding)
        if nvfp4:
            global_amax.copy_(torch.stack([binding.local_row_amax for binding in bindings]).amax(0))
        for binding in bindings:
            moe.run_phase2(binding=binding)
        torch.cuda.synchronize(device)

        live = ((ids_ref >= 0) & (ids_ref < text.experts)).flatten()
        assert bool(live.all())
        worst = {"activation": 0.0, "routes": 0.0}
        for rank, (binding, record, weights) in enumerate(zip(bindings, records_ref, canonical, strict=True)):
            activation = binding.views["intermediate"][:tokens * top_k]
            reference_activation = torch.zeros_like(activation)
            gate_bound = torch.zeros(tokens * top_k, width, dtype=torch.float64, device=device)
            situ_bound = torch.zeros_like(gate_bound)
            for expert, (rows, slots, canonical_activation, _) in record.down_inputs.items():
                route = rows * top_k + slots
                reference_activation[route] = canonical_activation
                rows_q = ref._rows(quantized, rows)
                gate_up = ref._project(weights, "w13", expert, rows_q)
                gain = 1.0
                if nvfp4:
                    blocks = rows_q.blocks().double()
                    decoded = ref.decode_nvfp4_blocks(weights.w13[expert], weights.w13_scale[expert]).double()
                    gain = ((rows_q.amax * weights.w13_amax[expert].float())
                            * (1.0 / (ref.NVFP4_OUTER * weights.w13_denominator[0].float()))).double()[:, None]
                else:
                    from ignition.reference.qwen38_quant import decode_mxfp8

                    blocks = rows_q.dequantized().double()
                    decoded = decode_mxfp8(weights.w13[expert], weights.w13_scale[expert]).double()
                accumulation = hidden * FP32 * (blocks.abs() @ decoded.abs().t()) * gain
                projection_error = 2 * _spacing(gate_up.double().abs() + accumulation) + accumulation
                gate_error, up_error = projection_error[:, 0::2], projection_error[:, 1::2]
                situ_bound[route] = situ_error(gate_up[:, 0::2], gate_up[:, 1::2], gamma_ref[rows, slots],
                                               gate_error, up_error, canonical_activation)
            difference = (activation.double() - reference_activation.double()).abs()
            worst["activation"] = max(worst["activation"], float((difference / situ_bound.clamp_min(1e-300)).max()))
            assert bool((difference <= situ_bound).all()), (rank, float((difference - situ_bound).max()))
            # FC2 input: the canonical quantizer applied to the prepared activation and the every-rank amax.
            down = ref._quantize(activation, expert_format, global_amax if nvfp4 else None)
            intermediate_q = binding.views["intermediate_q"][:tokens * top_k]
            intermediate_sf = binding.views["intermediate_sf"][:tokens * top_k]
            if nvfp4:
                codes = torch.stack((intermediate_q & 15, intermediate_q >> 4), -1).flatten(-2).long()
                assert torch.equal(table[codes], down.values)
                assert torch.equal(intermediate_sf.float(), down.scales.float())
            else:
                assert torch.equal(intermediate_q.view(torch.float8_e4m3fn).float(), down.values.float())
                assert torch.equal(intermediate_sf, down.exponents)
            routes = binding.views["routes"][:tokens * top_k]
            expected_routes = torch.zeros(tokens * top_k, hidden, dtype=torch.bfloat16, device=device)
            route_bound = torch.zeros(tokens * top_k, hidden, dtype=torch.float64, device=device)
            for expert, (rows, slots, _, _) in record.down_inputs.items():
                route = rows * top_k + slots
                rows_q = ref._rows(down, route)
                projected = ref._project(weights, "w2", expert, rows_q)
                expected_routes[route] = projected
                if nvfp4:
                    blocks = rows_q.blocks().double()
                    decoded = ref.decode_nvfp4_blocks(weights.w2[expert], weights.w2_scale[expert]).double()
                    gain = ((rows_q.amax * weights.w2_amax[expert].float())
                            * (1.0 / (ref.NVFP4_OUTER * weights.w2_denominator[0].float()))).double()[:, None]
                else:
                    blocks = rows_q.dequantized().double()
                    decoded = decode_mxfp8(weights.w2[expert], weights.w2_scale[expert]).double()
                    gain = 1.0
                accumulation = width * FP32 * (blocks.abs() @ decoded.abs().t()) * gain
                route_bound[route] = 2 * _spacing(projected.double().abs() + accumulation) + accumulation
            difference = (routes.double() - expected_routes.double()).abs()
            worst["routes"] = max(worst["routes"], float((difference / route_bound.clamp_min(1e-300)).max()))
            assert bool((difference <= route_bound).all()), rank
            # Token sums: FP32 sums of the BF16 route rows in route order, rounded once.
            assert torch.equal(outputs[rank], ref._combine(routes.view(tokens, top_k, hidden)))
            report(f"layer {layer} rank {rank} partial expert output against the canonical four-rank experts",
                   outputs[rank], partials_ref[rank])
        print(f"layer {layer} {expert_format}: largest difference/bound, FC1 activation {worst['activation']:.3f}, "
              f"FC2 routes {worst['routes']:.3f}")

        # The compiled forward of one rank, captured, reproduces the eager phases on its own routes.
        module = modules[0]
        supplied = torch.zeros(tokens * top_k, dtype=torch.float32, device=device)
        input_rows = source.clone()

        def forward():
            ids, gamma = module.route(input_rows, valid)
            records = module.quantize(input_rows)
            pending = module.phase1(records, ids, gamma)
            output = module.phase2(pending, ids, gamma, supplied if module.supplies_amax else None)
            return output, ids, gamma, pending.row_amax if module.supplies_amax else gamma

        compiled = compile_forward(forward)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            captured = compiled()
        graph.replay()
        torch.cuda.synchronize(device)
        ids_captured, gamma_captured = captured[1], captured[2]
        if module.supplies_amax:
            supplied.copy_(captured[3])
        output = torch.empty_like(source)
        scratch = torch.empty(module.experts.workspace_nbytes, dtype=torch.uint8, device=device)
        eager = moe.bind(module.experts.plan, a=values, input_block_scales=scales, input_row_scale=amax,
                         topk_ids=ids_captured.clone(), topk_weights=gamma_captured.clone(), output=output, scratch=scratch,
                         **(dict(intermediate_row_amax=supplied) if nvfp4 else {}))
        moe.run(binding=eager) if not nvfp4 else (moe.run_phase1(binding=eager), moe.run_phase2(binding=eager))
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(captured[0], output)


def residual_bound(sources, queries, eps, mixtures, normalized, norm_weight):
    """Bound of |prepared normalized mixture - canonical| for FP32 scores and mixtures over `sources`.

    The FP32 dot products over `D` channels move each score by at most
    `inv D 2^-24 sum |x q| + |score| (D 2^-24 + 2^-22)` (the last term covers
    the FP32 sum of squares and rsqrt), each probability by twice the largest
    score error, and the mixture by those probabilities plus the FP32 sum over
    sources. The RMS normalization of a row moves by the row's relative error,
    and both sides round once to BF16.
    """
    from tests.recipes.inkling2_bounds import FP32

    values = torch.stack([source.double() for source in sources])
    count, _, hidden = values.shape
    relative = hidden * FP32 + 2.0 ** -22
    inverse = torch.rsqrt(values.square().mean(-1) + eps)
    bounds = []
    for query, mixture, output in zip(queries.double().reshape(-1, hidden), mixtures, normalized, strict=True):
        scaled = query * hidden ** -0.5
        scores = (values @ scaled) * inverse
        score_error = inverse * hidden * FP32 * (values.abs() @ scaled.abs()) + scores.abs() * relative
        p = torch.softmax(scores, 0)
        moved = 2 * score_error.amax(0)
        weights = p * inverse
        magnitude = torch.einsum("nt,ntd->td", weights, values.abs())
        error = magnitude * (moved[:, None] + relative + count * FP32)
        m = mixture.double()
        rs = torch.rsqrt(m.square().mean(-1, keepdim=True) + eps)
        row = error.norm(dim=-1, keepdim=True) / (m.norm(dim=-1, keepdim=True) - error.norm(dim=-1, keepdim=True)).clamp_min(1e-30)
        weight = norm_weight.double().abs()
        bounds.append(weight * rs * (error + m.abs() * row) + output.double().abs() * relative
                      + 2 * _spacing(output.double().abs() + weight * rs * (error + m.abs() * row)))
    return bounds


@pytest.mark.parametrize("layer", LAYERS)
@torch.no_grad()
def test_residual_mixtures_and_norms_match_the_canonical_streams(checkpoint, activations, layer):
    """Attention and MLP inputs from the layer's real residual state, compiled and captured."""
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.inkling2.arithmetic import residual_mix, rms_norm
    from ignition.recipes.inkling2.layer import scaled_query
    from ignition.reference import inkling2 as ref

    text = checkpoint.config.text
    placed = load_rank(checkpoint, layer, 0, experts=False)
    weights = reference_weights(placed)
    entry = activations[layer]
    blocks, partial, attention = list(entry["blocks"]), entry["partial"], entry["attention_output"]
    state = ref.ResidualState(tuple(blocks), partial)
    expected_streams = ref.normalized_streams(weights, text, state)
    expected_mlp = entry["mlp_input"]
    norms = (weights.attention_norm,) if weights.local else (weights.query_norm, weights.key_norm, weights.value_norm)
    mlp_query = scaled_query(weights.mlp_query, text.hidden)
    attention_query = None if not layer else scaled_query(weights.attention_query, text.hidden)

    def forward():
        if layer:
            streams = residual_mix((*blocks, partial), attention_query)
        else:
            streams = (partial,) * len(norms)
        normalized = tuple(rms_norm(stream, norm, text.eps) for stream, norm in zip(streams, norms, strict=True))
        if layer % text.residual_block == 0:
            mixed = residual_mix((*blocks, partial, attention), mlp_query)[0]
        else:
            mixed = residual_mix((*blocks, partial), mlp_query, addend=attention)[0]
        return (*normalized, rms_norm(mixed, weights.mlp_norm, text.eps))

    compiled = compile_forward(forward)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = compiled()
    graph.replay()
    torch.cuda.synchronize()
    if layer:
        mixtures = ref.residual_mix((*blocks, partial), weights.attention_query)
        limits = residual_bound((*blocks, partial), weights.attention_query, 1e-6, mixtures, expected_streams,
                                norms[0]) if len(norms) == 1 else [
            residual_bound((*blocks, partial), weights.attention_query[index], 1e-6, (mixtures[index],),
                           (expected_streams[index],), norms[index])[0] for index in range(len(norms))]
    else:
        limits = [2 * _spacing(stream.double().abs()) + stream.double().abs() * 2.0 ** -20 for stream in expected_streams]
    for index, (value, expected, limit) in enumerate(zip(outputs[:-1], expected_streams, limits, strict=True)):
        difference = report(f"layer {layer} attention input stream {index}", value, expected, limit)
        assert bool((difference <= limit).all())
    if layer % text.residual_block == 0:
        sources = (*blocks, partial, attention)
    else:
        sources = (*blocks, partial.float() + attention.float())
    mixture = ref.residual_mix(sources, weights.mlp_query)
    limit = residual_bound(sources, weights.mlp_query, 1e-6, mixture, (expected_mlp,), weights.mlp_norm)[0]
    difference = report(f"layer {layer} MLP input", outputs[-1], expected_mlp, limit)
    assert bool((difference <= limit).all())


@pytest.mark.parametrize("layer", (0, 5))
@torch.no_grad()
def test_output_convolutions_continue_across_chunks_rows_and_decode(checkpoint, activations, layer):
    """The attention-output and MLP-output convolutions with per-row history equal one canonical pass.

    Inputs are the layer's real attention and MLP outputs. Request A runs chunks of
    1100 and 305 rows and three decode steps; request B forks A's row after the first
    chunk and decodes from position 1100. Both sides convolve identical BF16 inputs.
    """
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.inkling2.arithmetic import commit_history, short_convolution
    from ignition.recipes.inkling2.pool import ContinuationPool, RankGeometry
    from ignition.reference import inkling2 as ref
    from tests.recipes.inkling2_bounds import FP32

    device = checkpoint.device
    text = checkpoint.config.text
    placed = load_rank(checkpoint, layer, 0, experts=False)
    entry = activations[layer]
    inputs = {"attention_convolution": entry["attention_output"], "mlp_convolution": entry["mlp_input"]}
    pool = ContinuationPool.allocate(text, RankGeometry.of(text, 4), 4, 4, device)
    pool.backing.zero_()
    histories = {kind: pool.history(layer, kind) for kind in inputs}
    shapes = {"prefill": (1152, 1), "decode": (3, 3)}
    metadata = {mode: Metadata.allocate(rows, requests, 12, device) for mode, (rows, requests) in shapes.items()}
    sources = {mode: {kind: torch.zeros(rows, text.hidden, dtype=torch.bfloat16, device=device) for kind in inputs}
               for mode, (rows, _) in shapes.items()}
    graphs, outputs = {}, {}
    for mode, m in metadata.items():
        def forward(mode=mode, m=m):
            results = []
            for kind, history in histories.items():
                x = sources[mode][kind]
                results.append(short_convolution(x, placed[kind], history, m))
                commit_history(history, x, m)
            return tuple(results)

        compiled = compile_forward(forward)
        pool.backing.zero_()
        graphs[mode] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[mode]):
            outputs[mode] = compiled()
    pool.backing.zero_()
    tokens = entry["mlp_input"].shape[0]
    results = {kind: torch.zeros(tokens, text.hidden, dtype=torch.bfloat16, device=device) for kind in inputs}
    table = torch.zeros(1, 12, dtype=torch.int32, device=device)
    for start, length in ((0, 1100), (1100, 305)):
        for kind, value in inputs.items():
            sources["prefill"][kind].zero_()
            sources["prefill"][kind][:length].copy_(value[start:start + length])
        metadata["prefill"].set(torch.tensor([start], device=device),
                                torch.tensor([length], dtype=torch.int32, device=device),
                                torch.tensor([1], device=device), table,
                                torch.zeros(1, 1152, dtype=torch.int64, device=device), torch.tensor([1], device=device))
        graphs["prefill"].replay()
        for index, kind in enumerate(inputs):
            results[kind][start:start + length] = outputs["prefill"][index][:length]
            assert not outputs["prefill"][index][length:].count_nonzero()
        if start == 0:
            pool.backing[2].copy_(pool.backing[1])
    decoded = {kind: {} for kind in inputs}
    for step in range(3):
        positions = (1405 + step, 1100 + step, 0)
        for kind, value in inputs.items():
            for row, position in enumerate(positions):
                sources["decode"][kind][row] = value[position] if row < 2 else 0
        metadata["decode"].set(torch.tensor(positions, device=device),
                               torch.tensor([1, 1, 0], dtype=torch.int32, device=device),
                               torch.tensor([1, 2, 0], device=device), table.expand(3, -1).contiguous(),
                               torch.zeros(3, 1, dtype=torch.int64, device=device), torch.tensor([2], device=device))
        graphs["decode"].replay()
        for index, kind in enumerate(inputs):
            for row in range(2):
                decoded[kind][positions[row]] = outputs["decode"][index][row].clone()
            assert not outputs["decode"][index][2].count_nonzero()
    torch.cuda.synchronize()
    for kind, value in inputs.items():
        weight = placed[kind]
        expected, _ = ref.short_convolution(value, weight)
        width = weight.shape[-1]
        magnitude = ref.short_convolution(value.double().abs(), weight.double().abs())[0]
        limit = 2 * _spacing(expected.double().abs()) + 2 * width * FP32 * magnitude
        sequence = results[kind][:1405]
        difference = report(f"layer {layer} {kind} over chunks", sequence, expected[:1405], limit[:1405])
        assert bool((difference <= limit[:1405]).all())
        rows = sorted(decoded[kind])
        actual = torch.stack([decoded[kind][row] for row in rows])
        difference = report(f"layer {layer} {kind} decode", actual, expected[rows], limit[rows])
        assert bool((difference <= limit[rows]).all())


@torch.no_grad()
def test_vocabulary_parallel_embedding_and_softcapped_logits(checkpoint, activations):
    """Four rank shards: exact embedding sums and FP32 logits within accumulation error, softcap and mask."""
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.inkling2.arithmetic import rms_norm
    from ignition.recipes.inkling2.vocabulary import PreparedLogits, embedding_rows
    from ignition.reference import inkling2 as ref
    from tests.recipes.inkling2_bounds import FP32

    device = checkpoint.device
    text = checkpoint.config.text
    ids = activations["ids"]
    shards = [checkpoint.global_weights(rank) for rank in range(4)]
    rows = sum(embedding_rows(shard.embedding, ids, torch.full((1,), rank * shard.embedding.shape[0], device=device))
               .float() for rank, shard in enumerate(shards))
    whole = checkpoint.global_weights()
    assert torch.equal(rows.to(torch.bfloat16), whole.embedding[ids])
    embedded = rms_norm(rows.to(torch.bfloat16), shards[0].embedding_norm, text.eps)
    assert torch.equal(embedded, ref.embed(ids, whole, text))
    hidden = activations[5]["partial"]
    tokens = hidden.shape[0]
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="inkling-2-small", workload_schema="inkling2-components-v1",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        heads = [PreparedLogits(shard.unembedding, text, rank, tokens=tokens, session=session, name=f"logits{rank}")
                 for rank, shard in enumerate(shards)]
        session.tune()
        arena.release_preparation()
        logits = torch.cat([head(hidden) for head in heads], -1)[:, :text.vocabulary]
        # The last rank's shard holds the masked columns and the zero rows past the table.
        compiled = compile_forward(lambda: heads[3](hidden))
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            captured = compiled()
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(captured, heads[3](hidden))
    expected = ref.logits_ranks(hidden, shards, text)
    assert torch.equal(torch.isinf(logits), torch.isinf(expected))
    assert bool(torch.isinf(logits[:, text.valid_vocabulary:]).all())
    valid = slice(0, text.valid_vocabulary)
    magnitude = hidden.double().abs() @ whole.unembedding[valid].double().abs().t()
    limit = 2 * text.hidden * FP32 * magnitude + 2.0 ** -22 * text.softcap
    difference = report("softcapped logits", logits[:, valid], expected[:, valid], limit)
    assert bool((difference <= limit).all())
    agreement = (logits[:, valid].argmax(-1) == expected[:, valid].argmax(-1)).float().mean()
    print(f"greedy token agreement over {tokens} rows: {float(agreement):.4f}")


@torch.no_grad()
def test_model_weights_load_a_rank_layer_prefix_and_vocabulary_shard(checkpoint):
    """`ModelWeights.load` places rank 2's first layer, embedding and unembedding rows as the layout reads them."""
    from ignition.recipes.inkling2.attention import attention_weights
    from ignition.recipes.inkling2.model import ModelWeights

    text = checkpoint.config.text
    weights = ModelWeights.load(checkpoint.manifest, checkpoint.device, config=checkpoint.config, rank=2, world=4,
                                layers=1)
    assert len(weights.layers) == 1 and weights.rank == 2
    placed = load_rank(checkpoint, 0, 2, experts=False)
    expected = attention_weights(placed, local=True, heads=12, kv_heads=2, dim=text.head_dim,
                                 width=text.convolution_width)
    for actual, reference in zip(weights.layers[0].attention.projections, expected.projections, strict=True):
        assert torch.equal(actual, reference)
    shard = checkpoint.global_weights(2)
    assert torch.equal(weights.embedding, shard.embedding) and torch.equal(weights.unembedding, shard.unembedding)
    assert torch.equal(weights.output_query, shard.output_query.float().reshape(1, -1) * text.hidden ** -0.5)
