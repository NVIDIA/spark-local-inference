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
