# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 checkpoint layers loaded through the four-rank placement and run by the reference layer.

Requires `FLARE_RUN_CUDA_TESTS=1` and `FLARE_INKLING2_SNAPSHOT`, a directory with the
checkpoint's config.json, hf_quant_config.json and model.safetensors.index.json and
the shards that hold the tested layers. A layer whose shards are absent is skipped:
layer 0 (local, NVFP4 experts) is in model-00001, layer 5 (global, NVFP4) in
model-00022 and layer 40 (local, MXFP8) in model-00015.
"""

import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from ignition.preparation.checkpoint import IndexedCheckpoint, _read_header
from ignition.recipes.inkling2.config import ModelConfig
from ignition.recipes.inkling2.weights import PlacedWeightLoader, checkpoint_layout
from ignition.reference import inkling2 as ref
from tests.reference.inkling2_support import attention_bound, mlp_bound, shard

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an assigned CUDA device")

SNAPSHOT = Path(os.environ.get("FLARE_INKLING2_SNAPSHOT", Path.home() / ".cache/huggingface/hub" /
    "models--thinkingmachines--Inkling-2-Small/snapshots/b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b"))
BF16 = torch.bfloat16


def load_layer(layout, layer, device):
    placements = layout.layer(layer)
    names = {placement.name for placement in placements.values()}
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({index[name] for name in names})
    if any(not (SNAPSHOT / name).is_file() for name in shards):
        pytest.skip(f"layer {layer} shards are unavailable: {shards}")
    tensors, records = {}, []
    for name in shards:
        record, entries = _read_header(SNAPSHOT / name, name)
        records.append(record)
        tensors.update(entries)
    manifest = IndexedCheckpoint(SNAPSHOT, MappingProxyType(tensors), tuple(records))
    with PlacedWeightLoader(manifest, device, layout=layout, names=names) as loader:
        whole = {role: loader.load(placement.name) for role, placement in placements.items()}
        ranks = [{role: loader.load(placement.name, rank=rank) for role, placement in placements.items()}
                 for rank in range(4)]
    for role, placement in placements.items():
        for rank in range(4):
            expected = shard(whole[role], placement, rank)
            assert ranks[rank][role].dtype == expected.dtype
            assert torch.equal(ranks[rank][role].view(torch.uint8), expected.view(torch.uint8)), (role, rank)
    return ref.LayerWeights(**whole), [ref.LayerWeights(**values) for values in ranks]


def residual_state(config, layer, tokens, device):
    generator = torch.Generator(device).manual_seed(layer)

    def sample(scale):
        return (torch.randn(tokens, config.hidden, generator=generator, device=device) * scale).to(BF16)

    if not layer:
        return ref.ResidualState((), sample(1.0))
    return ref.ResidualState(tuple(sample(scale) for scale in [1.0] + [8.0] * (layer // config.residual_block)),
                             sample(4.0))


@pytest.mark.parametrize("layer,start", [(0, 0), (5, 200_000), (40, 1000)])
@torch.no_grad()
def test_checkpoint_layer_runs_and_four_ranks_match_one_rank(layer, start):
    if not (SNAPSHOT / "config.json").is_file():
        pytest.skip("Inkling 2 Small configuration is unavailable; set FLARE_INKLING2_SNAPSHOT")
    device = torch.device("cuda", torch.cuda.current_device())
    config = ModelConfig.load(SNAPSHOT)
    text = config.text
    layout = checkpoint_layout(config, 4)
    whole, ranks = load_layer(layout, layer, device)
    assert whole.expert_format == config.expert_format(layer) and whole.local == text.local(layer)
    tokens = 16
    state = residual_state(text, layer, tokens, device)
    positions = torch.arange(start, start + tokens, device=device)
    single_state, _, single = ref.decoder_layer(whole, text, layer, state, positions,
                                                ref.LayerCache.empty(whole, text.head_dim))
    parallel_state, caches, parallel = ref.decoder_layer_ranks(
        ranks, text, layer, state, positions, [ref.LayerCache.empty(weights, text.head_dim) for weights in ranks])
    for value in (single.attention_output, single.mlp_output, single_state.partial, parallel.attention_output,
                  parallel.mlp_output, parallel_state.partial):
        assert bool(torch.isfinite(value.float()).all())
    heads = text.heads // 4 * text.head_dim
    for rank, record in enumerate(parallel.attention):
        assert torch.equal(record.gated, single.attention[0].gated[:, rank * heads:(rank + 1) * heads])
    assert caches[0].key.shape == (tokens, text.kv_heads // 4, text.head_dim)

    attention = (parallel.attention_output.float() - single.attention_output.float()).abs()
    attention_limit = attention_bound(single, parallel, whole)
    assert bool((attention <= attention_limit).all()), float((attention - attention_limit).max())

    # The expert stage on the four-rank path's own MLP input, against one rank on that input.
    outputs, records = ref.routed_experts_ranks((whole,), text, parallel.mlp_input)
    width = text.intermediate // 4
    for expert, (_, _, activation, down) in records[0].down_inputs.items():
        for rank, record in enumerate(parallel.experts):
            part = record.down_inputs[expert]
            assert torch.equal(part[2], activation[:, rank * width:(rank + 1) * width])
            columns = slice(rank * width, (rank + 1) * width)
            assert torch.equal(part[3].values.float(), down.values.float()[:, columns])
    reference_m = ref.short_convolution(outputs[0], whole.mlp_convolution)[0]
    reference = SimpleNamespace(expert_partials=outputs, mlp_convolved=[reference_m], mlp_output=reference_m,
                                experts=records)
    mlp = (parallel.mlp_output.float() - reference_m.float()).abs()
    mlp_limit = mlp_bound(reference, parallel, whole)
    assert bool((mlp <= mlp_limit).all()), float((mlp - mlp_limit).max())
    rms = lambda value: float(value.float().square().mean().sqrt())  # noqa: E731
    ratio = lambda difference, limit: float((difference / limit).max())  # noqa: E731
    print(f"layer {layer} {whole.expert_format}: attention rms {rms(single.attention_output):.4g}, "
          f"four-rank difference max {float(attention.max()):.3g}, "
          f"largest difference/bound {ratio(attention, attention_limit):.3f}; "
          f"expert rms {rms(reference_m):.4g}, difference max {float(mlp.max()):.3g}, "
          f"largest difference/bound {ratio(mlp, mlp_limit):.3f}; experts routed {len(records[0].down_inputs)}")
