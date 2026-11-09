# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inferact DSpark checkpoint, context storage, and target-weight sharing."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from .test_kimi31 import config, raw_config


@pytest.fixture(scope="module")
def draft_snapshot():
    from ignition.recipes.kimi31.dspark_config import DSPARK_REVISION

    snapshot = Path.home() / ".cache/huggingface/hub/models--Inferact--Kimi-K3.1-Pro-DSpark-preview/snapshots" / DSPARK_REVISION
    if not (snapshot / "model.safetensors").is_file():
        pytest.skip("Inferact Kimi K3.1 DSpark checkpoint is not installed")
    return snapshot


@pytest.fixture(scope="module")
def draft_config(config, draft_snapshot):
    from ignition.recipes.kimi31.dspark_config import DSparkConfig

    return DSparkConfig.load(draft_snapshot, config)


def test_actual_checkpoint_geometry_and_headers(config, draft_config, draft_snapshot):
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.kimi31.dspark_weights import _validate_checkpoint

    _validate_checkpoint(inspect_indexed_checkpoint(draft_snapshot), draft_config)
    assert draft_config.auxiliary_layers == (76, 80, 84, 88, 92)
    assert draft_config.auxiliary_stream == "prefix"
    assert draft_config.window is None
    for rank in range(16):
        draft_config.validate_shard(TensorParallelShard(rank, 16))


@pytest.mark.parametrize("field,value", [
    ("full_attention_causal", False), ("sliding_window", 1024),
    ("sample_from_anchor", False), ("qk_rope_head_dim", 0),
    ("target_layer_ids", [75, 79, 83, 87, 92]),
    ("target_layer_ids", [75, 79, 83, 87, 87]),
    ("dflash_config", {"causal": False}),
])
def test_incompatible_draft_semantics_rejected(config, draft_config, field, value):
    from ignition.recipes.kimi31.dspark_config import DSparkConfig

    raw = deepcopy(draft_config.raw)
    raw[field] = value
    with pytest.raises(ValueError, match="Kimi K3.1 DSpark"):
        DSparkConfig.from_dict(raw, config.text)


def test_rotary_attention_factor_cannot_be_dropped(config, draft_config):
    from ignition.recipes.kimi31.dspark_config import DSparkConfig

    raw = deepcopy(draft_config.raw)
    del raw["rope_parameters"]["attn_factor"]
    with pytest.raises(ValueError, match="rotary"):
        DSparkConfig.from_dict(raw, config.text)


def test_redhat_draft_cannot_attach_to_k31(config, tmp_path):
    from ignition.recipes.kimi31.dspark_config import DSparkConfig

    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DSparkDraftModel"]}))
    with pytest.raises(ValueError, match="architectures"):
        DSparkConfig.load(tmp_path, config)


def test_draft_pages_and_partial_snapshots_are_budgeted(config, draft_config):
    import torch
    from ignition.recipes.kimi3.pool import ContinuationPool, continuation_layout, page_bytes
    from ignition.recipes.kimi31.recipe import RECIPE

    config = replace(config, draft=draft_config)
    costs = RECIPE.costs(16, RECIPE.speculation)
    layout = continuation_layout(config, 16, speculative_tokens=7)
    assert layout.row_bytes == costs.state_row_bytes == 33_238_272
    assert page_bytes(config) == costs.page_bytes == 2_912_256
    pool = ContinuationPool.for_model(config, 16, 2, 3, "meta", speculative_tokens=7)
    for layer in range(5):
        cache = pool.draft_attention(layer)
        assert cache.shape == (3, 128, 576) and cache.dtype == torch.bfloat16
        assert layout.region(f"draft{layer}.latent").nbytes == cache[0].nbytes
    assert set(dict(pool.partial_pages())) == set(pool.pages)


def test_draft_weight_declaration_preserves_target_embedding_alias():
    import torch
    from dataclasses import dataclass
    from ignition.components.weights import MLPWeights
    from ignition.preparation.rebuild_manifest import declare_weights, weight_payload_declaration
    from ignition.recipes.kimi31.dspark_weights import DraftLayerWeights, DraftWeights

    def tensor():
        return torch.zeros((2, 2), dtype=torch.bfloat16)

    embedding = tensor()
    layer = DraftLayerWeights({"q_b": tensor()}, tensor(), tensor(), MLPWeights(tensor(), tensor()))
    draft = DraftWeights(tensor(), tensor(), tensor(), tensor(), embedding, None,
                        tensor(), tensor(), (layer,))
    @dataclass
    class Model:
        embedding: torch.Tensor
        draft: DraftWeights
        layers: tuple = ()

    model = Model(embedding, draft)
    declaration = declare_weights(model, components=("embedding", "draft"),
        aliases=(("weights.draft.embedding", "weights.embedding"),))
    views = {view["name"]: view for view in weight_payload_declaration(declaration)["views"]}
    assert views["weights.draft.embedding"]["alias_of"] == "weights.embedding"
    assert "weights.draft.layers[0].attention['q_b']" in views
    assert "weights.draft.layers[0].mlp.gate_up" in views
    assert "weights.draft.markov_embedding" in views
