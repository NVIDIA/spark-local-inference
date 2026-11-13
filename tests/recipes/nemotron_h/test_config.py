# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint structure and concrete continuation storage declarations."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def checkpoint():
    path = Path(os.environ.get("FLARE_NEMOTRON_CHECKPOINT", "/data/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"))
    if not path.is_dir():
        pytest.skip("requires Nemotron Bolt checkpoint metadata")
    return path


def test_checkpoint_storage_matches_recipe(checkpoint):
    from ignition.recipes.nemotron_h.config import ModelConfig
    from ignition.recipes.nemotron_h.pool import continuation_layout, page_bytes
    from ignition.recipes.nemotron_h.recipe import RECIPE

    config = ModelConfig.load(checkpoint)
    layout = continuation_layout(config)
    assert config.layers.count("mamba") == 40
    assert config.layers.count("attention") == 8
    assert config.layers.count("moe") == 40
    assert layout.row_bytes == RECIPE.costs(1, None).state_row_bytes
    assert page_bytes(config) == RECIPE.costs(1, None).page_bytes
    for index, kind in enumerate(config.layers):
        if kind == "mamba":
            assert layout.region(f"layer{index}.recurrent").shape == (128, 64, 128)
            assert layout.region(f"layer{index}.convolution").shape == (10240, 3)
    assert RECIPE.sampling_defaults.temperature == 1
    assert RECIPE.sampling_defaults.top_p == .95


@pytest.mark.parametrize("field,value", [("use_qk_norm", True), ("time_step_limit", [.001, 1.]),
                                         ("residual_in_fp32", True), ("conv_kernel", 5)])
def test_rejects_different_recurrent_or_attention_semantics(checkpoint, field, value):
    from ignition.recipes.nemotron_h.config import ModelConfig

    config = json.loads((checkpoint / "config.json").read_text())
    config["llm_config"][field] = value
    with pytest.raises(ValueError):
        ModelConfig.from_dict(config)


def test_draft_taps_and_quantization_exclusions(checkpoint):
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.nemotron_h.config import ModelConfig
    from ignition.recipes.nemotron_h.draft_config import DraftConfig
    from ignition.recipes.nemotron_h.draft_weights import DraftWeightLoader

    path = Path(str(checkpoint) + "-LiLiCorr")
    if not path.is_dir():
        pytest.skip("requires LiLiCorr checkpoint metadata")
    target = ModelConfig.load(checkpoint)
    draft = DraftConfig.load(path, target)
    from ignition.recipes.nemotron_h.pool import continuation_layout
    from ignition.recipes.nemotron_h.recipe import RECIPE
    layout = continuation_layout(replace(target, draft=draft))
    assert layout.row_bytes == RECIPE.costs(1, RECIPE.speculation).state_row_bytes
    assert layout.region("layer0.convolution").shape == (10240, 10)
    quant = DraftWeightLoader(inspect_indexed_checkpoint(path), "cuda:0").quantization
    assert draft.target_layers == (1, 18, 35, 51, 68, 85)
    assert quant["fc"] == dict(quant_algo="W4A16_NVFP4", group_size=16)
    assert "layers.0.mlp.gate_proj" in quant and "layers.0.mlp.up_proj" in quant
    assert "layers.0.self_attn.q_proj" not in quant
    assert "lilicorr.factor_input_proj" not in quant
    assert "lilicorr.feature_mlp.up_proj" not in quant
    invalid = deepcopy(draft.text)
    invalid["dflash_config"]["target_layer_ids"] = [len(target.layers)]
    with pytest.raises(ValueError, match="feature taps"):
        DraftConfig.from_dict(invalid, target)


def test_image_frontend_matches_checkpoint(checkpoint):
    from ignition.preparation.frontend import prepare_frontend
    from ignition.recipes.nemotron_h.vision import VisionConfig
    from ignition.recipes.nemotron_h.contract import FRONTEND, VISION
    from ignition.preparation import runtime_core_pb2 as p

    raw = json.loads((checkpoint / "config.json").read_text())
    vision = VisionConfig.from_dict(raw)
    assert vision.hidden == 4096
    frontend = prepare_frontend(checkpoint, eos_token_ids=[2], image_profile="radio")
    assert frontend["image"] == {"radio": {}}
    assert FRONTEND["image_layout"] == p.RADIO_IMAGE_LAYOUT
    assert FRONTEND["image_tokens"]["feature_token"] == 18
    assert VISION["features_per_request"] == 4 * 13312 // 4
    raw["ps_version"] = "v1"
    with pytest.raises(ValueError, match="ps_version"):
        VisionConfig.from_dict(raw)


def test_host_ragged_rounds_separate_target_rows_from_sample_rows():
    from ignition.recipes.nemotron_h.bootstrap import PreparationConfig
    from ignition.recipes.nemotron_h.recipe import RECIPE

    config = PreparationConfig(max_active=4, draft_checkpoint=Path("lilicorr"),
        speculative_widths=tuple(range(1, 9)), speculative_token_capacity=8)
    assert (4, 8, 5) in config.speculative_round_shapes
    assert config.decode_tokens == 8
    assert config.decode_samples == 32
    assert RECIPE.speculation.packed_verification
    fixed = replace(config, speculative_fixed_width=4)
    assert {width for _, width, _ in fixed.speculative_round_shapes} == {1, 4}
    assert fixed.decode_tokens == 8 and fixed.decode_samples == 16
    plain = replace(config, draft_checkpoint=None)
    assert plain.speculative_round_shapes == ()
    assert plain.decode_tokens == plain.decode_samples == 4
