# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The Inkling 2 recipe declaration: text-only TP4 plans, rejected settings and declared costs.

The cost test reads `FLARE_INKLING2_SNAPSHOT` (default: the pinned revision in the
Hugging Face hub cache), which needs only config.json and hf_quant_config.json,
and skips without it.
"""

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.errors import UsageError
from ignition.recipes import recipes
from ignition.recipes.declaration import Speculation
from ignition.recipes.inkling2.config import ModelConfig
from ignition.recipes.inkling2.pool import RankGeometry, continuation_layout, page_specs
from ignition.recipes.inkling2.recipe import RECIPE
from tests.reference.inkling2_support import synthetic_config

SNAPSHOT = Path(os.environ.get("FLARE_INKLING2_SNAPSHOT", Path.home() / ".cache/huggingface/hub" /
    "models--thinkingmachines--Inkling-2-Small/snapshots/b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b"))
GB10 = 128_526_520_320


def sparks():
    return tuple(planning.Gpu(index, "NVIDIA GB10", f"GPU-{index}", GB10, GB10 - (6 << 30), (12, 1))
                 for index in range(4))


def planned(**settings):
    return planning.plan(RECIPE, RECIPE.checkpoints[0], sparks(), selected=(0, 1, 2, 3), concurrency=16,
                         reserve_bytes=0, speculation=None, **settings)


def test_recipe_is_registered_for_text_only_tp4_without_speculation():
    assert {recipe.name: recipe for recipe in recipes()}["inkling2"] is RECIPE
    assert RECIPE.tensor_parallel == (4,) and RECIPE.speculation is None and RECIPE.max_draft_tokens == 0
    assert RECIPE.max_images == 0 and not RECIPE.language_model_only and not RECIPE.mixed_prefill
    plan = planned()
    assert plan.fits and plan.max_images == 0 and plan.image_concurrency == 0
    assert plan.decoder_layers is None and plan.to_dict()["decoder_layers"] is None


@pytest.mark.parametrize("settings,message", [
    (dict(image_concurrency=1), "--image-concurrency requires a vision model"),
    (dict(mixed_prefill=True), "does not capture mixed"),
    (dict(language_model_only=True), "prepares no separable vision encoder"),
    (dict(online_mxfp8=True), "names no projections"),
    (dict(expert_scale_compression="csf"), "expert"),
])
def test_planning_rejects_images_mixed_prefill_and_weight_conversions(settings, message):
    with pytest.raises(UsageError, match=message):
        planned(**settings)


@pytest.mark.parametrize("change,ranks,message", [
    (dict(speculation=Speculation(draft_tokens=1)), True, "no drafter"),
    (dict(prompt_logprobs=True), True, "scored prefill"),
    (dict(roce_chain=True), True, "chains"),
    ({}, False, "four hosts"),
])
def test_preparation_rejects_speculation_scored_prefill_chains_and_one_host(change, ranks, message):
    plan = replace(planned(), **change)
    group = SimpleNamespace(hosts=("ampere", "faraday", "maxwell", "hertz")) if ranks else None
    with pytest.raises(ValueError, match=message):
        RECIPE.prepare(None, plan, compile_forwards=True, kernel_cache=None, progress=None, ranks=group)


def test_decoder_layer_prefix_is_recorded_in_the_plan():
    plan = replace(planned(), decoder_layers=6)
    assert plan.decoder_layers == 6 and plan.to_dict()["decoder_layers"] == 6
    with pytest.raises(UsageError, match="positive integer"):
        replace(plan, decoder_layers=0)


def test_text_prefix_keeps_the_first_layers_and_their_attention_kinds():
    text = ModelConfig.from_dict(*synthetic_config()).text
    prefix = text.prefix(6)
    assert (prefix.layers, prefix.global_layers, sorted(prefix.local_layers)) == (6, (5,), [0, 1, 2, 3, 4])
    assert text.prefix(5).global_layers == ()
    assert replace(prefix, layers=text.layers, local_layers=text.local_layers) == text
    for layers in (0, text.layers + 1):
        with pytest.raises(ValueError, match="prefix"):
            text.prefix(layers)


def test_declared_costs_match_the_continuation_rows_and_pages():
    if not (SNAPSHOT / "config.json").is_file() or not (SNAPSHOT / "hf_quant_config.json").is_file():
        pytest.skip("Inkling 2 Small configuration is unavailable; set FLARE_INKLING2_SNAPSHOT")
    text = ModelConfig.load(SNAPSHOT).text
    geometry = RankGeometry.of(text, 4)
    costs = RECIPE.costs(4, None)
    assert costs.state_row_bytes == continuation_layout(text, geometry).row_bytes
    assert costs.page_bytes == sum(shape[0] for _, shape in page_specs(text, geometry))
