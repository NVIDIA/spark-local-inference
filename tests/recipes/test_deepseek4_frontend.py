# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native prompt metadata exported from the cached checkpoint tokenizer."""

from pathlib import Path

import pytest

from ignition.preparation.frontend import PromptEncoder, prepare_frontend
from ignition.preparation.core_proto import serving
from ignition.preparation.runtime_core_pb2 import ModelIndex
from ignition.recipes.deepseek4.recipe import RECIPE
from ignition.recipes.deepseek4.config import MODEL_ID, REVISION


def test_native_prompt_export():
    snapshot = (Path.home() / ".cache/huggingface/hub"
                / ("models--" + MODEL_ID.replace("/", "--"))
                / "snapshots" / REVISION)
    if not (snapshot / "tokenizer.json").exists():
        pytest.skip("checkpoint tokenizer is not cached")
    frontend = prepare_frontend(snapshot, eos_token_ids=[1],
        prompt_encoder=PromptEncoder.DEEPSEEK_V4,
        image_profile="deepseek4",
        preprocessor=Path("ignition/recipes/deepseek4/assets/preprocessor_config.json"))
    assert frontend["prompt_encoder"] == PromptEncoder.DEEPSEEK_V4
    assert frontend.get("image_placeholder_token_id") == 129264
    index = ModelIndex()
    serving(index, dict(assets={"tokenizer": {}}, frontend=frontend,
                        sampling_defaults=RECIPE.sampling_defaults.as_dict()))
    restored = ModelIndex.FromString(index.SerializeToString())
    assert restored.frontend.prompt_encoder == PromptEncoder.DEEPSEEK_V4
    assert restored.frontend.HasField("image_placeholder_token_id")
    assert restored.frontend.image_placeholder_token_id == 129264
