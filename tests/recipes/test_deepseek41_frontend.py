# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native prompt metadata exported from the cached checkpoint tokenizer."""

import json
from pathlib import Path

import pytest

from ignition.preparation.frontend import PromptEncoder, prepare_frontend
from ignition.preparation.core_proto import serving
from ignition.preparation.runtime_core_pb2 import ModelIndex
from ignition.recipes.deepseek41.recipe import RECIPE


def test_native_prompt_export():
    checkpoint = RECIPE.checkpoints[0]
    snapshot = (Path.home() / ".cache/huggingface/hub"
                / ("models--" + checkpoint.repository.replace("/", "--"))
                / "snapshots" / checkpoint.revision)
    if not (snapshot / "tokenizer.json").exists():
        pytest.skip("checkpoint tokenizer is not cached")
    model = json.loads((snapshot / "config.json").read_text())
    vision = model.get("vision_config") is not None
    frontend = prepare_frontend(snapshot, eos_token_ids=[1],
        prompt_encoder=PromptEncoder.DEEPSEEK_V41,
        image_profile='deepseek_row_major' if vision else None,
        preprocessor=None)
    assert frontend["prompt_encoder"] == PromptEncoder.DEEPSEEK_V41
    assert frontend.get("image_placeholder_token_id") == (129264 if vision else None)
    index = ModelIndex()
    serving(index, dict(assets={"tokenizer": {}}, frontend=frontend,
                        sampling_defaults=RECIPE.sampling_defaults.as_dict()))
    restored = ModelIndex.FromString(index.SerializeToString())
    assert restored.frontend.prompt_encoder == PromptEncoder.DEEPSEEK_V41
    assert restored.frontend.HasField("image_placeholder_token_id") == vision
    assert restored.frontend.image_placeholder_token_id == (129264 if vision else 0)
