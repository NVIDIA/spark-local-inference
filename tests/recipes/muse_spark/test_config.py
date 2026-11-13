# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark text declaration geometry and precision exclusions."""

import json
from pathlib import Path

import pytest

from ignition.components.parallel import TensorParallelShard
from ignition.recipes import resolve
from ignition.recipes.muse_spark.config import ModelConfig
from ignition.recipes.muse_spark.pool import continuation_layout, page_bytes

CHECKPOINT = Path('/data/cache/huggingface/hub/models--meta-models--Muse-Spark-Open-816B-A42B-NVFP4/snapshots/4239e4db19376904f2e82c167e78f70abb5455fe')


@pytest.fixture
def config():
    if not CHECKPOINT.is_dir():
        pytest.skip("Muse Spark checkpoint configuration is unavailable")
    return ModelConfig.load(CHECKPOINT)


def test_text_metadata_partitions_and_cache_costs(config):
    recipe = resolve("muse-spark-open-nvfp4").recipe
    assert (recipe.context, recipe.concurrency, recipe.prefill_chunk) == (262144, 4, 4096)
    assert recipe.max_images == 0
    assert recipe.max_draft_tokens == 7
    assert recipe.sampling_defaults.temperature == recipe.sampling_defaults.top_p == 1
    assert recipe.sampling_defaults.top_k == -1
    shard = TensorParallelShard(0, 8)
    config.validate_shard(shard)
    assert [i for i in range(config.layers) if config.expert_format(i) == "bf16"] == [0, 61]
    assert config.attention(0).rotary_dim == 64 and config.attention(1).rotary_dim == 0
    assert config.attention(0).local(shard) == (16, 2)
    partitions = [config.vocabulary_partition(TensorParallelShard(rank, 8)) for rank in range(8)]
    assert partitions[0] == (0, 25280, 25280)
    assert partitions[-1] == (176960, 202048, 25280)
    assert continuation_layout(config, shard).row_bytes == recipe.costs(8, None).state_row_bytes
    assert page_bytes(config, shard) == recipe.costs(8, None).page_bytes


def test_expert_precision_follows_exclusions(config):
    raw = json.loads((CHECKPOINT / "config.json").read_text())
    raw["quantization_config"]["modules_to_not_convert"].append("model.language_model.layers.4.mlp.experts")
    assert ModelConfig.from_dict(raw).expert_format(4) == "bf16"
