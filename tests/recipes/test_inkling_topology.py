# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inkling NVFP4 preparation limits and memory planning through the shared planner."""

import os
from pathlib import Path

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from ignition import planning
from ignition.components.parallel import TensorParallelShard
from ignition.recipes import resolve
from ignition.recipes.inkling import bootstrap
from ignition.recipes.inkling.config import ModelConfig
from ignition.recipes.inkling.pool import continuation_layout, page_bytes
from ignition.recipes.inkling.recipe import RECIPE


def test_minimum_prefill_chunk_is_accepted():
    assert bootstrap.PreparationConfig(prefill_chunk_size=64).prefill_chunk_size == 64


@pytest.mark.parametrize('values,match', [
    (dict(context=100), '128-aligned'), (dict(prefill_chunk_size=96), '64-aligned'),
])
def test_invalid_preparation_config_rejects_before_loading(values, match):
    with pytest.raises(ValueError, match=match):
        bootstrap.PreparationConfig(**values)


def test_unsupported_gpu_is_rejected_before_checkpoint_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap.torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr(bootstrap.torch.cuda, 'get_device_capability', lambda device: (12, 1))
    with pytest.raises(ValueError, match='requires an SM103'):
        bootstrap.prepare(tmp_path / 'not-downloaded', bootstrap.PreparationConfig(), package_storage=None)


def test_nvfp4_plan_fits_one_station():
    model = resolve('inkling-small-nvfp4')
    assert model.recipe is RECIPE and model.checkpoint.repository == 'thinkingmachines/Inkling-Small-NVFP4'
    station = planning.Gpu(0, 'NVIDIA GB300', 'GPU-0', 256 << 30, 240 << 30, (10, 3))
    plan = planning.plan(RECIPE, model.checkpoint, (station,), selected=None,
                         concurrency=4, reserve_bytes=4 << 30, speculation=None)
    assert plan.world_size == 1 and plan.fits
    assert plan.context == plan.guaranteed_tokens == 262144


def test_declared_costs_match_publisher_cache_geometry():
    snapshot = Path(os.environ.get('FLARE_INKLING_SNAPSHOT') or resolve('inkling-small-nvfp4').snapshot(Path(HF_HUB_CACHE)))
    if not (snapshot / 'hf_quant_config.json').is_file():
        pytest.skip('Inkling checkpoint metadata is unavailable')
    config, shard = ModelConfig.load(snapshot), TensorParallelShard(0, 1)
    costs = RECIPE.costs(1, None)
    assert costs.state_row_bytes == continuation_layout(config, shard).row_bytes
    assert costs.page_bytes == page_bytes(config, shard)
