# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 partial pages and normalized hidden states remain exact through snapshot forks."""

import os

import pytest
import torch

from ignition.recipes.muse_glimmer.config import ModelConfig
from ignition.recipes.muse_glimmer.pool import ContinuationPool
from tests.recipes.test_muse_spark_state_cuda import assert_snapshot_forks

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("endpoint", [127, 128, 129, 2047, 2048, 2049])
@torch.no_grad()
def test_captured_snapshot_forks_preserve_bf16_histories(endpoint):
    config = ModelConfig(dict(num_hidden_layers=2, hidden_size=256, num_key_value_heads=2, head_dim=128), {})
    pool = ContinuationPool.allocate(config, 8, 5, "cuda:0")
    assert_snapshot_forks(pool, endpoint)
