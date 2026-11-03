# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 proposal scratch for attention widths distinct from model hidden widths."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.dflash2 import DraftScratch
from ignition.components.dflash2_geometry import DraftAttentionGeometry
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.scratch import GraphBufferPool


@pytest.mark.parametrize('world', [1, 2, 4, 8])
def test_non_square_attention_scratch_and_proposal_views(world):
    config = SimpleNamespace(block_size=8,
        text=dict(hidden_size=5120, intermediate_size=12288, vocab_size=128),
        attention=DraftAttentionGeometry(32, 8, 128, 128, 128, 10000000., 2048))
    buffers = GraphBufferPool('meta').cursor()
    scratch = DraftScratch.allocate(config, TensorParallelShard(0, world), 32, 'meta', buffers=buffers)
    assert scratch.embedding.shape == (32, 5120)
    assert scratch.local_embedding.shape == (32, 5120 // world)
    assert scratch.query.shape == scratch.attended.shape == (32, 32 // world, 128)
    assert scratch.key.shape == scratch.value.shape == (32, 8 // world, 128)
    assert scratch.qkv.shape == (32, 6144 // world)
    assert scratch.partial.shape == scratch.output.shape == (32, 5120)
    view = scratch.rows(8, 8)
    assert view.query.shape == (8, 32 // world, 128)
    assert view.embedding.shape == (8, 5120)
    assert view.window_key is scratch.window_key
    assert view.convolution is scratch.convolution
    assert view.selectors is scratch.selectors
    with pytest.raises(ValueError, match='complete proposal blocks'):
        scratch.rows(7, 7)


@pytest.mark.parametrize("width", [8, 13, 16])
def test_effective_block_preserves_checkpoint_and_scratch_fields(width):
    config = SimpleNamespace(block_size=8,
        text=dict(hidden_size=5120, intermediate_size=17408, vocab_size=128),
        attention=DraftAttentionGeometry(32, 8, 128, 128, 128, 10000000., 2048))
    scratch = DraftScratch.allocate(config, TensorParallelShard(0, 1), width, "meta",
                                   buffers=GraphBufferPool("meta").cursor(), block_size=width)
    view = scratch.rows(width, width)
    assert config.block_size == 8
    assert view.block_size == width
    assert view.query.shape == (width, 32, 128)
    assert view.window_key.shape[0] == 17
    assert all(not isinstance(value, int) for value in view.scratch_tensors())
