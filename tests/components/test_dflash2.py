# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 proposal scratch for attention widths distinct from model hidden widths."""

from types import SimpleNamespace
import copy

import pytest
import torch

from ignition.components.dflash2 import DraftScratch, _draft_templates
from ignition.components.dflash2_geometry import DraftAttentionGeometry
from ignition.components.parallel import TensorParallelShard


@pytest.mark.parametrize('world', [1, 2, 4, 8])
def test_non_square_attention_scratch_and_proposal_views(world):
    config = SimpleNamespace(block_size=8,
        text=dict(hidden_size=5120, intermediate_size=12288, vocab_size=128),
        attention=DraftAttentionGeometry(32, 8, 128, 128, 128, 10000000., 2048))
    shard = TensorParallelShard(0, world)
    scratch = DraftScratch.allocate(config, shard, 32, 'meta')
    templates = _draft_templates(config, shard, scratch, 'meta')
    assert scratch.logit_shape == (32, 128)
    assert templates.embedding.shape == (32, 5120)
    assert templates.local_embedding.shape == (32, 5120 // world)
    assert templates.query.shape == templates.attended.shape == (32, 32 // world, 128)
    assert templates.key.shape == templates.value.shape == (32, 8 // world, 128)
    assert templates.qkv.shape == (32, 6144 // world)
    assert templates.partial.shape == templates.output.shape == (32, 5120)
    assert templates.window_key.shape == (68, 128, 8 // world, 128)
    assert templates.window_value.shape == (68, 128, 8 // world, 128)
    assert templates.local_logits.shape == (32, 128 if world == 1 else 64)
    assert templates.local_logits.dtype == torch.bfloat16
    view = scratch.rows(8, 8)
    view_templates = _draft_templates(config, shard, view, 'meta')
    assert view.logit_shape == (8, 128)
    assert view_templates.query.shape == (8, 32 // world, 128)
    assert view_templates.embedding.shape == (8, 5120)
    assert view_templates.window_key.shape == (17, 128, 8 // world, 128)
    assert view_templates.convolution["before"].shape == (8, 5120)
    assert scratch.scratch_tensors() == view.scratch_tensors() == ()
    with pytest.raises(ValueError, match='complete proposal blocks'):
        scratch.rows(7, 7)


@pytest.mark.parametrize("width", [8, 13, 16])
def test_effective_block_preserves_checkpoint_and_scratch_fields(width):
    config = SimpleNamespace(block_size=8,
        text=dict(hidden_size=5120, intermediate_size=17408, vocab_size=128),
        attention=DraftAttentionGeometry(32, 8, 128, 128, 128, 10000000., 2048))
    shard = TensorParallelShard(0, 1)
    scratch = DraftScratch.allocate(config, shard, width, "meta", block_size=width)
    view = scratch.rows(width, width)
    templates = _draft_templates(config, shard, view, "meta")
    assert config.block_size == 8
    assert view.block_size == width
    assert templates.query.shape == (width, 32, 128)
    assert templates.window_key.shape[0] == 17
    assert templates.mlp.intermediate.shape == (width, 17408)
    assert templates.local_logits.shape == (width, 128)
    assert templates.gathered_logits.dtype == torch.bfloat16
    assert view.scratch_tensors() == ()


@pytest.mark.parametrize("model", ["dense", "flash"])
@pytest.mark.parametrize("world", [1, 2, 4])
def test_checkpoint_geometry_drives_shared_context_storage(model, world):
    from ignition.components.dflash2_geometry import DFlash2Config, draft_context_specs
    from ignition.recipes.qwen38.draft_config import EXPECTED
    from ignition.recipes.qwen38_dense.draft_config import PROFILE

    profile = EXPECTED if model == "flash" else PROFILE
    text = dict(model_type="qwen3", architectures=["DFlash2DraftModel"], hidden_act="silu",
        dtype="bfloat16", attention_bias=False, attention_dropout=0., tie_word_embeddings=False,
        use_sliding_window=True, is_causal=False, rms_norm_eps=1e-6, sliding_window=2048,
        layer_types=["sliding_attention"] * 5) | copy.deepcopy(profile)
    hidden, layers = (2560, 48) if model == "flash" else (5120, 64)
    config = DFlash2Config.from_dict(text, target_hidden=hidden, target_vocabulary=248320,
        target_layers=layers, expected=profile)
    shard = TensorParallelShard(0, world)
    config.validate_shard(shard)
    regions = dict((name, (shape, dtype)) for name, shape, dtype in draft_context_specs(config, shard))
    assert set(regions) == {f"draft{i}.{part}" for i in range(5) for part in ("key", "value")}
    assert set(regions.values()) == {((2048, 8 // world, 128), torch.float8_e4m3fn)}
    scratch = DraftScratch.allocate(config, shard, 8, "meta")
    templates = _draft_templates(config, shard, scratch, "meta")
    assert scratch.hidden_size == hidden
    assert templates.embedding.shape == (8, hidden)
    assert templates.query.shape == (8, 32 // world, 128)
    assert templates.local_embedding.shape == (8, hidden // world)
    assert templates.qkv.shape == (8, 6144 // world)
    with pytest.raises(ValueError, match="does not match the target"):
        DFlash2Config.from_dict(text, target_hidden=hidden, target_vocabulary=248320,
            target_layers=layers + 1, expected=profile)
