# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned DFlash2 configuration and rank-local BF16 checkpoint transfers."""

from dataclasses import replace
import copy
import os
from pathlib import Path

import pytest

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.muse_spark.config import ModelConfig
from ignition.recipes.muse_spark.draft_config import DraftConfig, REVISION

CACHE = Path('/data/cache/huggingface/hub')
TARGET = CACHE / 'models--meta-models--Muse-Spark-Open-816B-A42B-NVFP4/snapshots/4239e4db19376904f2e82c167e78f70abb5455fe'
DRAFT = CACHE / 'models--nvidia--Muse-Spark-Open-816B-A42B-DFlash2/snapshots' / REVISION


@pytest.fixture
def config():
    if not TARGET.is_dir() or not DRAFT.is_dir():
        pytest.skip("Muse Spark target and DFlash2 checkpoint configurations are unavailable")
    return ModelConfig.load(TARGET, draft_snapshot=DRAFT)


def test_draft_selection_and_geometry(config):
    assert ModelConfig.load(TARGET).draft is None
    d = config.draft
    assert (d.layers, d.block_size, d.noise_token) == (5, 8, 201817)
    assert d.target_layers == (3, 19, 31, 47, 59)
    for rank in range(8):
        shard = TensorParallelShard(rank, 8)
        d.validate_shard(shard)
        assert d.attention.local(shard) == (16, 2)
    assert (d.attention.window, d.attention.theta) == (2048, 500000.)


@pytest.mark.parametrize('key,value', [('hidden_size', 4096), ('rope_scaling', {'rope_type': 'linear'}),
    ('is_causal', True), ('intermediate_size', 16384), ('torch_dtype', 'float32'), ('vocab_size', 202049)])
def test_incompatible_decoder_rejected(config, key, value):
    raw = copy.deepcopy(config.draft.text)
    raw[key] = value
    with pytest.raises(ValueError):
        DraftConfig.from_dict(raw, config)


@pytest.mark.parametrize('key,value', [('target_layer_ids', [3, 19, 31, 47, 60]),
    ('mask_token_id', 0), ('conv_group_size', 32), ('selector_top_k', 8), ('causal', True),
    ('output_multiplier', 1.)])
def test_incompatible_projector_rejected(config, key, value):
    raw = copy.deepcopy(config.draft.text)
    raw['dflash_config'][key] = value
    with pytest.raises(ValueError):
        DraftConfig.from_dict(raw, config)


def test_tensor_metadata_checked_before_loading(config):
    from ignition.recipes.muse_spark.draft_weights import validate_metadata
    manifest = inspect_indexed_checkpoint(DRAFT)
    validate_metadata(manifest, config.draft)
    for shape, dtype in [((1, 8192), 'BF16'), ((8192, 40960), 'F32')]:
        tensors = dict(manifest.tensors)
        tensors['fc.weight'] = replace(tensors['fc.weight'], shape=shape, dtype=dtype)
        with pytest.raises(ValueError, match='fc.weight'):
            validate_metadata(replace(manifest, tensors=tensors), config.draft)
    tensors = dict(manifest.tensors)
    del tensors['candidate_selector.predecessor_codebook']
    with pytest.raises(ValueError, match='tensor names'):
        validate_metadata(replace(manifest, tensors=tensors), config.draft)


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires assigned CUDA_VISIBLE_DEVICES')
def test_checkpoint_shards_and_replicated_weights(config):
    import torch
    from safetensors import safe_open
    from ignition.recipes.muse_spark.draft_weights import DraftWeights

    rank = 3
    weights = DraftWeights.load(DRAFT, config.draft, torch.device('cuda:0'), TensorParallelShard(rank, 8))
    with safe_open(DRAFT / 'model.safetensors', framework='pt', device='cpu') as source:
        def check(actual, name, extent=None):
            expected = source.get_tensor(name) if extent is None else source.get_slice(name)[extent]
            assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
            assert torch.equal(actual.cpu(), expected), name

        check(weights.projection, 'fc.weight', (slice(3072, 4096), slice(None)))
        check(weights.hidden_norm, 'hidden_norm.weight')
        check(weights.norm, 'norm.weight')
        for i, layer in enumerate(weights.layers):
            p = f'layers.{i}.'
            for j, name in enumerate(('q', 'k', 'v')):
                first, last = (3072, 4096) if j == 0 else (384, 512)
                part = layer.qkv[:1024] if j == 0 else layer.qkv[1024 + (j-1)*128:1024+j*128]
                check(part, p + 'self_attn.' + name + '_proj.weight', (slice(first, last), slice(None)))
            assert torch.equal(weights.context_kv[i*256:(i+1)*256], layer.qkv[1024:])
            check(layer.output, p + 'self_attn.o_proj.weight', (slice(None), slice(3072, 4096)))
            for actual, name in ((layer.query_norm, 'self_attn.q_norm'), (layer.key_norm, 'self_attn.k_norm'),
                                 (layer.input_norm, 'input_layernorm'), (layer.post_attention_norm, 'post_attention_layernorm')):
                check(actual, p + name + '.weight')
            for j, name in enumerate(('gate', 'up')):
                check(layer.mlp.gate_up[j*3072:(j+1)*3072], p + 'mlp.' + name + '_proj.weight',
                      (slice(9216, 12288), slice(None)))
            check(layer.mlp.down, p + 'mlp.down_proj.weight', (slice(None), slice(9216, 12288)))
            for name, conv in (('attention_conv', layer.attention_conv), ('mlp_conv', layer.mlp_conv)):
                check(conv['base'], p + name + '.base_kernel')
                check(conv['projection'], p + name + '.kernel_projection.weight')
        for key, name in (('projection', 'hidden_projection.weight'), ('predecessor', 'predecessor_codebook'),
                          ('successor', 'successor_codebook')):
            check(weights.selector[key], 'candidate_selector.' + name)
