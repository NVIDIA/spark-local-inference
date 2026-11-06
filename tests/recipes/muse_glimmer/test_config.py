# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Glimmer composition and checkpoint-derived planning geometry."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes import resolve
from ignition.recipes.muse_glimmer.config import ModelConfig
from ignition.recipes.muse_glimmer.model import decoder_layer
from ignition.recipes.muse_glimmer.pool import continuation_layout, page_bytes

SNAPSHOT = Path(os.environ.get('FLARE_MUSE_GLIMMER_SNAPSHOT',
    '/data/cache/huggingface/hub/models--nvidia--Muse-Glimmer-30B-NVFP4/snapshots/47818374517751c48c55cde2621594926b1888b6'))


def test_checkpoint_geometry_and_sampling():
    if not (SNAPSHOT / 'config.json').is_file():
        pytest.skip('requires pinned Muse Glimmer checkpoint metadata')
    config = ModelConfig.load(SNAPSHOT)
    model = resolve('muse-glimmer-30b-nvfp4')
    assert model.checkpoint.architecture == 'MuseGlimmerForConditionalGeneration'
    assert resolve(str(SNAPSHOT)).recipe.name == model.recipe.name
    assert (config.hidden, config.layers, config.vocabulary) == (6656, 52, 202048)
    assert [config.window(i) for i in range(4)] == [2047, 2047, 2047, -1]
    assert config.text['layer_rope_theta'][:4] == [500000, 500000, 500000, 0]
    assert config.quantization['model.language_model.layers.0.self_attn.q_proj']['quant_algo'] == 'FP8'
    assert 'model.language_model.layers.7.self_attn.q_proj' not in config.quantization
    costs = model.recipe.costs(1, None)
    assert (continuation_layout(config).row_bytes, page_bytes(config)) == (costs.state_row_bytes, costs.page_bytes)
    assert model.recipe.max_images == model.recipe.max_draft_tokens == 0
    assert model.recipe.speculation is None
    sampling = model.recipe.sampling_defaults
    assert (sampling.temperature, sampling.top_p, sampling.top_k) == (1., .95, 64)


def test_reject_changed_quantization_and_geometry():
    if not (SNAPSHOT / 'config.json').is_file():
        pytest.skip('requires pinned Muse Glimmer checkpoint metadata')
    original = json.loads((SNAPSHOT / 'config.json').read_text())
    for section, key, value in [('quantization_config', 'kv_cache_quant_algo', 'FP8'),
                                ('text_config', 'hidden_activation', 'gelu'),
                                ('text_config', 'layer_rope_theta', [0]),
                                ('text_config', 'attention_bias', True)]:
        changed = copy.deepcopy(original)
        changed[section][key] = value
        with pytest.raises(ValueError):
            ModelConfig.from_dict(changed)


@torch.no_grad()
def test_canonical_centered_norms_and_residual_forward():
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextDecoderLayer

    torch.manual_seed(332)
    config = MuseGlimmerTextConfig(hidden_size=256, intermediate_size=512,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=128,
        layer_types=['full_attention'], layer_rope_theta=[0])
    config._attn_implementation = 'eager'
    reference = MuseGlimmerTextDecoderLayer(config, layer_idx=0).eval().bfloat16()
    for name, parameter in reference.named_parameters():
        if 'layernorm' in name:
            parameter.copy_(torch.randn_like(parameter) * .15)
    weights = SimpleNamespace(input_norm=reference.input_layernorm.weight,
        post_attention_norm=reference.post_attention_layernorm.weight,
        pre_feedforward_norm=reference.pre_feedforward_layernorm.weight,
        post_feedforward_norm=reference.post_feedforward_layernorm.weight)
    prepared = decoder_layer(SimpleNamespace(text=config.to_dict()), weights,
        copy.deepcopy(reference.self_attn), copy.deepcopy(reference.mlp))
    source = torch.randn((2, 11, config.hidden_size), dtype=torch.bfloat16)
    mask = torch.full((11, 11), float('-inf')).triu(1)[None, None]
    actual = prepared(source, attention_mask=mask)
    expected = reference(source, attention_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_text_contract_preserves_all_paged_histories():
    from google.protobuf.json_format import ParseDict
    from ignition.preparation import contract as c, runtime_package_pb2 as p
    from ignition.preparation.execution import GraphSignature
    from ignition.recipes.muse_glimmer.contract import prepare
    from ignition.recipes.muse_glimmer.pool import ContinuationPool

    config = ModelConfig(dict(hidden_size=256, num_hidden_layers=2, num_key_value_heads=2, head_dim=128), {})
    pool = ContinuationPool.allocate(config, 3, 2, 'cpu')
    buffers = {}
    signatures = {kind: GraphSignature(kind, 1, 128 if 'prefill' in kind else 1, tuple(c.text_inputs(kind)))
        for kind in ('init', 'restore', 'snapshot', 'prefill', 'scored_prefill', 'decode', 'resume')}
    encoded = ParseDict(prepare(signatures, buffers, pool, vocabulary=202048), p.ModelContract())
    assert encoded.vision.max_images == 0
    assert encoded.frontend.tools.kind == p.ATEM
    assert encoded.frontend.reasoning.kind == p.RECIPIENT_HEADERS
    histories = [r for r in encoded.cache.regions if r.kind == p.PAGED_KEY_VALUE]
    assert {r.storage.buffer for r in histories} == {f'cache.layer{i}.{kind}'
        for i in range(2) for kind in ('key', 'value')}
    assert all(r.storage.stride == 128 * 2 * 128 * 2 for r in histories)
    for graph in encoded.graphs:
        assert not graph.HasField('draft_cursor')
        assert any(s.view.nbytes == pool.layout.row_bytes and s.access == p.READ_WRITE for s in graph.state)
    decode = next(g for g in encoded.graphs if g.kind == p.DECODE)
    assert decode.pending_token.buffer == 'request_pending_token'
    assert all(b.source not in (p.PROMPT_TOKEN, p.PENDING_TOKEN) for b in decode.inputs)
