# SPDX-License-Identifier: Apache-2.0

"""MLA draft geometry, TP storage and accepted-window semantics."""

from dataclasses import replace
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.kimi3.config import DSparkConfig, ModelConfig
from ignition.recipes.kimi3.pool import continuation_layout, page_bytes
from ignition.recipes.kimi31.dflash_config import DFlash2Config
from ignition.recipes.kimi31.recipe import RECIPE


def configuration():
    checkpoint = Path(os.environ.get("FLARE_KIMI31_CHECKPOINT", str(Path.home() /
        ".cache/huggingface/hub/models--moonshotai--Kimi-K3.1-Pro/snapshots/"
        "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e")))
    if not (checkpoint / "config.json").is_file():
        pytest.skip("Kimi K3.1 checkpoint configuration is not installed")
    target = ModelConfig.load(checkpoint)
    raw = dict(architectures=["DFlash2DraftModel"], model_type="qwen3", dtype="bfloat16", hidden_act="silu",
        attention_bias=False, attention_dropout=0., is_causal=True, use_sliding_window=True,
        tie_word_embeddings=False, mla_use_nope=False, rope_interleave=True, q_lora_rank=1536,
        kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, qk_head_dim=192, v_head_dim=128,
        hidden_size=7168, num_attention_heads=96, vocab_size=163840, draft_vocab_size=163840,
        max_position_embeddings=1048576, num_hidden_layers=5, intermediate_size=14336, num_target_layers=93,
        sliding_window=1024, layer_types=["sliding_attention"]*5, rms_norm_eps=1e-5,
        dflash_config=dict(attention_mode="mla", aux_hidden_stream="attn_res", block_size=5,
            conv_kernel_size=2, conv_group_size=16, selector_rank=256, selector_top_k=32,
            mask_token_id=163824, target_layer_ids=[7,23,51,67,83,91]),
        rope_parameters=dict(rope_type="yarn", factor=32., rope_theta=50000.,
            original_max_position_embeddings=32768, mscale=1., mscale_all_dim=1., beta_fast=32, beta_slow=1))
    return target, raw


@pytest.mark.parametrize("world", [16])
@pytest.mark.parametrize("draft_tokens", [0, 7])
def test_sharded_draft_preserves_target_pages(world, draft_tokens):
    target, raw = configuration()
    draft = DSparkConfig.from_dict(raw, target.text)
    assert isinstance(draft, DFlash2Config)
    assert draft.auxiliary_layers == (8,24,52,68,84,92)
    assert draft.auxiliary_source == "attention_input"
    assert draft.auxiliary_stream == "attn_res"
    draft.validate_shard(TensorParallelShard(0, world))
    model = replace(target, draft=draft)
    layout = continuation_layout(model, world, speculative_tokens=draft_tokens)
    costs = RECIPE.speculative_methods["dflash2"].ranks[world]
    assert page_bytes(model) == page_bytes(target) == costs.page_bytes
    assert layout.row_bytes <= costs.state_row_bytes
    if draft_tokens == 7:
        assert layout.row_bytes == costs.state_row_bytes
    for i in range(draft.layers):
        region = layout.region(f"draft{i}.latent")
        assert region.shape == (1024,576)
        assert region.dtype == torch.float8_e4m3fn
    assert draft.attention_scale == pytest.approx(192**-.5*(1+.1*__import__('math').log(32))**2)


@pytest.mark.parametrize("embedded", [False, True])
def test_preparation_selects_dflash_without_loading_inferact(monkeypatch, tmp_path, embedded):
    from ignition.recipes import resolve
    from ignition.recipes.kimi31.dflash_config import REPOSITORY, REVISION
    from .test_declarations import fitted

    model = resolve("kimi-k3.1-pro")
    plan = fitted(model, 16, concurrency=4,
                  online_mxfp8=model.recipe.online_mxfp8,
                  online_mxfp8_scope=model.recipe.online_mxfp8_scope,
                  speculation=replace(model.recipe.speculation, method="dflash2"))
    # An available DSpark must not shadow the explicitly selected DFlash checkpoint.
    (tmp_path / "dspark").mkdir()
    if embedded:
        (tmp_path / "dflash2").mkdir()
    downloaded = []

    def download(repository, *, revision, cache_dir, allow_patterns):
        downloaded.append((repository, revision))
        assert allow_patterns == ("config.json", "model.safetensors")
        return str(tmp_path / "external")

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr("ignition.preparation.rank_group.prepare_group",
                        lambda snapshot, config, prepare, **kwargs: config)
    prepared = model.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert downloaded == ([] if embedded else [(REPOSITORY, REVISION)])
    assert prepared.draft_checkpoint == str(tmp_path / ("dflash2" if embedded else "external"))
    assert prepared.online_mxfp8_scope == "kda-input"
    assert prepared.draft_head_quantization is None
    assert prepared.snapshot_budget_bytes == plan.snapshots * RECIPE.speculative_methods["dflash2"].ranks[16].state_row_bytes


@pytest.mark.parametrize("online,scope", [(True, "kda-input"), (False, None), (True, "all")])
def test_dflash_body_and_head_precision_reach_config_and_rebuild_metadata(
        monkeypatch, tmp_path, online, scope):
    from google.protobuf import json_format
    from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto
    from ignition.recipes import resolve
    from .test_declarations import fitted

    model = resolve("kimi-k3.1-pro")
    plan = fitted(model, 16, concurrency=4, online_mxfp8=online, online_mxfp8_scope=scope,
        speculation=replace(model.recipe.speculation, method="dflash2",
                            draft_body_quantization="mxfp8", draft_head_quantization="nvfp4"))
    (tmp_path / "dflash2").mkdir()
    monkeypatch.setattr("ignition.preparation.rank_group.prepare_group",
                        lambda snapshot, config, prepare, **kwargs: config)
    prepared = model.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert prepared.online_mxfp8 == online
    assert prepared.online_mxfp8_scope == (scope or "all")
    assert prepared.draft_body_quantization == "mxfp8"
    assert prepared.draft_head_quantization == "nvfp4"
    manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
    encoded = proto.RebuildManifest()
    json_format.ParseDict(manifest, encoded)
    assert encoded.speculation.draft_body_quantization == "mxfp8"
    assert encoded.speculation.draft_head_quantization == "nvfp4"
    assert encoded.online_mxfp8_scope == (scope or "all")


@pytest.mark.parametrize("change", [dict(is_causal=False),dict(mla_use_nope=True),dict(sliding_window=1000),
    dict(num_target_layers=92),dict(hidden_size=8192),dict(rope_interleave=False)])
def test_rejects_incompatible_draft(change):
    target, raw = configuration()
    raw.update(change)
    with pytest.raises(ValueError):
        DFlash2Config.from_dict(raw, target.text)


@pytest.mark.parametrize("start,count", [(0,5),(17,5),(1023,5),(1024,5),(1025,3),(500001,5),(1025,0)])
def test_private_latent_window_excludes_inactive_and_preserves_history(start, count):
    from ignition.recipes.kimi31.dflash import stage_latent_window

    window, block, hidden = 1024,5,576
    history = torch.arange(window).remainder(61).bfloat16()[None,:,None].expand(2,window,hidden).clone()
    latent = torch.arange(block).add(100).bfloat16()[:,None].expand(block,hidden).clone()
    saved = history.clone()
    output = torch.full((9,128,576), float('nan'), dtype=torch.bfloat16)
    metadata = SimpleNamespace(starts=torch.tensor([start]), lengths=torch.tensor([count],dtype=torch.int32),
        state_rows=torch.tensor([1]), query_start_loc=torch.tensor([0,count],dtype=torch.int32))
    stage_latent_window(history,latent,output,metadata)
    actual = output.view(-1,576)
    prefix = min(start,window)
    expected = torch.zeros_like(actual)
    if count:
        expected[:prefix] = history[1,torch.arange(start-prefix,start).remainder(window)]
        expected[prefix:prefix+count] = latent[:count]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(history,saved,rtol=0,atol=0)


def test_checkpoint_context_regions_survive_zero_draft_depth():
    checkpoint = Path(os.environ.get("FLARE_KIMI31_DFLASH_CHECKPOINT", str(Path.home() /
        ".cache/huggingface/hub/models--RadixArk--Kimi-K3.1-Dflash2-Preview/snapshots/"
        "ea257b38a8ee755b65287bb8ccd0938b519ce26b")))
    if not (checkpoint / "config.json").is_file():
        pytest.skip("RadixArk Kimi K3.1 DFlash2 checkpoint is not installed")
    target, _ = configuration()
    draft = DSparkConfig.load(checkpoint, target)
    attached = replace(target, draft=draft)
    layout = continuation_layout(attached, 16, speculative_tokens=0)
    target_layout = continuation_layout(target, 16)
    assert layout.row_bytes > target_layout.row_bytes
    for index in range(draft.layers):
        region = layout.region(f"draft{index}.latent")
        assert region.shape == (draft.window, 576)
        assert region.dtype == torch.float8_e4m3fn
        with pytest.raises(KeyError):
            target_layout.region(region.name)
