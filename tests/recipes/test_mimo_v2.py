# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiMo partitions retain each checkpoint row's original quantization scale."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.mimo_v2.attention import AttentionScratch
from ignition.recipes.mimo_v2.config import ModelConfig
from ignition.recipes.mimo_v2.weights import MiMoWeightLoader


def config(*, checkpoint_ranks=8, heads=128, kv_heads=(8, 8)):
    """A two-layer MiMo text geometry: one global-attention layer, one sliding-window layer.

    `kv_heads` gives the global and sliding KV head counts; the Pro checkpoint uses 8 and 8,
    the Flash RL checkpoint 4 and 8.
    """
    return ModelConfig.from_dict({
        "model_type": "mimo_v2", "architectures": ["MiMoV2ForCausalLM"],
        "attention_projection_layout": "fused_qkv", "attention_bias": False,
        "hidden_size": 128, "hidden_act": "silu", "num_hidden_layers": 2,
        "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, 1],
        "num_attention_heads": heads, "num_key_value_heads": kv_heads[0], "head_dim": 192,
        "v_head_dim": 128, "swa_num_attention_heads": heads, "swa_num_key_value_heads": kv_heads[1],
        "swa_head_dim": 192, "swa_v_head_dim": 128, "partial_rotary_factor": .334,
        "rope_theta": 10000000, "swa_rope_theta": 10000, "sliding_window_size": 128,
        "add_swa_attention_sink_bias": True, "n_shared_experts": None,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1, "topk_group": 1,
        "quantization_config": {"quant_method": "fp8", "store_dtype": "mxfp4",
            "weight_block_size": [128, 128], "mxfp4_block_size": 32,
            "activation_scheme": "dynamic", "fmt": "e4m3",
            "ignored_layers": ["model.layers.0.self_attn.o_proj"]},
    }, checkpoint_ranks=checkpoint_ranks)


def flash_config():
    return config(checkpoint_ranks=4, heads=64, kv_heads=(4, 8))


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_qkv_keeps_training_chunk_padding_and_scale_ownership(world):
    c = config()
    # Every training rank stores 3392 weight rows and 27 scale blocks (3456 padded rows).
    source = torch.ones((27136, 128), dtype=torch.float8_e4m3fn)
    scales = torch.arange(1, 217, dtype=torch.float32).reshape(216, 1)
    restored = []
    for rank in range(world):
        loader = object.__new__(MiMoWeightLoader)
        loader.config, loader.shard, loader.device = c, TensorParallelShard(rank, world), "cpu"
        loader._session = SimpleNamespace(flush=lambda: None)
        loader._sources = {"model.layers.0.self_attn.qkv_proj.weight": source,
                           "model.layers.0.self_attn.qkv_proj.weight_scale_inv": scales}
        loader._copy = lambda target, value: target.copy_(value)
        result = loader.load_qkv(0)
        chunks = c.qkv_chunks(0, loader.shard)
        offset = 0
        for chunk in chunks:
            n = chunk.weight_rows[1] - chunk.weight_rows[0]
            values = result.values[offset:offset + chunk.padded_rows].float()
            scale = result.scales[offset // 128:(offset + chunk.padded_rows) // 128].repeat_interleave(128, 0)
            restored.append((values * scale)[:n])
            assert torch.count_nonzero(values[n:]) == 0
            assert (chunk.query_rows, chunk.key_rows, chunk.value_rows) == (3072, 192, 128)
            offset += chunk.padded_rows
    reference = torch.cat([scales[i * 27:(i + 1) * 27].repeat_interleave(128, 0)[:3392]
                           for i in range(8)]).expand(-1, 128)
    torch.testing.assert_close(torch.cat(restored), reference, rtol=0, atol=0)


def test_format_follows_metadata():
    c = config()
    assert c.projection_format("model.layers.0.self_attn.o_proj") == "bf16"
    assert c.projection_format("model.layers.1.self_attn.o_proj") == "block_fp8"
    assert c.projection_format("model.layers.1.mlp.experts.0.gate_proj") == "mxfp4"


def test_attention_schedule_has_independent_rotary_and_window_geometry():
    global_attention, window = config().attention(0), config().attention(1)
    assert (global_attention.window, global_attention.theta, global_attention.sinks) == (-1, 10000000, False)
    assert (window.window, window.theta, window.sinks) == (128, 10000, True)
    assert global_attention.rotary_dim == window.rotary_dim == 64
    assert global_attention.local(TensorParallelShard(7, 8)) == (16, 1)


def test_checkpoint_chunk_geometry_rejects_inexact_sharding():
    with pytest.raises(ValueError, match="checkpoint's fused-QKV shard count"):
        config().qkv_chunks(0, TensorParallelShard(0, 3))
    with pytest.raises(ValueError, match="checkpoint's fused-QKV shard count"):
        config().qkv_chunks(0, TensorParallelShard(0, 16))


@pytest.mark.parametrize("world", [2, 4])
def test_flash_geometry_partitions_four_checkpoint_shards(world):
    c = flash_config()
    global_attention, window = c.attention(0), c.attention(1)
    assert (global_attention.heads, global_attention.kv_heads) == (64, 4)
    assert (window.heads, window.kv_heads) == (64, 8)
    shard = TensorParallelShard(0, world)
    assert global_attention.local(shard) == (64 // world, 4 // world)
    assert window.local(shard) == (64 // world, 8 // world)
    # Each of the four training shards stores 16 query heads and one or two KV heads per projection.
    chunks = c.qkv_chunks(0, shard)
    assert len(chunks) == 4 // world
    assert all((chunk.query_rows, chunk.key_rows, chunk.value_rows, chunk.padded_rows) == (3072, 192, 128, 3456)
               for chunk in chunks)
    chunks = c.qkv_chunks(1, shard)
    assert all((chunk.query_rows, chunk.key_rows, chunk.value_rows, chunk.padded_rows) == (3072, 384, 256, 3712)
               for chunk in chunks)
    with pytest.raises(ValueError, match="checkpoint's fused-QKV shard count"):
        c.qkv_chunks(0, TensorParallelShard(0, 8))


def test_attention_scratch_views_narrower_kv_layers_over_the_widest_allocation():
    c = flash_config()
    shard = TensorParallelShard(0, 2)
    scratch = AttentionScratch.allocate(c, shard, 16, "cpu", requests=2)
    # The scratch holds the sliding layers' four local KV heads; the global layer views two of them.
    assert scratch.key.shape == (16, 4, 192) and scratch.value.shape == (16, 4, 128)
    assert scratch.window_key.shape[2:] == (4, 192) and scratch.window_value.shape[2:] == (4, 128)
    rows = scratch.rows(8)
    narrow = rows.for_kv_heads(c.attention(0).local(shard)[1])
    assert narrow.key.shape == (8, 2, 192) and narrow.value.shape == (8, 2, 128)
    assert narrow.window_key.shape == (*scratch.window_key.shape[:2], 2, 192)
    assert narrow.window_value.shape == (*scratch.window_value.shape[:2], 2, 128)
    for name in ("key", "value", "window_key", "window_value"):
        wide, view = getattr(rows, name), getattr(narrow, name)
        assert view.is_contiguous() and view.data_ptr() == wide.data_ptr()
    assert narrow.query is rows.query and narrow.attended is rows.attended
    assert rows.for_kv_heads(4).key.shape == (8, 4, 192)
    with pytest.raises(ValueError, match="must fit"):
        rows.for_kv_heads(8)


def test_attention_scratch_requires_shared_query_geometry():
    c = config(heads=128, kv_heads=(8, 8))
    text = dict(c.text, swa_head_dim=128)
    with pytest.raises(ValueError, match="matching query head counts and head dimensions"):
        AttentionScratch.allocate(ModelConfig.from_dict(text, checkpoint_ranks=8), TensorParallelShard(0, 2), 4, "cpu")
