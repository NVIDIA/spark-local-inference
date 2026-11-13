# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact checkpoint partitioning for GLM tensor-parallel weights."""

import os
from itertools import pairwise
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader, _expert_scales
from ignition.reference.qwen38_quant import decode_csf_scales, decode_csf_slab_scales


@pytest.fixture(scope="module")
def manifest():
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    return inspect_indexed_checkpoint(snapshot)


def read(manifest, name, extent=None):
    with safe_open(str(manifest.snapshot / manifest.tensors[name].shard), framework="pt") as source:
        if extent is not None:
            return source.get_slice(name)[extent].clone()
        return source.get_tensor(name).clone()


@pytest.mark.parametrize("world", [2, 4])
def test_tp_partitions_cover_checkpoint_exactly(world):
    for extent, alignment in ((64, 1), (8192, 128), (16384, 256), (32768, 512), (2048, 128)):
        partitions = [TensorParallelShard(rank, world).partition(extent, alignment=alignment)
                      for rank in range(world)]
        assert partitions[0][0] == 0 and partitions[-1][1] == extent
        assert all(a[1] == b[0] for a, b in pairwise(partitions))
        assert all(a % alignment == b % alignment == 0 for a, b in partitions)


@pytest.mark.parametrize("rank,world", [(1, 1), (0, 0), (-1, 2), (2, 2), (True, 2)])
def test_invalid_tp(rank, world):
    with pytest.raises(ValueError, match="positive world size and a rank within it"):
        TensorParallelShard(rank, world)


def test_loader_construction_leaves_cuda_uninitialized(manifest):
    initialized = torch.cuda.is_initialized()
    loader = GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 2))
    with pytest.raises(RuntimeError, match="open GlmWeightLoader"):
        loader.load_tensor(f"{TEXT_PREFIX}.embed_tokens.weight")
    assert torch.cuda.is_initialized() == initialized


def test_gate_up_activation_scale_uses_larger_calibration():
    values = {f"expert.{name}.{suffix}": torch.tensor(value) for name in ("up_proj", "gate_proj", "down_proj")
              for suffix, value in (("input_scale", .25), ("weight_scale_2", .01))}
    values["expert.gate_proj.input_scale"] = torch.tensor(.5)
    scales = _expert_scales(values, ("expert",))
    assert scales[2].item() == 2
    assert scales[3].item() == 4
    values["expert.gate_proj.weight_scale_2"] *= 2
    with pytest.raises(ValueError, match="global scales differ"):
        _expert_scales(values, ("expert",))


cuda = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")


def assert_mxfp8(manifest, packed, name, *, rows=None, columns=None, output_rows=None):
    rows = slice(None) if rows is None else slice(*rows)
    columns = slice(None) if columns is None else slice(*columns)
    expected = read(manifest, name + ".weight", (rows, columns)).view(torch.uint8)
    scale_columns = slice(None) if columns.start is None else slice(columns.start // 32, columns.stop // 32)
    expected_scales = read(manifest, name + ".weight_scale", (rows, scale_columns))
    actual = packed.weight.values.view(torch.uint8).reshape(packed.out_features, -1).cpu()
    scales = packed.weight.scale_rows.view(torch.uint8).reshape(packed.out_features, -1).cpu()
    if output_rows is not None:
        actual, scales = actual[slice(*output_rows)], scales[slice(*output_rows)]
    assert torch.equal(actual[:, :expected.shape[1]], expected)
    assert torch.equal(scales[:, :expected_scales.shape[1]], expected_scales)


@cuda
@pytest.mark.parametrize("world,rank", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_actual_dense_and_attention_shards(manifest, world, rank):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    shard = TensorParallelShard(rank, world)
    prefixes = tuple(f"{TEXT_PREFIX}.layers.{layer}." for layer in (0, 3))
    with GlmWeightLoader(manifest, "cuda:0", shard=shard, prefixes=prefixes) as loader:
        kda = loader.load_attention(0)
        mla = loader.load_attention(3)
        dense = loader.load_mlp(0)
        residual = loader.load_residual(3)
        assert loader._session.torch_copy_bytes == 0
    channels = shard.partition(8192)
    prefix = f"{TEXT_PREFIX}.layers.0.self_attn"
    width = channels[1] - channels[0]
    for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
        assert_mxfp8(manifest, kda["input_proj"], f"{prefix}.{name}", rows=channels,
                     output_rows=(index * width, (index + 1) * width))
    for name in ("f_b_proj", "g_b_proj"):
        assert_mxfp8(manifest, kda[name], f"{prefix}.{name}", rows=channels)
    heads = 64 // world
    assert_mxfp8(manifest, kda["input_proj"], f"{prefix}.b_proj", rows=shard.partition(64),
                 output_rows=(3 * width, 3 * width + heads))
    assert_mxfp8(manifest, kda["input_proj"], f"{prefix}.f_a_proj",
                 output_rows=(3 * width + heads, 3 * width + heads + 128))
    assert_mxfp8(manifest, kda["o_proj"], f"{prefix}.o_proj", columns=channels)
    assert_mxfp8(manifest, kda["g_a_proj"], f"{prefix}.g_a_proj")
    expected_conv = torch.cat([read(manifest, f"{prefix}.{name}_conv1d.weight", slice(*channels))
                               for name in ("q", "k", "v")]).squeeze(1)
    assert torch.equal(kda["conv1d.weight"].cpu(), expected_conv)
    assert torch.equal(kda["A_log"].cpu(), read(manifest, prefix + ".A_log", slice(*shard.partition(64))))
    assert torch.equal(kda["dt_bias"].cpu(), read(manifest, prefix + ".dt_bias", slice(*channels)))
    prefix = f"{TEXT_PREFIX}.layers.3.self_attn"
    for name, width in (("q_b_proj", 16384), ("kv_b_proj", 32768)):
        assert_mxfp8(manifest, mla[name], f"{prefix}.{name}", rows=shard.partition(width))
    assert_mxfp8(manifest, mla["o_proj"], f"{prefix}.o_proj", columns=shard.partition(16384))
    assert_mxfp8(manifest, mla["qkv_a_proj"], f"{prefix}.q_a_proj", output_rows=(0, 1536))
    assert_mxfp8(manifest, mla["qkv_a_proj"], f"{prefix}.kv_a_proj_with_mqa", output_rows=(1536, 2048))
    assert_mxfp8(manifest, mla["qkv_a_proj"], f"{prefix}.indexer.wk", output_rows=(2048, 2176))
    assert_mxfp8(manifest, mla["qkv_a_proj"], f"{prefix}.indexer.weights_proj", output_rows=(2176, 2208))
    assert_mxfp8(manifest, mla["indexer.wq_b"], f"{prefix}.indexer.wq_b")
    prefix = f"{TEXT_PREFIX}.layers.0.mlp"
    extent = slice(*shard.partition(12288))
    expected = torch.cat([read(manifest, f"{prefix}.{name}.weight", extent) for name in ("gate_proj", "up_proj")])
    assert torch.equal(dense.gate_up.cpu(), expected)
    assert torch.equal(dense.down.cpu(), read(manifest, prefix + ".down_proj.weight", (slice(None), extent)))
    for name, value in residual.items():
        assert torch.equal(value.cpu(), read(manifest, f"{TEXT_PREFIX}.layers.3.{name}"))


@cuda
@pytest.mark.parametrize("world,rank", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_actual_nvfp4_shards_and_shared_scale_storage(manifest, world, rank):
    from b12x.moe import fused_moe as moe

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    shard = TensorParallelShard(rank, world)
    ids = (0, 287)
    prefix = f"{TEXT_PREFIX}.layers.3.mlp.experts"
    with GlmWeightLoader(manifest, "cuda:0", shard=shard,
                         prefixes=tuple(f"{prefix}.{i}." for i in ids)) as loader:
        scratch = loader.allocate_expert_scale_scratch(len(ids))
        experts = loader.load_experts(3, expert_ids=ids, scale_scratch=scratch)
        assert loader._session.torch_copy_bytes == 0
    assert experts.plan.activation.swiglu_limit == 10
    assert experts._impl.w1_blockscale.data_ptr() == scratch[0].data_ptr()
    assert experts._impl.w2_blockscale.data_ptr() == scratch[1].data_ptr()
    moe.expand_scales(experts)
    local = 2048 // world
    decoded = []
    for storage, rows, columns in ((scratch[0], 2 * local, 256), (scratch[1], 4096, local // 16)):
        logical = storage.view(torch.uint8).reshape(len(ids), rows // 128, columns // 4, 32, 4, 4)
        decoded.append(logical.permute(0, 1, 4, 3, 2, 5).reshape(len(ids), rows, columns).cpu())
    first, last = shard.partition(2048)
    for index, expert in enumerate(ids):
        scales, weights = {}, {}
        for name, rows, columns in (("up_proj", 2048, 256), ("gate_proj", 2048, 256), ("down_proj", 4096, 128)):
            name_prefix = f"{prefix}.{expert}.{name}"
            if f"{name_prefix}.weight_scale.csf32_bases" in manifest.tensors:
                planes = [read(manifest, f"{name_prefix}.weight_scale.csf32_{suffix}")
                          for suffix in ("bases", "codes", "page_bases", "offsets", "exceptions")]
                full = decode_csf_scales(*planes, rows=rows, columns=columns)
            else:
                full = decode_csf_slab_scales(
                    read(manifest, f"{name_prefix}.weight_scale.nvfp4_csf_fixed"),
                    read(manifest, f"{name_prefix}.weight_scale.nvfp4_csf_exceptions"),
                    rows=rows, columns=columns)
            scales[name] = full[:, first // 16:last // 16] if name == "down_proj" else full[first:last]
            extent = (slice(None), slice(first // 2, last // 2)) if name == "down_proj" else slice(first, last)
            weights[name] = read(manifest, f"{name_prefix}.weight", extent)
        assert torch.equal(experts._impl.w1_fp4[index].cpu(), torch.cat((weights["up_proj"], weights["gate_proj"])))
        assert torch.equal(experts._impl.w2_fp4[index].cpu(), weights["down_proj"])
        assert torch.equal(decoded[0][index], torch.cat((scales["up_proj"], scales["gate_proj"])))
        assert torch.equal(decoded[1][index], scales["down_proj"])


@cuda
@pytest.mark.parametrize("world,rank", [(2, 1), (4, 3)])
def test_mtp_experts_preserve_packed_weights_and_e4m3_scales(manifest, world, rank, monkeypatch):
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.moe import fused_moe

    packed = {}
    prepare = fused_moe.prepare_weights

    def capture(*, plan, weights):
        for name in ("w13", "w2", "w13_block_scales", "w2_block_scales"):
            packed[name] = getattr(weights, name).clone()
        return prepare(plan=plan, weights=weights)

    monkeypatch.setattr(fused_moe, "prepare_weights", capture)

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    shard = TensorParallelShard(rank, world)
    ids = (0, 287)
    prefix = f"{TEXT_PREFIX}.layers.45.mlp.experts"
    with GlmWeightLoader(manifest, "cuda:0", shard=shard,
                         prefixes=tuple(f"{prefix}.{i}." for i in ids)) as loader:
        experts = loader.load_experts(45, expert_ids=ids)
        assert loader._session.torch_copy_bytes == 0
    assert experts.plan.activation.mode == "a16"
    first, last = shard.partition(2048)
    weights, scales = {}, {}
    for projection in ("up_proj", "gate_proj", "down_proj"):
        extent = (slice(None), slice(first // 2, last // 2)) if projection == "down_proj" else slice(first, last)
        scale_extent = (slice(None), slice(first // 16, last // 16)) if projection == "down_proj" else slice(first, last)
        weights[projection] = torch.stack([read(manifest, f"{prefix}.{expert}.{projection}.weight", extent) for expert in ids])
        scales[projection] = torch.stack([read(manifest, f"{prefix}.{expert}.{projection}.weight_scale", scale_extent) for expert in ids])
    assert torch.equal(packed["w13"].cpu(), torch.cat((weights["up_proj"], weights["gate_proj"]), dim=1))
    assert torch.equal(packed["w2"].cpu(), weights["down_proj"])
    for actual, expected in ((packed["w13_block_scales"], torch.cat((scales["up_proj"], scales["gate_proj"]), dim=1)),
                             (packed["w2_block_scales"], scales["down_proj"])):
        expected = swizzle_block_scale(expected.cuda())
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
