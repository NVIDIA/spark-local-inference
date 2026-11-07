# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 drafter checkpoint checks and block-16 scratch through the shared DFlash2 components."""

from dataclasses import MISSING, dataclass, fields, replace
import os

import pytest
import torch

from ignition.components.dflash2 import DraftScratch
from ignition.components.dflash2_geometry import DraftAttentionGeometry
from ignition.components.dflash2_weights import DraftWeights, validate_metadata
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import GraphBufferPool

# z-lab/Muse-Glimmer-30B-DFlash2: hidden 6656, intermediate 19968, 32/8 heads x 128, five sliding layers
# tapping target layers [1, 13, 25, 37, 49], block 16, mask token 201818, two-tap group-16 convolutions,
# rank-256 top-16 candidate selector over the target's 202048-token vocabulary.
DFLASH_CONFIG = dict(block_size=16, conv_kernel_size=2, conv_group_size=16, selector_rank=256, selector_top_k=16,
    target_layer_ids=[1, 13, 25, 37, 49], mask_token_id=201818, output_multiplier=0.19611613513818404,
    final_logit_softcapping=20.0)
DFLASH2 = dict(hidden_size=6656, intermediate_size=19968, num_attention_heads=32, num_key_value_heads=8,
    head_dim=128, num_hidden_layers=5, rms_norm_eps=1e-5, sliding_window=2048, vocab_size=202048,
    dflash_config=DFLASH_CONFIG)
SMALL = {**DFLASH2, "hidden_size": 128, "intermediate_size": 256, "num_attention_heads": 4,
         "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 256,
         "dflash_config": {**DFLASH_CONFIG, "selector_rank": 16}}


@dataclass(frozen=True)
class StubDraftConfig:
    """The slice of a draft configuration the shared DFlash2 components read."""
    text: dict

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def target_layers(self):
        return tuple(self.text["dflash_config"]["target_layer_ids"])

    @property
    def block_size(self):
        return self.text["dflash_config"]["block_size"]

    @property
    def attention(self):
        c = self.text
        return DraftAttentionGeometry(c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
            c["head_dim"], c["head_dim"], 500000., c["sliding_window"])

    def validate_shard(self, shard):
        self.attention.local(shard)
        shard.partition(self.text["hidden_size"], alignment=64)
        shard.partition(self.text["intermediate_size"], alignment=128)


def expected_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """The DFlash2 tensor set a config declares: projection + norm, final norm, three selector tensors, 15 per layer."""
    h, inter, dflash = config["hidden_size"], config["intermediate_size"], config["dflash_config"]
    heads, kv, d = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    vocab, rank = config["vocab_size"], dflash["selector_rank"]
    taps, channels = dflash["conv_kernel_size"], 2 * dflash["conv_kernel_size"] * h // dflash["conv_group_size"]
    shapes = {
        "fc.weight": (h, h * len(dflash["target_layer_ids"])),
        "hidden_norm.weight": (h,), "norm.weight": (h,),
        "candidate_selector.hidden_projection.weight": (rank, h),
        "candidate_selector.predecessor_codebook": (vocab, rank),
        "candidate_selector.successor_codebook": (vocab, rank),
    }
    for i in range(config["num_hidden_layers"]):
        p = f"layers.{i}."
        shapes.update({
            p + "input_layernorm.weight": (h,), p + "post_attention_layernorm.weight": (h,),
            p + "self_attn.q_proj.weight": (heads * d, h), p + "self_attn.k_proj.weight": (kv * d, h),
            p + "self_attn.v_proj.weight": (kv * d, h), p + "self_attn.o_proj.weight": (h, heads * d),
            p + "self_attn.q_norm.weight": (d,), p + "self_attn.k_norm.weight": (d,),
            p + "mlp.gate_proj.weight": (inter, h), p + "mlp.up_proj.weight": (inter, h),
            p + "mlp.down_proj.weight": (h, inter),
            p + "attention_conv.base_kernel": (2, taps, h), p + "attention_conv.kernel_projection.weight": (channels, h),
            p + "mlp_conv.base_kernel": (2, taps, h), p + "mlp_conv.kernel_projection.weight": (channels, h),
        })
    return shapes


def fabricate(path, c, *, dtype=torch.bfloat16, random=False):
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(0)
    tensors = {name: (torch.randn(shape, generator=generator) if random else torch.zeros(shape)).to(dtype)
               for name, shape in expected_shapes(c).items()}
    save_file(tensors, str(path / "model.safetensors"))
    return tensors


def test_dflash2_tensor_set_is_81_bf16_tensors_with_the_published_shapes(tmp_path):
    fabricate(tmp_path, SMALL)
    manifest = inspect_indexed_checkpoint(tmp_path)
    assert len(manifest.tensors) == 81 and set(manifest.tensors) == set(expected_shapes(SMALL))
    validate_metadata(manifest, StubDraftConfig(SMALL))
    # The published checkpoint is 5.7 GB, so its header is reproduced as metadata only.
    template = manifest.tensors["norm.weight"]
    full = {name: replace(template, name=name, shape=shape) for name, shape in expected_shapes(DFLASH2).items()}
    assert len(full) == 81
    assert full["fc.weight"].shape == (6656, 33280) and full["layers.4.mlp.down_proj.weight"].shape == (6656, 19968)
    assert full["layers.0.self_attn.q_proj.weight"].shape == (4096, 6656)
    assert full["layers.0.attention_conv.base_kernel"].shape == (2, 2, 6656)
    assert full["layers.0.mlp_conv.kernel_projection.weight"].shape == (1664, 6656)
    assert full["candidate_selector.hidden_projection.weight"].shape == (256, 6656)
    assert full["candidate_selector.successor_codebook"].shape == (202048, 256)
    validate_metadata(replace(manifest, tensors=full), StubDraftConfig(DFLASH2))


@pytest.mark.parametrize("fault,message", [
    (dict(drop="layers.2.self_attn.k_norm.weight"), "tensor names disagree"),
    (dict(drop="candidate_selector.predecessor_codebook"), "tensor names disagree"),
    (dict(rename=("fc.weight", "encoder.fc.weight")), "tensor names disagree"),
    (dict(extra="encoder.output_norm_enc.weight"), "tensor names disagree"),
    (dict(reshape=("fc.weight", (128, 128))), r"fc.weight: DFlash2 requires BF16 storage with shape \(128, 640\)"),
    (dict(reshape=("layers.1.mlp_conv.kernel_projection.weight", (16, 128))),
     r"layers.1.mlp_conv.kernel_projection.weight: DFlash2 requires BF16 storage with shape \(32, 128\)"),
    (dict(dtype=torch.float32), "requires BF16 storage"),
])
def test_dflash2_check_rejects_missing_renamed_extra_misshaped_or_non_bf16_tensors(tmp_path, fault, message):
    fabricate(tmp_path, SMALL, dtype=fault.get("dtype", torch.bfloat16))
    manifest = inspect_indexed_checkpoint(tmp_path)
    tensors = dict(manifest.tensors)
    if "drop" in fault:
        del tensors[fault["drop"]]
    if "rename" in fault:
        old, new = fault["rename"]
        tensors[new] = replace(tensors.pop(old), name=new)
    if "extra" in fault:
        tensors[fault["extra"]] = replace(tensors["norm.weight"], name=fault["extra"])
    if "reshape" in fault:
        name, shape = fault["reshape"]
        tensors[name] = replace(tensors[name], shape=shape)
    with pytest.raises(ValueError, match=message):
        validate_metadata(replace(manifest, tensors=tensors), StubDraftConfig(SMALL))


def test_block_16_scratch_carries_the_convolution_buffers_and_has_no_default_block():
    assert next(f for f in fields(DraftScratch) if f.name == "block_size").default is MISSING
    config, shard = StubDraftConfig(SMALL), TensorParallelShard(0, 1)
    scratch = DraftScratch.allocate(config, shard, 32, "meta", buffers=GraphBufferPool("meta").cursor())
    assert scratch.block_size == 16 and scratch.convolution["coefficients"].shape == (32, 32)
    assert scratch.embedding.shape == (32, 128) and scratch.query.shape == (32, 4, 32)
    assert scratch.window_key.shape == (2 * 17, 128, 2, 32)
    rows = scratch.rows(16, 16)
    # Row views keep the whole convolution scratch; only per-token buffers narrow.
    assert rows.block_size == 16 and rows.convolution is scratch.convolution
    assert rows.embedding.shape == (16, 128)
    with pytest.raises(ValueError, match="complete proposal blocks"):
        DraftScratch.allocate(config, shard, 24, "meta", buffers=GraphBufferPool("meta").cursor())


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA_VISIBLE_DEVICES")
def test_dflash2_load_reads_the_projection_convolutions_and_selector(tmp_path):
    source = fabricate(tmp_path, SMALL, random=True)
    weights = DraftWeights.load(tmp_path, StubDraftConfig(SMALL), torch.device("cuda:0"), TensorParallelShard(0, 1))
    def check(actual, name):
        assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
        assert torch.equal(actual.cpu(), source[name]), name

    check(weights.projection, "fc.weight")
    check(weights.hidden_norm, "hidden_norm.weight")
    check(weights.norm, "norm.weight")
    assert len(weights.layers) == 5
    for i, layer in enumerate(weights.layers):
        p = f"layers.{i}."
        check(layer.qkv[:128], p + "self_attn.q_proj.weight")
        check(layer.qkv[128:192], p + "self_attn.k_proj.weight")
        check(layer.qkv[192:], p + "self_attn.v_proj.weight")
        assert torch.equal(weights.context_kv[i * 128:(i + 1) * 128], layer.qkv[128:])
        check(layer.output, p + "self_attn.o_proj.weight")
        check(layer.query_norm, p + "self_attn.q_norm.weight")
        check(layer.key_norm, p + "self_attn.k_norm.weight")
        check(layer.input_norm, p + "input_layernorm.weight")
        check(layer.post_attention_norm, p + "post_attention_layernorm.weight")
        check(layer.mlp.gate_up[:256], p + "mlp.gate_proj.weight")
        check(layer.mlp.gate_up[256:], p + "mlp.up_proj.weight")
        check(layer.mlp.down, p + "mlp.down_proj.weight")
        for name, conv in (("attention_conv", layer.attention_conv), ("mlp_conv", layer.mlp_conv)):
            assert conv is not None
            check(conv["base"], p + name + ".base_kernel")
            check(conv["projection"], p + name + ".kernel_projection.weight")
    assert set(weights.selector) == {"projection", "predecessor", "successor"}
    check(weights.selector["projection"], "candidate_selector.hidden_projection.weight")
    check(weights.selector["predecessor"], "candidate_selector.predecessor_codebook")
    check(weights.selector["successor"], "candidate_selector.successor_codebook")
