# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiMo text geometry and lossless partitions of chunked checkpoint projections."""

from dataclasses import dataclass
import json
from pathlib import Path

from ignition.components.parallel import TensorParallelShard


@dataclass(frozen=True)
class AttentionGeometry:
    heads: int
    kv_heads: int
    key_dim: int
    value_dim: int
    rotary_dim: int
    theta: float
    window: int
    sinks: bool

    def local(self, shard):
        shard.partition(self.heads)
        first, last = shard.replicated_partition(self.kv_heads)
        return self.heads // shard.world_size, last - first


@dataclass(frozen=True)
class QKVChunk:
    weight_rows: tuple[int, int]
    scale_rows: tuple[int, int]
    query_rows: int
    key_rows: int
    value_rows: int
    padded_rows: int


@dataclass(frozen=True)
class DFlashConfig:
    text: dict

    @classmethod
    def from_dict(cls, text, target):
        draft = text.get("dflash_config", {})
        layers = tuple(draft.get("target_layer_ids", ()))
        if (text.get("architectures") != ["DFlashDraftModel"]
                or text.get("hidden_size") != target["hidden_size"]
                or text.get("vocab_size") != target["vocab_size"]
                or text.get("hidden_act") != "silu" or text.get("attention_bias")
                or text.get("is_causal") is not False
                or text.get("layer_types") != ["sliding_attention"] * text["num_hidden_layers"]
                or text.get("sliding_window", 0) % 128 or text.get("sliding_window", 0) <= 0
                or not 2 <= text.get("block_size", 0) <= 8
                or not layers or layers != tuple(sorted(set(layers)))
                or any(not 0 <= i < target["num_hidden_layers"] for i in layers)
                or not 0 <= draft.get("mask_token_id", -1) < target["vocab_size"]):
            raise ValueError("unsupported MiMo DFlash geometry")
        return cls(text)

    @property
    def target_layers(self):
        return tuple(self.text["dflash_config"]["target_layer_ids"])

    @property
    def noise_token(self):
        return self.text["dflash_config"]["mask_token_id"]

    @property
    def block_size(self):
        return self.text["block_size"]

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def attention(self):
        c = self.text
        return AttentionGeometry(c["num_attention_heads"], c["num_key_value_heads"],
            c["head_dim"], c["v_head_dim"], int(c["head_dim"] * c["partial_rotary_factor"]),
            c["rope_theta"], c["sliding_window"], c["dflash_config"].get("attention_sink_bias", False))


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict
    checkpoint_ranks: int
    draft: DFlashConfig | None = None

    @classmethod
    def load(cls, snapshot):
        snapshot = Path(snapshot)
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        text = json.loads((snapshot / "config.json").read_text())
        return cls.from_dict(text, checkpoint_ranks=index["metadata"]["tp_size"],
            draft=DFlashConfig.from_dict(json.loads((snapshot / "dflash/config.json").read_text()), text))

    @classmethod
    def from_dict(cls, text, *, checkpoint_ranks, draft=None):
        if (text.get("model_type") != "mimo_v2"
                or text.get("architectures") != ["MiMoV2ForCausalLM"]):
            raise ValueError("MiMo preparation requires MiMoV2ForCausalLM")
        quant = text.get("quantization_config", {})
        if (quant.get("quant_method") != "fp8" or quant.get("store_dtype") != "mxfp4"
                or quant.get("weight_block_size") != [128, 128]
                or quant.get("mxfp4_block_size") != 32
                or quant.get("activation_scheme") != "dynamic" or quant.get("fmt") != "e4m3"):
            raise ValueError("MiMo requires block-128 FP8 projections and block-32 MXFP4 experts")
        TensorParallelShard(0, checkpoint_ranks)
        if (text.get("attention_projection_layout") != "fused_qkv"
                or text.get("hidden_act") != "silu" or text.get("attention_bias")
                or text.get("scoring_func") != "sigmoid" or text.get("topk_method") != "noaux_tc"
                or text.get("n_group") != 1 or text.get("topk_group") != 1
                or text.get("n_shared_experts") not in (None, 0)):
            raise ValueError("unsupported MiMo projection, activation or routing geometry")
        layers = text["num_hidden_layers"]
        for name in ("hybrid_layer_pattern", "moe_layer_freq"):
            if len(text[name]) != layers or any(value not in (0, 1) for value in text[name]):
                raise ValueError(f"{name} must describe every decoder layer")
        result = cls(text, quant, checkpoint_ranks, draft)
        for layer in range(layers):
            geometry = result.attention(layer)
            if (geometry.heads % checkpoint_ranks or geometry.kv_heads % checkpoint_ranks
                    or geometry.heads % geometry.kv_heads or geometry.rotary_dim % 2
                    or not 0 < geometry.rotary_dim <= geometry.key_dim
                    or geometry.key_dim % 16 or geometry.value_dim % 16):
                raise ValueError("attention geometry must preserve complete heads and rotary pairs")
        return result

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    def attention(self, layer):
        c = self.text
        sliding = bool(c["hybrid_layer_pattern"][layer])
        prefix = "swa_" if sliding else ""
        key_dim = c[prefix + "head_dim"]
        return AttentionGeometry(c[prefix + "num_attention_heads"], c[prefix + "num_key_value_heads"],
            key_dim, c[prefix + "v_head_dim"], int(key_dim * c["partial_rotary_factor"]),
            c[prefix + "rope_theta"], c["sliding_window_size"] if sliding else -1,
            c.get("add_swa_attention_sink_bias" if sliding else "add_full_attention_sink_bias", False))

    def qkv_chunks(self, layer, shard):
        """Keep each training shard's scale blocks intact, including its padded final block."""
        if self.checkpoint_ranks % shard.world_size:
            raise ValueError("runtime ranks must divide the checkpoint's fused-QKV shard count")
        a = self.attention(layer)
        q = a.heads // self.checkpoint_ranks * a.key_dim
        k = a.kv_heads // self.checkpoint_ranks * a.key_dim
        v = a.kv_heads // self.checkpoint_ranks * a.value_dim
        rows = q + k + v
        blocks = (rows + 127) // 128
        first, last = shard.partition(self.checkpoint_ranks)
        return tuple(QKVChunk((i * rows, (i + 1) * rows), (i * blocks, (i + 1) * blocks),
                              q, k, v, blocks * 128) for i in range(first, last))

    def projection_format(self, prefix):
        if prefix in self.quantization["ignored_layers"]:
            return "bf16"
        return "mxfp4" if ".experts." in prefix else "block_fp8"
