# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry and compression contracts for DeepSeek V4 Flash Vision."""

import json
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
REVISION = "6821d6ad3681a4b137b066b76094fa82ebd0a380"


@dataclass(frozen=True)
class Compressor:
    layer: int
    ratio: int
    head_dim: int
    index: bool = False

    @property
    def prefix(self):
        return f"layer{self.layer}.{'index_' if self.index else ''}compressor"

    @property
    def overlap(self):
        return self.ratio == 4

    @property
    def state_shape(self):
        factor = 2 if self.overlap else 1
        return self.ratio * factor, self.head_dim * factor


@dataclass(frozen=True)
class DSparkConfig:
    layers: int
    block_size: int
    noise_token: int
    target_layers: tuple[int, ...]
    markov_rank: int
    experts: int
    top_k: int


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, config):
        if config.get("model_type") != "deepseek_v4" or config.get("architectures") != [
            "DeepseekV4ForCausalLM"
        ]:
            raise ValueError("DeepSeek V4 requires DeepseekV4ForCausalLM")
        for name, expected in {
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "vocab_size": 129280,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "num_hash_layers": 3,
            "scoring_func": "sqrtsoftplus",
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
            "hc_mult": 4,
            "sliding_window": 128,
            "vision_n_layers": 32,
            "vision_dim": 1024,
            "vision_n_heads": 16,
            "vision_inter_dim": 2816,
            "vision_patch_size": 14,
            "vision_downsample_ratio": 3,
            "vision_max_n_token": 384,
            "num_nextn_predict_layers": 3,
            "dspark_markov_rank": 256,
        }.items():
            if config.get(name) != expected:
                raise ValueError(
                    f"DeepSeek V4 Flash Vision {name} requires {expected}, got {config.get(name)}"
                )
        quant = config["quantization_config"]
        if (
            quant.get("quant_method") != "fp8"
            or quant.get("weight_block_size") != [128, 128]
            or quant.get("scale_fmt") != "ue8m0"
            or config.get("expert_dtype") != "fp4"
            or quant.get("activation_scheme") != "dynamic"
        ):
            raise ValueError("DeepSeek V4 requires block-128 FP8 dense weights and native MXFP4 experts")
        result = cls(config, quant)
        ratios = config.get("compress_ratios", [])
        if (
            len(ratios) != result.layers + config["num_nextn_predict_layers"]
            or any(type(r) is not int or r not in (0, 4, 128) for r in ratios)
            or ratios[result.layers :] != [0] * config["num_nextn_predict_layers"]
        ):
            raise ValueError("DeepSeek V4 requires C4/C128 target compression and uncompressed DSpark stages")
        targets = config.get("dspark_target_layer_ids", [])
        noise = config.get("dspark_noise_token_id", -1)
        block = config.get("dspark_block_size", 0)
        if (
            type(block) is not int
            or block <= 0
            or type(noise) is not int
            or not 0 <= noise < config["vocab_size"]
            or not targets
            or any(type(i) is not int or not 0 <= i < result.layers for i in targets)
            or targets != sorted(set(targets))
        ):
            raise ValueError("DeepSeek V4 requires valid checkpoint-integrated DSpark metadata")
        return result

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def draft(self):
        c = self.text
        return DSparkConfig(
            c["num_nextn_predict_layers"],
            c["dspark_block_size"],
            c["dspark_noise_token_id"],
            tuple(c["dspark_target_layer_ids"]),
            c["dspark_markov_rank"],
            c["n_routed_experts"],
            c["num_experts_per_tok"],
        )

    @property
    def compressors(self):
        result = []
        for layer, ratio in enumerate(self.text["compress_ratios"][: self.layers]):
            if ratio:
                result.append(Compressor(layer, ratio, self.text["head_dim"]))
            if ratio == 4:
                result.append(Compressor(layer, ratio, self.text["index_head_dim"], index=True))
        return tuple(result)
