# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint geometry and source ownership for DeepSeek V4.1 Flash."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheSource:
    layer: int
    ratio: int

    def records(self, tokens):
        if tokens % self.ratio:
            raise ValueError("cache page tokens must be divisible by the compression ratio")
        return tokens // self.ratio


@dataclass(frozen=True)
class DSparkConfig:
    layers: int
    block_size: int
    noise_token: int
    target_layers: tuple[int, ...]
    markov_rank: int
    experts: int
    top_k: int

    @classmethod
    def from_text(cls, text):
        result = cls(text["num_nextn_predict_layers"], text["dspark_block_size"],
            text["dspark_noise_token_id"], tuple(text["dspark_target_layer_ids"]),
            text["dspark_markov_rank"], text["dspark_n_routed_experts"],
            text["dspark_num_experts_per_tok"])
        if (result.layers != 3 or result.block_size < 1 or result.markov_rank != 256
                or result.experts != 128 or result.top_k != 3
                or not 0 <= result.noise_token < text["vocab_size"]
                or not result.target_layers or tuple(sorted(set(result.target_layers))) != result.target_layers
                or any(not 0 <= i < text["num_hidden_layers"] for i in result.target_layers)):
            raise ValueError("unsupported DeepSeek V4.1 Flash DSpark geometry")
        start = text["num_hidden_layers"]
        if text["compress_ratios"][start:start + result.layers] != [0] * result.layers:
            raise ValueError("DSpark stages require uncompressed sliding-window attention")
        return result


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict
    vision: dict | None = None

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, config):
        if (config.get("model_type") != "deepseek_v41"
                or config.get("architectures") != ["DeepseekV41ForCausalLM"]):
            raise ValueError("DeepSeek V4.1 requires DeepseekV41ForCausalLM")
        text, quant = config["text_config"], config["quantization_config"]
        for name, value in {
            "hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 64,
            "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64,
            "q_lora_rank": 1280, "o_lora_rank": 1024, "o_groups": 8,
            "vocab_size": 129280, "moe_intermediate_size": 2304, "n_routed_experts": 384,
            "n_shared_experts": 1, "num_experts_per_tok": 6, "scoring_func": "sqrtsoftplus",
            "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512,
            "candidate_topk_blocks": 2048, "candidate_block_size": 8, "hc_mult": 4,
            "sliding_window": 128, "engram_max_ngram_size": 4, "engram_n_heads": 8,
            "engram_head_dim": 256,
        }.items():
            if text.get(name) != value:
                raise ValueError(f"DeepSeek V4.1 Flash {name} requires {value}, got {text.get(name)}")
        if (quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [32, 32]
                or quant.get("scale_fmt") != "ue8m0" or quant.get("expert_dtype") != "fp4"
                or quant.get("activation_scheme") != "dynamic"):
            raise ValueError("DeepSeek V4.1 requires block-32 FP8 weights and native MXFP4 experts")
        vision = config.get("vision_config")
        if vision is not None:
            for key, expected in {"num_hidden_layers": 32, "hidden_size": 1024,
                    "num_attention_heads": 16, "intermediate_size": 2816, "patch_size": 14,
                    "rope_theta": 10000, "downsample_ratio": 3, "max_image_tokens": 1024,
                    "min_pixels": 295936, "max_wh_ratio": None}.items():
                if vision.get(key) != expected:
                    raise ValueError(f"unsupported DeepSeek V4.1 vision {key}: {vision.get(key)}")
            if config.get("image_token_id") != 129264:
                raise ValueError("DeepSeek V4.1 image placeholder requires token 129264")
        result = cls(text, quant, vision)
        ratios = text["compress_ratios"]
        if len(ratios) < result.layers or any(r not in (0, 1, 2) for r in ratios):
            raise ValueError("unsupported DeepSeek compression schedule")
        for name in ("kv_source_layer_ids", "index_source_layer_ids"):
            ids = text[name]
            if not ids or ids != sorted(set(ids)) or any(not 0 <= i < result.layers for i in ids):
                raise ValueError(f"{name} must be ordered unique backbone layers")
        if not set(text["kv_source_layer_ids"]) <= set(text["index_source_layer_ids"]):
            raise ValueError("KV source layers must produce index keys")
        for layer in range(result.layers):
            source = result.source(layer)
            if ratios[layer] and (source is None or source.ratio != ratios[layer]):
                raise ValueError("shared KV consumers must match their source's compression ratio")
            if layer in text["kv_source_layer_ids"] and not ratios[layer]:
                raise ValueError("KV sources require compressed attention")
        candidate = text["candidate_source_layer_id"]
        if candidate not in text["index_source_layer_ids"]:
            raise ValueError("candidate selection must belong to an index source")
        if any(result.source(i) != result.source(candidate)
               for i in text["index_source_layer_ids"] if i > candidate):
            raise ValueError("candidate consumers must use the candidate source's KV history")
        return result

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def draft(self):
        return DSparkConfig.from_text(self.text)

    @property
    def sources(self):
        return tuple(CacheSource(i, self.text["compress_ratios"][i])
                     for i in self.text["kv_source_layer_ids"])

    @property
    def decoder_start(self):
        ratios = self.text["compress_ratios"]
        return next((source.layer for source in self.sources if source.ratio == 1
                     and any(ratio > 1 for ratio in ratios[:source.layer])), None)

    def source(self, layer):
        if not self.text["compress_ratios"][layer]:
            return None
        return next((source for source in reversed(self.sources) if source.layer <= layer), None)

    def index_source(self, layer):
        if self.source(layer) is None:
            return None
        return next((i for i in reversed(self.text["index_source_layer_ids"]) if i <= layer), None)
