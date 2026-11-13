# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural checkpoint configuration for the Qwen3.5 dense architecture."""

from dataclasses import dataclass
import json
from pathlib import Path

from ignition.preparation.quantization import quantized_layers

MODEL_ID = "nvidia/Qwen3.8-27B-NVFP4"
REVISION = "482ca0f3832238542f8f5295dde86b5f22711d80"
TEXT_PREFIX = "model.language_model"
PAGE_TOKENS = 128


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict
    draft: object = None

    @classmethod
    def load(cls, snapshot):
        raw = json.loads((Path(snapshot) / "config.json").read_text())
        if raw.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
            raise ValueError("Qwen dense preparation requires Qwen3_5ForConditionalGeneration")
        text = raw.get("text_config", {})
        required = {"model_type": "qwen3_5_text", "hidden_act": "silu", "attention_bias": False,
                    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                    "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                    "linear_value_head_dim": 128, "head_dim": 256, "tie_word_embeddings": False}
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"unsupported Qwen dense {name}: {text.get(name)!r}")
        for name in ("hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "linear_num_key_heads",
                     "linear_num_value_heads", "max_position_embeddings"):
            if type(text.get(name)) is not int or text[name] <= 0:
                raise ValueError(f"Qwen dense requires a positive {name}")
        layers = text.get("layer_types", ())
        if (len(layers) != text["num_hidden_layers"]
                or any(kind not in ("linear_attention", "full_attention") for kind in layers)):
            raise ValueError("Qwen dense layer types must describe every attention layer")
        if (text["num_attention_heads"] % text["num_key_value_heads"]
                or text["linear_num_value_heads"] % text["linear_num_key_heads"]):
            raise ValueError("Qwen query and value head counts must divide their key head counts")
        rope = text.get("rope_parameters", {})
        if rope.get("rope_type") != "default" or rope.get("partial_rotary_factor") != .25:
            raise ValueError("Qwen dense preparation requires default partial rotary embeddings")
        if not 0 < float(rope.get("rope_theta", 0)) or not 0 < float(text.get("rms_norm_eps", 0)):
            raise ValueError("Qwen rotary frequency base and normalization epsilon must be positive")
        return cls(text, dict(quantized_layers(raw)))

    @property
    def layers(self):
        return tuple(self.text["layer_types"])

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def pack_gdn_inputs(self):
        from ignition.preparation.quantization import linear_format

        names = tuple(f"{TEXT_PREFIX}.layers.{index}.linear_attn.in_proj_{part}"
                      for index, kind in enumerate(self.layers) if kind == "linear_attention"
                      for part in ("qkv", "z", "b", "a"))
        return bool(names) and all(name in self.quantization
                                   and linear_format(self.quantization, name) == "mxfp8"
                                   for name in names)
