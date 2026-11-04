# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical text geometry and checkpoint-declared mixed projection precision."""

from dataclasses import dataclass
import json
from pathlib import Path

from ignition.preparation.quantization import linear_format, quantized_layers


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, config):
        if (config.get("model_type") != "muse_glimmer"
                or config.get("architectures") != ["MuseGlimmerForConditionalGeneration"]):
            raise ValueError("Muse Glimmer requires MuseGlimmerForConditionalGeneration")
        c, q = config["text_config"], quantized_layers(config)
        if (c.get("model_type") != "muse_glimmer_text" or c.get("hidden_activation") != "silu"
                or c.get("attention_bias") or c.get("tie_word_embeddings")):
            raise ValueError("unsupported Muse Glimmer projection arithmetic")
        for name in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size",
                     "num_attention_heads", "num_key_value_heads", "head_dim", "sliding_window"):
            if type(c[name]) is not int or c[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (c["num_attention_heads"] % c["num_key_value_heads"] or c["head_dim"] != 128
                or c.get("rope_parameters", {}).get("rope_type") != "default"):
            raise ValueError("unsupported Muse Glimmer attention geometry")
        if any(len(c[name]) != c["num_hidden_layers"] for name in ("layer_types", "layer_rope_theta")):
            raise ValueError("attention types and RoPE must describe every decoder layer")
        if any(v not in ("sliding_attention", "full_attention") for v in c["layer_types"]):
            raise ValueError("unsupported Muse Glimmer attention type")
        if any(theta < 0 for theta in c["layer_rope_theta"]):
            raise ValueError("layer RoPE theta must be nonnegative")
        if any(c[name] <= 0 for name in ("rms_norm_eps", "post_norm_eps", "qk_scale_factor",
                                         "output_multiplier", "final_logit_softcapping")):
            raise ValueError("normalization and output scales must be positive")
        for prefix in ("lm_head", *(f"model.language_model.layers.{i}.{part}" for i in range(c["num_hidden_layers"])
                for part in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj",
                             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))):
            if prefix in q and linear_format(q, prefix) not in ("nvfp4", "tensor_fp8"):
                raise ValueError(f"{prefix}: expected NVFP4 or calibrated FP8")
        if q["lm_head"]["quant_algo"] != "W4A16_NVFP4":
            raise ValueError("Muse Glimmer requires an NVFP4 A16 vocabulary projection")
        return cls(c, q)

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    def window(self, layer):
        return self.text["sliding_window"] - 1 if self.text["layer_types"][layer] == "sliding_attention" else -1
