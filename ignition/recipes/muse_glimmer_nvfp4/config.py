# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer text geometry and ModelOpt NVFP4 projection declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from ignition.preparation.quantization import modelopt_nvfp4_layers
from ignition.recipes.muse_spark.config import AttentionGeometry
from .draft_config import DraftConfig

PREFIX = "model.language_model"
# The eight NVFP4 projections of every decoder layer, in checkpoint naming.
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def candidate_linear_names(layers):
    return ["lm_head"] + [f"{PREFIX}.layers.{i}.{p}" for i in range(layers) for p in PROJECTIONS]


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict
    draft: DraftConfig | None = None

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, config):
        if (config.get("model_type") != "muse_glimmer"
                or config.get("architectures") != ["MuseGlimmerForConditionalGeneration"]):
            raise ValueError("Muse Glimmer requires MuseGlimmerForConditionalGeneration")
        c = config["text_config"]
        if (c.get("model_type") != "muse_glimmer_text" or c.get("hidden_activation") != "silu"
                or c.get("attention_bias") or c.get("tie_word_embeddings")
                or c.get("rope_parameters", {}).get("rope_type") != "default"):
            raise ValueError("unsupported Muse Glimmer projection or embedding arithmetic")
        for name in ("layer_types", "layer_rope_theta"):
            if len(c[name]) != c["num_hidden_layers"]:
                raise ValueError(f"{name} must describe every decoder layer")
        if any(v not in ("sliding_attention", "full_attention") for v in c["layer_types"]):
            raise ValueError("unsupported Muse Glimmer attention type")
        if any(bool(theta) != (kind == "sliding_attention")
               for theta, kind in zip(c["layer_rope_theta"], c["layer_types"], strict=True)):
            raise ValueError("Muse Glimmer rotates sliding layers only")
        if (c["num_attention_heads"] % c["num_key_value_heads"] or c["head_dim"] != 128
                or c["sliding_window"] <= 0 or c["sliding_window"] % 128
                or c["intermediate_size"] % 128 or c["hidden_size"] % 64):
            raise ValueError("unsupported Muse Glimmer attention or feed-forward geometry")
        for name in ("qk_scale_factor", "output_multiplier", "final_logit_softcapping", "post_norm_eps",
                     "rms_norm_eps", "vocab_size"):
            if not c.get(name):
                raise ValueError(f"Muse Glimmer text config requires {name}")
        # The checkpoint quantizes activations as well as weights (W4A4); the shared parser reads weights only.
        quant = config.get("quantization_config")
        groups = quant.get("config_groups") if isinstance(quant, Mapping) else None
        group = groups.get("group_0") if isinstance(groups, Mapping) else None
        activations = group.get("input_activations") if isinstance(group, Mapping) else None
        if not isinstance(activations, Mapping) or activations.get("num_bits") != 4 or activations.get("group_size") != 16:
            raise ValueError("Muse Glimmer requires 4-bit group_size=16 NVFP4 input activations")
        quantization = modelopt_nvfp4_layers(config, candidate_linear_names(c["num_hidden_layers"]))
        if "lm_head" in quantization or len(quantization) != len(PROJECTIONS) * c["num_hidden_layers"]:
            raise ValueError("Muse Glimmer requires NVFP4 decoder projections and a BF16 head")
        return cls(c, quantization)

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def intermediate(self):
        return self.text["intermediate_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    def attention(self, layer):
        c = self.text
        theta = c["layer_rope_theta"][layer] or 0.
        return AttentionGeometry(c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
            c["head_dim"], c["head_dim"] if theta else 0, theta,
            c["sliding_window"] if c["layer_types"][layer] == "sliding_attention" else -1)

    def validate_shard(self, shard):
        for layer in range(self.layers):
            self.attention(layer).local(shard)
        shard.partition(self.hidden, alignment=64)
        shard.partition(self.intermediate, alignment=128)

    def vocabulary_partition(self, shard):
        width = (self.vocabulary + shard.world_size * 64 - 1) // (shard.world_size * 64) * 64
        return shard.rank * width, min((shard.rank + 1) * width, self.vocabulary), width
