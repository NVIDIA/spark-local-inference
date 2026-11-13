# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark text geometry and checkpoint-declared projection precision."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
import json
from pathlib import Path


if TYPE_CHECKING:
    from .draft_config import DraftConfig


@dataclass(frozen=True)
class AttentionGeometry:
    heads: int
    kv_heads: int
    key_dim: int
    value_dim: int
    rotary_dim: int
    theta: float
    window: int

    def local(self, shard):
        shard.partition(self.heads)
        first, last = shard.replicated_partition(self.kv_heads)
        return self.heads // shard.world_size, last - first


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict
    draft: DraftConfig | None = None

    @classmethod
    def load(cls, snapshot, *, draft_snapshot=None):
        config = cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))
        if draft_snapshot is not None:
            from .draft_config import DraftConfig
            config = replace(config, draft=DraftConfig.load(draft_snapshot, config))
        return config

    @classmethod
    def from_dict(cls, config):
        if (config.get("model_type") != "muse_spark"
                or config.get("architectures") != ["MuseSparkForConditionalGeneration"]):
            raise ValueError("Muse Spark requires MuseSparkForConditionalGeneration")
        c, q = config["text_config"], config["quantization_config"]
        if (c.get("model_type") != "muse_spark_text" or c.get("hidden_act") != "silu"
                or c.get("attention_bias") or c.get("num_shared_experts") != 0
                or c.get("tie_word_embeddings") or c.get("routed_scaling_factor") != 1.):
            raise ValueError("unsupported Muse Spark projection or routing arithmetic")
        if (q.get("quant_method") != "modelopt" or q.get("quant_algo") != "NVFP4"
                or q.get("group_size") != 16 or not isinstance(q.get("modules_to_not_convert"), list)):
            raise ValueError("Muse Spark requires ModelOpt NVFP4 with explicit precision exclusions")
        for name in ("layer_types", "layer_rope_theta", "mlp_layer_types"):
            if len(c[name]) != c["num_hidden_layers"]:
                raise ValueError(f"{name} must describe every decoder layer")
        if any(v != "sparse" for v in c["mlp_layer_types"]):
            raise ValueError("Muse Spark preparation requires routed decoder layers")
        if any(v not in ("sliding_attention", "full_attention") for v in c["layer_types"]):
            raise ValueError("unsupported Muse Spark attention type")
        if (c["num_attention_heads"] % c["num_key_value_heads"] or c["head_dim"] != 64
                or c["sliding_window"] <= 0 or c["sliding_window"] % 128
                or c.get("rope_parameters", {}).get("rope_type") != "default"
                or c["num_experts_per_tok"] != 8):
            raise ValueError("unsupported Muse Spark attention or expert geometry")
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

    def attention(self, layer):
        c = self.text
        theta = c["layer_rope_theta"][layer]
        return AttentionGeometry(c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
            c["head_dim"], c["head_dim"] if theta else 0, theta,
            c["sliding_window"] if c["layer_types"][layer] == "sliding_attention" else -1)

    def projection_format(self, prefix):
        excluded = self.quantization["modules_to_not_convert"]
        return "bf16" if any(prefix == p or prefix.startswith(p + ".") for p in excluded) else "nvfp4"

    def expert_format(self, layer):
        return self.projection_format(f"model.language_model.layers.{layer}.mlp.experts")

    def validate_shard(self, shard):
        for layer in range(self.layers):
            self.attention(layer).local(shard)
        for extent in (self.hidden, self.text["moe_hidden_size"], self.text["moe_intermediate_size"]):
            shard.partition(extent, alignment=64)

    def vocabulary_partition(self, shard):
        width = (self.vocabulary + shard.world_size * 64 - 1) // (shard.world_size * 64) * 64
        return shard.rank * width, min((shard.rank + 1) * width, self.vocabulary), width
