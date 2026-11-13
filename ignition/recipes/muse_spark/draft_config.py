# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark DFlash2 draft geometry and target-model compatibility."""

from dataclasses import dataclass
import json
from pathlib import Path

from ignition.components.dflash2_geometry import DraftAttentionGeometry

REPOSITORY = "nvidia/Muse-Spark-Open-816B-A42B-DFlash2"
REVISION = "fbec887a79866e9ecff6c57adfd9fd02aa3cadea"


@dataclass(frozen=True)
class DraftConfig:
    text: dict

    @classmethod
    def load(cls, snapshot, target):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target)

    @classmethod
    def from_dict(cls, text, target):
        d = text.get("dflash_config", {})
        required = dict(model_type="qwen3", architectures=["DFlash2DraftModel"], hidden_size=8192,
            num_hidden_layers=5, num_attention_heads=128, num_key_value_heads=16, head_dim=64,
            intermediate_size=24576, hidden_act="silu", torch_dtype="bfloat16", rms_norm_eps=1e-6,
            attention_bias=False, attention_dropout=0., tie_word_embeddings=False, rope_scaling=None,
            rope_theta=500000., sliding_window=2048, use_sliding_window=True, is_causal=False,
            layer_types=["sliding_attention"] * 5, block_size=8)
        if any(name not in text or text[name] != value for name, value in required.items()):
            raise ValueError("unsupported Muse Spark DFlash2 decoder geometry or arithmetic")
        options = dict(projector_type="dflash2", conv_kernel_size=2, conv_group_size=16,
            selector_rank=256, selector_top_k=16, block_size=8, use_swa=True,
            swa_window_size=2048, causal=False, target_layer_ids=[3, 19, 31, 47, 59],
            mask_token_id=201817)
        if any(name not in d or d[name] != value for name, value in options.items()):
            raise ValueError("unsupported Muse Spark DFlash2 projector, selector, or context geometry")
        if (text["hidden_size"] != target.hidden or text.get("vocab_size") != target.vocabulary
                or text.get("num_target_layers") != target.layers
                or max(d["target_layer_ids"]) >= target.layers
                or any(d.get(name) != target.text[name]
                       for name in ("output_multiplier", "final_logit_softcapping"))):
            raise ValueError("Muse Spark DFlash2 draft does not match the target model")
        return cls(text)

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def target_layers(self):
        return tuple(self.text["dflash_config"]["target_layer_ids"])

    @property
    def block_size(self):
        return self.text["block_size"]

    @property
    def noise_token(self):
        return self.text["dflash_config"]["mask_token_id"]

    @property
    def attention(self):
        c = self.text
        return DraftAttentionGeometry(c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
            c["head_dim"], c["head_dim"], c["rope_theta"], c["sliding_window"])

    def validate_shard(self, shard):
        self.attention.local(shard)
        shard.partition(self.text["hidden_size"], alignment=64)
        shard.partition(self.text["intermediate_size"], alignment=128)
