# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer 30B DFlash2 geometry and target compatibility."""

from dataclasses import dataclass
import json
from pathlib import Path

from ignition.components.dflash2_geometry import DraftAttentionGeometry

REPOSITORY = "z-lab/Muse-Glimmer-30B-DFlash2"
REVISION = "b54ffdd11fa9cfe2af370012e5763d492c904128"


@dataclass(frozen=True)
class DraftConfig:
    """The z-lab DFlash2 drafter configuration in the standard DFlash2 form.

    ``text`` carries a ``dflash_config`` with the convolution, selector and block geometry the
    shared DFlash2 components build, plus the target's output transform.
    """
    text: dict

    @classmethod
    def load(cls, snapshot, target):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target)

    @classmethod
    def from_dict(cls, text, target):
        required = dict(model_type="qwen3", architectures=["DFlash2DraftModel"], hidden_size=6656,
            intermediate_size=19968, num_hidden_layers=5, num_attention_heads=32, num_key_value_heads=8,
            head_dim=128, vocab_size=202048, num_target_layers=52, hidden_act="silu", dtype="bfloat16",
            rms_norm_eps=1e-5, attention_bias=False, is_causal=False, sliding_window=2048,
            use_sliding_window=True, layer_types=["sliding_attention"] * 5,
            rope_parameters=dict(rope_type="default", rope_theta=500000.))
        if any(text.get(name) != value for name, value in required.items()):
            raise ValueError("unsupported Muse Glimmer DFlash2 decoder geometry or arithmetic")
        options = dict(block_size=16, conv_kernel_size=2, conv_group_size=16, selector_rank=256,
            selector_top_k=16, mask_token_id=201818, target_layer_ids=[1, 13, 25, 37, 49])
        if any(text.get("dflash_config", {}).get(name) != value for name, value in options.items()):
            raise ValueError("unsupported Muse Glimmer DFlash2 projector, selector, or context geometry")
        for name in ("output_multiplier", "final_logit_softcapping"):
            expected = target.text[name]
            if text.get(name, expected) != expected or text["dflash_config"].get(name) != expected:
                raise ValueError("unsupported Muse Glimmer DFlash2 output transform")
        if (text["hidden_size"] != target.hidden or text["vocab_size"] != target.vocabulary
                or text["num_target_layers"] != target.layers):
            raise ValueError("Muse Glimmer DFlash2 draft does not match the target model")
        return cls(text)

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
    def noise_token(self):
        return self.text["dflash_config"]["mask_token_id"]

    @property
    def attention(self):
        c = self.text
        return DraftAttentionGeometry(c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
            c["head_dim"], c["head_dim"], c["rope_parameters"]["rope_theta"], c["sliding_window"])

    def validate_shard(self, shard):
        self.attention.local(shard)
        shard.partition(self.text["hidden_size"], alignment=64)
        shard.partition(self.text["intermediate_size"], alignment=128)
