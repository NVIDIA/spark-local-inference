# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H heterogeneous block geometry and execution semantics."""

from dataclasses import dataclass
import json
from pathlib import Path

PAGE_TOKENS = 128


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    vision: dict
    image_token: int
    eos_token: int
    draft: object = None

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw):
        if (raw.get("model_type") != "nemotron_h_omni"
                or raw.get("architectures") != ["NemotronH_Omni_Reasoning_V3"]):
            raise ValueError("Nemotron-H requires NemotronH_Omni_Reasoning_V3")
        text = raw["llm_config"]
        required = dict(model_type="nemotron_h_puzzle", mamba_hidden_act="silu", mlp_hidden_act="relu2",
            use_bias=False, mlp_bias=False, attention_bias=False, use_qk_norm=False,
            use_conv_bias=True, conv_kernel=4, n_group=1, topk_group=1,
            norm_topk_prob=True, scoring_func="sigmoid", residual_in_fp32=False,
            tie_word_embeddings=False, num_nextn_predict_layers=0, sliding_window=None)
        for key, expected in required.items():
            if text.get(key) != expected:
                raise ValueError(f"unsupported Nemotron-H {key}: {text.get(key)!r}")
        for key in ("hidden_size", "vocab_size", "mamba_num_heads", "mamba_head_dim", "ssm_state_size",
                    "n_groups", "num_attention_heads", "num_key_value_heads", "head_dim", "moe_latent_size",
                    "moe_shared_expert_intermediate_size", "max_position_embeddings"):
            if type(text.get(key)) is not int or text[key] <= 0:
                raise ValueError(f"Nemotron-H requires positive {key}")
        if (text["mamba_head_dim"] not in (64, 128) or text["ssm_state_size"] not in (64, 128)
                or text["mamba_num_heads"] % text["n_groups"]
                or text["num_attention_heads"] % text["num_key_value_heads"]):
            raise ValueError("unsupported Nemotron-H attention or state geometry")
        limits = text.get("time_step_limit")
        if limits is not None and limits != [0.0, {"__float__": "Infinity"}] and limits != [0.0, float("inf")]:
            raise ValueError("Nemotron-H requires an unclipped softplus time step")
        blocks = text["block_configs"]
        if not blocks or [b["block_type"] for b in blocks] != text["layers_block_type"]:
            raise ValueError("Nemotron-H block configurations must agree with the layer sequence")
        for block in blocks:
            if block["block_type"] not in ("mamba", "attention", "moe"):
                raise ValueError(f"unsupported Nemotron-H block: {block['block_type']}")
            if block["block_type"] == "moe":
                n, k, width = (block.get(key) for key in
                    ("n_routed_experts", "num_experts_per_tok", "moe_intermediate_size"))
                if (any(type(v) is not int for v in (n, k, width)) or not 0 < k <= n <= 1024
                        or width <= 0 or width % 256):
                    raise ValueError("Nemotron-H experts require top_k <= experts <= 1024 and 256-aligned widths")
        return cls(text, raw["vision_config"], raw["img_context_token_id"], raw["eos_token_id"])

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def layers(self):
        return tuple(self.text["layers_block_type"])

    @property
    def mamba_width(self):
        return self.text["mamba_num_heads"] * self.text["mamba_head_dim"]

    @property
    def convolution_width(self):
        return self.mamba_width + 2 * self.text["n_groups"] * self.text["ssm_state_size"]

    def validate_shard(self, shard):
        if shard.world_size != 1:
            raise ValueError("Nemotron-H preparation requires one GPU")
