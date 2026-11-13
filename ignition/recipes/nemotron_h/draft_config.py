# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiLiCorr draft backbone, candidate lattice, and target feature geometry."""

from dataclasses import dataclass
import json
from pathlib import Path

REPOSITORY = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"
REVISION = "abd478065678ee83a93821c4b3f512da24e4cecb"


@dataclass(frozen=True)
class DraftConfig:
    text: dict

    @classmethod
    def load(cls, snapshot, target):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target)

    @classmethod
    def from_dict(cls, c, target):
        expected = dict(architectures=["LiLiCorrDraftModel"], model_type="qwen3", block_size=8,
            hidden_size=target.hidden, vocab_size=target.vocabulary, hidden_act="silu",
            attention_bias=False, tie_word_embeddings=False, rope_scaling=None,
            num_attention_heads=32, num_key_value_heads=2, head_dim=128, num_hidden_layers=5,
            intermediate_size=12288, sliding_window=1024, attention_sink_bias=True,
            use_sliding_window=True, layer_types=["sliding_attention"] * 5)
        for field, value in expected.items():
            if c.get(field) != value:
                raise ValueError(f"unsupported LiLiCorr {field}: {c.get(field)!r}")
        d = c["dflash_config"]
        expected = dict(use_swa=True, swa_window_size=1024, causal=False, attention_sink_bias=True,
            projector_type="lilicorr", lilicorr_enabled=True, lilicorr_candidate_topk=8,
            lilicorr_hidden_size=1024, lilicorr_num_layers=2, lilicorr_num_heads=8,
            lilicorr_mlp_ratio=2., lilicorr_factor_dim=1024, conv_kernel_size=2, conv_group_size=16)
        for field, value in expected.items():
            if d.get(field) != value:
                raise ValueError(f"unsupported LiLiCorr {field}: {d.get(field)!r}")
        taps = d["target_layer_ids"]
        if not taps or sorted(set(taps)) != taps or not all(type(i) is int and 0 <= i < len(target.layers) for i in taps):
            raise ValueError("LiLiCorr feature taps must be ordered target decoder indices")
        if not 0 <= d["mask_token_id"] < target.vocabulary:
            raise ValueError("LiLiCorr mask token must belong to the target vocabulary")
        if min(c["rope_theta"], c["rms_norm_eps"], d["lilicorr_vector_eps"], d["lilicorr_logit_scale"]) <= 0:
            raise ValueError("LiLiCorr rotary base, normalization epsilons, and logit scale must be positive")
        return cls(c)

    @property
    def block_size(self):
        return self.text["block_size"]

    @property
    def noise_token(self):
        return self.text["dflash_config"]["mask_token_id"]

    @property
    def target_layers(self):
        return tuple(self.text["dflash_config"]["target_layer_ids"])

    @property
    def layers(self):
        return self.text["num_hidden_layers"]
