# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inferact Kimi K3.1 DSpark geometry and target feature boundaries."""

from dataclasses import dataclass
import json
from pathlib import Path

from ignition.components.parallel import TensorParallelShard

DSPARK_REPOSITORY = "Inferact/Kimi-K3.1-Pro-DSpark-preview"
DSPARK_REVISION = "6a2ef20a05efe88aabbf9ecaa12d020c3b0edf94"


@dataclass(frozen=True)
class DSparkConfig:
    """Five causal MLA draft layers conditioned on target prefix residual streams."""

    raw: dict

    @classmethod
    def load(cls, snapshot, target):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target.text)

    @classmethod
    def from_dict(cls, raw, target):
        if not isinstance(raw, dict) or not isinstance(target, dict):
            raise ValueError("Kimi K3.1 DSpark requires draft and target configuration objects")
        expected = dict(architectures=["K3DSparkModel"], model_type="k3_dspark",
            hidden_act="silu", hidden_size=7168, intermediate_size=14336,
            num_hidden_layers=5, num_attention_heads=96, num_key_value_heads=96,
            q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128,
            qk_rope_head_dim=64, qk_head_dim=192, v_head_dim=128, head_dim=64,
            attention_bias=False, attention_dropout=0.0, rms_norm_eps=1e-5,
            rope_interleave=True, rope_theta=1_000_000.0,
            layer_types=["full_attention"] * 5, sliding_window=None,
            use_sliding_window=False, full_attention_causal=True,
            sliding_window_non_causal=False, sample_from_anchor=True,
            vocab_size=163840, draft_vocab_size=163840, mask_token_id=163839,
            markov_rank=512, markov_head_type="vanilla", block_size=7,
            num_target_layers=5, target_hidden_size=7168, target_num_hidden_layers=93,
            max_position_embeddings=1048576, tie_word_embeddings=False, torch_dtype="bfloat16",
            enable_confidence_head=True, confidence_head_with_markov=True)
        for name, value in expected.items():
            actual = raw.get(name)
            if (type(actual) is not type(value) or actual != value):
                raise ValueError(f"unsupported Kimi K3.1 DSpark {name}: {actual!r}")
        if (raw["hidden_size"] != target.get("hidden_size")
                or raw["vocab_size"] != target.get("vocab_size")
                or raw["target_num_hidden_layers"] != target.get("num_hidden_layers")):
            raise ValueError("Kimi K3.1 DSpark target geometry differs from its checkpoint configuration")
        taps = raw.get("target_layer_ids")
        if (not isinstance(taps, list) or len(taps) != raw["num_target_layers"]
                or any(type(index) is not int or not 0 <= index < target["num_hidden_layers"] - 1 for index in taps)
                or taps != sorted(set(taps))):
            raise ValueError("Kimi K3.1 DSpark target_layer_ids must be ordered unique target decoder indices")
        causal = raw.get("dflash_config")
        if not isinstance(causal, dict) or causal != {"causal": True} or causal.get("causal") is not True:
            raise ValueError("Kimi K3.1 DSpark requires causal full attention")
        rope = raw.get("rope_parameters")
        expected_rope = dict(rope_type="yarn", rope_theta=1_000_000.0, factor=32.0,
            original_max_position_embeddings=32768, beta_fast=32, beta_slow=1,
            mscale=1.0, mscale_all_dim=0.0, attn_factor=0.7426255848312643)
        if (not isinstance(rope, dict) or set(rope) != set(expected_rope)
                or any(type(rope.get(name)) is not type(value) or rope[name] != value
                       for name, value in expected_rope.items())):
            raise ValueError("unsupported Kimi K3.1 DSpark YaRN rotary parameters")
        if raw.get("quantization_config") is not None:
            raise ValueError("Kimi K3.1 DSpark requires unquantized BF16 checkpoint weights")
        return cls(raw)

    @property
    def text(self):
        return self.raw

    @property
    def auxiliary_layers(self):
        """One-based decoder boundaries after the configured target layers."""
        return tuple(index + 1 for index in self.raw["target_layer_ids"])

    @property
    def auxiliary_stream(self):
        return "prefix"

    @property
    def noise_token(self):
        return self.raw["mask_token_id"]

    @property
    def markov_rank(self):
        return self.raw["markov_rank"]

    @property
    def layers(self):
        return self.raw["num_hidden_layers"]

    @property
    def heads(self):
        return self.raw["num_attention_heads"]

    @property
    def window(self):
        return None

    @property
    def block_size(self):
        return self.raw["block_size"]

    @property
    def theta(self):
        return self.raw["rope_parameters"]["rope_theta"]

    def validate_shard(self, shard: TensorParallelShard):
        c = self.text
        for extent, alignment in ((self.heads, 1), (c["q_lora_rank"], 1),
                (c["kv_lora_rank"] + c["qk_rope_head_dim"], 1),
                (c["hidden_size"], 64), (c["intermediate_size"], 32), (c["vocab_size"], 64)):
            shard.partition(extent, alignment=alignment)
