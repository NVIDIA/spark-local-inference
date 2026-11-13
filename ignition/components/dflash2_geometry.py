# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped-query attention geometry for full-rotary DFlash2 drafts."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DraftAttentionGeometry:
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
class DFlash2Config:
    """Validated block drafter geometry and its target feature contract."""

    text: dict

    @classmethod
    def load(cls, snapshot, **kwargs):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), **kwargs)

    @classmethod
    def from_dict(cls, text, *, target_hidden, target_vocabulary, target_layers, expected):
        required = dict(model_type="qwen3", architectures=["DFlash2DraftModel"], hidden_act="silu",
            dtype="bfloat16", attention_bias=False, attention_dropout=0., tie_word_embeddings=False,
            use_sliding_window=True, is_causal=False)
        if any(text.get(name) != value for name, value in required.items()):
            raise ValueError("unsupported DFlash2 decoder arithmetic")
        options = text.get("dflash_config", {})
        for name, value in expected.items():
            actual = options if name == "dflash_config" else text.get(name)
            matches = (all(actual.get(key) == entry for key, entry in value.items())
                       if name == "dflash_config" else actual == value)
            if not matches:
                raise ValueError(f"DFlash2 checkpoint disagrees with its declared {name}")
        for name in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                     "num_key_value_heads", "head_dim", "sliding_window"):
            if type(text.get(name)) is not int or text[name] <= 0:
                raise ValueError(f"DFlash2 requires a positive {name}")
        if (text["num_attention_heads"] % text["num_key_value_heads"]
                or text.get("layer_types") != ["sliding_attention"] * text["num_hidden_layers"]):
            raise ValueError("DFlash2 requires grouped-query sliding attention in every layer")
        rope = text.get("rope_parameters", {})
        if (rope.get("rope_type") != "default" or float(rope.get("rope_theta", 0)) <= 0
                or float(text.get("rms_norm_eps", 0)) <= 0):
            raise ValueError("DFlash2 requires default rotary embeddings and positive normalization epsilon")
        if (text["hidden_size"] != target_hidden or text.get("vocab_size") != target_vocabulary
                or text.get("num_target_layers") != target_layers):
            raise ValueError("DFlash2 draft does not match the target model")
        taps = options.get("target_layer_ids", [])
        if (not taps or any(type(index) is not int or not 0 <= index < target_layers for index in taps)
                or taps != sorted(set(taps))):
            raise ValueError("DFlash2 target layers must be distinct ordered decoder indices")
        if not 0 <= options.get("mask_token_id", -1) < target_vocabulary:
            raise ValueError("DFlash2 mask token must belong to the target vocabulary")
        for name, value in (("output_multiplier", 1.), ("final_logit_softcapping", None)):
            if text.get(name, value) != value or options.get(name, value) != value:
                raise ValueError("unsupported DFlash2 output transform")
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


def draft_context_specs(config, shard):
    """Continuation storage for the drafter's accepted target KV rings."""
    import torch

    geometry = config.attention
    _, heads = geometry.local(shard)
    for layer in range(config.layers):
        for name, width in (("key", geometry.key_dim), ("value", geometry.value_dim)):
            yield f"draft{layer}.{name}", (geometry.window, heads, width), torch.float8_e4m3fn
