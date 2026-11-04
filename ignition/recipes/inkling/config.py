# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inkling text geometry, normalized with the checkpoint's canonical config."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class AttentionGeometry:
    heads: int
    kv_heads: int
    key_dim: int
    window: int
    relative_extent: int

    @property
    def value_dim(self):
        return self.key_dim

    def local(self, shard):
        shard.partition(self.heads)
        first, last = shard.replicated_partition(self.kv_heads)
        return self.heads // shard.world_size, last - first


@dataclass(frozen=True)
class ModelConfig:
    text: dict

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, config):
        from transformers.models.inkling.configuration_inkling import InklingTextConfig

        if (config.get("model_type") != "inkling_mm_model"
                or config.get("architectures") != ["InklingForConditionalGeneration"]):
            raise ValueError("Inkling requires InklingForConditionalGeneration")
        if config.get("quantization_config"):
            raise ValueError("Inkling text preparation requires the BF16 checkpoint")
        c = InklingTextConfig.from_dict(config["text_config"]).to_dict()
        if (c.get("dtype") != "bfloat16" or c["hidden_act"] != "silu" or c["q_bias"] or c["o_bias"]
                or not c["use_sconv"] or not c["use_embed_norm"] or not c["shared_expert_sink"]
                or c["gate_activation"] != "sigmoid" or not c["norm_after_topk"]
                or not c["use_gate_bias"] or not c["use_global_scale"]
                or c["final_logit_softcapping"] is not None or c.get("tie_word_embeddings", False)):
            raise ValueError("unsupported Inkling normalization, convolution, or routing arithmetic")
        if (len(c["layer_types"]) != c["num_hidden_layers"]
                or len(c["mlp_layer_types"]) != c["num_hidden_layers"]
                or any(t not in ("hybrid", "hybrid_sliding") for t in c["layer_types"])
                or any(t not in ("dense", "sparse") for t in c["mlp_layer_types"])
                or c["conv_kernel_size"] != 4 or c["head_dim"] != 128
                or c["swa_head_dim"] != 128 or c["sliding_window_size"] % 128
                or c["n_shared_experts"] <= 0 or c["n_routed_experts"] & (c["n_routed_experts"] - 1)):
            raise ValueError("unsupported Inkling layer or cache geometry")
        c["eos_token_id"] = config["eos_token_id"]
        return cls(c)

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["unpadded_vocab_size"]

    def attention(self, layer):
        c = self.text
        local = c["layer_types"][layer] == "hybrid_sliding"
        return AttentionGeometry(c["swa_num_attention_heads"] if local else c["num_attention_heads"],
            c["swa_num_key_value_heads"] if local else c["num_key_value_heads"],
            c["swa_head_dim"] if local else c["head_dim"], c["sliding_window_size"] if local else -1,
            c["sliding_window_size"] if local else c["rel_extent"])

    def validate_shard(self, shard):
        for layer in range(self.layers):
            self.attention(layer).local(shard)
        for extent in (self.hidden, self.text["intermediate_size"], self.text["moe_intermediate_size"]):
            shard.partition(extent, alignment=64)

    def vocabulary_partition(self, shard):
        width = -(-self.vocabulary // (shard.world_size * 64)) * 64
        return shard.rank * width, min((shard.rank + 1) * width, self.vocabulary), width
