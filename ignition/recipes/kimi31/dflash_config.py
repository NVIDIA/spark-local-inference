# SPDX-License-Identifier: Apache-2.0

"""Block-causal MLA DFlash2 geometry and Kimi K3.1 AttnRes feature mapping."""

from dataclasses import dataclass
import math


REPOSITORY = "RadixArk/Kimi-K3.1-Dflash2-Preview"
REVISION = "ea257b38a8ee755b65287bb8ccd0938b519ce26b"
WEIGHT_BYTES = 5_569_622_016


@dataclass(frozen=True)
class DFlash2Config:
    raw: dict

    @classmethod
    def from_dict(cls, raw, target):
        required = dict(architectures=["DFlash2DraftModel"], model_type="qwen3", dtype="bfloat16",
            hidden_act="silu", attention_bias=False, attention_dropout=0., is_causal=True,
            use_sliding_window=True, tie_word_embeddings=False, mla_use_nope=False, rope_interleave=True,
            q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
            qk_head_dim=192, v_head_dim=128)
        if any(raw.get(key) != value for key, value in required.items()):
            raise ValueError("Kimi DFlash2 requires causal interleaved-YaRN MLA with BF16 source weights")
        for key in ("hidden_size", "num_attention_heads", "vocab_size", "max_position_embeddings"):
            if raw.get(key) != target[key]:
                raise ValueError(f"Kimi DFlash2 {key} disagrees with the target")
        layers, window = raw.get("num_hidden_layers"), raw.get("sliding_window")
        if (type(layers) is not int or layers <= 0 or type(window) is not int or window <= 0 or window % 128
                or raw.get("layer_types") != ["sliding_attention"] * layers
                or raw.get("num_target_layers") != target["num_hidden_layers"]
                or raw.get("draft_vocab_size") != target["vocab_size"]
                or type(raw.get("intermediate_size")) is not int or raw["intermediate_size"] <= 0
                or float(raw.get("rms_norm_eps", 0)) <= 0):
            raise ValueError("Kimi DFlash2 requires positive sliding-window decoder geometry")
        options = raw.get("dflash_config", {})
        expected = dict(attention_mode="mla", aux_hidden_stream="attn_res", conv_kernel_size=2,
                        conv_group_size=16, selector_rank=256, selector_top_k=32)
        taps = options.get("target_layer_ids", [])
        if (any(options.get(key) != value for key, value in expected.items())
                or not taps or taps != sorted(set(taps))
                or any(type(i) is not int or not 0 <= i < target["num_hidden_layers"] - 1 for i in taps)
                or not 0 <= options.get("mask_token_id", -1) < target["vocab_size"]
                or type(options.get("block_size")) is not int or not 2 <= options["block_size"] <= 8):
            raise ValueError("Kimi DFlash2 requires valid AttnRes taps, block convolutions and a candidate selector")
        rope = raw.get("rope_parameters", {})
        if (rope.get("rope_type") != "yarn" or rope.get("factor", 0) <= 0
                or rope.get("rope_theta", 0) <= 0 or rope.get("original_max_position_embeddings", 0) <= 0
                or rope.get("mscale") != 1. or rope.get("mscale_all_dim") != 1.):
            raise ValueError("Kimi DFlash2 requires the declared DeepSeek YaRN magnitude scaling")
        return cls(raw)

    @property
    def text(self):
        return self.raw

    @property
    def auxiliary_layers(self):
        # A checkpoint tap names a layer output: the next layer's pre-norm AttnRes mixture.
        return tuple(i + 1 for i in self.raw["dflash_config"]["target_layer_ids"])

    @property
    def auxiliary_source(self):
        return "attention_input"

    @property
    def auxiliary_stream(self):
        return "attn_res"

    @property
    def noise_token(self):
        return self.raw["dflash_config"]["mask_token_id"]

    @property
    def layers(self):
        return self.raw["num_hidden_layers"]

    @property
    def heads(self):
        return self.raw["num_attention_heads"]

    @property
    def window(self):
        return self.raw["sliding_window"]

    @property
    def block_size(self):
        return self.raw["dflash_config"]["block_size"]

    @property
    def attention_scale(self):
        factor = self.raw["rope_parameters"]["factor"]
        magnitude = 1. if factor <= 1 else 1. + .1 * math.log(factor)
        return 192 ** -.5 * magnitude ** 2

    def validate_shard(self, shard):
        for extent, alignment in ((self.heads, 1), (self.text["hidden_size"], 64),
                (self.text["intermediate_size"], 32), (self.text["q_lora_rank"], 1), (576, 1)):
            shard.partition(extent, alignment=alignment)
