# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 geometry and checkpoint quantization contracts."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

from ignition.components.parallel import TensorParallelShard

if TYPE_CHECKING:
    from ignition.recipes.kimi31.dflash_config import DFlash2Config
    from ignition.recipes.kimi31.dspark_config import DSparkConfig as MLADSparkConfig

PAGE_TOKENS = 128
# Optional speculators-format draft directory within a target checkpoint.
DRAFT_DIRECTORY = "dspark"
DSPARK_REPOSITORY = "RedHatAI/Kimi-K3-speculator.dspark"
DSPARK_REVISION = "38a88101e0d46bb22134b9da340f381b954d40d4"


@dataclass(frozen=True)
class DSparkConfig:
    """A speculators-format DSpark draft: a Qwen3 sliding-window stack over Kimi K3 auxiliary states.

    The block's anchor predicts the first draft (``sample_from_anchor``), every later block position holds
    the mask token, and a rank-``markov_rank`` transition bias from the previous token joins each
    position's logits while drafts are sampled left to right.
    """

    raw: dict

    @classmethod
    def load(cls, snapshot, target):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target.text)

    @classmethod
    def from_dict(cls, raw, target):
        if raw.get("architectures") == ["DFlash2DraftModel"]:
            from ignition.recipes.kimi31.dflash_config import DFlash2Config
            return DFlash2Config.from_dict(raw, target)
        layer = raw.get("transformer_layer_config", {})
        aux = raw.get("aux_hidden_state_layer_ids", ())
        rope = layer.get("rope_parameters", {})
        if (raw.get("architectures") != ["DSparkDraftModel"] or raw.get("speculators_model_type") != "dspark"
                or layer.get("model_type") != "qwen3" or layer.get("hidden_act") != "silu"
                or layer.get("attention_bias") or layer.get("hidden_size") != target["hidden_size"]
                or layer.get("vocab_size") != target["vocab_size"]
                or raw.get("draft_vocab_size") != target["vocab_size"]
                or layer.get("layer_types") != ["sliding_attention"] * layer.get("num_hidden_layers", 0)
                or not 0 < layer.get("sliding_window", 0) or layer.get("sliding_window") % PAGE_TOKENS
                or raw.get("sliding_window_non_causal") is not False or raw.get("sample_from_anchor") is not True
                or rope.get("rope_type") != "default" or set(rope) != {"rope_type", "rope_theta"}
                or layer.get("num_attention_heads", 0) % max(1, layer.get("num_key_value_heads", 0))
                or raw.get("markov_head_type") != "vanilla" or not 0 < raw.get("markov_rank", 0)
                or not 0 <= raw.get("mask_token_id", -1) < target["vocab_size"]
                or not aux or list(aux) != sorted(set(aux))
                or any(not 0 < i < target["num_hidden_layers"] for i in aux)):
            raise ValueError("unsupported Kimi K3 DSpark draft geometry")
        return cls(raw)

    @property
    def text(self):
        return self.raw["transformer_layer_config"]

    @property
    def auxiliary_layers(self):
        """Decoder layers whose AttnRes input the draft reads; each follows the preceding layer's output."""
        return tuple(self.raw["aux_hidden_state_layer_ids"])

    @property
    def auxiliary_stream(self):
        return "attn_res"

    @property
    def noise_token(self):
        return self.raw["mask_token_id"]

    @property
    def markov_rank(self):
        return self.raw["markov_rank"]

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def heads(self):
        return self.text["num_attention_heads"]

    @property
    def kv_heads(self):
        return self.text["num_key_value_heads"]

    @property
    def head_dim(self):
        return self.text["head_dim"]

    @property
    def window(self):
        return self.text["sliding_window"]

    @property
    def theta(self):
        return self.text["rope_parameters"]["rope_theta"]

    def validate_shard(self, shard: TensorParallelShard):
        shard.partition(self.heads)
        shard.partition(self.kv_heads)
        shard.partition(self.text["intermediate_size"], alignment=32)
        shard.partition(self.text["hidden_size"], alignment=8)


@dataclass(frozen=True)
class DSAConfig:
    """Sparse indexer geometry and one-based decoder layer indices."""

    heads: int
    head_dim: int
    top_k: int
    layers: tuple[int, ...]

    @classmethod
    def from_dict(cls, raw, full_layers):
        expected = dict(index_num_heads=32, index_head_dim=128, index_topk=2048,
            index_rope_dim=0, index_k_norm_type="layernorm", index_topk_freq=1,
            index_skip_topk_offset=1)
        if (not isinstance(raw, dict) or set(raw) != {*expected, "dsa_layers"}
                or any(type(raw.get(key)) is not type(value) or raw[key] != value
                       for key, value in expected.items())
                or raw.get("dsa_layers") != full_layers
                or not full_layers or any(type(layer) is not int for layer in full_layers)
                or full_layers != sorted(set(full_layers))):
            raise ValueError("unsupported Kimi K3.1 DSA geometry or layer assignment")
        return cls(raw["index_num_heads"], raw["index_head_dim"], raw["index_topk"], tuple(full_layers))


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    vision: dict
    draft: "DSparkConfig | MLADSparkConfig | DFlash2Config | None" = None
    dsa: DSAConfig | None = None

    @classmethod
    def load(cls, snapshot):
        snapshot = Path(snapshot)
        config = cls.from_dict(json.loads((snapshot / "config.json").read_text()))
        draft = snapshot / DRAFT_DIRECTORY / "config.json"
        if config.dsa is not None or not draft.is_file():
            return config
        return replace(config, draft=DSparkConfig.from_dict(json.loads(draft.read_text()), config.text))

    @classmethod
    def from_dict(cls, raw):
        identity = (raw.get("model_type"), raw.get("architectures"))
        if identity not in (("kimi_k3", ["KimiK3ForConditionalGeneration"]),
                            ("kimi_k31", ["KimiK31ForConditionalGeneration"])):
            raise ValueError("Kimi requires matching KimiK3 or KimiK31 model type and architecture")
        c = raw["text_config"]
        required = dict(model_type="kimi_linear", hidden_act="situ", activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0, mla_use_nope=True, mla_use_output_gate=True,
            latent_moe_use_norm=True, moe_renormalize=True, moe_router_activation_func="sigmoid",
            num_expert_group=1, topk_group=1, topk_method="noaux_tc", moe_layer_freq=1,
            num_nextn_predict_layers=0, tie_word_embeddings=False)
        for name, value in required.items():
            if c.get(name) != value:
                raise ValueError(f"unsupported Kimi K3 {name}: {c.get(name)!r}")
        for name in ("hidden_size", "num_hidden_layers", "intermediate_size", "moe_intermediate_size",
                     "num_attention_heads", "routed_expert_hidden_size", "num_experts", "num_experts_per_token",
                     "num_shared_experts", "attn_res_block_size", "vocab_size"):
            if type(c.get(name)) is not int or c[name] <= 0:
                raise ValueError(f"Kimi K3 requires positive {name}")
        if not 0 < c["num_experts_per_token"] <= c["num_experts"] <= 1024:
            raise ValueError("Kimi K3 routing requires top_k <= experts <= 1024")
        if not 0 <= c["first_k_dense_replace"] <= c["num_hidden_layers"]:
            raise ValueError("dense layer count exceeds the decoder")
        if (c["kv_lora_rank"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]) != (512, 128, 64, 128):
            raise ValueError("Kimi K3 requires MLA dimensions (512,128,64,128)")
        kda = c["linear_attn_config"]
        if any(kda.get(k) != v for k, v in dict(head_dim=128, short_conv_kernel_size=4,
                use_full_rank_gate=True, gate_lower_bound=-5.0).items()):
            raise ValueError("unsupported Kimi K3 KDA geometry")
        linear, full = kda["kda_layers"], kda["full_attn_layers"]
        if sorted(linear + full) != list(range(1, c["num_hidden_layers"] + 1)):
            raise ValueError("Kimi K3 attention layers must partition the one-based decoder indices")
        dsa = None
        if raw["model_type"] == "kimi_k31":
            dsa = DSAConfig.from_dict(c.get("dsa_config"), full)
        elif "dsa_config" in c:
            raise ValueError("Kimi K3 does not declare sparse attention")
        q = c["quantization_config"]
        if (q.get("quant_method") != "compressed-tensors" or q.get("format") != "mxfp4-pack-quantized"
                or q.get("quantization_status") != "compressed" or q.get("kv_cache_scheme") is not None
                or q.get("weight_scale_encoding") not in (None, "csf_32x2_v1")):
            raise ValueError("Kimi K3 requires compressed-tensors MXFP4 with native or csf_32x2_v1 scales")
        groups = list(q.get("config_groups", {}).values())
        if len(groups) != 1:
            raise ValueError("Kimi K3 requires one MXFP4 quantization group")
        group, w = groups[0], groups[0].get("weights", {})
        if (group.get("targets") != ["Linear"] or group.get("format") != q["format"]
                or group.get("input_activations") is not None or group.get("output_activations") is not None
                or any(w.get(k) != v for k, v in dict(num_bits=4, group_size=32, strategy="group",
                    type="float", symmetric=True, dynamic=False, scale_dtype="torch.uint8").items())):
            raise ValueError("unsupported Kimi K3 MXFP4 group")
        vision = raw["vision_config"]
        expected = dict(vt_hidden_size=1024, vt_intermediate_size=4096, vt_num_attention_heads=12,
            vt_num_hidden_layers=27, qkv_hidden_size=1536, patch_size=14, norm_type="rmsnorm",
            mlp_type="mlp2", activation_func="gelu_pytorch_tanh", attn_bias=False, linear_bias=False,
            patch_embed_proj_bias=False, init_pos_emb_height=64, init_pos_emb_width=64, init_pos_emb_time=4,
            pos_emb_type="divided_fixed", pos_emb_interpolation_mode="bilinear", merge_kernel_size=[2, 2],
            merge_type="sd2_tpool", mm_projector_type="patchmergerv2", mm_hidden_size=1024,
            text_hidden_size=c["hidden_size"], projector_hidden_act="gelu", projector_ln_eps=1e-5)
        for name, value in expected.items():
            if vision.get(name) != value:
                raise ValueError(f"unsupported Kimi K3 vision {name}: {vision.get(name)!r}")
        if raw.get("media_placeholder_token_id") != 163605 or raw.get("image_placeholder") != "<|kimi_image_placeholder|>":
            raise ValueError("unsupported Kimi K3 image placeholder")
        return cls(c, vision, dsa=dsa)

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def mla_norm_eps(self):
        """Query/KV normalization epsilon declared by the checkpoint's attention implementation."""
        return 1e-6 if self.dsa is None else self.text["rms_norm_eps"]

    @property
    def layers(self):
        linear = set(self.text["linear_attn_config"]["kda_layers"])
        return tuple("kda" if i + 1 in linear else "dsa" if self.dsa is not None else "mla"
                     for i in range(self.text["num_hidden_layers"]))

    def validate_shard(self, shard: TensorParallelShard):
        c = self.text
        for extent, alignment in ((c["num_attention_heads"], 1),
                (c["linear_attn_config"]["num_heads"], 1), (c["hidden_size"], 8),
                (c["routed_expert_hidden_size"], 8), (c["moe_intermediate_size"], 32),
                (c["intermediate_size"], 32), (c["num_shared_experts"] * c["moe_intermediate_size"], 32),
                (c["num_experts"], 8),
                (c["vocab_size"], 64)):
            shard.partition(extent, alignment=alignment)

    def projection_format(self, name):
        q = self.text["quantization_config"]
        if any(re.search(pattern[3:], name) if pattern.startswith("re:") else pattern == name for pattern in q["ignore"]):
            return "bf16"
        return "mxfp4" if q.get("weight_scale_encoding") is None else "mxfp4_csf"
