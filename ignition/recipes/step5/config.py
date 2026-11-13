# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step-5 checkpoint geometry and ModelOpt storage declarations."""

import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

PAGE_TOKENS = 128


@dataclass(frozen=True)
class AttentionGeometry:
    heads: int
    kv_heads: int
    dimension: int
    rotary: int
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
    vision: dict

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, config):
        if (config.get("model_type") != "step5_vl"
                or config.get("architectures") != ["Step5ForConditionalGeneration"]):
            raise ValueError("Step-5 preparation requires Step5ForConditionalGeneration")
        c, q = config["text_config"], config["quantization_config"]
        if (c.get("model_type") != "step5" or c.get("att_impl_type") != "GQA"
                or c.get("torch_dtype") != "bfloat16" or c.get("norm_dtype") != "float32"
                or c.get("fp32_residual_connection") is not True
                or c.get("use_head_wise_attn_gate") is not True
                or c.get("need_fp32_gate") is not True
                or c.get("use_moe_router_bias") is not True
                or c.get("norm_expert_weight") is not True):
            raise ValueError("unsupported Step-5 attention, residual, or routing arithmetic")
        if (q.get("quant_method") != "modelopt" or q.get("quant_algo") != "NVFP4"
                or q.get("group_size") != 16 or q.get("kv_cache_quant_algo") is not None
                or not isinstance(q.get("ignore"), list)):
            raise ValueError("Step-5 requires ModelOpt NVFP4 with explicit exclusions and unquantized KV")
        total = c["num_hidden_layers"] + c.get("num_nextn_predict_layers", 0)
        for name in ("layer_types", "rope_theta", "partial_rotary_factors", "swiglu_limits", "swiglu_limits_shared"):
            if len(c[name]) != total:
                raise ValueError(f"{name} must describe the target and MTP layers")
        if any(t not in ("sliding_attention", "full_attention") for t in c["layer_types"]):
            raise ValueError("unsupported Step-5 attention type")
        sparse = c["sparse_config"]
        expected = {"enabled": True, "proxy_dim": 256, "sparse_indexer_rope_dim": 32,
            "sparse_indexer_use_rope": True, "sparse_indexer_num_k_heads": 1,
            "sparse_indexer_q_norm_type": "rmsnorm", "sparse_indexer_k_norm_type": "layernorm",
            "sparse_indexer_csa_z_norm_type": "none", "compression_method": "csa_block_compress",
            "attention_impl": "sparse_gqa", "apply_to_layer_types": ["full_attention"],
            "region_block_size": 8, "topk": 512}
        if any(sparse.get(k) != v for k, v in expected.items()):
            raise ValueError("unsupported Step-5 compressed-region indexer")
        if (sparse["sparse_indexer_num_heads"] % sparse["num_provider_groups"]
                or sparse["num_provider_groups"] != c["num_attention_groups_full_attention"]):
            raise ValueError("indexer provider groups must align with GQA key heads")
        if c["sliding_window"] <= 0 or c["sliding_window"] % PAGE_TOKENS:
            raise ValueError("sliding history must contain complete cache pages")
        scaling = c.get("rope_scaling")
        if scaling is not None and scaling.get("rope_type") != "llama3":
            raise ValueError("unsupported Step-5 rotary scaling")
        result = cls(c, q, config["vision_config"])
        if (config.get("image_token_id") != 128001 or config.get("image_token_len") != 169
                or config.get("patch_token_len") != 81 or config.get("understand_projector_stride") != 2
                or config.get("vision_select_layer") != -1 or config.get("projector_bias") is not False):
            raise ValueError("unsupported Step-5 image feature layout")
        expected_vision = {"model_type": "perception_encoder", "image_size": 728, "patch_size": 14,
            "width": 1536, "layers": 47, "heads": 16, "pool_type": "none", "output_dim": None, "use_cls_token": False,
            "use_ln_post": False, "hidden_act": "quick_gelu"}
        if any(result.vision.get(k) != v for k, v in expected_vision.items()):
            raise ValueError("unsupported Step-5 perception encoder")
        for layer in range(total):
            a = result.attention(layer)
            if (a.heads % a.kv_heads or a.heads // a.kv_heads > 16 or a.dimension != 192 or a.rotary % 2
                    or not 0 < a.rotary <= a.dimension):
                raise ValueError("attention geometry must preserve complete heads and rotary pairs")
            base = result.attention(0)
            if (a.heads, a.kv_heads, a.dimension) != (base.heads, base.kv_heads, base.dimension):
                raise ValueError("Step-5 attention layers must share QKV geometry")
        if c.get("num_nextn_predict_layers") != 3 or any(result.attention(i).window < 0 for i in range(result.layers, total)):
            raise ValueError("Step-5 requires three sliding-window MTP predictors")
        if (len(set(c["moe_layer_list"])) != len(c["moe_layer_list"])
                or any(not 0 <= i < result.layers for i in c["moe_layer_list"])):
            raise ValueError("MoE layer indices must be distinct target layers")
        return result

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def sparse(self):
        return self.text["sparse_config"]

    def attention(self, layer):
        c = self.text
        sliding = c["layer_types"][layer] == "sliding_attention"
        suffix = "sliding_attention" if sliding else "full_attention"
        dim = c["head_dim_" + suffix]
        return AttentionGeometry(c["num_sliding_attention_heads" if sliding else "num_attention_heads"],
            c["num_attention_groups_" + suffix], dim, round(dim * c["partial_rotary_factors"][layer]),
            c["rope_theta"][layer], c["sliding_window"] if sliding else -1)

    def projection_format(self, prefix):
        return "bf16" if any(fnmatchcase(prefix, pattern) for pattern in self.quantization["ignore"]) else "nvfp4"

    def expert_format(self, prefix):
        # Stacked checkpoint projections correspond to the canonical module's expert list.
        formats = {self.projection_format(f"{prefix}.experts.{i}.{p}_proj")
                   for i in range(self.text["moe_num_experts"]) for p in ("gate", "up", "down")}
        if len(formats) != 1:
            raise ValueError(f"{prefix}: stacked experts require one declared weight format")
        return formats.pop()

    def validate_shard(self, shard):
        for layer in range(self.layers):
            self.attention(layer).local(shard)
        for extent, alignment in ((self.hidden, 64), (self.text["intermediate_size"], 32),
                                  (self.text["moe_intermediate_size"], 32),
                                  (self.text["share_expert_dim"], 32)):
            shard.partition(extent, alignment=alignment)

    def vocabulary_partition(self, shard):
        width = (self.vocabulary + shard.world_size * 64 - 1) // (shard.world_size * 64) * 64
        return shard.rank * width, min((shard.rank + 1) * width, self.vocabulary), width
