# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small checkpoints with the publisher's original tensor names and mixed precision."""

import json

import torch
from safetensors.torch import save_file

from ignition.recipes.inkling.config import ModelConfig


def configuration():
    return dict(architectures=["InklingForConditionalGeneration"], model_type="inkling_mm_model", eos_token_id=200006,
        text_config=dict(hidden_size=256, num_hidden_layers=3, vocab_size=256, unpadded_vocab_size=250,
            num_attention_heads=2, num_key_value_heads=1, head_dim=128, swa_head_dim=128,
            swa_num_attention_heads=2, swa_num_key_value_heads=1, sliding_window_size=128,
            d_rel=16, rel_extent=256, local_layer_ids=[0], dense_mlp_idx=1,
            dense_intermediate_size=256, intermediate_size=128, n_routed_experts=4,
            num_experts_per_tok=2, n_shared_experts=2, shared_expert_sink=True,
            q_bias=False, o_bias=False, use_sconv=True, use_embed_norm=True,
            sconv_kernel_size=4, hidden_act="silu", route_scale=8., use_gate_bias=True,
            gate_activation="sigmoid", norm_after_topk=True, use_global_scale=True,
            logits_mup_width_multiplier=16., final_logit_softcapping=None,
            log_scaling_n_floor=128, log_scaling_alpha=.1, rms_norm_eps=1e-6, torch_dtype="bfloat16"))


def quantization():
    excluded = ["model.llm." + name for name in ("embed", "embed_norm", "norm", "unembed")]
    for layer in range(3):
        prefix = f"model.llm.layers.{layer}."
        excluded.extend(prefix + name for name in ("attn", "attn_norm", "attn_sconv", "mlp_norm", "mlp_sconv"))
        excluded.extend(prefix + name for name in
            (("mlp.w13_dn", "mlp.w2_md", "mlp.global_scale") if layer == 0
             else ("mlp.gate", "mlp.shared_experts")))
    excluded.append("model.llm.layers.1.mlp.experts")
    quantizer = dict(num_bits=[2, 1], block_sizes={"-1": 16, "type": "dynamic", "scale_bits": [4, 3]},
                     axis=None, enable=True)
    return dict(quant_algo="NVFP4", kv_cache_quant_algo="none", group_size=16, exclude_modules=excluded,
        modelopt_quant_config=dict(algorithm="max", quant_cfg={"*weight_quantizer": quantizer,
            "*input_quantizer": quantizer.copy(), "default": {"enable": False}}))


def checkpoint(path, *, moe_intermediate=128, routed_experts=4, top_k=2):
    torch.manual_seed(7821)
    raw = configuration()
    raw["text_config"]["intermediate_size"] = moe_intermediate
    raw["text_config"]["n_routed_experts"] = routed_experts
    raw["text_config"]["num_experts_per_tok"] = top_k
    quant = quantization()
    config = ModelConfig.from_dict(raw, quant)
    c, h = config.text, config.hidden
    tensors = {}
    def rand(name, shape, scale=.025, dtype=torch.bfloat16):
        tensors[name] = (torch.randn(shape) * scale).to(dtype)
    def ones(name, shape):
        tensors[name] = torch.ones(shape, dtype=torch.bfloat16)
    prefix = "model.llm"
    rand(prefix + ".embed.weight", (c["vocab_size"], h), .1)
    rand(prefix + ".unembed.weight", (c["vocab_size"], h), .1)
    for name in ("embed_norm", "norm"):
        ones(prefix + "." + name + ".weight", (h,))
    for i in range(config.layers):
        p = prefix + f".layers.{i}"
        a = config.attention(i)
        for name, rows in (("wq_du", a.heads * a.key_dim), ("wk_dv", a.kv_heads * a.key_dim),
                           ("wv_dv", a.kv_heads * a.key_dim), ("wr_du", a.heads * c["d_rel"])):
            rand(p + ".attn." + name + ".weight", (rows, h))
        rand(p + ".attn.wo_ud.weight", (h, a.heads * a.key_dim))
        rand(p + ".attn.rel_logits_proj.proj", (c["d_rel"], a.relative_extent))
        for name in ("q_norm", "k_norm"):
            ones(p + ".attn." + name + ".weight", (a.key_dim,))
        for name in ("attn_norm", "mlp_norm"):
            ones(p + "." + name + ".weight", (h,))
        for name, channels in (("attn.k_sconv", a.kv_heads * a.key_dim), ("attn.v_sconv", a.kv_heads * a.key_dim),
                               ("attn_sconv", h), ("mlp_sconv", h)):
            rand(p + "." + name + ".weight", (channels, 1, 4), .05)
        if c["mlp_layer_types"][i] == "dense":
            inter = c["intermediate_size"]
            rand(p + ".mlp.w13_dn.weight", (2 * inter, h))
            rand(p + ".mlp.w2_md.weight", (h, inter))
            tensors[p + ".mlp.global_scale"] = torch.tensor([1.125], dtype=torch.bfloat16)
        else:
            inter = c["moe_intermediate_size"]
            for name, count in (("experts", c["n_routed_experts"]), ("shared_experts", c["n_shared_experts"])):
                stem = "shared_" if name == "shared_experts" else ""
                rand(p + f".mlp.{name}.{stem}w13_weight", (count, 2 * inter, h))
                rand(p + f".mlp.{name}.{stem}w2_weight", (count, h, inter))
            rand(p + ".mlp.gate.weight", (c["n_routed_experts"] + c["n_shared_experts"], h))
            rand(p + ".mlp.gate.bias", (c["n_routed_experts"],), .01, torch.float32)
            tensors[p + ".mlp.gate.global_scale"] = torch.tensor([.14234226942062378], dtype=torch.float32)
            if i in config.quantized_routed_layers:
                for projection in ("w13_weight", "w2_weight"):
                    name = p + ".mlp.experts." + projection
                    original = tensors[name]
                    count, rows, columns = original.shape
                    # Independent exact checkpoint bytes: signed E2M1 codes, distinct
                    # block scales and per-expert gains expose row/column packing errors.
                    codes = torch.randint(0, 16, original.shape, dtype=torch.uint8)
                    tensors[name] = codes[..., ::2] | (codes[..., 1::2] << 4)
                    tensors[name + ".scale"] = (torch.rand(count, rows, columns // 16) * 2 + 1).to(torch.float8_e4m3fn)
                    tensors[name + ".scale2"] = torch.linspace(.001, .0025, count, dtype=torch.float32)
                    tensors[name + ".input_amax"] = torch.tensor([1.8359375 if projection == "w13_weight" else 187.], dtype=torch.bfloat16)
                    tensors[name + ".original_shape"] = torch.tensor(original.shape, dtype=torch.int64)
    (path / "config.json").write_text(json.dumps(raw))
    (path / "hf_quant_config.json").write_text(json.dumps(dict(quantization=quant)))
    save_file(tensors, path / "model.safetensors")
    return config, tensors
