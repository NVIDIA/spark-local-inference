# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored tensor contract for the pinned Qwen3.8 Flash Next checkpoint."""

MODEL_ID = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
REVISION = "b9ce18a6887e18c089fdea8509a33e63429a3300"
ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
TRANSFORMERS_REVISION = "1d9654e78a23fb167ef539cbd699dfbd3e4830cf"
CONFIG_SHA256 = "b50dbc52a0c83b0cd2a44a372c1fb6b5e0e6827d22f2611c97d4d030f92f6b28"
INDEX_SHA256 = "5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1"
HEADERS_SHA256 = "b4884a790482d83fc86cf97eb01054780817d39e866c29013ca63cae881441aa"
GENERATION_SHA256 = "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
FRONTEND_SHA256 = {
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
}
TEXT_PREFIX = "model.language_model"
PLE_PREFIX = f"{TEXT_PREFIX}.layers.1.ple.ple_embedding"
EXCLUDED_QUANTIZATION_ALIAS = "mtp.layers.48.mlp.experts"

TEXT_GEOMETRY = {
    "hidden_size": 2560,
    "vocab_size": 248320,
    "num_hidden_layers": 48,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "hc_count": 4,
    "hc_lowrank": 320,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_compress_ratio": 4,
    "indexer_budget": 2048,
    "max_position_embeddings": 262144,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "split_ngram_parts": 128,
    "ple_embed_dim": 2560,
    "ple_conv_kernel_size": 4,
    "ple_layer_ids": [2],
    "ple_embedding_dtype": "nvfp4",
    "mtp_num_hidden_layers": 1,
}
VISION_GEOMETRY = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_heads": 16,
    "out_hidden_size": 2560,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
}


def expected_quantization():
    entries = {}
    for i in range(48):
        prefix = f"{TEXT_PREFIX}.layers.{i}"
        entries[f"{prefix}.mlp.experts"] = {
            "quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf_32x2_v1"
        }
        for proj in ("gate_proj", "up_proj", "down_proj"):
            entries[f"{prefix}.mlp.shared_expert.{proj}"] = {"quant_algo": "MXFP8", "group_size": 32}
        attention = (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj"))
            if i % 4 == 3 else
            ("linear_attn", ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"))
        )
        for proj in attention[1]:
            entries[f"{prefix}.{attention[0]}.{proj}"] = {"quant_algo": "MXFP8", "group_size": 32}
    entries[f"{PLE_PREFIX}.ngram_embedding"] = {"quant_algo": "NVFP4", "group_size": 16}
    for prefix in ("mtp.layers.0.mlp.experts", EXCLUDED_QUANTIZATION_ALIAS):
        entries[prefix] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    for i in range(27):
        for proj in ("proj", "qkv"):
            entries[f"model.visual.blocks.{i}.attn.{proj}"] = {"quant_algo": "MXFP8", "group_size": 32}
        for proj in ("linear_fc1", "linear_fc2"):
            entries[f"model.visual.blocks.{i}.mlp.{proj}"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    for proj in ("linear_fc1", "linear_fc2"):
        entries[f"model.visual.merger.{proj}"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    return entries


def _linear(prefix, rows, columns, quant=None, *, quantization=None):
    if quantization is not None and prefix in quantization:
        from .quantization import linear_format

        quant = {"mxfp8": "MXFP8", "mxfp4": "MXFP4", "nvfp4": quantization[prefix]["quant_algo"]}[linear_format(quantization, prefix)]
    if quant == "MXFP4":
        yield f"{prefix}.weight", "U8", (rows, columns // 2)
        yield f"{prefix}.weight_scale", "U8", (rows, columns // 32)
    elif quant == "MXFP8":
        yield f"{prefix}.weight", "F8_E4M3", (rows, columns)
        yield f"{prefix}.weight_scale", "U8", (rows, columns // 32)
    elif quant in ("CSF", "CSF_SLAB", "NVFP4", "W4A16_NVFP4"):
        yield f"{prefix}.weight", "U8", (rows, columns // 2)
        yield f"{prefix}.weight_scale_2", "F32", ()
        if quant == "CSF_SLAB":
            yield f"{prefix}.input_scale", "F32", ()
            yield f"{prefix}.weight_scale.nvfp4_csf_fixed", "U8", (rows // 16, 16 * (1 + columns // 32))
            yield f"{prefix}.weight_scale.nvfp4_csf_exceptions", "U32", (None,)
        elif quant == "CSF":
            yield f"{prefix}.input_scale", "F32", ()
            scale_columns = columns // 16
            groups = ((rows + 127) // 128) * ((scale_columns + 3) // 4)
            pages = (groups + 31) // 32
            blocks = (rows // 32) * ((scale_columns + 1) // 2)
            for field, dtype, size in (
                ("bases", "U8", rows), ("codes", "U8", rows * scale_columns // 2),
                ("page_bases", "U32", pages), ("offsets", "U16", blocks + pages),
                ("exceptions", "U16", None),
            ):
                yield f"{prefix}.weight_scale.csf32_{field}", dtype, (size,)
        else:
            yield f"{prefix}.weight_scale", "F8_E4M3", (rows, columns // 16)
            if quant == "NVFP4":
                yield f"{prefix}.input_scale", "F32", ()
    else:
        yield f"{prefix}.weight", "BF16", (rows, columns)


def _hyperconnection(prefix, *, block=True, norm_dtype="F32"):
    yield f"{prefix}.hc_norm.weight", norm_dtype, (10240,)
    yield from _linear(f"{prefix}.input_mix_weight_down", 320, 10240)
    yield from _linear(f"{prefix}.input_mix_weight_up", 10240, 320)
    if block:
        yield from _linear(f"{prefix}.block_inject_weight", 4, 10240)


def _sparse_attention(prefix, quant, *, quantization=None):
    for proj, rows, cols in (
        ("q_proj", 12288, 2560), ("k_proj", 512, 2560),
        ("v_proj", 512, 2560), ("o_proj", 2560, 6144),
        ("indexer.index_qk_proj", 640, 2560),
    ):
        yield from _linear(f"{prefix}.{proj}", rows, cols, quant, quantization=quantization)
    for norm in ("q_norm", "k_norm"):
        yield f"{prefix}.{norm}.weight", "BF16", (256,)
    for norm in ("q_layernorm", "k_layernorm"):
        yield f"{prefix}.indexer.{norm}.weight", "BF16", (128,)


def _mlp(prefix, *, mtp=False, quantization=None):
    yield from _linear(f"{prefix}.gate", 512, 2560)
    yield from _linear(f"{prefix}.shared_expert_gate", 1, 2560)
    expert_quant = "W4A16_NVFP4" if mtp else "CSF"
    if quantization is not None:
        from .quantization import expert_format

        fmt, mode = expert_format(quantization, f"{prefix}.experts")
        expert_quant = ("CSF_SLAB" if quantization[f"{prefix}.experts"]["weight_scale_encoding"] == "csf" else "CSF") if fmt == "nvfp4_csf" else "MXFP4" if fmt == "mxfp4" else "W4A16_NVFP4" if mode == "a16" else "NVFP4"
    for proj, rows, columns in (("gate_proj", 640, 2560), ("up_proj", 640, 2560), ("down_proj", 2560, 640)):
        yield from _linear(f"{prefix}.shared_expert.{proj}", rows, columns, None if mtp else "MXFP8",
                           quantization=quantization)
        for expert in range(512):
            yield from _linear(f"{prefix}.experts.{expert}.{proj}", rows, columns, expert_quant)


def expected_tensors(quantization=None):
    """Yield every required stored tensor; None marks the CSF exception count."""
    yield from _linear("lm_head", 248320, 2560)
    yield from _linear(f"{TEXT_PREFIX}.embed_tokens", 248320, 2560)
    yield from _hyperconnection(f"{TEXT_PREFIX}.hyper_connection_mixer", block=False)
    for i in range(48):
        prefix = f"{TEXT_PREFIX}.layers.{i}"
        for region in ("attn_hyper_connection", "mlp_hyper_connection"):
            yield from _hyperconnection(f"{prefix}.{region}")
        yield from _mlp(f"{prefix}.mlp", quantization=quantization)
        if i % 4 == 3:
            yield from _sparse_attention(f"{prefix}.self_attn", "MXFP8", quantization=quantization)
        else:
            prefix += ".linear_attn"
            for proj, rows, columns in (
                ("in_proj_a", 48, 2560), ("in_proj_b", 48, 2560),
                ("in_proj_qkv", 10240, 2560), ("in_proj_z", 6144, 2560),
                ("out_proj", 2560, 6144),
            ):
                yield from _linear(f"{prefix}.{proj}", rows, columns, "MXFP8", quantization=quantization)
            for name in ("A_log", "dt_bias"):
                yield f"{prefix}.{name}", "BF16", (48,)
            yield f"{prefix}.conv1d.weight", "BF16", (10240, 1, 4)
            yield f"{prefix}.norm.weight", "BF16", (128,)

    ple = f"{TEXT_PREFIX}.layers.1.ple"
    yield f"{ple}.conv1d.weight", "BF16", (10240, 1, 4)
    yield from _linear(f"{ple}.key_proj", 10240, 2560)
    yield from _linear(f"{ple}.value_proj", 2560, 2560)
    for norm in ("norm_conv", "norm_key", "norm_query"):
        yield f"{ple}.{norm}.weight", "F32", (10240,)
    for name, shape in (("layer_multipliers", (3,)), ("ngram_heads_offsets", (16,)), ("ngram_heads_vocab_sizes", (16,))):
        yield f"{PLE_PREFIX}.{name}", "I64", shape
    yield f"{PLE_PREFIX}.ngram_embedding.weight_scale_2", "F32", (1,)
    for i in range(128):
        prefix = f"{PLE_PREFIX}.ngram_embedding.shard_{i}"
        yield f"{prefix}.weight", "U8", (2500012, 80)
        yield f"{prefix}.weight_scale", "F8_E4M3", (2500012, 10)

    for i in range(27):
        prefix = f"model.visual.blocks.{i}"
        for name, rows, columns, quant in (
            ("attn.proj", 1152, 1152, "MXFP8"), ("attn.qkv", 3456, 1152, "MXFP8"),
            ("mlp.linear_fc1", 4304, 1152, "W4A16_NVFP4"),
            ("mlp.linear_fc2", 1152, 4304, "W4A16_NVFP4"),
        ):
            yield from _linear(f"{prefix}.{name}", rows, columns, quant, quantization=quantization)
            yield f"{prefix}.{name}.bias", "BF16", (rows,)
        for norm in ("norm1", "norm2"):
            for field in ("weight", "bias"):
                yield f"{prefix}.{norm}.{field}", "BF16", (1152,)
    for name, rows in (("linear_fc1", 4608), ("linear_fc2", 2560)):
        yield from _linear(f"model.visual.merger.{name}", rows, 4608, "W4A16_NVFP4", quantization=quantization)
        yield f"model.visual.merger.{name}.bias", "BF16", (rows,)
    for field in ("weight", "bias"):
        yield f"model.visual.merger.norm.{field}", "BF16", (1152,)
    yield "model.visual.patch_embed.proj.weight", "BF16", (1152, 3, 2, 16, 16)
    yield "model.visual.patch_embed.proj.bias", "BF16", (1152,)
    yield "model.visual.pos_embed.weight", "BF16", (2304, 1152)

    for proj in ("fc_embedding", "fc_hidden"):
        yield from _linear(f"mtp.{proj}", 2560, 2560)
    yield "mtp.pre_fc_norm_embedding.weight", "BF16", (2560,)
    yield "mtp.pre_fc_norm_hidden.weight", "BF16", (10240,)
    yield from _hyperconnection("mtp.hyper_connection_mixer", block=False, norm_dtype="BF16")
    for region in ("attn_hyper_connection", "mlp_hyper_connection"):
        yield from _hyperconnection(f"mtp.layers.0.{region}", norm_dtype="BF16")
    yield from _sparse_attention("mtp.layers.0.self_attn", None)
    yield from _mlp("mtp.layers.0.mlp", mtp=True, quantization=quantization)
