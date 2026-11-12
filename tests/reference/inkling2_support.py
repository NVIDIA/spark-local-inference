# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Inkling 2 checkpoint configurations, stored tensors and rank shards."""

import torch

from ignition.recipes.inkling2.config import ModelConfig
from ignition.recipes.inkling2.weights import checkpoint_layout


def synthetic_config(layers=7, local=(0, 1, 2, 3, 4, 6), mxfp8=(5, 6), **text):
    """A small checkpoint configuration of the Inkling 2 family and its precision groups."""
    text_config = dict(model_max_length=4096, torch_dtype="bfloat16", hidden_size=256, num_hidden_layers=layers,
        vocab_size=1000, num_attention_heads=8, num_key_value_heads=4, head_dim=32, d_rel=8, rel_extent=16,
        q_bias=False, o_bias=False, log_scaling_n_floor=20, log_scaling_alpha=0.1, rms_norm_eps=1e-6,
        pre_norm_pre_affine=False, final_norm_pre_affine=False, use_value_norm_no_gamma=True,
        qk_l2_clip_threshold=4.0, rel_l2_clip_threshold=2.0, local_layer_ids=list(local), sconv_kernel_size=4,
        unpadded_vocab_size=990, final_logit_softcapping=30.0, swa_head_dim=32, swa_num_attention_heads=8,
        swa_num_key_value_heads=4, sliding_window_size=16, n_layers_per_attn_res_block=6,
        attn_res_normalize_values=True, n_routed_experts=16, num_experts_per_tok=8, n_shared_experts=0,
        intermediate_size=256, mlp_gate_softcap=4.0, mlp_up_softcap=25.0, route_scale=8.0, use_gate_bias=True)
    text_config.update(text)
    config = dict(architectures=["Inkling2ForConditionalGeneration"], model_type="inkling2", eos_token_id=3,
        text_config=text_config,
        vision_config=dict(temporal_patch_size=2, native_patch_size=4, adapter_merge_size=2,
            projection_hidden_dim=64, projector_bias=False, hidden_size=64, num_hidden_layers=1,
            num_attention_heads=4, mlp_ratio=4.0, posemb_grid_size=4, layer_norm_eps=1e-5),
        mtp_config=dict(hidden_size=text_config["hidden_size"], num_hidden_layers=2, num_attention_heads=8,
            num_key_value_heads=4, head_dim=32, intermediate_size=512, max_position_embeddings=4096,
            torch_dtype="bfloat16", q_bias=False, o_bias=False, rms_norm_eps=1e-6, rope_theta=1e6,
            local_layer_ids=[0], sliding_window_size=16, block_size=8, markov_rank=16, markov_head_type="rnn_hidden",
            target_layer_ids=[1, 5, layers]))
    return config, synthetic_quantization(config, mxfp8)


def synthetic_quantization(config, mxfp8):
    """Precision groups naming each tensor as the Inkling 2 checkpoint's hf_quant_config.json does."""
    t, v, d = config["text_config"], config["vision_config"], config["mtp_config"]
    llm, layers, local = "model.llm", t["num_hidden_layers"], set(t["local_layer_ids"])
    fp32 = [f"{llm}.attn_res.attn_queries.{i}" for i in range(layers - 1)]
    fp32 += [f"{llm}.attn_res.mlp_queries.{i}" for i in range(layers)] + [f"{llm}.attn_res.output_query"]
    fp32 += [f"{llm}.layers.{i}.mlp.gate.bias" for i in range(layers)]
    bf16 = [f"{llm}.embed.weight", f"{llm}.embed_norm.weight", f"{llm}.norm.weight"]
    experts = {"nvfp4": [], "mxfp8": []}
    for i in range(layers):
        p, a = f"{llm}.layers.{i}", f"{llm}.layers.{i}.attn"
        bf16 += [f"{a}.{name}.weight" for name in ("wq_du", "gate_attn", "wr_du", "attn_sink_proj", "wk_dv", "wv_dv",
                                                    "wo_ud")]
        bf16 += [f"{a}.q_l2_clip.scale", f"{a}.rel_l2_clip.scale", f"{a}.rel_logits_proj.proj"]
        for stream in ("k", "v"):
            bf16 += ([f"{a}.{stream}_sconv.weight"] if i in local else
                     [f"{a}.{stream}_sconv.static_w", f"{a}.{stream}_sconv.weight_proj"])
        norms = ("attn_norm",) if i in local else ("q_input_norm", "k_input_norm", "v_input_norm")
        bf16 += [f"{p}.{name}.weight" for name in (*norms, "mlp_norm", "attn_sconv", "mlp_sconv", "mlp.gate")]
        experts["mxfp8" if i in mxfp8 else "nvfp4"] += [f"{p}.mlp.experts.w13_weight", f"{p}.mlp.experts.w2_weight"]
    b = "model.visual.backbone"
    bf16 += [f"{b}.embeddings.{name}" for name in ("patch_embedding.weight", "position_embedding", "class_embedding",
                                                   "cls_position_embedding")]
    bf16 += [f"{b}.ln_pre.weight", f"{b}.ln_pre.bias", "model.visual.proj_in.weight", "model.visual.proj_out.weight"]
    for i in range(v["num_hidden_layers"]):
        block = f"{b}.layers.{i}"
        bf16 += [f"{block}.{name}.{suffix}" for name in ("ln_1", "ln_2", "self_attn.q_proj", "self_attn.k_proj",
                 "self_attn.v_proj", "self_attn.out_proj", "mlp.c_fc", "mlp.c_proj") for suffix in ("weight", "bias")]
        bf16 += [f"{block}.ls_1.gamma", f"{block}.ls_2.gamma"]
    draft = ["fuse_weight", "mask_embed", "norm.weight"]
    for i in range(d["num_hidden_layers"]):
        draft += [f"layers.{i}.{name}.weight" for name in ("input_layernorm", "post_attention_layernorm",
                  "dflare_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
        draft += [f"layers.{i}.self_attn.{name}.weight" for name in ("q_norm", "k_norm", "k_norm_ctx", "q_proj",
                  "k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx", "o_proj")]
    draft += [f"markov_head.{name}" for name in ("markov_w1.weight", "rnn_proj.weight", "rnn_proj.bias",
              "hidden_out.weight", "hidden_out.bias", "dflare_norm.weight")]
    plain = lambda kind, **extra: dict(compute_dtype=kind, output_dtype=kind, required_activation_quantization=None,  # noqa: E731
                                       weight_block_shape=[], weight_transform="cast", **extra)
    groups = [plain("float32", tensors=fp32), plain("bfloat16", tensors=bf16),
        dict(compute_dtype="nvfp4", output_dtype="uint8", required_activation_quantization=dict(
            amax_type="per_token", block_size=16, format="nvfp4_e2m1", rounding="nearest",
            scale_dtype="float8_e4m3fn", scale_selection_rule="fouroversix_mae"), weight_block_shape=[1, 16],
            weight_scale_format="amax_denominator", weight_transform="nvfp4_nearest", tensors=experts["nvfp4"]),
        dict(compute_dtype="mxfp8", output_dtype="float8_e4m3fn", required_activation_quantization=dict(
            block_size=32, format="mxfp8_e4m3", rounding="nearest", scale_dtype="ue8m0"),
            weight_block_shape=[32, 32], weight_transform="mxfp8_nearest", tensors=experts["mxfp8"]),
        plain("bfloat16", fwd_output_dtype="float32", tensors=[f"{llm}.unembed.weight"])]
    return dict(quant_method="inkling2", model=dict(parameter_groups=[g for g in groups if g["tensors"]]),
                mtp=dict(parameter_groups=[plain("bfloat16", tensors=draft)]))


def random_tensor(placement, generator):
    """Stored-format random data with magnitudes like the checkpoint's for the placement's role."""
    shape, role = placement.shape, placement.role
    if placement.dtype == "U8" and role in ("w13", "w2"):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
    if role in ("w13_scale", "w2_scale") and placement.dtype == "F8_E4M3":
        return (16 + 432 * torch.rand(shape, generator=generator)).to(torch.float8_e4m3fn)
    if role in ("w13_scale", "w2_scale"):
        return torch.randint(114, 121, shape, dtype=torch.uint8, generator=generator)
    if role in ("w13", "w2"):
        # E4M3 codes without the NaN encodings 0x7F and 0xFF.
        codes = torch.randint(0, 126, shape, dtype=torch.uint8, generator=generator)
        return (codes | (torch.randint(0, 2, shape, dtype=torch.uint8, generator=generator) << 7)).view(
            torch.float8_e4m3fn)
    if role in ("w13_amax", "w2_amax"):
        return 0.05 + 0.5 * torch.rand(shape, generator=generator)
    if role in ("w13_denominator", "w2_denominator"):
        return torch.full(shape, 2688.0)
    values = torch.randn(shape, generator=generator)
    if role.endswith("_norm") or role.endswith("clip_scale") or role in ("embedding_norm", "final_norm"):
        values = 1 + 0.1 * values
    elif role.endswith("_query") and placement.precision == "float32":
        values = 0.5 * values
    elif role == "router_bias":
        values = 0.02 * values
    elif "convolution" in role and not role.endswith("dynamic"):
        values = 0.3 * values
    elif len(shape) == 2:
        values = values * shape[-1] ** -0.5
    return values.to(torch.float32 if placement.dtype == "F32" else torch.bfloat16)


def shard(tensor, placement, rank):
    """Rank `rank`'s extent of a stored tensor, zero-filled past the stored extent."""
    if placement.axis is None:
        return tensor.clone()
    first, last = placement.rows(rank)
    result = torch.zeros(placement.local_shape, dtype=tensor.dtype, device=tensor.device)
    result.narrow(placement.axis, 0, last - first).copy_(tensor[placement.extent(rank)])
    return result


def synthetic_model(seed=0, world_size=4, **settings):
    """A validated config, its four-rank layout and random stored tensors for every model placement."""
    config = ModelConfig.from_dict(*synthetic_config(**settings))
    layout = checkpoint_layout(config, world_size)
    generator = torch.Generator().manual_seed(seed)
    tensors = {name: random_tensor(placement, generator) for name, placement in sorted(layout.model.items())}
    return config, layout, tensors
