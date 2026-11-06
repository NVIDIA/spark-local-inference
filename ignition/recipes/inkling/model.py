# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inkling normalization, FP32 sink routing, route mixing and decoder composition."""

import torch


def norm(weight, eps):
    from transformers.models.inkling.modeling_inkling import InklingRMSNorm

    with torch.device("meta"):
        module = InklingRMSNorm(weight.numel(), eps=eps)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module.eval()


def route(logits, correction, scale, *, top_k, shared_experts, route_scale):
    """Select top-k routed experts and normalize their FP32 weights jointly with the shared-expert sinks."""
    routed = logits[..., :-shared_experts]
    ids = torch.topk(routed.sigmoid() + correction, top_k, dim=-1, sorted=False).indices
    selected = torch.cat((routed.gather(-1, ids), logits[..., -shared_experts:]), dim=-1)
    log_probabilities = torch.nn.functional.logsigmoid(selected)
    weights = torch.exp(log_probabilities - torch.logsumexp(log_probabilities, dim=-1, keepdim=True))
    weights = weights * route_scale * scale
    return ids, weights[..., :top_k], weights[..., top_k:]


def mix_routes(routes, weights, ids):
    """Accumulate weighted BF16 expert outputs in canonical ascending expert order."""
    order = ids.argsort(dim=-1)
    weighted = (routes * weights[..., None]).to(routes.dtype)
    ordered = weighted.gather(1, order[..., None].expand_as(weighted))
    result = torch.zeros_like(ordered[:, 0])
    for index in range(ordered.shape[1]):
        result = result + ordered[:, index]
    return result


def decoder_layer(config, weights, attention, mlp, attention_conv, mlp_conv):
    from transformers.models.inkling.configuration_inkling import InklingTextConfig
    from transformers.models.inkling.modeling_inkling import InklingDecoderLayer

    layer = InklingDecoderLayer.__new__(InklingDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.config = InklingTextConfig.from_dict(config.text)
    layer.hidden_size = config.hidden
    layer.self_attn, layer.mlp = attention, mlp
    layer.attn_sconv, layer.mlp_sconv = attention_conv, mlp_conv
    layer.input_layernorm = norm(weights.input_norm, config.text["rms_norm_eps"])
    layer.post_attention_layernorm = norm(weights.post_attention_norm, config.text["rms_norm_eps"])
    return layer.eval()
