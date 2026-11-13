# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Muse Spark normalization and residual-gated decoder composition."""

import torch


def canonical_types():
    from transformers.models.muse_spark.modeling_muse_spark import (
        MuseSparkRMSNorm, MuseSparkTextCenteredRMSNorm, MuseSparkTextInputScaledRMSNorm,
        MuseSparkResidualGate, MuseSparkTextDecoderLayer,
    )
    return (MuseSparkRMSNorm, MuseSparkTextCenteredRMSNorm, MuseSparkTextInputScaledRMSNorm,
            MuseSparkResidualGate, MuseSparkTextDecoderLayer)


def norm(kind, weight=None, *, dimension=None, eps, with_scale=True):
    rms, centered, scaled, _, _ = canonical_types()
    cls = {"rms": rms, "centered": centered, "input_scaled": scaled}[kind]
    with torch.device("meta"):
        module = (cls(dimension, eps=eps, with_scale=with_scale) if kind == "rms"
                  else cls(weight.numel(), eps=eps))
    if weight is not None:
        module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module.eval()


def decoder_layer(config, weights, attention, mlp):
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig

    _, _, _, gate_type, layer_type = canonical_types()
    c = MuseSparkTextConfig.from_dict(config.text)
    layer = layer_type.__new__(layer_type)
    torch.nn.Module.__init__(layer)
    layer.hidden_size, layer.config = c.hidden_size, c
    layer.self_attn, layer.mlp = attention, mlp
    layer.input_layernorm = norm("centered", weights.input_norm, eps=c.rms_norm_eps)
    layer.post_attention_layernorm = norm("rms", dimension=c.hidden_size, eps=c.post_norm_eps, with_scale=False)
    layer.pre_feedforward_layernorm = norm("centered", weights.pre_feedforward_norm, eps=c.rms_norm_eps)
    layer.post_feedforward_layernorm = norm("input_scaled", weights.post_feedforward_norm, eps=c.post_norm_eps)
    for name, value in (("post_attention_residual_gate", weights.attention_gate),
                        ("post_feedforward_residual_gate", weights.feedforward_gate)):
        with torch.device("meta"):
            gate = gate_type(c)
        gate.gate = torch.nn.Parameter(value, requires_grad=False)
        setattr(layer, name, gate)
    return layer.eval()
