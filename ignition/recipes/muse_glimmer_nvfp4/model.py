# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Muse Glimmer normalization and plain-residual decoder composition."""

import torch


def canonical_types():
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerRMSNorm, MuseGlimmerTextCenteredRMSNorm, MuseGlimmerTextDecoderLayer,
    )
    return MuseGlimmerRMSNorm, MuseGlimmerTextCenteredRMSNorm, MuseGlimmerTextDecoderLayer


def norm(kind, weight=None, *, dimension=None, eps, with_scale=True):
    rms, centered, _ = canonical_types()
    cls = {"rms": rms, "centered": centered}[kind]
    with torch.device("meta"):
        module = cls(dimension, eps=eps, with_scale=with_scale) if kind == "rms" else cls(weight.numel(), eps=eps)
    if weight is not None:
        module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module.eval()


def decoder_layer(config, weights, attention, mlp):
    """input_layernorm → attn → post_attention_layernorm → add; pre_feedforward → mlp → post_feedforward → add."""
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig

    _, _, layer_type = canonical_types()
    c = MuseGlimmerTextConfig.from_dict(config.text)
    layer = layer_type.__new__(layer_type)
    torch.nn.Module.__init__(layer)
    layer.hidden_size, layer.config = c.hidden_size, c
    layer.self_attn, layer.mlp = attention, mlp
    layer.input_layernorm = norm("centered", weights.input_norm, eps=c.rms_norm_eps)
    layer.post_attention_layernorm = norm("centered", weights.post_attention_norm, eps=c.post_norm_eps)
    layer.pre_feedforward_layernorm = norm("centered", weights.pre_feedforward_norm, eps=c.rms_norm_eps)
    layer.post_feedforward_layernorm = norm("centered", weights.post_feedforward_norm, eps=c.post_norm_eps)
    return layer.eval()
