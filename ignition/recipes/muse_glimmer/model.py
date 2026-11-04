# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical centered normalization and dense decoder residual composition."""

import torch


def norm(kind, weight=None, *, eps):
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerRMSNorm, MuseGlimmerTextCenteredRMSNorm,
    )
    with torch.device("meta"):
        module = (MuseGlimmerTextCenteredRMSNorm(weight.numel(), eps=eps) if kind == "centered"
                  else MuseGlimmerRMSNorm(None if weight is None else weight.numel(), eps=eps,
                                         with_scale=weight is not None))
    if weight is not None:
        module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module.eval()


def decoder_layer(config, weights, attention, mlp):
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextDecoderLayer

    c = MuseGlimmerTextConfig.from_dict(config.text)
    layer = MuseGlimmerTextDecoderLayer.__new__(MuseGlimmerTextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.hidden_size, layer.config = c.hidden_size, c
    layer.self_attn, layer.mlp = attention, mlp
    for name, weight, eps in (("input_layernorm", weights.input_norm, c.rms_norm_eps),
            ("post_attention_layernorm", weights.post_attention_norm, c.post_norm_eps),
            ("pre_feedforward_layernorm", weights.pre_feedforward_norm, c.rms_norm_eps),
            ("post_feedforward_layernorm", weights.post_feedforward_norm, c.post_norm_eps)):
        setattr(layer, name, norm("centered", weight, eps=eps))
    return layer.eval()
