# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 canonical normalization and SiTU activation."""

import torch


def rms_norm(weight, eps):
    from transformers.models.kimi_linear.modeling_kimi_linear import KimiLinearRMSNorm

    with torch.device("meta"):
        module = KimiLinearRMSNorm(weight.numel(), eps=eps)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module


def situ(gate_up):
    gate, up = gate_up.float().chunk(2, dim=-1)
    return (4. * (gate / 4.).tanh() * gate.sigmoid() * (25. * (up / 25.).tanh())).to(gate_up.dtype)
