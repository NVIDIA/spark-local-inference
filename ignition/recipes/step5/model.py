# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step-5 decoder semantics from StepFun's inference implementation.

Reference: https://huggingface.co/stepfun-ai/Step-5-Preview/blob/main/inference/model.py.
Attention and feed-forward modules are supplied by the preparation recipe.
"""

import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, weight, epsilon):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.epsilon = epsilon

    def forward(self, hidden):
        x = hidden.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.epsilon)
                * (1 + self.weight.float())).to(hidden.dtype)


class DecoderLayer(torch.nn.Module):
    def __init__(self, weights, attention, feed_forward, epsilon):
        super().__init__()
        self.input_norm = RMSNorm(weights.input_norm, epsilon)
        self.post_attention_norm = RMSNorm(weights.post_attention_norm, epsilon)
        self.attention, self.feed_forward = attention, feed_forward

    def forward(self, residual):
        residual = residual + self.attention(self.input_norm(residual).bfloat16()).float()
        return residual + self.feed_forward(self.post_attention_norm(residual).bfloat16()).float()


def rotary_inverse(dimension, theta, scaling, device):
    inverse = theta ** (-torch.arange(0, dimension, 2, dtype=torch.float32, device=device) / dimension)
    if scaling is not None:
        factor = scaling["factor"]
        low, high = scaling["low_freq_factor"], scaling["high_freq_factor"]
        original = scaling["original_max_position_embeddings"]
        wavelength = 2 * torch.pi / inverse
        smooth = (original / wavelength - low) / (high - low)
        inverse = torch.where(wavelength > original / low, inverse / factor,
            torch.where(wavelength < original / high, inverse,
                        (1 - smooth) * inverse / factor + smooth * inverse))
    return inverse


def rotary(values, positions, inverse):
    width = inverse.numel()
    frequencies = positions.clamp_min(0).float()[:, None] * inverse[None]
    cosine, sine = frequencies.cos().bfloat16().float()[:, None], frequencies.sin().bfloat16().float()[:, None]
    left, right = values[..., :width].float(), values[..., width:2 * width].float()
    rotated = torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).to(values.dtype)
    return torch.cat((rotated, values[..., 2 * width:]), -1)
