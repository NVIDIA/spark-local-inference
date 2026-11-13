# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""One rank's serving tensors of an Inkling 2 decoder layer.

`load_layer` reads a rank's placed checkpoint extents, fuses the attention
projections by input stream, prepares the expert slice for b12x and prescales
the FP32 attention-residual queries by `hidden ** -0.5`, the factor the
canonical mixing applies to each query before its dot products.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ignition.preparation.persistent import persistent_empty

from .attention import AttentionWeights, attention_weights
from .experts import ExpertWeights, prepare_experts

_ATTENTION_ROLES = ("query", "output_gate", "relative_query", "sink", "key", "value", "key_convolution_dynamic",
                    "value_convolution_dynamic")
_EXPERT_ROLES = ("w13", "w13_scale", "w2", "w2_scale", "w13_amax", "w2_amax", "w13_denominator", "w2_denominator")


@dataclass(frozen=True)
class LayerWeights:
    attention: AttentionWeights
    experts: ExpertWeights
    attention_convolution: torch.Tensor
    mlp_convolution: torch.Tensor
    input_norms: tuple[torch.Tensor, ...]
    mlp_norm: torch.Tensor
    attention_query: torch.Tensor | None
    mlp_query: torch.Tensor


def scaled_query(query, hidden):
    """FP32 residual queries `[streams, hidden]` multiplied by `hidden ** -0.5` in FP32."""
    scaled = persistent_empty((query.numel() // hidden, hidden), dtype=torch.float32, device=query.device)
    scaled.copy_(query.float().reshape(-1, hidden) * hidden ** -0.5)
    return scaled


def layer_weights(placed, config, layer, world) -> LayerWeights:
    """Serving tensors of decoder `layer` from one rank's placed tensors by checkpoint role."""
    t = config.text
    local = t.local(layer)
    attention = attention_weights(placed, local=local, heads=t.heads // world, kv_heads=t.kv_heads // world,
                                  dim=t.head_dim, width=t.convolution_width)
    expert_format = config.expert_format(layer)
    experts, denominators = prepare_experts(placed, format=expert_format, hidden=t.hidden,
                                            intermediate=t.intermediate // world)
    norms = (placed["attention_norm"],) if local else (placed["query_norm"], placed["key_norm"],
                                                       placed["value_norm"])
    query = placed.get("attention_query")
    return LayerWeights(attention, ExpertWeights(placed["router"], placed["router_bias"], experts, expert_format,
                                                 denominators),
                        placed["attention_convolution"], placed["mlp_convolution"], norms, placed["mlp_norm"],
                        None if query is None else scaled_query(query, t.hidden), scaled_query(placed["mlp_query"], t.hidden))


def load_layer(loader, config, layer, rank) -> LayerWeights:
    """Rank `rank`'s tensors of decoder `layer` from a `PlacedWeightLoader` holding them.

    Tensors that fusion or expert preparation replaces load outside persistent storage.
    """
    placements = loader.layout.layer(layer)
    temporary = set(_ATTENTION_ROLES) | set(_EXPERT_ROLES)
    placed = {role: loader.load(placement.name, rank=rank, temporary=role in temporary)
              for role, placement in placements.items()}
    return layer_weights(placed, config, layer, loader.layout.world_size)
