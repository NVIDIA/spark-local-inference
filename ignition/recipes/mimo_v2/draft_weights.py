# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharded DFlash projections and the checkpoint's learned mask embedding."""

from dataclasses import dataclass

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint


@dataclass(frozen=True)
class DraftLayerWeights:
    qkv: torch.Tensor
    output: torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    sink: torch.Tensor | None
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights


@dataclass(frozen=True)
class DraftWeights:
    projection: torch.Tensor
    hidden_norm: torch.Tensor
    context_kv: torch.Tensor
    norm: torch.Tensor
    mask_embedding: torch.Tensor
    layers: tuple[DraftLayerWeights, ...]

    @classmethod
    def load(cls, snapshot, config, device, shard, *, progress=None):
        c, a = config.text, config.attention
        q_range = shard.partition(a.heads * a.key_dim, alignment=a.key_dim)
        first, last = shard.replicated_partition(a.kv_heads)
        k_range = first * a.key_dim, last * a.key_dim
        v_range = first * a.value_dim, last * a.value_dim
        intermediate = shard.partition(c["intermediate_size"], alignment=128)
        manifest = inspect_indexed_checkpoint(snapshot)
        layers, context = [], []
        with DirectWeightLoader(manifest, device, progress=progress, progress_label="Load draft weights") as loader:
            def matrix(name, rows=None, columns=None):
                value = loader.load_tensor(name + ".weight", extent=(slice(*(rows or (None, None))),
                                                                     slice(*(columns or (None, None)))))
                if value.dtype != torch.bfloat16 or value.ndim != 2:
                    raise ValueError(f"DFlash requires BF16 matrices: {name}")
                return value

            projection = matrix("fc", rows=shard.partition(c["hidden_size"], alignment=64))
            hidden_norm, norm = loader.load_tensors(("hidden_norm.weight", "norm.weight"))
            for i in range(config.layers):
                prefix = f"layers.{i}"
                qkv = torch.cat([matrix(prefix + ".self_attn." + name, rows=rows)
                                 for name, rows in (("q_proj", q_range), ("k_proj", k_range), ("v_proj", v_range))])
                q_rows = q_range[1] - q_range[0]
                context.append(qkv[q_rows:])
                output = matrix(prefix + ".self_attn.o_proj",
                                columns=shard.partition(a.heads * a.value_dim, alignment=a.value_dim))
                query_norm, key_norm, input_norm, post_norm = loader.load_tensors(tuple(prefix + name for name in
                    (".self_attn.q_norm.weight", ".self_attn.k_norm.weight", ".input_layernorm.weight",
                     ".post_attention_layernorm.weight")))
                sink = loader.load_tensor(prefix + ".self_attn.attention_sink_bias", dtype=torch.float32,
                    extent=slice(*shard.partition(a.heads))) if a.sinks else None
                mlp = MLPWeights(torch.cat([matrix(prefix + ".mlp." + name, rows=intermediate)
                                           for name in ("gate_proj", "up_proj")]),
                                 matrix(prefix + ".mlp.down_proj", columns=intermediate))
                layers.append(DraftLayerWeights(qkv, output, query_norm, key_norm, sink, input_norm, post_norm, mlp))
        mask = torch.load(snapshot / "mask_embedding.pt", map_location="cpu", weights_only=True)
        if (not isinstance(mask, dict) or mask.get("mask_token_id") != config.noise_token
                or not isinstance(mask.get("embedding"), torch.Tensor)
                or mask["embedding"].shape != (c["hidden_size"],)
                or mask["embedding"].dtype != torch.bfloat16):
            raise ValueError("DFlash mask_embedding.pt disagrees with the draft configuration")
        return cls(projection, hidden_norm, torch.cat(context), norm, mask["embedding"].to(device), tuple(layers))
