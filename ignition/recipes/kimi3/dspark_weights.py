# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sharded Kimi K3 DSpark draft: Qwen3 sliding-window layers, context projections and the Markov head."""

from dataclasses import dataclass

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.persistent import persistent_cat, persistent_copy


@dataclass(frozen=True)
class DraftLayerWeights:
    qkv: torch.Tensor
    output: torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights


@dataclass(frozen=True)
class DraftWeights:
    """One rank's draft weights.

    ``projection`` holds this rank's rows of the auxiliary-state combination, gathered across ranks;
    ``context_kv`` stacks every layer's key and value rows, so accepted target states reach all draft
    caches through one projection. The embedding is column-sharded like the target's. The output head
    is the target's when the checkpoint's head is identical, and the Markov transition keeps its
    embedding replicated while its projection is vocabulary-sharded like the head. A plan storing
    online MXFP8 holds the combination, attention outputs and MLPs as MXFP8; queries, keys, values,
    the embedding and the Markov transition keep the checkpoint's BF16.
    """

    projection: torch.Tensor
    hidden_norm: torch.Tensor
    context_kv: torch.Tensor
    norm: torch.Tensor
    embedding: torch.Tensor
    head: torch.Tensor | None
    markov_embedding: torch.Tensor
    markov_head: torch.Tensor
    layers: tuple[DraftLayerWeights, ...]

    @classmethod
    def load(cls, snapshot, config, device, shard, *, target_head, progress=None, online_mxfp8=False):
        from ignition.components.weights import quantize_bf16_linear_as_mxfp8 as mxfp8_rows

        stored = mxfp8_rows if online_mxfp8 else lambda weight: weight
        d = config.draft
        d.validate_shard(shard)
        c, hidden = d.text, config.hidden
        q_rows = shard.partition(d.heads * d.head_dim, alignment=d.head_dim)
        kv_rows = shard.partition(d.kv_heads * d.head_dim, alignment=d.head_dim)
        intermediate = shard.partition(c["intermediate_size"], alignment=32)
        vocabulary = shard.partition(config.vocabulary, alignment=64)
        manifest = inspect_indexed_checkpoint(snapshot)
        layers, context = [], []
        with DirectWeightLoader(manifest, device, progress=progress, progress_label="Load draft weights") as loader:
            def matrix(name, rows=None, columns=None, *, temporary=False):
                value = loader.load_tensor(name + ".weight", extent=(slice(*(rows or (None, None))),
                                                                     slice(*(columns or (None, None)))),
                                           temporary=temporary)
                if value.dtype != torch.bfloat16 or value.ndim != 2:
                    raise ValueError(f"Kimi K3 DSpark requires BF16 matrices: {name}")
                return value

            projection = matrix("fc", rows=shard.partition(hidden, alignment=64), temporary=online_mxfp8)
            if projection.shape[1] != len(d.auxiliary_layers) * hidden:
                raise ValueError("DSpark fc must combine one hidden state per auxiliary layer")
            projection = stored(projection)
            hidden_norm, norm = loader.load_tensors(("hidden_norm.weight", "norm.weight"))
            for index in range(d.layers):
                prefix = f"layers.{index}"
                qkv = loader.load_linear_group(
                    tuple(prefix + ".self_attn." + name for name in ("q_proj", "k_proj", "v_proj")),
                    row_ranges=(q_rows, kv_rows, kv_rows))
                context.extend((qkv[q_rows[1] - q_rows[0]:q_rows[1] - q_rows[0] + kv_rows[1] - kv_rows[0]],
                                qkv[q_rows[1] - q_rows[0] + kv_rows[1] - kv_rows[0]:]))
                output = stored(matrix(prefix + ".self_attn.o_proj", columns=q_rows, temporary=online_mxfp8))
                query_norm, key_norm, input_norm, post_norm = loader.load_tensors(tuple(prefix + name for name in
                    (".self_attn.q_norm.weight", ".self_attn.k_norm.weight", ".input_layernorm.weight",
                     ".post_attention_layernorm.weight")))
                if online_mxfp8:
                    gate_up = stored(torch.cat([matrix(prefix + ".mlp." + name, rows=intermediate, temporary=True)
                                                for name in ("gate_proj", "up_proj")]))
                else:
                    gate_up = loader.load_linear_group(
                        tuple(prefix + ".mlp." + name for name in ("gate_proj", "up_proj")),
                        row_ranges=(intermediate, intermediate))
                mlp = MLPWeights(gate_up,
                                 stored(matrix(prefix + ".mlp.down_proj", columns=intermediate,
                                               temporary=online_mxfp8)))
                layers.append(DraftLayerWeights(qkv, output, query_norm, key_norm, input_norm, post_norm, mlp))
            embedding = matrix("embed_tokens", columns=shard.partition(hidden, alignment=64))
            head = matrix("lm_head", rows=vocabulary, temporary=True)
            # Kimi K3's DSpark draft reuses the target's output head; keep a copy only for drafts that differ.
            head = None if torch.equal(head, target_head) else persistent_copy(head)
            markov_embedding = matrix("markov_head.markov_w1")
            markov_head = matrix("markov_head.markov_w2", rows=vocabulary)
            if markov_embedding.shape[1] != d.markov_rank or markov_head.shape[1] != d.markov_rank:
                raise ValueError("DSpark Markov projections disagree with markov_rank")
        return cls(projection, hidden_norm, persistent_cat(context), norm, embedding, head, markov_embedding,
                   markov_head, tuple(layers))
