# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sharded Inferact Kimi K3.1 DSpark MLA, context, and Markov weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint

if TYPE_CHECKING:
    from b12x.gemm.blockscaled import Weight


@dataclass(frozen=True)
class DraftLayerWeights:
    attention: dict
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights


@dataclass(frozen=True)
class DraftWeights:
    """One rank's MLA draft with the target embedding and output head shared.

    ``context_kv`` concatenates every layer's complete KV latent projection. The
    draft input projections reuse rank-local rows of those same weights. Query
    and KV normalization, absorbed keys and values, and Markov weights retain
    BF16 storage when the remaining projections use online MXFP8.
    """

    projection: torch.Tensor | Weight
    hidden_norm: torch.Tensor
    context_kv: torch.Tensor
    norm: torch.Tensor
    embedding: torch.Tensor
    head: torch.Tensor | None
    markov_embedding: torch.Tensor
    markov_head: torch.Tensor
    layers: tuple[DraftLayerWeights, ...]

    @classmethod
    def load(cls, snapshot, config, device, shard, *, target_embedding, target_head,
             progress=None, online_mxfp8=False):
        from ignition.components.weights import quantize_bf16_linear_as_mxfp8

        d = config.draft
        d.validate_shard(shard)
        c, hidden = d.text, config.hidden
        h0, h1 = shard.partition(d.heads)
        q_rows = shard.partition(c["q_lora_rank"])
        kv_width = c["kv_lora_rank"] + c["qk_rope_head_dim"]
        kv_rows = shard.partition(kv_width)
        hidden_rows = shard.partition(hidden, alignment=64)
        intermediate = shard.partition(c["intermediate_size"], alignment=32)
        vocabulary = shard.partition(config.vocabulary, alignment=64)
        for name, tensor, shape in (("embedding", target_embedding,
                (config.vocabulary, hidden_rows[1] - hidden_rows[0])),
                ("head", target_head, (vocabulary[1] - vocabulary[0], hidden))):
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
                raise ValueError(f"Kimi K3.1 DSpark target {name} must be a BF16 tensor of shape {shape}")
        if target_embedding.device != target_head.device:
            raise ValueError("Kimi K3.1 DSpark target embedding and head must share a device")
        selected_device = torch.device(device)
        if (target_embedding.device.type != selected_device.type
                or selected_device.index is not None and target_embedding.device != selected_device):
            raise ValueError("Kimi K3.1 DSpark target weights must be on the draft device")
        manifest = inspect_indexed_checkpoint(snapshot)
        _validate_checkpoint(manifest, d)

        def stored(weight):
            return quantize_bf16_linear_as_mxfp8(weight) if online_mxfp8 else weight

        layers = []
        with DirectWeightLoader(manifest, device,
                prefixes=("context_", "final_norm.", "layers.", "markov_head."),
                progress=progress, progress_label="Load draft weights") as loader:
            def matrix(name, *, rows=None, columns=None):
                return loader.load_tensor(name + ".weight", extent=(slice(*(rows or (None, None))),
                    slice(*(columns or (None, None)))))

            projection = stored(matrix("context_proj", rows=hidden_rows))
            hidden_norm, norm = loader.load_tensors(("context_norm.weight", "final_norm.weight"))
            context_kv = torch.cat(loader.load_tensors(tuple(
                f"layers.{index}.self_attn.kv_a_proj_with_mqa.weight" for index in range(d.layers))))
            for index in range(d.layers):
                prefix = f"layers.{index}"
                attention = prefix + ".self_attn"
                query = matrix(attention + ".q_a_proj", rows=q_rows)
                key_value = context_kv[index * kv_width + kv_rows[0]:index * kv_width + kv_rows[1]]
                latent_width = query.shape[0] + key_value.shape[0]
                latent = torch.cat((query, key_value, query.new_zeros((-latent_width % 8, hidden))))
                kv = matrix(attention + ".kv_b_proj", rows=(h0 * 256, h1 * 256)).view(h1 - h0, 256, 512)
                query_norm, kv_norm, input_norm, post_norm = loader.load_tensors(tuple(prefix + name for name in
                    (".self_attn.q_a_layernorm.weight", ".self_attn.kv_a_layernorm.weight",
                     ".input_layernorm.weight", ".post_attention_layernorm.weight")))
                projections = dict(latent_gate=stored(latent),
                    q_b=stored(matrix(attention + ".q_b_proj", rows=(h0 * 192, h1 * 192))),
                    key=kv[:, :128].contiguous(), value=kv[:, 128:].transpose(1, 2).contiguous(),
                    query_norm=query_norm, kv_norm=kv_norm,
                    output=stored(matrix(attention + ".o_proj", columns=(h0 * 128, h1 * 128))))
                mlp = MLPWeights(stored(torch.cat(tuple(matrix(prefix + ".mlp." + name, rows=intermediate)
                    for name in ("gate_proj", "up_proj")))),
                    stored(matrix(prefix + ".mlp.down_proj", columns=intermediate)))
                layers.append(DraftLayerWeights(projections, input_norm, post_norm, mlp))
            markov_embedding = matrix("markov_head.markov_w1")
            markov_head = matrix("markov_head.markov_w2", rows=vocabulary)
        return cls(projection, hidden_norm, context_kv, norm, target_embedding, None,
                   markov_embedding, markov_head, tuple(layers))


def _validate_checkpoint(manifest, config):
    c, hidden = config.text, config.text["hidden_size"]
    shapes = {"context_proj.weight": (hidden, len(config.auxiliary_layers) * hidden),
              "context_norm.weight": (hidden,), "final_norm.weight": (hidden,),
              "markov_head.markov_w1.weight": (c["vocab_size"], config.markov_rank),
              "markov_head.markov_w2.weight": (c["vocab_size"], config.markov_rank)}
    if c.get("enable_confidence_head") is True:
        shapes["confidence_head.proj.weight"] = (1, hidden + (config.markov_rank
            if c.get("confidence_head_with_markov") is True else 0))
        shapes["confidence_head.proj.bias"] = (1,)
    attention = {"q_a_proj": (c["q_lora_rank"], hidden),
                 "q_b_proj": (config.heads * c["qk_head_dim"], c["q_lora_rank"]),
                 "q_a_layernorm": (c["q_lora_rank"],),
                 "kv_a_proj_with_mqa": (c["kv_lora_rank"] + c["qk_rope_head_dim"], hidden),
                 "kv_b_proj": (config.heads * (c["qk_nope_head_dim"] + c["v_head_dim"]), c["kv_lora_rank"]),
                 "kv_a_layernorm": (c["kv_lora_rank"],),
                 "o_proj": (hidden, config.heads * c["v_head_dim"])}
    for index in range(config.layers):
        prefix = f"layers.{index}"
        for name, shape in attention.items():
            shapes[f"{prefix}.self_attn.{name}.weight"] = shape
        for name in ("input_layernorm", "post_attention_layernorm"):
            shapes[f"{prefix}.{name}.weight"] = (hidden,)
        for name in ("gate_proj", "up_proj"):
            shapes[f"{prefix}.mlp.{name}.weight"] = (c["intermediate_size"], hidden)
        shapes[f"{prefix}.mlp.down_proj.weight"] = (hidden, c["intermediate_size"])
    missing, extra = shapes.keys() - manifest.tensors.keys(), manifest.tensors.keys() - shapes.keys()
    if missing or extra:
        raise ValueError(f"Kimi K3.1 DSpark tensor coverage differs from its configuration: "
                         f"missing={sorted(missing)}, unexpected={sorted(extra)}")
    for name, shape in shapes.items():
        tensor = manifest.tensors[name]
        if tensor.dtype != "BF16" or tuple(tensor.shape) != shape:
            raise ValueError(f"Kimi K3.1 DSpark {name} requires BF16 shape {shape}, "
                             f"found {tensor.dtype} {tuple(tensor.shape)}")
