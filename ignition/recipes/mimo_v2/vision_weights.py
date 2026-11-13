# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 vision projections with tensor-parallel attention and MLP partitions."""

from dataclasses import dataclass

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights


@dataclass(frozen=True)
class VisionLayerWeights:
    norm1: torch.Tensor
    norm2: torch.Tensor
    qkv: torch.Tensor
    qkv_bias: torch.Tensor
    output: torch.Tensor
    output_bias: torch.Tensor
    sink: torch.Tensor | None
    mlp: MLPWeights
    gate_up_bias: torch.Tensor
    down_bias: torch.Tensor


@dataclass(frozen=True)
class VisionWeights:
    config: dict
    shard: object
    patch: torch.Tensor
    layers: tuple[VisionLayerWeights, ...]
    norm: torch.Tensor
    merge_up: torch.Tensor
    merge_down: torch.Tensor

    @classmethod
    def load(cls, manifest, config, device, shard, *, progress=None):
        c = config["vision_config"]
        if (c["hidden_size"] != 1280 or c["num_heads"] != 32 or c["num_key_value_heads"] != 8
                or c.get("qk_channels", 64) != 64 or c.get("kv_channels", 64) != 64
                or c["spatial_merge_size"] != 2 or c["temporal_patch_size"] != 2 or c["patch_size"] != 16
                or c["out_hidden_size"] != config["hidden_size"] or c["hidden_act"] != "silu"):
            raise ValueError("unsupported MiMo vision geometry")
        q = shard.partition(c["num_heads"] * 64, alignment=64)
        first, last = shard.replicated_partition(c["num_key_value_heads"])
        k = first * 64, last * 64
        intermediate = shard.partition(c["intermediate_size"], alignment=64)
        merge = shard.partition(c["hidden_size"] * 4, alignment=64)
        layers = []
        with DirectWeightLoader(manifest, device, prefixes=("visual.",), progress=progress,
                                progress_label="Load image weights") as loader:
            def tensor(name, extent=None):
                value = loader.load_tensor("visual." + name, extent=extent)
                if value.dtype != torch.bfloat16:
                    raise ValueError(f"MiMo vision requires BF16 weights: {name}")
                return value

            def matrix(name, rows=None, columns=None):
                return tensor(name + ".weight", extent=(slice(*(rows or (None, None))),
                                                        slice(*(columns or (None, None)))))

            patch = tensor("patch_embed.proj.weight").flatten(1)
            for i in range(c["depth"]):
                p = f"blocks.{i}"
                ranges = (q, (2048 + k[0], 2048 + k[1]), (2560 + k[0], 2560 + k[1]))
                qkv = torch.cat([matrix(p + ".attn.qkv", rows=extent) for extent in ranges])
                bias = torch.cat([tensor(p + ".attn.qkv.bias", extent=slice(*extent)) for extent in ranges])
                sink = (tensor(p + ".attn.sinks", extent=slice(*shard.partition(c["num_heads"]))).float()
                        if c["use_sink"] and i not in c["fullatt_block_indexes"] else None)
                mlp = MLPWeights(torch.cat([matrix(p + ".mlp." + kind, rows=intermediate)
                                           for kind in ("gate_proj", "up_proj")]),
                                 matrix(p + ".mlp.down_proj", columns=intermediate))
                gate_bias = torch.cat([tensor(p + ".mlp." + kind + ".bias", extent=slice(*intermediate))
                                       for kind in ("gate_proj", "up_proj")])
                layers.append(VisionLayerWeights(tensor(p + ".norm1.weight"), tensor(p + ".norm2.weight"),
                    qkv, bias, matrix(p + ".attn.proj", columns=q), tensor(p + ".attn.proj.bias"), sink,
                    mlp, gate_bias, tensor(p + ".mlp.down_proj.bias")))
            return cls(c, shard, patch, tuple(layers), tensor("merger.ln_q.weight"),
                       matrix("merger.mlp.0", rows=merge), matrix("merger.mlp.2", columns=merge))
