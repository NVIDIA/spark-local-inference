# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel BF16 perception-encoder projections and convolution weights."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VisionWeights:
    config: dict
    patch: torch.Tensor
    position: torch.Tensor
    norm: tuple
    layers: tuple
    first: tuple
    second: tuple
    projector: torch.Tensor

    @classmethod
    def load(cls, loader):
        c, shard = loader.config.vision, loader.shard
        h = c["width"]
        first, last = shard.partition(h)
        m0, m1 = shard.partition(8960)
        d0, d1 = shard.partition(2 * h)

        def tensor(name, extent=None):
            value = loader.load_tensor("vision_model." + name, extent=extent)
            if value.dtype != torch.bfloat16:
                raise ValueError(f"vision_model.{name}: expected BF16 storage")
            return value

        def norm(name):
            return tensor(name + ".weight"), tensor(name + ".bias")

        def affine(name, rows=None, columns=None):
            return (tensor(name + ".weight", (slice(None) if rows is None else slice(*rows),
                       slice(None) if columns is None else slice(*columns))),
                    tensor(name + ".bias", slice(None) if rows is None else slice(*rows)))

        layers = []
        for i in range(c["layers"]):
            prefix = f"transformer.resblocks.{i}."
            shape = (3 * (last - first), h)
            qkv = torch.empty(shape, dtype=torch.bfloat16, device=loader.device)
            bias = torch.empty(shape[0], dtype=torch.bfloat16, device=loader.device)
            for j in range(3):
                output = slice(j * (last - first), (j + 1) * (last - first))
                source = slice(j * h + first, j * h + last)
                loader._copy(qkv[output], loader._sources["vision_model." + prefix + "attn.in_proj_weight"][source])
                loader._copy(bias[output], loader._sources["vision_model." + prefix + "attn.in_proj_bias"][source])
            loader._session.flush()
            layers.append({"norm1": norm(prefix + "ln_1"), "norm2": norm(prefix + "ln_2"),
                "scale1": tensor(prefix + "ls_1.gamma"), "scale2": tensor(prefix + "ls_2.gamma"),
                "qkv": (qkv, bias), "output": affine(prefix + "attn.out_proj", columns=(first, last)),
                "up": affine(prefix + "mlp.c_fc", rows=(m0, m1)),
                "down": affine(prefix + "mlp.c_proj", columns=(m0, m1))})
        conv1, bias1 = affine("vit_downsampler1", rows=(d0, d1))
        conv2, bias2 = affine("vit_downsampler2", columns=(d0, d1))
        return cls(c, tensor("conv1.weight").flatten(1), tensor("positional_embedding"), norm("ln_pre"),
            tuple(layers), (conv1.flatten(1), bias1), (conv2.flatten(1), bias2),
            loader.load_tensor("vit_large_projector.weight", extent=(slice(*shard.partition(loader.config.hidden)), slice(None))))
