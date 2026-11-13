# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""One rank's Inkling 2 text model: decoder layers and vocabulary shards.

`ModelWeights.load` reads a rank's placed extents of every decoder layer, or of
the first `layers` of them, with the embedding and unembedding rows of its
vocabulary shard and the replicated normalization and output-mixture tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import TEXT_PREFIX
from .layer import LayerWeights, load_layer, scaled_query
from .weights import PlacedWeightLoader, checkpoint_layout


@dataclass(frozen=True)
class ModelWeights:
    config: object
    layers: tuple[LayerWeights, ...]
    embedding: torch.Tensor
    embedding_norm: torch.Tensor
    final_norm: torch.Tensor
    output_query: torch.Tensor
    unembedding: torch.Tensor
    rank: int
    world: int

    @classmethod
    def load(cls, manifest, device, *, config, rank, world, layers=None, progress=None):
        layout = checkpoint_layout(config, world)
        count = config.text.layers if layers is None else layers
        if type(count) is not int or not 1 <= count <= config.text.layers:
            raise ValueError("loaded decoder layers must be a positive prefix of the model")
        names = {f"{TEXT_PREFIX}.{name}" for name in ("embed.weight", "embed_norm.weight", "norm.weight",
                                                       "attn_res.output_query", "unembed.weight")}
        for layer in range(count):
            names.update(placement.name for placement in layout.layer(layer).values())
        with PlacedWeightLoader(manifest, device, layout=layout, names=names, progress=progress) as loader:
            def placed(name):
                return loader.load(f"{TEXT_PREFIX}.{name}", rank=rank)

            loaded = tuple(load_layer(loader, config, layer, rank) for layer in range(count))
            return cls(config, loaded, placed("embed.weight"), placed("embed_norm.weight"), placed("norm.weight"),
                       scaled_query(placed("attn_res.output_query"), config.text.hidden), placed("unembed.weight"),
                       rank, world)
