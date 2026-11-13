# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Rank-local persistent GLM state and invocation-local text metadata."""

from dataclasses import dataclass

import torch

from ignition.components.recurrent import DeltaMetadata, DeltaState

from .indexer import IndexMetadata
from .mla import MLAState


@dataclass(frozen=True)
class DecoderState:
    layers: tuple[DeltaState | MLAState, ...]

    @classmethod
    def allocate(cls, layer_types, world_size, slots, pages, device):
        if world_size not in (1, 2, 4):
            raise ValueError("GLM state requires one, two or four ranks")
        heads = 64 // world_size
        layers = []
        for kind in layer_types:
            if kind == "linear_attention":
                layers.append(DeltaState.allocate(slots, device, heads=heads, channels=3 * heads * 128))
            elif kind == "indexed_attention":
                layers.append(MLAState.allocate(pages, slots, device))
            else:
                raise ValueError(f"unsupported canonical GLM attention type: {kind}")
        return cls(tuple(layers))

    def named_tensors(self):
        for index, state in enumerate(self.layers):
            if isinstance(state, DeltaState):
                values = {"convolution": state.convolution, "recurrent": state.recurrent}
            else:
                values = {"latent_cache": state.cache, "index_cache": state.index.cache, "index_tail": state.index.tail}
            for name, value in values.items():
                yield f"layer{index}.{name}", value

    def clear_for_preparation(self):
        for _, tensor in self.named_tensors():
            tensor.zero_()


@dataclass(frozen=True)
class DecoderMetadata:
    token_ids: torch.Tensor
    sample_indices: torch.Tensor
    recurrent: DeltaMetadata
    indexed: IndexMetadata

    @classmethod
    def allocate(cls, tokens, sequences, blocks, device, *, samples=None, verification_columns=1):
        recurrent = DeltaMetadata.allocate(sequences, device, verification_columns=verification_columns)
        indexed = IndexMetadata.allocate(tokens, sequences, blocks, device)
        return cls(torch.zeros(tokens, dtype=torch.int64, device=device),
                   torch.zeros(sequences if samples is None else samples, dtype=torch.int64, device=device),
                   recurrent, indexed)
