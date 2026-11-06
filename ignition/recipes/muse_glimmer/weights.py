# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text weights with checkpoint NVFP4 and FP8 calibration."""

from contextlib import nullcontext
from dataclasses import dataclass
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import DirectWeightLoader
from .config import ModelConfig

PREFIX = "model.language_model"


@dataclass(frozen=True)
class LayerWeights:
    qkv: object
    output: object
    gate_up: object
    down: object
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    pre_feedforward_norm: torch.Tensor
    post_feedforward_norm: torch.Tensor


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: object
    layers: tuple[LayerWeights, ...]
    allocated_bytes: int
    draft: object | None = None

    @classmethod
    def load(cls, manifest, device, *, progress=None, package_storage=None):
        before = torch.cuda.memory_allocated(device)
        loading = nullcontext if package_storage is None else package_storage.transient
        with loading(), MuseGlimmerWeightLoader(manifest, device, progress=progress) as loader:
            config = ModelConfig.load(manifest.snapshot)
            embedding = loader.load_tensor(PREFIX + ".embed_tokens.weight")
            norm = loader.load_tensor(PREFIX + ".norm.weight")
            head = loader.load_linear("lm_head")
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
        return cls(config, TensorParallelShard(0, 1), embedding, norm, head, layers,
                   torch.cuda.memory_allocated(device) - before)


class MuseGlimmerWeightLoader(DirectWeightLoader):
    def _selected_names(self):
        return {n for n in super()._selected_names() if n.startswith((PREFIX + ".", "lm_head."))}

    def load_layer(self, layer):
        prefix = f"{PREFIX}.layers.{layer}"
        attn, mlp = prefix + ".self_attn", prefix + ".mlp"
        return LayerWeights(
            self.load_mixed_linear_group(tuple(attn + "." + name + "_proj" for name in ("q", "k", "v", "gate"))),
            self.load_linear(attn + ".o_proj"),
            self.load_mixed_linear_group((mlp + ".gate_proj", mlp + ".up_proj")),
            self.load_linear(mlp + ".down_proj"),
            *(self.load_tensor(prefix + "." + name + ".weight") for name in
              ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")))
