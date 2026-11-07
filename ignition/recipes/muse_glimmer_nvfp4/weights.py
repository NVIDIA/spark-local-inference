# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local direct transfers of Muse Glimmer NVFP4 projections and BF16 norms."""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import cached_property
import json
from pathlib import Path

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights
from .config import PREFIX, ModelConfig
from .draft_config import DraftConfig

VISION_PREFIX = "model.vision_"


@dataclass(frozen=True)
class LayerWeights:
    """One decoder layer.

    ``qkv`` rows are ``[q | k | v]`` (``heads × head_dim``, then two ``kv_heads × head_dim`` blocks) and
    ``gate`` is the attention gate on its own: the checkpoint calibrates ``gate_proj`` with a
    ``weight_scale_2`` that differs from q/k/v, which forbids one fused NVFP4 group. Packed NVFP4 gives
    no shape signal on a swapped order, so the loader asserts the fused width instead.
    """
    qkv: object
    gate: object
    output: object
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    pre_feedforward_norm: torch.Tensor
    post_feedforward_norm: torch.Tensor
    mlp: MLPWeights


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    layers: tuple[LayerWeights, ...]
    allocated_bytes: int
    draft: object | None = None
    vision: object | None = None
    scale_scratch: tuple = ()

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, package_storage=None, draft_snapshot=None,
             vision=False):
        """``vision`` loads the BF16 tower, adapter and projection; the text-only path reads none of them."""
        before = torch.cuda.memory_allocated(device)
        loading = nullcontext if package_storage is None else package_storage.transient
        with loading(), MuseGlimmerWeightLoader(manifest, device, shard=shard, progress=progress) as loader:
            config = loader.config
            if draft_snapshot is not None:
                config = replace(config, draft=DraftConfig.load(draft_snapshot, config))
            first, last = shard.partition(config.hidden)
            embedding = loader.load_tensor(PREFIX + ".embed_tokens.weight", extent=(slice(None), slice(first, last)))
            norm = loader.load_tensor(PREFIX + ".norm.weight")
            first, last, width = config.vocabulary_partition(shard)
            head = torch.zeros((width, config.hidden), dtype=torch.bfloat16, device=device)
            loader._copy(head[:last-first], loader._sources["lm_head.weight"][first:last])
            loader._session.flush()
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
        draft = None
        if draft_snapshot is not None:
            from ignition.components.dflash2_weights import DraftWeights
            with loading():
                draft = DraftWeights.load(draft_snapshot, config.draft, device, shard, progress=progress)
        tower = None
        if vision:
            from .vision import VisionWeights
            with loading(), DirectWeightLoader(manifest, device, prefixes=(VISION_PREFIX,), progress=progress,
                                               progress_label="Load vision weights") as loader:
                tower = VisionWeights.load(vision_model(manifest.snapshot), loader)
        return cls(config, shard, embedding, norm, head, layers, torch.cuda.memory_allocated(device) - before,
                   draft=draft, vision=tower)


def vision_model(snapshot):
    """The canonical tower on the meta device: parameter names for the loader, modules for each vision graph."""
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionModel

    config = MuseGlimmerVisionConfig(**json.loads((Path(snapshot) / "config.json").read_text())["vision_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        module = MuseGlimmerVisionModel(config)
    return module.eval().requires_grad_(False)


class MuseGlimmerWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None):
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config, self.shard = ModelConfig.load(manifest.snapshot), shard
        self.config.validate_shard(shard)

    @cached_property
    def quantization(self):
        return self.config.quantization

    def _selected_names(self):
        return {n for n in super()._selected_names() if n.startswith(PREFIX + ".") or n == "lm_head.weight"}

    def load_layer(self, layer):
        shard = self.shard
        prefix = f"{PREFIX}.layers.{layer}"
        a = self.config.attention(layer)
        q0, q1 = shard.partition(a.heads * a.key_dim, alignment=a.key_dim)
        k0, k1 = shard.replicated_partition(a.kv_heads)
        kv = (k0 * a.key_dim, k1 * a.key_dim)
        attn = prefix + ".self_attn"
        qkv = self.load_linear_group(tuple(attn + "." + name + "_proj" for name in ("q", "k", "v")),
                                     row_ranges=((q0, q1), kv, kv))
        gate = self.load_linear(attn + ".gate_proj", rows=(q0, q1))
        heads, kv_heads = a.local(shard)
        if qkv.out_features + gate.out_features != 2 * heads * a.key_dim + kv_heads * (a.key_dim + a.value_dim):
            raise ValueError(f"{attn}: fused q/k/v and gate rows must fill the attention projection scratch")
        intermediate = shard.partition(self.config.intermediate, alignment=128)
        mlp = prefix + ".mlp"
        return LayerWeights(qkv, gate, self.load_linear(attn + ".o_proj", columns=(q0, q1)),
            *self.load_tensors(tuple(prefix + "." + name + ".weight" for name in (
                "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"))),
            MLPWeights(self.load_linear_group((mlp + ".gate_proj", mlp + ".up_proj"), row_ranges=(intermediate, intermediate)),
                       self.load_linear(mlp + ".down_proj", columns=intermediate)))
