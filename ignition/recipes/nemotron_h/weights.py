# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H checkpoint transfers preserving each declared weight format."""

from dataclasses import dataclass, replace

import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.persistent import persistent_copy, allocated_tensor_bytes

from .config import ModelConfig

PREFIX = "language_model.backbone"


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: object
    layers: tuple[dict, ...]
    allocated_bytes: int
    vision: object = None
    draft: object = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, draft_checkpoint=None):
        before = allocated_tensor_bytes(device)
        with NemotronWeightLoader(manifest, device, shard=shard, progress=progress) as loader:
            config = loader.config
            embedding, norm = loader.load_tensors((PREFIX + ".embeddings.weight", PREFIX + ".norm_f.weight"))
            if embedding.dtype != torch.bfloat16 or embedding.shape != (config.vocabulary, config.hidden):
                raise ValueError("Nemotron-H requires a BF16 embedding matching the configured vocabulary")
            head = loader.load_linear("language_model.lm_head")
            layers = tuple(loader.load_layer(i) for i in range(len(config.layers)))
        from .vision import VisionWeights
        vision = VisionWeights.load(manifest, device, progress=progress)
        draft = None
        if draft_checkpoint is not None:
            from .draft_config import DraftConfig
            from .draft_weights import DraftWeights

            config = replace(config, draft=DraftConfig.load(draft_checkpoint, config))
            draft = DraftWeights.load(draft_checkpoint, config.draft, device, progress=progress)
        return cls(config, shard, embedding, norm, head, layers, allocated_tensor_bytes(device) - before, vision=vision, draft=draft)


class NemotronWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None):
        super().__init__(manifest, device, prefixes=prefixes, progress=progress)
        self.config, self.shard = ModelConfig.load(manifest.snapshot), shard
        self.config.validate_shard(shard)

    def _selected_names(self):
        return {name for name in super()._selected_names() if name.startswith("language_model.")}

    def load_layer(self, index):
        c = self.config.text
        prefix = f"{PREFIX}.layers.{index}"
        mixer = prefix + ".mixer"
        result = dict(norm=self.load_tensor(prefix + ".norm.weight"))
        kind = self.config.layers[index]
        if kind == "mamba":
            a = self.load_tensor(mixer + ".A_log", dtype=torch.float32, temporary=True)
            d, bias, convolution, conv_bias, norm = self.load_tensors(tuple(mixer + "." + suffix
                for suffix in ("D", "dt_bias", "conv1d.weight", "conv1d.bias", "norm.weight")))
            heads, channels = c["mamba_num_heads"], self.config.convolution_width
            if (a.shape != (heads,) or d.shape != (heads,) or bias.shape != (heads,)
                    or convolution.shape != (channels, 1, c["conv_kernel"])
                    or conv_bias.shape != (channels,) or norm.shape != (self.config.mamba_width,)):
                raise ValueError(f"{mixer}: Mamba parameter geometry differs from configuration")
            result.update(input=self.load_linear(mixer + ".in_proj"),
                output=self.load_linear(mixer + ".out_proj"), A=persistent_copy(-a.exp()), D=d,
                dt_bias=bias, convolution=convolution.squeeze(1), convolution_bias=conv_bias, output_norm=norm)
        elif kind == "attention":
            result.update(qkv=self.load_linear_group(tuple(mixer + "." + name
                          for name in ("q_proj", "k_proj", "v_proj"))),
                          output=self.load_linear(mixer + ".o_proj"))
        else:
            block = c["block_configs"][index]
            router, correction = self.load_tensors((mixer + ".gate.weight", mixer + ".gate.e_score_correction_bias"))
            count = block["n_routed_experts"]
            if router.shape != (count, self.config.hidden) or correction.shape != (count,):
                raise ValueError(f"{mixer}: router geometry differs from block configuration")
            result.update(router=router, correction=correction,
                latent_down=self.load_linear(mixer + ".fc1_latent_proj"),
                latent_up=self.load_linear(mixer + ".fc2_latent_proj"),
                shared_up=self.load_linear(mixer + ".shared_experts.up_proj"),
                shared_down=self.load_linear(mixer + ".shared_experts.down_proj"),
                experts=self.load_block_quant_experts(
                    tuple(f"{mixer}.experts.{i}" for i in range(count)),
                    hidden_size=c["moe_latent_size"], intermediate_size=block["moe_intermediate_size"]))
        return result
