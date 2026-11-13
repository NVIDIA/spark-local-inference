# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiMo checkpoint transfers preserving MXFP4 experts and FP32 block scales."""

from dataclasses import dataclass

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights, _aligned_range
from ignition.components.block_fp8 import BlockFP8Weight

from .config import ModelConfig


@dataclass(frozen=True)
class LayerWeights:
    qkv: BlockFP8Weight
    output: torch.Tensor | BlockFP8Weight
    sink: torch.Tensor | None
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights | None
    router: torch.Tensor | None
    correction: torch.Tensor | None
    experts: object | None


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

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, dflash=True):
        before = torch.cuda.memory_allocated(device)
        with MiMoWeightLoader(manifest, device, shard=shard, progress=progress) as loader:
            config = loader.config
            embedding, norm = loader.load_tensors(("model.embed_tokens.weight", "model.norm.weight"))
            head = loader.load_tensor("lm_head.weight",
                extent=slice(*shard.partition(config.text["vocab_size"], alignment=64)))
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
        draft = None
        if dflash:
            from .draft_weights import DraftWeights

            draft = DraftWeights.load(manifest.snapshot / "dflash", config.draft, device, shard, progress=progress)
        from .vision_weights import VisionWeights

        vision = VisionWeights.load(manifest, config.text, device, shard, progress=progress)
        return cls(config, shard, embedding, norm, head, layers,
                   torch.cuda.memory_allocated(device) - before, draft, vision)


class MiMoWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None):
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config, self.shard = ModelConfig.load(manifest.snapshot), shard
        self.config.qkv_chunks(0, shard)

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if name.startswith("model.layers.")
                or name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")}

    def load_linear_group(self, prefixes, *, row_ranges=None, column_range=None):
        prefixes = tuple(prefixes)
        formats = {self.config.projection_format(prefix) for prefix in prefixes}
        if not prefixes or len(formats) != 1 or formats == {"mxfp4"}:
            raise ValueError("dense projection fusion requires matching BF16 or block-FP8 formats")
        ranges = (None,) * len(prefixes) if row_ranges is None else row_ranges
        values, scales = [], []
        for prefix, rows in zip(prefixes, ranges, strict=True):
            source = self._sources[prefix + ".weight"]
            quantized = formats == {"block_fp8"}
            if source.ndim != 2 or source.dtype != (torch.float8_e4m3fn if quantized else torch.bfloat16):
                raise ValueError(f"{prefix}: weight dtype disagrees with quantization metadata")
            n, k = source.shape
            r0, r1 = _aligned_range(rows, n, 128 if quantized else 1)
            c0, c1 = _aligned_range(column_range, k, 128 if quantized else 1)
            value = torch.empty((r1 - r0, c1 - c0), dtype=source.dtype, device=self.device)
            self._copy(value, source[r0:r1, c0:c1])
            values.append(value)
            if quantized:
                scale = self._sources[prefix + ".weight_scale_inv"]
                if scale.dtype != torch.float32 or scale.shape != ((n + 127) // 128, (k + 127) // 128):
                    raise ValueError(f"{prefix}: invalid FP32 block scale geometry")
                target = torch.empty(((r1 - r0) // 128, (c1 - c0) // 128),
                                     dtype=torch.float32, device=self.device)
                self._copy(target, scale[r0 // 128:r1 // 128, c0 // 128:c1 // 128])
                scales.append(target)
        self._session.flush()
        weight = torch.cat(values) if len(values) > 1 else values[0]
        return weight if not scales else BlockFP8Weight(weight, torch.cat(scales) if len(scales) > 1 else scales[0])

    def load_qkv(self, layer):
        prefix = f"model.layers.{layer}.self_attn.qkv_proj"
        weight, scale = (self._sources[prefix + suffix] for suffix in (".weight", ".weight_scale_inv"))
        chunks = self.config.qkv_chunks(layer, self.shard)
        first = self.config.qkv_chunks(layer, type(self.shard)(0, 1))
        hidden = self.config.text["hidden_size"]
        if (self.config.projection_format(prefix) != "block_fp8"
                or weight.dtype != torch.float8_e4m3fn or weight.shape != (first[-1].weight_rows[1], hidden)
                or scale.dtype != torch.float32 or scale.shape != (first[-1].scale_rows[1], hidden // 128)):
            raise ValueError(f"{prefix}: chunked QKV weights or padded scale rows differ from checkpoint geometry")
        rows = sum(c.padded_rows for c in chunks)
        values = torch.zeros((rows, hidden), dtype=weight.dtype, device=self.device)
        scales = torch.empty((rows // 128, hidden // 128), dtype=torch.float32, device=self.device)
        offset = 0
        for chunk in chunks:
            start, end = chunk.weight_rows
            self._copy(values[offset:offset + end - start], weight[start:end])
            self._copy(scales[offset // 128:(offset + chunk.padded_rows) // 128],
                       scale[slice(*chunk.scale_rows)])
            offset += chunk.padded_rows
        self._session.flush()
        return BlockFP8Weight(values, scales)

    def load_layer(self, layer):
        prefix = f"model.layers.{layer}"
        c, shard = self.config.text, self.shard
        a = self.config.attention(layer)
        qkv = self.load_qkv(layer)
        output = self.load_linear(prefix + ".self_attn.o_proj",
                                  columns=shard.partition(a.heads * a.value_dim, alignment=a.value_dim))
        sink = self.load_tensor(prefix + ".self_attn.attention_sink_bias", dtype=torch.float32,
                               extent=slice(*shard.partition(a.heads))) if a.sinks else None
        input_norm, post_norm = self.load_tensors((prefix + ".input_layernorm.weight",
                                                 prefix + ".post_attention_layernorm.weight"))
        mlp = router = correction = experts = None
        if c["moe_layer_freq"][layer]:
            experts = self.load_fp4_experts(
                [f"{prefix}.mlp.experts.{i}" for i in range(c["n_routed_experts"])],
                hidden_size=c["hidden_size"], intermediate_size=c["moe_intermediate_size"],
                format="mxfp4", activation_mode="a8",
                intermediate_range=shard.partition(c["moe_intermediate_size"], alignment=32))
            router, correction = self.load_tensors((prefix + ".mlp.gate.weight",
                                                    prefix + ".mlp.gate.e_score_correction_bias"))
        else:
            interval = shard.partition(c["intermediate_size"], alignment=128)
            mlp = MLPWeights(self.load_linear_group((prefix + ".mlp.gate_proj", prefix + ".mlp.up_proj"),
                                                    row_ranges=(interval, interval)),
                             self.load_linear(prefix + ".mlp.down_proj", columns=interval))
        return LayerWeights(qkv, output, sink, input_norm, post_norm, mlp, router, correction, experts)
