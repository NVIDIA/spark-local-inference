# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct Step-5 transfers, packed projections, and load-time weight conversion."""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property

import torch

from ignition.components.weights import (
    DirectWeightLoader,
    MLPWeights,
    _aligned_range,
    quantize_bf16_linear_as_mxfp8,
)

from .config import ModelConfig


@dataclass(frozen=True)
class LayerWeights:
    qkvg: torch.Tensor
    output: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights
    router: torch.Tensor | None
    correction: torch.Tensor | None
    experts: object | None
    indexer: dict | None


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
    scale_scratch: tuple[torch.Tensor, ...] = ()

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, mtp=True,
             expert_scale_compression=None, core_storage=None):
        before = torch.cuda.memory_allocated(device)
        loading = nullcontext if core_storage is None else core_storage.transient
        # Reuse conversion buffers across layers before retiring the loading pool.
        with loading(), Step5WeightLoader(manifest, device, shard=shard, progress=progress, mtp=mtp,
                                         expert_scale_compression=expert_scale_compression) as loader:
            config = loader.config
            scale_scratch = loader.scale_scratch
            first, last = shard.partition(config.hidden)
            embedding = loader.load_tensor("model.embed_tokens.weight", extent=(slice(None), slice(first, last)))
            norm = loader.norm("model.norm.weight", config.hidden)
            head = loader.head("lm_head.weight")
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
            draft = tuple(loader.load_predictor(i) for i in range(config.text["num_nextn_predict_layers"])) if mtp else None
            from .vision_weights import VisionWeights
            vision = VisionWeights.load(loader)
        return cls(config, shard, embedding, norm, head, layers, torch.cuda.memory_allocated(device) - before,
                   draft, vision, scale_scratch)


class Step5WeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None, mtp=True,
                 expert_scale_compression=None):
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config, self.shard = ModelConfig.load(manifest.snapshot), shard
        self.mtp = mtp
        if expert_scale_compression not in (None, "csf"):
            raise ValueError("Step-5 expert scale compression must be None or 'csf'")
        self.expert_scale_compression = expert_scale_compression
        self.config.validate_shard(shard)

    @cached_property
    def scale_scratch(self):
        if self.expert_scale_compression is None:
            return ()
        c = self.config.text
        first, last = self.shard.partition(c["moe_intermediate_size"], alignment=32)
        padded = (last - first + 127) // 128 * 128
        # Routed layers run serially; A4 and padded A16 layouts share these allocations.
        return tuple(torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device)
            for shape in ((c["moe_num_experts"], 2 * padded, c["hidden_size"] // 16),
                          (c["moe_num_experts"], c["hidden_size"], padded // 16)))

    def expert_scale_scratch(self, intermediate):
        if not self.scale_scratch:
            return None
        c = self.config.text
        shapes = ((c["moe_num_experts"], (2 * intermediate + 127) // 128 * 128, c["hidden_size"] // 16),
                  (c["moe_num_experts"], c["hidden_size"], (intermediate // 16 + 3) // 4 * 4))
        return tuple(buffer.flatten()[:count * rows * columns].view(count, rows, columns)
                     for buffer, (count, rows, columns) in zip(self.scale_scratch, shapes, strict=True))

    def _selected_names(self):
        def selected(name):
            if name.startswith(("vision_model.", "vit_large_projector.")):
                return True
            if name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
                return True
            parts = name.split(".")
            total = self.config.layers + (self.config.text["num_nextn_predict_layers"] if self.mtp else 0)
            return (parts[:2] == ["model", "layers"] and int(parts[2]) < total
                    and not name.endswith(".ssmax_s"))
        return {name for name in super()._selected_names() if selected(name)}

    @cached_property
    def quantization(self):
        return {name.removesuffix(".weight"): {"quant_algo": "NVFP4", "group_size": 16}
                for name in self._sources if name.endswith(".weight")
                and self.config.projection_format(name.removesuffix(".weight")) == "nvfp4"}

    def norm(self, name, width):
        source = self._sources[name]
        if source.shape != (width,) or source.dtype != torch.float32:
            raise ValueError(f"{name}: expected a {width}-element FP32 normalization parameter")
        return self.load_tensor(name)

    def head(self, name):
        source = self._sources[name]
        if source.dtype != torch.bfloat16 or source.shape != (self.config.vocabulary, self.config.hidden):
            raise ValueError(f"{name}: expected the checkpoint's BF16 vocabulary projection")
        first, last, width = self.config.vocabulary_partition(self.shard)
        output = torch.zeros((width, self.config.hidden), dtype=torch.bfloat16, device=self.device)
        self._copy(output[:last-first], source[first:last])
        self._session.flush()
        return output

    def load_predictor(self, index):
        layer = self.config.layers + index
        prefix = f"model.layers.{layer}"
        return {"layer": self.load_layer(layer),
            "enorm": self.norm(prefix + ".enorm.weight", self.config.hidden),
            "hnorm": self.norm(prefix + ".hnorm.weight", self.config.hidden),
            "projection": self.load_linear(prefix + ".eh_proj"),
            "norm": self.norm(prefix + ".transformer.shared_head.norm.weight", self.config.hidden),
            "head": self.head(prefix + ".transformer.shared_head.output.weight")}

    def load_linear_group(self, prefixes, *, row_ranges=None, column_range=None):
        prefixes = tuple(prefixes)
        formats = {self.config.projection_format(p) for p in prefixes}
        if formats != {"bf16"}:
            return super().load_linear_group(prefixes, row_ranges=row_ranges, column_range=column_range)
        ranges = (None,) * len(prefixes) if row_ranges is None else row_ranges
        views = []
        for prefix, rows in zip(prefixes, ranges, strict=True):
            source = self._sources[prefix + ".weight"]
            # The indexer's head-weight GEMM explicitly rounds its FP32 parameter to BF16.
            expected = torch.float32 if prefix.endswith(".sparse_indexer_w") else torch.bfloat16
            if source.ndim != 2 or source.dtype != expected:
                raise ValueError(f"{prefix}: projection storage disagrees with checkpoint metadata")
            r0, r1 = _aligned_range(rows, source.shape[0], 1)
            c0, c1 = _aligned_range(column_range, source.shape[1], 1)
            views.append(source[r0:r1, c0:c1])
        output = torch.empty((sum(v.shape[0] for v in views), views[0].shape[1]),
                             dtype=torch.bfloat16, device=self.device)
        offset = 0
        for source in views:
            self._copy(output[offset:offset + source.shape[0]], source)
            offset += source.shape[0]
        self._session.flush()
        return output

    def mlp(self, prefix, intermediate):
        interval = self.shard.partition(intermediate, alignment=32)
        gate_up = self.load_linear_group((prefix + ".gate_proj", prefix + ".up_proj"),
                                        row_ranges=(interval, interval))
        if isinstance(gate_up, torch.Tensor):
            gate_up = quantize_bf16_linear_as_mxfp8(gate_up)
        return MLPWeights(gate_up,
                          self.load_linear(prefix + ".down_proj", columns=interval))

    def load_layer(self, layer):
        c, shard = self.config.text, self.shard
        prefix = f"model.layers.{layer}"
        a = self.config.attention(layer)
        q0, q1 = shard.partition(a.heads)
        k0, k1 = shard.replicated_partition(a.kv_heads)
        attention = prefix + ".self_attn"
        rows = ((q0 * a.dimension, q1 * a.dimension), (k0 * a.dimension, k1 * a.dimension),
                (k0 * a.dimension, k1 * a.dimension), (q0, q1))
        projections = tuple(attention + "." + p + "_proj" for p in ("q", "k", "v", "g"))
        projection_rows = rows
        output = self.load_linear(attention + ".o_proj", columns=rows[0])
        indexer = None
        if a.window < 0:
            sparse = self.config.sparse
            dim = sparse["proxy_dim"]
            per_group = sparse["sparse_indexer_num_heads"] // sparse["num_provider_groups"]
            h0, h1 = k0 * per_group, k1 * per_group
            projections += tuple(attention + ".sparse_indexer_" + p for p in ("q", "k", "z", "w"))
            projection_rows += ((h0 * dim, h1 * dim), None, None, (h0, h1))
            indexer = {"q_norm": self.norm(attention + ".sparse_indexer_q_norm.weight", dim).bfloat16(),
                "k_norm": self.norm(attention + ".sparse_indexer_k_norm.weight", dim).bfloat16(),
                "k_bias": self.norm(attention + ".sparse_indexer_k_norm.bias", dim).bfloat16()}
        qkvg = self.load_linear_group(projections, row_ranges=projection_rows)
        routed = layer in c["moe_layer_list"]
        mlp = self.mlp(prefix + (".share_expert" if routed else ".mlp"),
                       c["share_expert_dim"] if routed else c["intermediate_size"])
        router = correction = experts = None
        if routed:
            router = self.load_linear(prefix + ".moe.gate")
            correction = self.norm(prefix + ".moe.router_bias", c["moe_num_experts"])
            experts = self.load_experts(prefix + ".moe", c["swiglu_limits"][layer])
        return LayerWeights(qkvg, output, self.norm(attention + ".q_norm.weight", a.dimension),
            self.norm(attention + ".k_norm.weight", a.dimension), self.norm(prefix + ".input_layernorm.weight", c["hidden_size"]),
            self.norm(prefix + ".post_attention_layernorm.weight", c["hidden_size"]),
            mlp, router, correction, experts, indexer)

    def load_experts(self, prefix, limit):
        c = self.config.text
        count = c["moe_num_experts"]
        interval = self.shard.partition(c["moe_intermediate_size"], alignment=32)
        local = interval[1] - interval[0]
        if self.config.expert_format(prefix) == "bf16":
            return self.load_bf16_experts_as_nvfp4(prefix, num_experts=count,
                hidden_size=c["hidden_size"], intermediate_size=c["moe_intermediate_size"],
                intermediate_range=interval, swiglu_limit=limit, swiglu_clamp_after_silu=True,
                scale_compression=self.expert_scale_compression,
                scale_scratch=self.expert_scale_scratch((local + 127) // 128 * 128))
        projections = ("up_proj", "gate_proj", "down_proj")
        vectors = {}
        for p in projections:
            for suffix in ("input_scale", "weight_scale_2"):
                value = self._sources[f"{prefix}.{p}.{suffix}"]
                if (value.dtype != torch.float32 or value.shape != (count,)
                        or not bool((torch.isfinite(value) & (value > 0)).all())):
                    raise ValueError(f"{prefix}.{p}.{suffix}: expected positive finite FP32 expert scales")
                vectors[p, suffix] = value
        for suffix in ("input_scale", "weight_scale_2"):
            if not torch.equal(vectors["up_proj", suffix], vectors["gate_proj", suffix]):
                raise ValueError(f"{prefix}: gate/up {suffix} differ; NVFP4 fusion would change calibration")
        aliases = []
        prefixes = tuple(f"{prefix}.experts.{i}" for i in range(count))
        try:
            for p in projections:
                for suffix in ("weight", "weight_scale", "weight_scale_2"):
                    value = self._sources[f"{prefix}.{p}.{suffix}"]
                    if value.shape[0] != count:
                        raise ValueError(f"{prefix}.{p}.{suffix}: wrong expert count")
                    for i, target in enumerate(prefixes):
                        name = f"{target}.{p}.{suffix}"
                        self._sources[name] = value[i]
                        aliases.append(name)
            return self.load_fp4_experts(prefixes, hidden_size=c["hidden_size"],
                intermediate_size=c["moe_intermediate_size"],
                intermediate_range=interval,
                format="nvfp4", activation_mode="a4", swiglu_limit=limit, swiglu_clamp_after_silu=True,
                scale_compression=self.expert_scale_compression, scale_scratch=self.expert_scale_scratch(local),
                scale_vectors=(vectors["up_proj", "weight_scale_2"], vectors["down_proj", "weight_scale_2"],
                               vectors["up_proj", "input_scale"].reciprocal(), vectors["down_proj", "input_scale"].reciprocal()))
        finally:
            for name in aliases:
                del self._sources[name]
