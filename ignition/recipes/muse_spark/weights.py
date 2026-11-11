# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local direct transfers of stacked Muse Spark checkpoint weights."""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import cached_property

import torch

from ignition.components.weights import DirectWeightLoader
from .config import ModelConfig

PREFIX = "model.language_model"


@dataclass(frozen=True)
class BF16Experts:
    gate_up: torch.Tensor
    down: torch.Tensor


@dataclass(frozen=True)
class LayerWeights:
    qkv: torch.Tensor
    output: torch.Tensor
    input_norm: torch.Tensor
    pre_feedforward_norm: torch.Tensor
    post_feedforward_norm: torch.Tensor
    attention_gate: torch.Tensor
    feedforward_gate: torch.Tensor
    router: torch.Tensor
    correction: torch.Tensor
    pre_expert: torch.Tensor
    pre_expert_norm: torch.Tensor
    experts: object
    post_expert_norm: torch.Tensor
    post_expert: torch.Tensor


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
    def load(cls, manifest, device, *, shard, progress=None, core_storage=None, draft_snapshot=None):
        before = torch.cuda.memory_allocated(device)
        loading = nullcontext if core_storage is None else core_storage.transient
        with loading(), MuseSparkWeightLoader(manifest, device, shard=shard, progress=progress) as loader:
            config = loader.config
            if draft_snapshot is not None:
                from .draft_config import DraftConfig
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
            from .draft_weights import DraftWeights
            with loading():
                draft = DraftWeights.load(draft_snapshot, config.draft, device, shard, progress=progress)
        return cls(config, shard, embedding, norm, head, layers, torch.cuda.memory_allocated(device) - before, draft=draft)


class MuseSparkWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None):
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config, self.shard = ModelConfig.load(manifest.snapshot), shard
        self.config.validate_shard(shard)

    @cached_property
    def quantization(self):
        return {}

    def _selected_names(self):
        return {n for n in super()._selected_names() if n.startswith(PREFIX + ".") or n == "lm_head.weight"}

    def load_layer(self, layer):
        c, shard = self.config.text, self.shard
        prefix = f"{PREFIX}.layers.{layer}"
        a = self.config.attention(layer)
        q0, q1 = shard.partition(a.heads * a.key_dim, alignment=a.key_dim)
        k0, k1 = shard.replicated_partition(a.kv_heads)
        kv = (k0 * a.key_dim, k1 * a.key_dim)
        attn = prefix + ".self_attn"
        qkv = self.load_linear_group(tuple(attn + "." + name + "_proj" for name in ("q", "k", "v", "gate")),
                                     row_ranges=((q0, q1), kv, kv, (q0, q1)))
        mlp = prefix + ".mlp"
        def tensor(suffix, *, fp32=False):
            value = self.load_tensor(prefix + "." + suffix)
            return value.float() if fp32 else value
        return LayerWeights(qkv, self.load_linear(attn + ".o_proj", columns=(q0, q1)),
            tensor("input_layernorm.weight"), tensor("pre_feedforward_layernorm.weight"),
            tensor("post_feedforward_layernorm.weight"), tensor("post_attention_residual_gate.gate", fp32=True),
            tensor("post_feedforward_residual_gate.gate", fp32=True), tensor("mlp.gate.weight"),
            tensor("mlp.gate.e_score_correction_bias"),
            self.load_linear(mlp + ".pre_expert_proj", rows=shard.partition(c["moe_hidden_size"], alignment=64)),
            tensor("mlp.pre_expert_norm.weight"), self.load_experts(layer),
            tensor("mlp.experts.post_expert_norm.weight"),
            self.load_linear(mlp + ".post_expert_proj", rows=shard.partition(c["hidden_size"], alignment=64)))

    def load_experts(self, layer, *, expert_ids=None):
        c = self.config.text
        prefix = f"{PREFIX}.layers.{layer}.mlp.experts"
        count, hidden, inter = c["num_local_experts"], c["moe_hidden_size"], c["moe_intermediate_size"]
        ids = tuple(range(count)) if expert_ids is None else tuple(expert_ids)
        if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < count for i in ids):
            raise ValueError("expert IDs must be distinct valid checkpoint indices")
        first, last = self.shard.partition(inter, alignment=64)
        local = last - first
        gate_up, down = self._sources[prefix + ".gate_up_proj"], self._sources[prefix + ".down_proj"]
        fmt = self.config.expert_format(layer)
        if fmt == "bf16":
            if (gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16
                    or gate_up.shape != (count, 2 * inter, hidden) or down.shape != (count, hidden, inter)):
                raise ValueError(f"{prefix}: declared BF16 experts disagree with checkpoint storage")
            gate_target = torch.empty((len(ids), local * 2, hidden), dtype=torch.bfloat16, device=self.device)
            down_target = torch.empty((len(ids), hidden, local), dtype=torch.bfloat16, device=self.device)
            for target, expert in enumerate(ids):
                self._copy(gate_target[target, :local], gate_up[expert, first:last])
                self._copy(gate_target[target, local:], gate_up[expert, inter+first:inter+last])
                self._copy(down_target[target], down[expert, :, first:last])
            self._session.flush()
            return BF16Experts(gate_target, down_target)
        if (gate_up.dtype != torch.uint8 or down.dtype != torch.uint8
                or gate_up.shape != (count, 2 * inter, hidden // 2) or down.shape != (count, hidden, inter // 2)):
            raise ValueError(f"{prefix}: declared NVFP4 experts disagree with checkpoint storage")
        aliases = {}
        projections = ("gate_up_proj", "down_proj")
        vectors = {p: {suffix: self._sources[f"{prefix}.{p}_{suffix}"]
                      for suffix in ("input_scale", "weight_scale_2")} for p in projections}
        for p in projections:
            for suffix, value in vectors[p].items():
                shape = (count, 2) if p == "gate_up_proj" else (count,)
                if (value.shape != shape or value.dtype != torch.float32
                        or not bool((torch.isfinite(value) & (value > 0)).all())):
                    raise ValueError(f"{prefix}.{p}_{suffix}: expected positive finite FP32 calibration")
        gains = vectors["gate_up_proj"]["weight_scale_2"]
        activation = vectors["gate_up_proj"]["input_scale"]
        if not torch.equal(activation[:, 0], activation[:, 1]):
            raise ValueError(f"{prefix}: fused gate/up requires equal calibrated activation scales")
        prefixes = tuple(f"{prefix}.{i}" for i in ids)
        for target, expert in zip(prefixes, ids, strict=True):
            for name, plane, section in (("gate_proj", "gate_up_proj", slice(0, inter)),
                                         ("up_proj", "gate_up_proj", slice(inter, inter*2)),
                                         ("down_proj", "down_proj", slice(None))):
                for suffix in ("weight", "weight_scale", "weight_scale_2"):
                    source = self._sources[f"{prefix}.{plane}" + ("" if suffix == "weight" else "_" + suffix)]
                    aliases[f"{target}.{name}.{suffix}"] = (source[expert, section] if suffix != "weight_scale_2"
                        else source[expert, 0 if name == "gate_proj" else 1] if plane == "gate_up_proj" else source[expert])
        self._sources.update(aliases)
        try:
            selected = torch.tensor(ids, dtype=torch.long)
            # Physical w13 halves are up then gate; calibration follows the same order.
            return self.load_fp4_experts(prefixes, hidden_size=hidden, intermediate_size=inter,
                intermediate_range=(first, last), format="nvfp4", activation_mode="a4",
                scale_vectors=(gains[selected][:, [1, 0]].contiguous(),
                    vectors["down_proj"]["weight_scale_2"][selected].contiguous(),
                    activation[selected, 0].reciprocal(), vectors["down_proj"]["input_scale"][selected].reciprocal()))
        finally:
            for name in aliases:
                del self._sources[name]
