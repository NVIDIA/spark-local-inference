# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiLiCorr checkpoint projections with separate NVFP4 gate and up weights."""

from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import cached_property
import json

import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.persistent import persistent_cat, persistent_copy


class DraftWeightLoader(DirectWeightLoader):
    @cached_property
    def quantization(self):
        config = json.loads((self.manifest.metadata_path / "config.json").read_text())
        q = config["quantization_config"]
        if (q.get("quant_method") != "modelopt_fp4" or q.get("quant_algo") != "W4A16_NVFP4"
                or q.get("group_size") != 16 or q.get("kv_cache_quant_algo") is not None):
            raise ValueError("LiLiCorr requires W4A16 NVFP4 with 16-value scales")
        excluded = q["exclude_modules"]
        if q.get("ignore", excluded) != excluded:
            raise ValueError("LiLiCorr quantization exclusion declarations disagree")
        return {name.removesuffix(".weight"): dict(quant_algo=q["quant_algo"], group_size=q["group_size"])
                for name in self.manifest.tensors if name.endswith(".weight")
                and not any(fnmatchcase(name.removesuffix(".weight"), pattern) for pattern in excluded)}

    def projection(self, prefix, *, temporary=False):
        if prefix in self.quantization:
            return self.load_linear(prefix, temporary=temporary)
        return self.tensor(prefix + ".weight", temporary=temporary)

    def tensor(self, name, *, temporary=False):
        value = self.load_tensor(name, temporary=temporary)
        if value.dtype != torch.bfloat16:
            raise ValueError(f"LiLiCorr requires BF16 unquantized parameters: {name}")
        return value

    def biased(self, prefix, *, temporary=False):
        return self.projection(prefix, temporary=temporary), self.tensor(prefix + ".bias", temporary=temporary)

    def lattice(self, config):
        prefix, c = "lilicorr.", config.text["dflash_config"]
        weights = {name: self.tensor(prefix + name) for name in
                   ("slot_embedding", "rank_embedding", "relative_slot_bias", "same_slot_bias")}
        for name in ("token_proj", "pass_hidden_proj", "context_proj", "anchor_out_head"):
            weights[name] = self.biased(prefix + name)
        weights["feature_norm"] = self.tensor(prefix + "feature_norm.weight"), self.tensor(prefix + "feature_norm.bias")
        for name in ("output_norm", "anchor_norm"):
            weights[name] = self.tensor(prefix + name + ".weight")
        weights["feature_mlp"] = tuple(self.biased(prefix + "feature_mlp." + name) for name in ("up_proj", "down_proj"))
        factor = self.projection(prefix + "factor_input_proj", temporary=True)
        bias = self.tensor(prefix + "factor_input_proj.bias")
        hidden = c["lilicorr_hidden_size"]
        if factor.shape != (hidden, 3 * hidden):
            raise ValueError("LiLiCorr factor input must project candidate, anchor, and their product")
        weights["factor"] = tuple(persistent_copy(part.contiguous())
                                   for part in factor.split(hidden, dim=1)), bias
        edges = [self.biased(prefix + name, temporary=True) for name in ("out_head", "in_head")]
        weights["edges"] = tuple(persistent_cat([edge[i] for edge in edges], dim=0) for i in (0, 1))
        layers = []
        for index in range(c["lilicorr_num_layers"]):
            p = f"{prefix}layers.{index}."
            layers.append(dict(attn_norm=self.tensor(p + "attn_norm.weight"),
                qkv=(self.tensor(p + "attn.in_proj_weight"),
                     self.tensor(p + "attn.in_proj_bias")),
                output=self.biased(p + "attn.out_proj"), mlp_norm=self.tensor(p + "mlp_norm.weight"),
                mlp=tuple(self.biased(p + "mlp." + name) for name in ("up_proj", "down_proj"))))
        weights["layers"] = tuple(layers)
        return weights


@dataclass(frozen=True)
class DraftWeights:
    projection: object
    hidden_norm: torch.Tensor
    norm: torch.Tensor
    context_kv: torch.Tensor
    layers: tuple[dict, ...]
    lattice: dict

    @classmethod
    def load(cls, snapshot, config, device, *, progress=None):
        manifest = inspect_indexed_checkpoint(snapshot)
        with DraftWeightLoader(manifest, device, progress=progress, progress_label="Load draft weights") as loader:
            projection = loader.projection("fc")
            hidden_norm, norm = loader.tensor("hidden_norm.weight"), loader.tensor("norm.weight")
            layers = []
            for index in range(config.layers):
                p = f"layers.{index}."
                qkv = loader.load_linear_group(tuple(p + "self_attn." + name
                    for name in ("q_proj", "k_proj", "v_proj")))
                layers.append(dict(qkv=qkv, context=qkv[config.text["num_attention_heads"] * config.text["head_dim"]:],
                    output=loader.projection(p + "self_attn.o_proj"),
                    q_norm=loader.tensor(p + "self_attn.q_norm.weight"), k_norm=loader.tensor(p + "self_attn.k_norm.weight"),
                    sink=persistent_copy(loader.tensor(
                        p + "self_attn.attention_sink_bias", temporary=True).float()),
                    input_norm=loader.tensor(p + "input_layernorm.weight"),
                    post_norm=loader.tensor(p + "post_attention_layernorm.weight"),
                    mlp=tuple(loader.projection(p + "mlp." + name) for name in ("gate_proj", "up_proj", "down_proj")),
                    **{kind: dict(base=loader.tensor(p + kind + ".base_kernel"),
                        projection=loader.projection(p + kind + ".kernel_projection"))
                       for kind in ("attention_conv", "mlp_conv")}))
            lattice = loader.lattice(config)
        context_kv = persistent_cat([layer["context"] for layer in layers])
        return cls(projection, hidden_norm, norm, context_kv, tuple(layers), lattice)
