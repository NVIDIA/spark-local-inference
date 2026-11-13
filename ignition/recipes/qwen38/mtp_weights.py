# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Stored BF16 and NVFP4 W4A16 weights for the checkpoint's single MTP layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json

import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.vocabulary import nvfp4_draft_head
from ignition.preparation.checkpoint import CheckpointError, CheckpointManifest
from ignition.preparation.qwen38 import TRANSFORMERS_REVISION
from ignition.preparation.persistent import persistent_cat, allocated_tensor_bytes

from .moe import QwenMoeWeights, SharedExpertWeights
from .hyperconnection import packed_down_projection, hyperconnection_extents
from .weights import QwenWeightLoader, _expert_ids, _expert_scale_vectors


def _global_scales(sources, expert_ids):
    w13, w2 = [], []
    for expert in expert_ids:
        prefix = f"mtp.layers.0.mlp.experts.{expert}"
        values = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{prefix}.{projection}.weight_scale_2"
            value = sources[name]
            if value.device.type != "cpu" or value.dtype != torch.float32 or value.shape != ():
                raise ValueError(f"{name} requires a CPU float32 scalar")
            if not bool(torch.isfinite(value) & (value > 0)):
                raise ValueError(f"{name} must be finite and positive")
            values[projection] = value.item()
        if values["gate_proj"] != values["up_proj"]:
            raise ValueError(f"{prefix}: gate/up global scales differ")
        w13.append(values["up_proj"])
        w2.append(values["down_proj"])
    return torch.tensor(w13, dtype=torch.float32), torch.tensor(w2, dtype=torch.float32)


class MTPWeightLoader(QwenWeightLoader):
    """Reuse b12x direct transfers while selecting only stored ``mtp.*`` tensors."""

    def _selected_names(self):
        return {name for name, tensor in self.manifest.tensors.items()
                if tensor.component == "mtp"
                and (self.prefixes is None or name.startswith(self.prefixes))}

    def load_experts(self, *, expert_ids=None):
        """Prepare ordinary FP4 experts using their checkpoint quantization.

        Explicit expert subsets are component-test inputs. The executable MTP
        layer requires all 512 experts in checkpoint order.
        """
        from ignition.preparation.quantization import expert_format

        self._require_open()
        ids = _expert_ids(0, expert_ids)
        prefix = "mtp.layers.0.mlp.experts"
        fmt, mode = expert_format(self.quantization, prefix)
        if fmt == "nvfp4_csf":
            raise ValueError("Qwen MTP experts require ordinary NVFP4 or MXFP4 storage")
        return self.load_fp4_experts(tuple(f"{prefix}.{expert}" for expert in ids),
            hidden_size=2560, intermediate_size=640, format=fmt, activation_mode=mode,
            intermediate_range=self.shard.partition(640, alignment=32 if fmt == "mxfp4" else 16),
            scale_vectors=_expert_scale_vectors(self._sources, 0, ids, prefix=prefix) if mode == "a4" else None)

    def load_shared(self):
        prefix = "mtp.layers.0.mlp"
        first, last = self.shard.partition(640, alignment=16)
        gate_up = persistent_cat(self.load_tensors(tuple(f"{prefix}.shared_expert.{p}.weight"
            for p in ("gate_proj", "up_proj")), extents=(slice(first, last),) * 2, temporary=True))
        return SharedExpertWeights(gate_up,
            self.load_tensor(f"{prefix}.shared_expert.down_proj.weight", extent=(slice(None), slice(first, last))),
            self.load_tensor(f"{prefix}.shared_expert_gate.weight"))


def create_mtp_modules(manifest: CheckpointManifest):
    """Create canonical HC/QSA parameter owners without a duplicate target model."""
    distribution = importlib.metadata.distribution("transformers")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_REVISION:
        raise RuntimeError(f"MTP preparation requires Transformers {TRANSFORMERS_REVISION}")
    raw = (manifest.snapshot / "config.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.metadata_sha256["config.json"]:
        raise CheckpointError("config.json changed after checkpoint inspection")
    from transformers import Qwen4ExpConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextAttention, Qwen4ExpTextGatedResidual, Qwen4ExpTextRMSNorm,
    )

    config = Qwen4ExpConfig.from_dict(json.loads(raw)).text_config
    if config.mtp_num_hidden_layers != 1 or config.hc_count != 4 or config.hidden_size != 2560:
        raise ValueError("MTP requires one layer and four 2560-wide streams")
    with torch.device("meta"):
        module = torch.nn.Module()
        module.config = config
        module.fc_embedding = torch.nn.Linear(2560, 2560, bias=False)
        module.fc_hidden = torch.nn.Linear(2560, 2560, bias=False)
        module.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(2560, eps=config.rms_norm_eps)
        module.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(10240, eps=config.rms_norm_eps)
        module.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
        layer = torch.nn.Module()
        layer.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
        layer.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config)
        layer.self_attn = Qwen4ExpTextAttention(config, layer_idx=48)
        module.layers = torch.nn.ModuleList([layer])
    module.eval().requires_grad_(False)
    return module


@dataclass(frozen=True)
class MTPWeights:
    module: torch.nn.Module
    attention: dict[str, torch.Tensor]
    moe: QwenMoeWeights
    embedding: torch.Tensor
    lm_head: torch.Tensor
    stored_bytes: int
    allocated_bytes: int
    shard: TensorParallelShard = TensorParallelShard(0, 1)
    draft_head: object | None = None

    @classmethod
    def load(cls, manifest: CheckpointManifest, device, *, embedding_weight, lm_head_weight, progress=None,
             shard: TensorParallelShard = TensorParallelShard(0, 1),
             shard_hyperconnections: bool = False):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("MTP weights require a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        for label, tensor in (("embedding", embedding_weight), ("lm_head", lm_head_weight)):
            rows = 248320 if label == "embedding" else 248320 // shard.world_size
            if (tensor.shape != (rows, 2560) or tensor.dtype != torch.bfloat16
                    or tensor.device != device or not tensor.is_contiguous()):
                raise ValueError(f"shared MTP {label} must be contiguous BF16 [{rows},2560] on {device}")
        stored = sum(t.nbytes for t in manifest.tensors.values() if t.component == "mtp")
        module = create_mtp_modules(manifest)
        before = allocated_tensor_bytes(device)
        with MTPWeightLoader(manifest, device, progress=progress, progress_label="Load MTP weights", shard=shard) as loader:
            names = tuple(name for name, _ in module.named_parameters())
            kv = tuple(head * 256 for head in shard.replicated_partition(2))
            extents = {
                "layers.0.self_attn.q_proj.weight": slice(*shard.partition(12288)),
                "layers.0.self_attn.k_proj.weight": slice(*kv),
                "layers.0.self_attn.v_proj.weight": slice(*kv),
                "layers.0.self_attn.o_proj.weight": (slice(None), slice(*shard.partition(6144))),
            }
            for prefix in ("layers.0.attn_hyper_connection", "layers.0.mlp_hyper_connection", "hyper_connection_mixer"):
                extents.update({f"{prefix}.{name}": extent
                                for name, extent in hyperconnection_extents(shard, shard_hyperconnections).items()})
            temporary_names = {f"layers.0.self_attn.{name}.weight" for name in ("q_proj", "k_proj", "v_proj")}
            temporary_names.update(f"layers.0.{mixer}.{name}.weight"
                for mixer in ("attn_hyper_connection", "mlp_hyper_connection")
                for name in ("input_mix_weight_down", "block_inject_weight"))
            for temporary in (False, True):
                selected = tuple(name for name in names if (name in temporary_names) == temporary)
                if not selected:
                    continue
                values = loader.load_tensors(tuple(f"mtp.{name}" for name in selected),
                    extents=tuple(extents.get(name) for name in selected), temporary=temporary)
                for name, value in zip(selected, values, strict=True):
                    parent, _, attribute = name.rpartition(".")
                    setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))
                del values
            for mixer in (module.layers[0].attn_hyper_connection, module.layers[0].mlp_hyper_connection):
                packed_down_projection(mixer)
            shared = loader.load_shared()
            moe = QwenMoeWeights(loader.load_tensor("mtp.layers.0.mlp.gate.weight"),
                                loader.load_experts(), shared)
        attention_module = module.layers[0].self_attn
        qkv = persistent_cat([attention_module.get_submodule(name).weight.detach()
                         for name in ("q_proj", "k_proj", "v_proj")])
        for name, value in zip(("q_proj", "k_proj", "v_proj"), qkv.split((12288 // shard.world_size, max(1, 2 // shard.world_size) * 256, max(1, 2 // shard.world_size) * 256)), strict=True):
            attention_module.get_submodule(name).weight = torch.nn.Parameter(value, requires_grad=False)
        attention_module.register_buffer("packed_qkv", qkv, persistent=False)
        attention = {"qkv_proj": qkv, **{name: attention_module.get_submodule(name).weight
                     for name in ("o_proj", "indexer.index_qk_proj")}}
        # Drafts read the vocabulary through an NVFP4 copy of the BF16 head: a proposal
        # distribution only has to be the q that generated it; verification keeps BF16.
        draft_head = nvfp4_draft_head(lm_head_weight)
        torch.cuda.synchronize(device)
        return cls(module, attention, moe, embedding_weight, lm_head_weight,
                   stored, allocated_tensor_bytes(device) - before, shard, draft_head)
