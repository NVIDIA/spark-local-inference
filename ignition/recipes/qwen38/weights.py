# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned Qwen checkpoint for b12x without dequantizing weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import CheckpointManifest
from ignition.preparation.qwen38 import TEXT_PREFIX

if TYPE_CHECKING:
    import torch
    from b12x.gemm.blockscaled import Weight
    from b12x.moe.fused_moe import PreparedExperts


def _expert_ids(layer: int, expert_ids: Sequence[int] | None) -> tuple[int, ...]:
    if type(layer) is not int or not 0 <= layer < 48:
        raise ValueError("decoder layer must be an integer in [0, 48)")
    ids = tuple(range(512)) if expert_ids is None else tuple(expert_ids)
    if not ids or any(type(i) is not int or not 0 <= i < 512 for i in ids):
        raise ValueError("expert IDs must be a nonempty sequence of integers in [0, 512)")
    if len(set(ids)) != len(ids):
        raise ValueError("expert IDs must be unique")
    return ids


@dataclass(frozen=True, slots=True)
class ExpertMemoryPlan:
    """Exact source sizes and shared scratch; excludes b12x derived metadata."""

    expert_ids: tuple[int, ...]
    packed_bytes: int
    csf_bytes: int
    scalar_bytes: int
    scale_scratch_bytes: int


def expert_memory_plan(
    manifest: CheckpointManifest, layer: int, expert_ids: Sequence[int] | None = None
) -> ExpertMemoryPlan:
    ids = _expert_ids(layer, expert_ids)
    sizes = {"packed": 0, "csf": 0, "scalar": 0}
    for expert in ids:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp.experts.{expert}.{projection}"
            for suffix, kind in (
                ("weight", "packed"), ("weight_scale_2", "scalar"),
                ("input_scale", "scalar"),
                ("weight_scale.nvfp4_csf_fixed", "csf"), ("weight_scale.nvfp4_csf_exceptions", "csf"),
                *((f"weight_scale.csf32_{field}", "csf")
                  for field in ("bases", "codes", "page_bases", "offsets", "exceptions")),
            ):
                tensor = manifest.tensors.get(f"{prefix}.{suffix}")
                if tensor is not None:
                    sizes[kind] += tensor.nbytes
    return ExpertMemoryPlan(ids, sizes["packed"], sizes["csf"], sizes["scalar"], len(ids) * 307200)


def allocate_expert_scale_scratch(
    device: torch.device | str, expert_count: int = 512, *, shard: TensorParallelShard = TensorParallelShard(0, 1)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate scale expansion storage shared by serial decoder layer calls."""
    import torch

    if type(expert_count) is not int or not 1 <= expert_count <= 512:
        raise ValueError("expert_count must be an integer in [1, 512]")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("expert scale scratch requires a CUDA device")
    return (
        torch.empty((expert_count, ((1280 // shard.world_size + 127) // 128) * 128, 160), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((expert_count, 2560, ((40 // shard.world_size + 3) // 4) * 4), dtype=torch.float8_e4m3fn, device=device),
    )


def _expert_scale_vectors(sources: Mapping[str, torch.Tensor], layer: int, ids: tuple[int, ...], *, prefix=None):
    import torch

    w13, w2, a1, a2 = [], [], [], []
    root = prefix or f"{TEXT_PREFIX}.layers.{layer}.mlp.experts"
    for expert in ids:
        prefix = f"{root}.{expert}"
        values = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_scale_2", "input_scale"):
                name = f"{prefix}.{projection}.{suffix}"
                if name not in sources:
                    raise ValueError(f"{name} is required for calibrated NVFP4 activations")
                value = sources[name]
                if value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() != 1:
                    raise ValueError(f"{name} requires a CPU float32 scalar")
                scalar = value.item()
                if not isfinite(scalar) or scalar <= 0:
                    raise ValueError(f"{name} must be finite and positive")
                values[projection, suffix] = scalar
        for suffix in ("weight_scale_2", "input_scale"):
            if values["gate_proj", suffix] != values["up_proj", suffix]:
                raise ValueError(f"{prefix}: gate/up {suffix} differ; fused NVFP4 requires equal scales")
        w13.append(values["up_proj", "weight_scale_2"])
        w2.append(values["down_proj", "weight_scale_2"])
        a1.append(values["up_proj", "input_scale"])
        a2.append(values["down_proj", "input_scale"])
    vectors = (
        torch.tensor(w13, dtype=torch.float32), torch.tensor(w2, dtype=torch.float32),
        torch.tensor(a1, dtype=torch.float32).reciprocal(),
        torch.tensor(a2, dtype=torch.float32).reciprocal(),
    )
    if any(not bool(torch.isfinite(vector).all()) for vector in vectors):
        raise ValueError("expert activation reciprocal scales must remain finite in float32")
    return vectors


class QwenWeightLoader(DirectWeightLoader):
    """Direct checkpoint transfers and explicit, bounded b12x weight preparation.

    ``prefixes`` bounds component qualification; omit it for a complete recipe.
    MTP and PLE table payloads are excluded. Returned tensors own GPU storage
    beyond this context's lifetime. Compressed expert scales use at most one
    layer of GPU staging for partitioning and native-layout preparation.
    """

    def __init__(self, *args, shard=TensorParallelShard(0, 1), force_moe_decode_a16=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shard = shard
        self.force_moe_decode_a16 = force_moe_decode_a16

    def _selected_names(self):
        return {
            name for name, tensor in self.manifest.tensors.items()
            if tensor.component not in ("mtp", "ple_tables")
            and (self.prefixes is None or name.startswith(self.prefixes))
        }

    def load_gdn_projections(self, prefix: str) -> dict[str, Weight]:
        qkv = self.shard.fused_partitions((2048, 2048, 6144))
        value = self.shard.partition(6144)
        heads = self.shard.partition(48)
        return {
            "in_proj_qkvz": self.load_linear_group(
                (f"{prefix}.in_proj_qkv",) * 3 + (f"{prefix}.in_proj_z",),
                row_ranges=(*qkv, value)),
            "in_proj_ba": self.load_linear_group(
                (f"{prefix}.in_proj_b", f"{prefix}.in_proj_a"), row_ranges=(heads, heads)),
            "out_proj": self.load_linear(f"{prefix}.out_proj", columns=value),
        }

    def load_qsa_projections(self, prefix: str) -> dict[str, Weight]:
        kv = tuple(head * 256 for head in self.shard.replicated_partition(2))
        return {
            "qkv_proj": self.load_linear_group(
                tuple(f"{prefix}.{name}" for name in ("q_proj", "k_proj", "v_proj")),
                row_ranges=(self.shard.partition(12288), kv, kv)),
            "o_proj": self.load_linear(f"{prefix}.o_proj", columns=self.shard.partition(6144)),
            "indexer.index_qk_proj": self.load_linear(f"{prefix}.indexer.index_qk_proj"),
        }

    def load_experts(
        self, layer: int, *, expert_ids: Sequence[int] | None = None,
        scale_scratch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> PreparedExperts:
        """Prepare routed experts; subset IDs define local ordering in tests.

        Complete model layers require all 512 experts. Reusing ``scale_scratch``
        across layers requires serialized execution on one CUDA stream.
        """
        self._require_open()
        from ignition.preparation.quantization import expert_format

        ids = _expert_ids(layer, expert_ids)
        prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp.experts"
        fmt, mode = expert_format(self.quantization, prefix)
        activation_options = {}
        if self.force_moe_decode_a16:
            if fmt not in ("nvfp4", "nvfp4_csf"):
                raise ValueError("force_moe_decode_a16 requires calibrated NVFP4 routed experts")
            if (640 // self.shard.world_size) % 64:
                raise ValueError("force_moe_decode_a16 requires rank-local expert intermediate width divisible by 64")
            mode = "a16"
            activation_options["a4_prefill"] = True
        if fmt != "nvfp4_csf":
            return self.load_fp4_experts(
                tuple(f"{prefix}.{expert}" for expert in ids), hidden_size=2560, intermediate_size=640,
                format=fmt, activation_mode=mode, intermediate_range=self.shard.partition(640, alignment=32 if fmt == "mxfp4" else 16),
                scale_vectors=_expert_scale_vectors(self._sources, layer, ids) if mode == "a4" or self.force_moe_decode_a16 else None,
                **activation_options)
        return self.load_nvfp4_experts(
            tuple(f"{TEXT_PREFIX}.layers.{layer}.mlp.experts.{expert}" for expert in ids),
            hidden_size=2560, intermediate_size=640,
            intermediate_range=self.shard.partition(640, alignment=16),
            scale_vectors=_expert_scale_vectors(self._sources, layer, ids),
            scale_scratch=scale_scratch, activation_mode=mode, **activation_options,
        )
