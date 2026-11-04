# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Qwen routed NVFP4 experts and the shared MXFP8 expert."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from b12x.gemm import blockscaled
from b12x.moe import fused_moe
from b12x.preparation import PreparationSession

from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import PreparedMLP
from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.components.weights import MLPWeights
from ignition.preparation.qwen38 import TEXT_PREFIX
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer

from .weights import QwenWeightLoader


@dataclass(frozen=True)
class SharedExpertWeights:
    gate_up: blockscaled.Weight | torch.Tensor
    down: blockscaled.Weight | torch.Tensor
    gate: torch.Tensor

    @classmethod
    def load(cls, loader: QwenWeightLoader, layer: int) -> SharedExpertWeights:
        if type(layer) is not int or not 0 <= layer < 48:
            raise ValueError("decoder layer must be an integer in [0, 48)")
        prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp"
        extent = loader.shard.partition(640, alignment=32)
        return cls(
            loader.load_linear_group(tuple(f"{prefix}.shared_expert.{name}" for name in ("gate_proj", "up_proj")),
                                    row_ranges=(extent, extent)),
            loader.load_linear(f"{prefix}.shared_expert.down_proj", columns=extent),
            loader.load_tensor(f"{prefix}.shared_expert_gate.weight"),
        )


@dataclass(frozen=True)
class QwenMoeWeights:
    router: torch.Tensor
    experts: fused_moe.PreparedExperts
    shared: SharedExpertWeights
    experts_a4_prefill: bool = False

    @classmethod
    def load(cls, loader: QwenWeightLoader, layer: int, *, scale_scratch=None) -> QwenMoeWeights:
        shared = SharedExpertWeights.load(loader, layer)
        return cls(
            loader.load_tensor(f"{TEXT_PREFIX}.layers.{layer}.mlp.gate.weight"),
            loader.load_experts(layer, scale_scratch=scale_scratch),
            shared,
            experts_a4_prefill=loader.force_moe_decode_a16,
        )


@dataclass(frozen=True)
class SharedExpertScratch:
    gate_up: torch.Tensor
    intermediate: torch.Tensor
    projected: torch.Tensor
    gate_logits: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device, *, intermediate=640, buffers=None) -> SharedExpertScratch:
        if tokens <= 0:
            raise ValueError("shared expert capacity must be positive")
        return cls(*((torch.zeros((tokens, width), dtype=torch.bfloat16, device=device) if buffers is None
                      else buffers.zeros((tokens, width), torch.bfloat16))
                     for width in (2 * intermediate, intermediate, 2560, 1, 2560)))

    def rows(self, tokens: int) -> SharedExpertScratch:
        if not 0 < tokens <= self.gate_up.shape[0]:
            raise ValueError("shared expert rows exceed scratch capacity")
        return SharedExpertScratch(*(getattr(self, field)[:tokens] for field in self.__dataclass_fields__))


@dataclass(frozen=True)
class QwenMoeScratch:
    """Capacity-backed intermediates shared by serial decoder layers.

    The shared-expert output remains live while routed experts execute, so
    these are distinct buffers. Views for smaller graph buckets share storage;
    graphs using this scratch must execute in one ordered stream.
    """

    router_logits: torch.Tensor
    router: RouterScratch
    routed_output: torch.Tensor
    output: torch.Tensor
    shared: SharedExpertScratch

    @classmethod
    def allocate(cls, tokens: int, device, *, intermediate=640, buffers=None, output=None) -> QwenMoeScratch:
        if tokens <= 0:
            raise ValueError("MoE capacity must be positive")
        def zeros(width):
            return (torch.zeros((tokens, width), dtype=torch.bfloat16, device=device) if buffers is None
                    else buffers.zeros((tokens, width), torch.bfloat16))
        return cls(zeros(512), RouterScratch.allocate(tokens, device, experts=512, top_k=10, buffers=buffers),
            zeros(2560), zeros(2560) if output is None else output,
            SharedExpertScratch.allocate(tokens, device, intermediate=intermediate, buffers=buffers))

    def rows(self, tokens: int) -> QwenMoeScratch:
        if not 0 < tokens <= self.router_logits.shape[0]:
            raise ValueError("MoE rows exceed scratch capacity")
        return QwenMoeScratch(self.router_logits[:tokens], self.router.rows(tokens),
            self.routed_output[:tokens], self.output[:tokens], self.shared.rows(tokens))


class PreparedSharedExpert(PreparedMLP):
    """MXFP8 or stored BF16 SwiGLU with BF16 SiLU and sigmoid boundaries."""

    def __init__(self, source, weights: SharedExpertWeights, *, session: PreparationSession, name: str,
                 scratch: SharedExpertScratch | None = None, projection_workspace: ScratchArena | torch.Tensor | None = None):
        if source.ndim != 2 or source.shape[1] != 2560 or source.dtype != torch.bfloat16:
            raise ValueError("shared expert input must be BF16 [tokens, 2560]")
        def geometry(weight):
            return tuple(reversed(weight.shape)) if isinstance(weight, torch.Tensor) else (weight.in_features, weight.out_features)

        intermediate = geometry(weights.down)[0]
        if geometry(weights.gate_up) != (2560, 2 * intermediate):
            raise ValueError("shared gate/up projection must have geometry [2*intermediate,2560]")
        if geometry(weights.down) != (intermediate, 2560):
            raise ValueError("shared down projection must have geometry [2560,intermediate]")
        if tuple(weights.gate.shape) != (1, 2560):
            raise ValueError("shared expert gate must have geometry [1, 2560]")
        tokens, device = source.shape[0], source.device
        scratch = SharedExpertScratch.allocate(tokens, device, intermediate=intermediate) if scratch is None else scratch.rows(tokens)
        for field, width in zip(scratch.__dataclass_fields__, (2 * intermediate, intermediate, 2560, 1, 2560), strict=True):
            tensor = getattr(scratch, field)
            if tensor.shape != (tokens, width) or tensor.dtype != torch.bfloat16 or tensor.device != device or not tensor.is_contiguous():
                raise ValueError(f"shared expert {field} requires contiguous BF16 [{tokens}, {width}] on {device}")
            setattr(self, field, tensor)
        self.scratch = scratch
        super().__init__(source, MLPWeights(weights.gate_up, weights.down), self.gate_up,
            self.intermediate, self.projected, session=session, name=name,
            workspace=projection_workspace, activation_mode="auto")
        self.output = scratch.output
        self.gate_projection = UnquantizedProjection(source, weights.gate, self.gate_logits,
            session=session, name=f"{name}.shared_gate")

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        self.project(source)
        return self.finish()

    def project(self, source: torch.Tensor) -> None:
        super().__call__(source)
        self.gate_projection(source)

    def finish(self, routed: torch.Tensor | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
        output = self.output if out is None else out
        shared = self.projected * torch.sigmoid(self.gate_logits)
        output.copy_(shared if routed is None else routed + shared)
        return output


class PreparedQwenMoe:
    """Native FP32 routing and shared-expert composition around routed experts.

    Construction allocates and prepares all buffers and plans. Calls are
    recorded into the model CUDA graph; native replay executes its CUDA nodes
    without calling this Python object. Source storage is fixed per graph.
    """

    def __init__(self, source, weights: QwenMoeWeights, *, session: PreparationSession, name: str,
                 scratch: QwenMoeScratch | None = None, workspace: ScratchArena | torch.Tensor | None = None,
                 projection_workspace: ScratchArena | torch.Tensor | None = None, prefill: bool = False):
        if (weights.experts.num_experts, weights.experts.hidden_size) != (512, 2560):
            raise ValueError("Qwen MoE requires all 512 experts with hidden size 2560")
        if tuple(weights.router.shape) != (512, 2560):
            raise ValueError("Qwen router weight must have geometry [512, 2560]")
        if source.ndim != 2 or source.shape[1] != 2560 or source.dtype != torch.bfloat16:
            raise ValueError("Qwen MoE source must be BF16 [tokens, 2560]")
        self.source = source
        scratch = QwenMoeScratch.allocate(source.shape[0], source.device, intermediate=weights.experts.intermediate_size) if scratch is None else scratch.rows(source.shape[0])
        for tensor, width in ((scratch.router_logits, 512), (scratch.routed_output, 2560), (scratch.output, 2560)):
            if tensor.shape != (source.shape[0], width) or tensor.dtype != torch.bfloat16 or tensor.device != source.device or not tensor.is_contiguous():
                raise ValueError("MoE scratch geometry, BF16 dtype and device must match source")
        self.scratch = scratch
        self.router_logits, self.router = scratch.router_logits, scratch.router
        self.routed_output, self.output = scratch.routed_output, scratch.output
        self.router_projection = UnquantizedProjection(source, weights.router, self.router_logits,
            session=session, name=f"{name}.router")
        self.routing = PreparedRouter(self.router_logits, self.router, session=session, name=f"{name}.route_topk")
        self.shared = PreparedSharedExpert(source, weights.shared, session=session, name=f"{name}.shared",
            scratch=scratch.shared, projection_workspace=projection_workspace)
        self.experts = PreparedRoutedExperts(source, weights.experts, self.router.ids,
            self.router.weights, self.routed_output, session=session, name=f"{name}.experts", workspace=workspace,
            router_plan=self.routing.plan, a4_prefill=prefill and weights.experts_a4_prefill)
        self.concurrent_shared = None
        if (source.shape[0] <= 256 and isinstance(workspace, ScratchArena) and workspace is projection_workspace
                and workspace.overlap):
            self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                parallel_plans=(self.router_projection.plan, self.routing.plan, self.experts.plan),
                gate_projection=self.shared.gate_projection, session=session, workspace=workspace)

            def finalize_overlap():
                self.concurrent_shared.defer_launch = (
                    source.shape[0] <= 2 and self.experts.binding.implementation == "micro")

            register_finalizer(session, finalize_overlap, workspace=workspace)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if source is not self.source and (
                source.data_ptr() != self.source.data_ptr() or source.shape != self.source.shape):
            raise ValueError("prepared Qwen MoE requires its fixed source storage")
        if self.concurrent_shared is None:
            self.shared.project(source)
        else:
            self.concurrent_shared.start(source)
        self.router_projection(source)
        self.routing(self.router_logits)
        if getattr(self.experts, "reduces_into_shared", False):
            # SM103: one kernel reduces the routed FC2 rows into the shared-expert sum.
            staged = self.experts.stage_rows()
            if self.concurrent_shared is not None:
                self.concurrent_shared.finish(source, staged)
            return self.experts.reduce(self.shared.finish(), self.output)
        routed = self.experts()
        if self.concurrent_shared is not None:
            self.concurrent_shared.finish(source, routed)
        return self.shared.finish(routed, out=self.output)
