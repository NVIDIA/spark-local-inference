# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Qwen hyperconnection mixing and residual injection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from b12x.norm import hyperconnection as hc
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall

from ignition.components.hyperconnection_gate import PreparedHyperconnectionGate, PreparedShardedHyperconnectionGate
from ignition.components.linear import UnquantizedProjection
from ignition.preparation.fixtures import hyperconnection as hc_fixtures
from ignition.preparation.tuning import declare, register_finalizer
from ignition.preparation.persistent import persistent_copy, persistent_zeros


def hyperconnection_extents(shard, sharded: bool):
    """Partition the lowrank axis while retaining normalization and injection weights."""
    if not sharded:
        return {}
    if shard.world_size < 2:
        raise ValueError("hyperconnection sharding requires tensor parallelism")
    lowrank = slice(*shard.partition(320, alignment=16))
    return {"input_mix_weight_down.weight": lowrank,
            "input_mix_weight_up.weight": (slice(None), lowrank)}


def normalization_gain(module):
    """Retain one FP32 gain for every prepared use of an immutable mixer."""
    gain = getattr(module, "_flare_norm_gain", None)
    if gain is None:
        gain = persistent_copy((1.0 + module.hc_norm.weight.detach().float()).contiguous())
        module._flare_norm_gain = gain
    return gain


def packed_down_projection(module: torch.nn.Module) -> torch.Tensor:
    if module.block_inject_weight is None:
        return module.input_mix_weight_down.weight
    packed = getattr(module, "packed_down_injection", None)
    if packed is None:
        down = module.input_mix_weight_down.weight
        injection = module.block_inject_weight.weight
        rows = down.shape[0] + injection.shape[0]
        packed = persistent_zeros(((rows + 15) // 16 * 16, down.shape[1]), dtype=down.dtype, device=down.device)
        with torch.no_grad():
            packed[:down.shape[0]].copy_(down)
            packed[down.shape[0]:rows].copy_(injection)
        module.input_mix_weight_down.weight = torch.nn.Parameter(packed[:down.shape[0]], requires_grad=False)
        module.block_inject_weight.weight = torch.nn.Parameter(packed[down.shape[0]:rows], requires_grad=False)
        module.register_buffer("packed_down_injection", packed, persistent=False)
    return packed


@dataclass(frozen=True)
class HyperconnectionScratch:
    """Shared activations and transient storage for serialized mixers.

    Gate workspace holds local partial logits for sharded mixers and private
    scratch for fused replicated mixers. Reduced logits have separate storage.
    """

    normalized: torch.Tensor
    projected_down: torch.Tensor
    bottleneck: torch.Tensor
    gate_workspace: torch.Tensor
    injection_logits: torch.Tensor
    block_input: torch.Tensor
    down_and_injection: torch.Tensor
    gate_logits: torch.Tensor | None = None

    @classmethod
    def allocate(cls, tokens: int, device: torch.device, *, buffers=None, projected=None,
                 block_input=None, lowrank: int = 320) -> HyperconnectionScratch:
        if tokens <= 0:
            raise ValueError("hyperconnection token capacity must be positive")
        if lowrank not in (80, 160, 320):
            raise ValueError("Qwen hyperconnection lowrank must be 320, 160 or 80")
        def zeros(width):
            return (torch.zeros((tokens, width), device=device, dtype=torch.bfloat16) if buffers is None
                    else buffers.zeros((tokens, width), torch.bfloat16))
        normalized = zeros(10240)
        projected = zeros((lowrank + 4 + 15) // 16 * 16) if projected is None else projected
        bottleneck, gate_workspace = zeros(lowrank), zeros(10240)
        block_input = zeros(2560) if block_input is None else block_input
        return cls(normalized, projected[:, :lowrank], bottleneck, gate_workspace,
                   projected[:, lowrank:lowrank + 4], block_input, projected,
                   zeros(10240) if lowrank < 320 else None)

    def rows(self, tokens: int) -> HyperconnectionScratch:
        if not 0 < tokens <= self.normalized.shape[0]:
            raise ValueError("hyperconnection rows exceed scratch capacity")
        return HyperconnectionScratch(*(None if (value := getattr(self, field)) is None else value[:tokens]
                                       for field in self.__dataclass_fields__))

    def priming_arguments(self, source: torch.Tensor, norm_gain: torch.Tensor):
        """Borrow a zero-initialized source kept unchanged until preparation ends."""
        if (source.shape != self.normalized.shape or source.stride() != self.normalized.stride()
                or source.dtype != torch.bfloat16 or source.device != self.normalized.device
                or not source.is_contiguous()):
            raise ValueError("hyperconnection priming source must match normalized shape, stride, dtype and device")
        start, end = source.data_ptr(), source.data_ptr() + source.nbytes
        def overlaps(tensor):
            if tensor is None:
                return False
            extent = (1 + sum((n - 1) * stride for n, stride in zip(tensor.shape, tensor.stride()))) * tensor.element_size()
            return tensor.numel() > 0 and start < tensor.data_ptr() + extent and tensor.data_ptr() < end
        if any(overlaps(getattr(self, field)) for field in self.__dataclass_fields__):
            raise ValueError("hyperconnection priming source requires storage separate from scratch outputs")
        return {
            "grouped_rmsnorm": (source, norm_gain, self.normalized),
            "scaled_silu": (self.projected_down, self.bottleneck),
            "combine": (source, self.block_input, self.injection_logits, self.normalized),
        }


class PreparedHyperconnection:
    """Use one shared scratch region during serialized decoder execution.

    ``mix`` returns the next block's input. Its injection logits remain live
    until ``combine``; another mixer must not overwrite the scratch in between.
    The final model mixer has no residual injection projection.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        scratch: HyperconnectionScratch,
        *,
        priming_state: torch.Tensor,
        session: PreparationSession,
        name: str,
        runtime=None,
        coordinator=None,
    ) -> None:
        if module.hc_count != 4 or module.hidden_size != 2560:
            raise ValueError("Qwen hyperconnection requires four streams of width 2560")
        self.scratch = scratch
        self.eps = module.hc_norm.eps
        self.norm_gain = normalization_gain(module)
        device = scratch.normalized.device
        if self.norm_gain.device != device or device.type != "cuda":
            raise ValueError("hyperconnection weights and scratch must share a CUDA device")
        if tuple(self.norm_gain.shape) != (10240,):
            raise ValueError("hyperconnection norm weight must have 10240 elements")
        self.tokens = scratch.normalized.shape[0]
        self.lowrank = module.input_mix_weight_down.weight.shape[0]
        if scratch.bottleneck.shape[1] != self.lowrank:
            raise ValueError("hyperconnection weights and scratch have different lowrank widths")
        sharded = self.lowrank < 320
        if sharded and (runtime is None or self.lowrank * runtime.world_size != 320):
            raise ValueError("sharded hyperconnections require a matching TP collective")
        if sharded != (scratch.gate_logits is not None):
            raise ValueError("sharded hyperconnections require separate reduced-logit storage")
        caps = hc.Caps(device=device, max_tokens=self.tokens, hidden_size=2560, streams=4, lowrank=self.lowrank)
        self.plans = {}
        self.bindings = {}
        priming_arguments = scratch.priming_arguments(priming_state, self.norm_gain)
        invocations = {
            "grouped_rmsnorm": {"eps": self.eps, "zero_centered": False, "weight_dtype": "float32"},
            "scaled_silu": {},
        }
        if module.block_inject_weight is not None:
            invocations["combine"] = {"output_mode": "provided"}
        for operation, options in invocations.items():
            plan = hc.plan(caps, invocation=FrozenMapping({"operation": operation, **options}))
            arguments = priming_arguments[operation]
            kwargs = {"eps": self.eps} if operation == "grouped_rmsnorm" else {}
            declare(session, (plan.request(
                name=f"{name}.{operation}",
                prepare_call=lambda state, args=arguments, kw=kwargs: PreparedCall(run=lambda: state.launch(*args, **kw)),
                benchmark_call=partial(hc_fixtures.benchmark_call, operation=operation,
                                       arguments=arguments, kwargs=kwargs),
            ),))
            self.plans[operation] = plan
            if operation == "combine":
                continue

            def finalize(op=operation, selected_plan=plan):
                self.bindings[op] = hc.bind(selected_plan, tokens=self.tokens,
                    normalized=scratch.normalized, bottleneck=scratch.bottleneck, block_input=scratch.block_input)

            register_finalizer(session, finalize)

        self.injection = None if module.block_inject_weight is None else scratch.injection_logits
        down_output = scratch.projected_down if self.injection is None else scratch.down_and_injection
        self.down = UnquantizedProjection(scratch.normalized, packed_down_projection(module),
            down_output, session=session, name=f"{name}.down")
        if sharded:
            self.gate = PreparedShardedHyperconnectionGate(scratch.bottleneck, module.input_mix_weight_up.weight,
                scratch.normalized, scratch.block_input, scratch.gate_workspace, scratch.gate_logits,
                runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.gate")
        else:
            self.gate = PreparedHyperconnectionGate(scratch.bottleneck, module.input_mix_weight_up.weight,
                scratch.normalized, scratch.block_input, scratch.gate_workspace,
                session=session, name=f"{name}.gate")

    def prepare_combine_norm(self, following: PreparedHyperconnection, *,
                             priming_state: torch.Tensor, session: PreparationSession,
                             name: str) -> None:
        if self.injection is None:
            raise ValueError("the final hyperconnection mixer has no residual injection")
        self.following = following
        caps = hc.Caps(device=self.scratch.normalized.device, max_tokens=self.tokens,
                       hidden_size=2560, streams=4, lowrank=self.lowrank)
        plan = hc.plan(caps, invocation=FrozenMapping({"operation": "combine_norm",
            "eps": following.eps, "zero_centered": False, "weight_dtype": "float32",
            "output_mode": "provided"}))
        arguments = (priming_state, self.scratch.block_input, self.scratch.injection_logits,
                     following.norm_gain, self.scratch.gate_workspace, following.scratch.normalized)
        kwargs = {"eps": following.eps}
        declare(session, (plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.launch(*arguments, **kwargs)),
            benchmark_call=partial(hc_fixtures.benchmark_call, operation="combine_norm",
                                   arguments=arguments, kwargs=kwargs)),))
        self.plans["combine_norm"] = plan

    def mix(self, residual: torch.Tensor, *, pre_normalized: bool = False) -> torch.Tensor:
        normalized = self.scratch.normalized if pre_normalized else hc.run_grouped_rmsnorm(
            residual, self.norm_gain, eps=self.eps, zero_centered=False,
            binding=self.bindings["grouped_rmsnorm"])
        self.down(normalized)
        bottleneck = hc.run_scaled_silu(self.scratch.projected_down, binding=self.bindings["scaled_silu"])
        return self.gate(bottleneck, normalized)

    def trace_intermediates(self) -> dict[str, torch.Tensor]:
        """Expose activations produced by every eligible gate implementation."""
        values = {name: getattr(self.scratch, name)
                  for name in ("normalized", "projected_down", "bottleneck")}
        if self.injection is not None:
            values["injection_logits"] = self.injection
        return values

    def combine(self, residual: torch.Tensor, block_output: torch.Tensor) -> torch.Tensor:
        """Inject in FP32 and update the BF16 residual in place."""
        if self.injection is None:
            raise ValueError("the final hyperconnection mixer has no residual injection")
        return hc.run_combine(residual, block_output, self.scratch.injection_logits,
                              plan=self.plans["combine"], out=residual)

    def combine_norm(self, residual: torch.Tensor, block_output: torch.Tensor) -> torch.Tensor:
        """Update the residual in place and prepare the following mixer's input."""
        combined, _ = hc.run_combine_norm(residual, block_output, self.scratch.injection_logits,
            self.following.norm_gain, eps=self.following.eps, zero_centered=False,
            plan=self.plans["combine_norm"], out=(residual, self.following.scratch.normalized))
        return combined
