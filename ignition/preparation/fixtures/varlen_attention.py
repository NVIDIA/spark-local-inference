# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite varlen-attention operands with one full-capacity sequence."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import planned_scratch
from .resources import candidate_inputs, restore_seed


@dataclass(frozen=True)
class AttentionBenchmarkInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    seed_q: torch.Tensor
    seed_k: torch.Tensor
    seed_v: torch.Tensor
    cu_seqlens_q: torch.Tensor

    @property
    def nbytes(self):
        return sum(value.untyped_storage().nbytes() for value in vars(self).values())

    def produce(self):
        for value, seed in ((self.q, self.seed_q), (self.k, self.seed_k), (self.v, self.seed_v)):
            restore_seed(value, seed)


def attention_benchmark_inputs(arguments):
    """Private finite QKV and one full-capacity image for attention tuning."""
    query = arguments["q"]
    generator = torch.Generator(device=query.device).manual_seed(89017)
    result = {}
    for name in ("q", "k", "v"):
        source = arguments[name]
        value = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype,
                                    device=source.device).normal_(generator=generator)
        seed = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype,
                                   device="cpu", pin_memory=source.is_cuda)
        seed.copy_(value)
        result[f"seed_{name}"] = seed
        result[name] = value
    count = arguments["cu_seqlens_q"].numel()
    maximum = arguments["max_seqlen_q"]
    if (count - 1) * maximum < query.shape[0]:
        raise ValueError("attention benchmark segments cannot hold the prepared query rows")
    lengths = (torch.arange(count, device=query.device, dtype=torch.int32) * maximum).clamp_max(query.shape[0])
    return AttentionBenchmarkInputs(**result, cu_seqlens_q=lengths)


def input_factory(arguments, *, storage=None, session=None):
    def allocate():
        with (nullcontext() if storage is None else storage.transient()):
            return attention_benchmark_inputs(arguments)

    return candidate_inputs(allocate, session=session)


def benchmark_call(state, *, arguments, inputs, workspace):
    """Keep candidate inputs private and reuse the caller's disposable output."""
    inputs = inputs()
    arguments = {**arguments, **{key: getattr(inputs, key)
                                  for key in ("q", "k", "v", "cu_seqlens_q")}}
    scratch = planned_scratch(state.scratch_plan.scratch_specs(), workspace)
    binding = state.bind(scratch=scratch, **arguments)
    return PreparedCall(run=lambda: state.run(binding), produce=inputs.produce,
                        owners=(inputs, scratch, binding))
