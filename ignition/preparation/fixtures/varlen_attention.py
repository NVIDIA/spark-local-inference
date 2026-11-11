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
        return sum({value.untyped_storage()._cdata: value.untyped_storage().nbytes()
                    for value in vars(self).values()}.values())

    def produce(self):
        restored = set()
        for value, seed in ((self.q, self.seed_q), (self.k, self.seed_k), (self.v, self.seed_v)):
            storage = value.untyped_storage()._cdata
            if storage not in restored:
                restore_seed(value, seed)
                restored.add(storage)


def attention_benchmark_inputs(arguments):
    """Private finite QKV and one full-capacity image for attention tuning."""
    query = arguments["q"]
    generator = torch.Generator(device=query.device).manual_seed(89017)
    result, groups = {}, {}
    for name in ("q", "k", "v"):
        source = arguments[name]
        groups.setdefault(source.untyped_storage()._cdata, []).append((name, source))
    backings = []
    for operands in groups.values():
        first = min(source.storage_offset() for _, source in operands)
        last = max(source.storage_offset() + 1 + sum((size - 1) * stride
                   for size, stride in zip(source.shape, source.stride(), strict=True))
                   for _, source in operands)
        source = operands[0][1]
        backing = torch.zeros(last - first, dtype=source.dtype, device=source.device)
        for name, source in operands:
            value = backing.as_strided(source.shape, source.stride(), source.storage_offset() - first)
            result[name] = value
        backings.append((backing, operands, first))
    for name in ("q", "k", "v"):
        result[name].normal_(generator=generator)
    for backing, operands, first in backings:
        seed = torch.empty(backing.shape, dtype=backing.dtype, device="cpu", pin_memory=backing.is_cuda)
        seed.copy_(backing)
        for name, source in operands:
            result[f"seed_{name}"] = seed.as_strided(source.shape, source.stride(), source.storage_offset() - first)
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
