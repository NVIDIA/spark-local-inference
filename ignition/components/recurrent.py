# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixed-address metadata and preparation for recurrent sequence kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from b12x.preparation import PreparedCall

from ignition.preparation.fixtures import recurrent as recurrent_fixtures
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer


@triton.jit
def _commit_convolution(Backing, Offsets, Rows, Counts, STRIDE: tl.constexpr,
                        CHANNELS: tl.constexpr, COLUMNS: tl.constexpr, BLOCK: tl.constexpr):
    layer, row, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    count = tl.load(Counts + row)
    state = tl.load(Rows + row).to(tl.int64)
    channel = tile * BLOCK + tl.arange(0, BLOCK)
    live = (count > 1) & (state > 0) & (channel < CHANNELS)
    base = Backing + state * STRIDE + tl.load(Offsets + layer) + channel * COLUMNS
    first = tl.load(base + count - 1, live, other=0)
    second = tl.load(base + count, live, other=0)
    third = tl.load(base + count + 1, live, other=0)
    tl.store(base, first, live)
    tl.store(base + 1, second, live)
    tl.store(base + 2, third, live)


@dataclass(frozen=True)
class DeltaState:
    """Chronological BF16 convolution history and FP32 value-by-key recurrence."""

    convolution: torch.Tensor
    recurrent: torch.Tensor

    @classmethod
    def allocate(cls, slots, device, *, heads, channels, speculative_tokens=0):
        if type(slots) is not int or slots < 2:
            raise ValueError("delta state requires a null slot and at least one live state slot")
        if any(type(value) is not int or value <= 0 for value in (heads, channels)):
            raise ValueError("delta state requires positive head and convolution channel counts")
        if type(speculative_tokens) is not int or not 0 <= speculative_tokens <= 7:
            raise ValueError("delta state supports zero through seven speculative tokens")
        return cls(
            torch.zeros((slots, channels, 3 + speculative_tokens), dtype=torch.bfloat16, device=device),
            torch.zeros((slots, heads, 128, 128), dtype=torch.float32, device=device),
        )


@dataclass(frozen=True)
class DeltaMetadata:
    """Fixed-address device metadata updated by the native scheduler.

    Decode consumes ``state_indices[:, accepted_tokens - 1]`` and writes each
    query token's checkpoint to its corresponding column. Prefill uses explicit
    initial/final indices and can export a checkpoint at a multiple of 16 tokens.
    Live nonnegative slots must be unique and in range. Counts and packed query
    boundaries must fit their buffer capacities. These are scheduler invariants.
    """

    query_start_loc: torch.Tensor
    convolution_slots: torch.Tensor
    fresh: torch.Tensor
    accepted_tokens: torch.Tensor
    state_indices: torch.Tensor
    initial_state_indices: torch.Tensor
    final_state_indices: torch.Tensor
    checkpoint_state_indices: torch.Tensor
    checkpoint_offsets: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor

    @classmethod
    def allocate(
        cls, sequences: int, device: torch.device, *, verification_columns: int = 1
    ) -> DeltaMetadata:
        if type(sequences) is not int or sequences <= 0:
            raise ValueError("delta sequence capacity must be positive")
        if type(verification_columns) is not int or not 1 <= verification_columns <= 16:
            raise ValueError("delta verification columns must be in [1, 16]")

        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((sequences + 1,)),
            torch.full((sequences,), -1, dtype=torch.int64, device=device),
            zeros((sequences,), torch.bool),
            torch.ones(sequences, dtype=torch.int32, device=device),
            zeros((sequences, verification_columns), torch.int64),
            zeros((sequences,), torch.int64),
            zeros((sequences,), torch.int64),
            zeros((sequences,), torch.int64),
            zeros((sequences,)),
            zeros((1,)),
            zeros((1,)),
        )


def prepare_sequence_binding(
    plan, args, *, session, name, public_bind, mode, set_binding,
    bind_plan=False, workspace=None, run_options=None, bind_method="bind",
):
    run_options = {} if run_options is None else dict(run_options)

    def selected_scratch(specs):
        return planned_scratch(specs, workspace)

    def candidate_specs(prepared):
        return plan.scratch_specs() if bind_plan else prepared.layout.scratch_specs()

    def prepare_call(prepared):
        scratch = selected_scratch(candidate_specs(prepared))
        binding = getattr(prepared, bind_method)(scratch=scratch, **args, **({"plan": plan} if bind_plan else {}))
        return PreparedCall(run=lambda: prepared.run(binding, **run_options), owners=prime_owners(workspace, scratch, binding))

    def benchmark_call(prepared):
        return recurrent_fixtures.benchmark_call(
            prepared, arguments=args, session=session, name=name, mode=mode,
            scratch_specs=candidate_specs, workspace=workspace,
            run_options=run_options, binding_plan=plan if bind_plan else None, bind_method=bind_method,
        )

    declare(session, (plan.request(name=name, prepare_call=prepare_call,
        benchmark_call=benchmark_call),))
    selected = None

    def finalize():
        nonlocal selected
        scratch = selected_scratch(plan.scratch_specs())
        selected = public_bind(plan, scratch=scratch, **args)
        set_binding(selected)

    register_finalizer(session, finalize, workspace=workspace)
    return selected


