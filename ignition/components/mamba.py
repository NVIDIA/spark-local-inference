# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared Mamba-2 scans and FlashInfer selective state updates."""

from dataclasses import dataclass, replace

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared
from b12x.sequence import mamba_ssd

from ignition.preparation.fixtures.recurrent import (
    BenchmarkStateArena, RecurrentBenchmarkInputs, _benchmark_state_key,
)
from ignition.preparation.scratch import planned_scratch
from ignition.preparation.tuning import benchmark_resource, declare

from .native_operator import native_contract, native_plan


def _operands(args):
    return tuple(args[name] for name in ("x", "dt", "A", "B", "C", "D", "dt_bias", "recurrent_state",
        "cu_seqlens", "initial_state_indices", "final_state_indices", "state_is_fresh", "num_seqs", "num_tokens"))


def _fixture(args, *, session, name, mode):
    state = args["recurrent_state"]
    arena = benchmark_resource(session, _benchmark_state_key(state), lambda: BenchmarkStateArena(state))
    return benchmark_resource(session, ("mamba_inputs", name),
        lambda: RecurrentBenchmarkInputs(args, mode=mode, arena=arena))


class PreparedMambaScan:
    """Packed prefill with BF16 inputs and caller-owned recurrent state and scratch."""

    def __init__(self, args, *, session, workspace, name):
        output = args["out"]
        self.output_shape, self.output_dtype = tuple(output.shape), output.dtype
        query = mamba_ssd.query_from_call(*_operands(args), out=output)
        self.plan = mamba_ssd.plan(query, device=output.device)

        def benchmark(state):
            operands = _fixture(args, session=session, name=name, mode="prefill")
            scratch = planned_scratch(self.plan.scratch_specs(), workspace) if state.required_workspace else None
            return PreparedCall(run=lambda: state.run(*_operands(operands.args), out=operands.args["out"], workspace=scratch),
                output=operands.args["out"], produce=operands.producers[0], benchmark_producers=operands.producers,
                reset=operands.reset, restore=operands.reset, owners=(operands, scratch))

        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: replace(benchmark(state), owners=()), benchmark_call=benchmark),))

    def __call__(self, operands):
        output = torch.empty(self.output_shape, dtype=self.output_dtype, device=operands["x"].device)
        specs = self.plan.scratch_specs()
        workspace = planned_scratch(specs) if specs else None
        return mamba_ssd.run(*_operands(operands), out=output, plan=self.plan, workspace=workspace)


@dataclass(frozen=True)
class UpdateQuery:
    heads: int
    groups: int
    head_dim: int
    dstate: int
    sequences: int
    tokens: int
    state_dtype: str
    weight_dtype: str
    index_dtype: str
    state_stride: int
    x_stride: int
    dt_stride: int
    b_stride: int
    c_stride: int


UPDATE_CONTRACT = native_contract(component_id="flare.mamba_state_update", query_type=UpdateQuery,
                                  tactics=lambda q: (1,), semantic_version=2)


@dataclass(frozen=True)
class UpdateState:
    module: object
    algorithm: int

    def run(self, operands, output):
        x, dt, A, B, C, D, bias, state, cu, src, dst, fresh, ns, nt = operands
        heads, dim, dstate = state.shape[1:]
        self.module.selective_state_update(state, x, dt.unsqueeze(-1).expand(-1, -1, dim),
            A[:, None, None].expand(heads, dim, dstate), B, C, D[:, None].expand(heads, dim), None,
            bias[:, None].expand(heads, dim), True, src, dst, -1, None, output,
            False, None, None, None, None, 1, cu, None, self.algorithm)


@torch.library.custom_op("flare::mamba_state_update", mutates_args=("state", "out"))
def _update(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
            D: torch.Tensor, bias: torch.Tensor, state: torch.Tensor, cu: torch.Tensor,
            src: torch.Tensor, dst: torch.Tensor, fresh: torch.Tensor, ns: torch.Tensor, nt: torch.Tensor,
            out: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, x.device).run(
        (x, dt, A, B, C, D, bias, state, cu, src, dst, fresh, ns, nt), out)


@_update.register_fake
def _update_fake(x, dt, A, B, C, D, bias, state, cu, src, dst, fresh, ns, nt, out, handle):
    pass


class PreparedMambaUpdate:
    """Single-token packed decode using FlashInfer's native SM120 SSU.

    Each sequence contains at most one token. Fresh requests read a zeroed null
    state through their initial index; inactive sequences have empty boundaries.
    State mutation and output storage are visible to whole-forward compilation.
    """

    def __init__(self, args, *, session, name):
        output = args["out"]
        self.output_shape, self.output_dtype = tuple(output.shape), output.dtype
        q = mamba_ssd.query_from_call(*_operands(args), out=output)
        if q.max_tokens > q.max_seqs:
            raise ValueError("single-token Mamba update requires at most one token of capacity per sequence")
        if args["dt"].dtype != args["D"].dtype or args["dt"].dtype != args["dt_bias"].dtype:
            raise ValueError("FlashInfer SSU requires dt, D, and dt_bias to share their dtype")
        query = UpdateQuery(q.heads, q.groups, q.head_dim, q.dstate, q.max_seqs, q.max_tokens,
            q.state_dtype, q.dt_dtype, q.index_dtype, q.state_stride, q.x_stride, q.dt_stride, q.b_stride, q.c_stride)
        device = output.device

        def materialize(algorithm):
            from flashinfer.mamba.selective_state_update import get_selective_state_update_module

            module = get_selective_state_update_module(device,
                getattr(torch, q.state_dtype), torch.bfloat16, getattr(torch, q.dt_dtype), torch.float32,
                getattr(torch, q.index_dtype), q.head_dim, q.dstate, 1, torch.int32, torch.int64)
            return UpdateState(module, algorithm)

        self.plan = native_plan(contract=UPDATE_CONTRACT, query=query, device=device,
            scratch_bytes=lambda t: 0, materialize=materialize)

        def benchmark(state):
            operands = _fixture(args, session=session, name=name, mode="decode")
            return PreparedCall(run=lambda: state.run(_operands(operands.args), operands.args["out"]),
                output=operands.args["out"], produce=operands.producers[0], reset=operands.reset,
                restore=operands.reset, owners=(operands,))

        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: replace(benchmark(state), owners=()), benchmark_call=benchmark),))

    def __call__(self, operands):
        output = torch.empty(self.output_shape, dtype=self.output_dtype, device=operands["x"].device)
        _update(*_operands(operands), output, self.plan.handle)
        return output
