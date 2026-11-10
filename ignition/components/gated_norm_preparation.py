# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch compilation and autotuning for grouped gated RMS normalization."""

from dataclasses import dataclass, replace

import torch
import triton
from b12x._lib.compile_plan import attach_programs, compile_only_launches, launch_triton, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.tuning import benchmark_resource, declare

from .gated_norm import _gated_rms_norm
from .native_operator import native_contract, native_plan


@dataclass(frozen=True)
class Query:
    operands: tuple
    options: tuple


CONTRACT = native_contract(component_id="flare.gated_rms_norm", query_type=Query,
                           tactics=lambda q: (4, 1, 2, 8))


def _launch_arguments(tensors, q, warps):
    options = dict(q.options, num_warps=warps)
    hidden, gate = tensors[:2]
    grid = (hidden.shape[0], triton.cdiv(hidden.shape[1], options["HEAD_BLOCK"]), 1)
    return (*tensors, gate.stride(0)), options, grid


@program_cache(scope="preparation")
def compile_norm(payload, warps, ordinal):
    q = Query(**dict(payload))
    tensors = tuple(torch.empty_strided(shape, stride, dtype=getattr(torch, dtype), device="meta")
                    for shape, stride, dtype in q.operands)
    args, options, grid = _launch_arguments(tensors, q, warps)
    with torch.cuda.device(ordinal), compile_only_launches():
        return launch_triton(_gated_rms_norm, grid, *args, **options)


@dataclass(frozen=True)
class State:
    query: Query
    warps: int
    program: object

    def run(self, tensors):
        args, options, grid = _launch_arguments(tensors, self.query, self.warps)
        self.program[grid](*args, *(value for key, value in options.items() if key != "num_warps"))


@torch.library.custom_op("flare::gated_rms_norm", mutates_args=("output",))
def execute(hidden: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
            live: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, hidden.device).run((hidden, gate, weight, output, live))


@execute.register_fake
def _fake(hidden, gate, weight, output, live, handle):
    pass


def declare_norm(args, options, *, session, name):
    tensors = args[:5]
    query = Query(tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype).removeprefix("torch."))
                        for t in tensors), tuple((k, v) for k, v in options.items() if k != "num_warps"))
    payload = CONTRACT.encode_query(query)
    ordinal = tensors[0].device.index

    def materialize(warps):
        program = compile_norm(payload, warps, ordinal)
        load_programs(program)
        return attach_programs(State(query, warps, program), program)

    plan = native_plan(contract=CONTRACT, query=query, device=tensors[0].device,
        scratch_bytes=lambda t: 0, materialize=materialize,
        compile_jobs=lambda t, d: (CompileJob.create(
            "ignition.components.gated_norm_preparation:compile_norm", payload, t, d.ordinal),))

    def benchmark(state):
        def inputs():
            h, g, w, o, live = (torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
                               for t in tensors)
            h.normal_()
            g.normal_()
            w.copy_(tensors[2])
            live.fill_(h.shape[0])
            return (h, g, w, o, live), h.clone()
        values, seed = benchmark_resource(session, ("gated_rms_norm", name), inputs)
        return PreparedCall(run=lambda: state.run(values), output=values[3],
                            produce=lambda: values[0].copy_(seed), owners=(values, seed))

    declare(session, (plan.request(name=name, prepare_call=lambda state: replace(benchmark(state), owners=()),
                                  benchmark_call=benchmark),))
    return plan
