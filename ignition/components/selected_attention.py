# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared b12x paged GQA over caller-selected logical token positions."""

from dataclasses import dataclass, replace

import torch
from b12x._lib.compile_plan import attach_programs, compile_only_launches, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec
from b12x.attention.qsa._sparse_gqa import (
    compile_sparse_paged_gqa,
    launch_sparse_paged_gqa,
)
from b12x.attention.qsa._sparse_gqa_cute_config import split_row_limit
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.persistent import persistent_empty
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .native_operator import native_contract, native_plan


def tensor_facts(tensor):
    return tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype).removeprefix("torch.")


def metadata_tensor(facts):
    shape, stride, dtype = facts
    return torch.empty_strided(shape, stride, dtype=getattr(torch, dtype), device="meta")


@dataclass(frozen=True)
class Query:
    operands: tuple
    output_gate: tuple | None = None

    @property
    def rows(self):
        return self.operands[0][0][0]

    @property
    def direct(self):
        return self.rows > split_row_limit(self.operands[1][0][2])


def _tactics(query):
    return tuple((1, warps) for warps in (1, 2, 4)) if query.direct else tuple((n, 2) for n in (4, 8, 16, 32, 64))


CONTRACT = native_contract(component_id="flare.selected_gqa", query_type=Query, tactics=_tactics)


@program_cache(scope="preparation")
def compile_attention(operands, gate_facts, warps, ordinal):
    q, k, v, requests, selected = map(metadata_tensor, operands)
    with torch.cuda.device(ordinal), compile_only_launches():
        return compile_sparse_paged_gqa(query=q, key_cache=k, value_cache=v,
            request_ids=requests, selected_positions=selected, direct_kv_warps=warps,
            output_gate=metadata_tensor(gate_facts) if gate_facts is not None else None)


@dataclass(frozen=True)
class State:
    query: Query
    tactic: tuple
    programs: dict


def _run(query, key, value, pages, requests, positions, selected, output, output_gate, descale, scratch, state):
    rows, heads, dim = query.shape
    splits, warps = state.tactic
    partial = lse = None
    if not state.query.direct:
        size = rows * splits * heads * dim
        values = scratch.view(torch.float32)
        partial = values[:size].view(rows, splits, heads, dim)
        lse = values[size:size + rows * splits * heads].view(rows, splits, heads)
    launch_sparse_paged_gqa(query=query, key_cache=key, value_cache=value, block_table=pages,
        request_ids=requests, query_positions=positions, selected_positions=selected, output=output,
        partial_output=partial, partial_lse=lse, softmax_scale=dim ** -.5, block_n=16,
        splits=splits, direct_kv_warps=warps, output_gate=output_gate,
        k_descale=descale, v_descale=descale, _prepared=state.programs)


@torch.library.custom_op("flare::selected_gqa", mutates_args=("output", "scratch"))
def _execute(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pages: torch.Tensor,
             requests: torch.Tensor, positions: torch.Tensor, selected: torch.Tensor,
             output: torch.Tensor, output_gate: torch.Tensor | None, descale: torch.Tensor, scratch: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    _run(query, key, value, pages, requests, positions, selected, output, output_gate, descale, scratch,
         require_prepared(plan, plan.component_id, query.device))


@_execute.register_fake
def _fake(query, key, value, pages, requests, positions, selected, output, output_gate, descale, scratch, handle):
    pass


class PreparedSelectedAttention:
    def __init__(self, query, key, value, pages, requests, positions, selected, output, *, output_gate=None, session, workspace, name):
        self.arguments = query, key, value, pages, requests, positions, selected, output
        from b12x.attention.paged._selected_forward import _output_gate_stride

        _output_gate_stride(output_gate, query)
        self.output_gate = output_gate
        self.descale = persistent_empty((1,), torch.float32, query.device).fill_(1)
        q = Query(tuple(tensor_facts(x) for x in (query, key, value, requests, selected)),
                  tensor_facts(output_gate) if output_gate is not None else None)

        device, heads, dim = query.device, query.shape[1], query.shape[2]

        def materialize(tactic):
            programs = compile_attention(q.operands, q.output_gate, tactic[1], device.index)
            load_programs(programs)
            return attach_programs(State(q, tactic, programs), *programs.values())

        self.plan = native_plan(contract=CONTRACT, query=q, device=query.device,
            scratch_bytes=lambda t: 0 if q.direct else q.rows * t[0] * heads * (dim + 1) * 4,
            materialize=materialize, compile_jobs=lambda t, d: (CompileJob.create(
                "ignition.components.selected_attention:compile_attention", q.operands, q.output_gate, t[1], d.ordinal),))

        declaration, declaration_gate = self.arguments, self.output_gate

        def call(state, benchmark=False):
            args, gate = declaration, declaration_gate
            produce = None
            if benchmark:
                def inputs():
                    a, k, v, pt, r, pos, ids, out = declaration
                    a, pt, r, pos, ids, out = (torch.empty_like(x) for x in (a, pt, r, pos, ids, out))
                    a.normal_(std=.2)
                    pt.zero_()
                    r.zero_()
                    length = min(4096, pt.shape[1] * key.shape[1])
                    pos.fill_(length - 1)
                    indices = torch.arange(ids.shape[-1], device=ids.device)
                    ids.copy_(torch.where(indices < length, indices, -1))
                    return (a, k, v, pt, r, pos, ids, out), a.clone()
                args, seed = benchmark_resource(session, ("selected_gqa", name), inputs)
                produce = lambda: args[0].copy_(seed)
                if gate is not None:
                    gate = benchmark_resource(session, ("selected_gqa_gate", name),
                        lambda: torch.empty_strided(output_gate.shape, output_gate.stride(),
                            device=output_gate.device, dtype=output_gate.dtype).normal_())
            nbytes = 0 if q.direct else q.rows * state.tactic[0] * query.shape[1] * (query.shape[2] + 1) * 4
            scratch = planned_scratch((scratch_buffer_spec(CONTRACT.component_id, nbytes=nbytes,
                                                          device=query.device),), workspace)
            return PreparedCall(run=lambda: _run(*args, gate, self.descale, scratch, state), output=args[-1], produce=produce,
                                owners=prime_owners(workspace, scratch, (args, gate)))

        declare(session, (self.plan.request(name=name, prepare_call=lambda state: replace(call(state, True), owners=()),
            benchmark_call=lambda state: call(state, True)),))
        register_finalizer(session, lambda: self._bind(workspace), workspace=workspace)

    def _bind(self, workspace):
        self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)

    def functional(self, *, session):
        self.cache = self.arguments[1:3]
        self.output_layout = tuple(self.arguments[-1].shape), tuple(self.arguments[-1].stride())
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.scratch_layout = tuple((tuple(spec.shape), spec.dtype) for spec in self.plan.scratch_specs())
        self.arguments = self.output_gate = self.scratch = None

    def __call__(self, query=None, pages=None, requests=None, positions=None, selected=None, *, output_gate=None):
        if self.arguments is not None:
            _execute(*self.arguments, self.output_gate, self.descale, self.scratch, self.plan.handle)
            return self.arguments[-1]
        output = torch.empty_strided(*self.output_layout, dtype=query.dtype, device=query.device).zero_()
        scratch, = (torch.empty(shape, dtype=dtype, device=query.device) for shape, dtype in self.scratch_layout)
        _execute(query, *self.cache, pages, requests, positions, selected, output, output_gate,
                 self.descale, scratch, self.plan.handle)
        return output
