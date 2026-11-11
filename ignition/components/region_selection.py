# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared b12x weighted representative scoring and stable region selection."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
from b12x._lib.compile_plan import attach_programs, compile_only_launches, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.attention.dsa_indexer.tiled_topk import run_row_topk
from b12x.attention.qsa._score_cute import (
    compile_score_representatives,
    launch_score_representatives,
)
from b12x.attention.qsa._stable_select_cute import (
    compile_stable_selection,
    launch_stable_selection,
)
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .native_operator import native_contract, native_plan
from .selected_attention import metadata_tensor, tensor_facts


@dataclass(frozen=True)
class Query:
    rows: int
    heads: int
    dimension: int
    regions: int
    ratio: int
    top_k: int
    cache: tuple
    table: tuple

    @property
    def tile(self):
        return min(self.rows, 64)

    @property
    def caps(self):
        return SimpleNamespace(index_heads=self.heads, index_head_dim=self.dimension,
            compress_ratio=self.ratio, compressed_page_size=self.cache[0][1], max_groups=self.regions,
            group_budget=self.top_k, dcp_size=1, dcp_rank=0, cp_kv_cache_interleave_size=self.ratio)

    @property
    def scratch_bytes(self):
        return self.tile * (self.regions + 3 * self.top_k + 2) * 4


CONTRACT = native_contract(component_id="flare.region_selection", query_type=Query, tactics=lambda q: (0,))


@program_cache(scope="preparation")
def compile_selection(payload, ordinal):
    q = Query(**dict(payload))
    cache, table = metadata_tensor(q.cache), metadata_tensor(q.table)
    def empty(shape, dtype=torch.float32):
        return torch.empty(shape, device="meta", dtype=dtype)
    query = empty((q.tile, q.heads, q.dimension), cache.dtype)
    weights = empty((q.tile, q.heads))
    positions, requests = empty((q.tile,), torch.int64), empty((q.tile,), torch.int32)
    lengths = empty((table.shape[0],), torch.int32)
    scores, counts = empty((q.tile, q.regions)), empty((q.tile,), torch.int32)
    merges = empty((q.tile,), torch.int32)
    top_values, top_ids = empty((q.tile, q.top_k)), empty((q.tile, q.top_k), torch.int32)
    with torch.cuda.device(ordinal), compile_only_launches():
        score = compile_score_representatives(prepared_query=query, weights=weights,
            query_positions=positions, request_ids=requests, sequence_lengths=lengths,
            compressed_cache=cache, compressed_block_table=table, scores=scores,
            eligible_counts=counts, merge_lengths=merges, caps=q.caps, position_offset=0,
            bounded=True, indices_only=True)
        topk = {}
        run_row_topk(row_logits=scores, lengths=counts, topk=q.top_k,
                     output_values=top_values, output_indices=top_ids, launcher_sink=topk)
        stable = compile_stable_selection(q.top_k, ordinal, input_order=True, indices_only=True)
    return {"score": score, "topk": topk, "stable": stable}


@dataclass(frozen=True)
class State:
    query: Query
    programs: dict


def _run(query, weights, cache, table, requests, positions, lengths, selected, scratch, state):
    q, p = state.query, state.programs
    storage = scratch.view(torch.float32)
    offset = 0
    def take(shape, dtype=torch.float32):
        nonlocal offset
        count = 1
        for size in shape:
            count *= size
        result = storage[offset:offset + count].view(dtype).view(shape)
        offset += count
        return result
    scores = take((q.tile, q.regions))
    values, ids = take((q.tile, q.top_k)), take((q.tile, q.top_k), torch.int32)
    stable_values = take((q.tile, q.top_k))
    counts, merges = take((q.tile,), torch.int32), take((q.tile,), torch.int32)
    for start in range(0, q.rows, q.tile):
        count = min(q.tile, q.rows - start)
        end = start + count
        launch_score_representatives(prepared_query=query[start:end], weights=weights[start:end],
            query_positions=positions[start:end], request_ids=requests[start:end], sequence_lengths=lengths,
            compressed_cache=cache, compressed_block_table=table, scores=scores[:count],
            eligible_counts=counts[:count], merge_lengths=merges[:count], caps=q.caps,
            position_offset=0, group_offset=0, group_count=q.regions, bounded=True,
            indices_only=True, _prepared=p["score"])
        run_row_topk(row_logits=scores[:count], lengths=counts[:count], topk=q.top_k,
                     output_values=values[:count], output_indices=ids[:count], launcher=p["topk"])
        launch_stable_selection(scores=scores[:count], merge_lengths=merges[:count],
            prior_ids=ids[:count], eligible_counts=counts[:count], topk_values=values[:count],
            stable_values=stable_values[:count], stable_ids=selected[start:end], group_offset=0,
            group_budget=q.top_k, input_order=True, indices_only=True, prepared=p["stable"])


@torch.library.custom_op("flare::region_selection", mutates_args=("selected", "scratch"))
def _execute(query: torch.Tensor, weights: torch.Tensor, cache: torch.Tensor, table: torch.Tensor,
             requests: torch.Tensor, positions: torch.Tensor, lengths: torch.Tensor,
             selected: torch.Tensor, scratch: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    _run(query, weights, cache, table, requests, positions, lengths, selected, scratch,
         require_prepared(plan, plan.component_id, query.device))


@_execute.register_fake
def _fake(query, weights, cache, table, requests, positions, lengths, selected, scratch, handle):
    pass


class PreparedRegionSelection:
    def __init__(self, query, weights, cache, metadata, selected, *, ratio, session, workspace, name, buffers=None):
        tokens, groups, heads, dim = query.shape
        self.metadata, self.groups = metadata, groups
        self.tokens, self.selected_shape, self.selected_dtype = tokens, tuple(selected.shape), selected.dtype
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=query.device) if buffers is None else buffers.zeros(shape, dtype)
        self.requests = zeros((tokens * groups,), torch.int32)
        self.positions = zeros((tokens * groups,), torch.int64)
        self.lengths = zeros(metadata.lengths.shape, metadata.lengths.dtype)
        self.arguments = (query.flatten(0, 1), weights.flatten(0, 1), cache, metadata.page_table,
                          self.requests, self.positions, self.lengths, selected.flatten(0, 1))
        q = Query(tokens * groups, heads, dim, metadata.page_table.shape[1] * cache.shape[1], ratio,
                  selected.shape[-1], tensor_facts(cache), tensor_facts(metadata.page_table))
        payload = CONTRACT.encode_query(q)

        def materialize(tactic):
            programs = compile_selection(payload, query.device.index)
            load_programs(programs)
            return attach_programs(State(q, programs), programs["score"], programs["stable"], *programs["topk"].values())

        self.plan = native_plan(contract=CONTRACT, query=q, device=query.device,
            scratch_bytes=lambda t: q.scratch_bytes, materialize=materialize,
            compile_jobs=lambda t, d: (CompileJob.create(
                "ignition.components.region_selection:compile_selection", payload, d.ordinal),))

        def call(state, benchmark=False):
            args = self.arguments
            produce = None
            if benchmark:
                def inputs():
                    a, w, cache, table, r, pos, lengths, ids = self.arguments
                    a, w, table, r, pos, lengths, ids = (torch.empty_like(x) for x in (a, w, table, r, pos, lengths, ids))
                    a.copy_(torch.randn(a.shape, device=a.device).to(a.dtype))
                    w.normal_()
                    table.zero_()
                    r.zero_()
                    pos.fill_(min(32768, q.regions * ratio - 1))
                    lengths.fill_(min(32768, q.regions * ratio))
                    return (a, w, cache, table, r, pos, lengths, ids), a.clone()
                args, seed = benchmark_resource(session, ("region_selection", name), inputs)
                produce = lambda: args[0].copy_(seed)
            scratch = planned_scratch(self.plan.memory_requirements().scratch, workspace)
            return PreparedCall(run=lambda: _run(*args, scratch, state), output=args[-1], produce=produce,
                                owners=prime_owners(workspace, scratch, args))

        declare(session, (self.plan.request(name=name, prepare_call=lambda state: replace(call(state, True), owners=()),
                                           benchmark_call=lambda state: call(state, True)),))
        register_finalizer(session, lambda: self._bind(workspace), workspace=workspace)

    def _bind(self, workspace):
        self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)

    def functional(self, *, session):
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.cache = self.arguments[2]
        self.scratch_layout = (tuple(self.scratch.shape), tuple(self.scratch.stride()), self.scratch.dtype)
        self.lengths_layout = (tuple(self.lengths.shape), self.lengths.dtype)
        self.metadata = self.requests = self.positions = self.lengths = self.arguments = self.scratch = None

    def __call__(self, query=None, weights=None, metadata=None):
        if self.arguments is None:
            device = query.device
            requests = metadata.requests[:, None].expand(self.tokens, self.groups).reshape(-1).to(torch.int32).contiguous()
            positions = metadata.positions[:, None].expand(self.tokens, self.groups).reshape(-1).to(torch.int64).contiguous()
            lengths = (metadata.starts + metadata.lengths).to(self.lengths_layout[1])
            selected = torch.empty(self.selected_shape, dtype=self.selected_dtype, device=device)
            shape, stride, dtype = self.scratch_layout
            scratch = torch.empty_strided(shape, stride, dtype=dtype, device=device)
            _execute(query.flatten(0, 1), weights.flatten(0, 1), self.cache, metadata.page_table,
                     requests, positions, lengths, selected.flatten(0, 1), scratch, self.plan.handle)
            return selected
        m = self.metadata
        self.requests.view(-1, self.groups).copy_(m.requests[:, None])
        self.positions.view(-1, self.groups).copy_(m.positions[:, None])
        self.lengths.copy_(m.starts + m.lengths)
        _execute(*self.arguments, self.scratch, self.plan.handle)
        return self.arguments[-1].view(self.selected_shape)
