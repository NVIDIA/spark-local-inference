# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared MXFP4 sparse selection and hierarchical candidate sharing."""

import torch
from functools import partial
from b12x.attention import dsa_indexer as dsa
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer
from ignition.preparation.fixtures.mxfp4_indexer import acquire_inputs, benchmark_call


@torch.library.custom_op("flare::index_quantize", mutates_args=("packed", "scales"))
def _quantize(query: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, handle: int) -> None:
    dsa.quantize_q_mxfp4(plan_from_handle(handle), query, q_mxfp4=packed, q_scales=scales)


@_quantize.register_fake
def _quantize_fake(query, packed, scales, handle):
    pass


@torch.library.custom_op("flare::index_cache_write", mutates_args=("cache",))
def _write(keys: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, handle: int) -> None:
    dsa.quantize_write_index_k_mxfp4(plan_from_handle(handle), keys, index_k_cache=cache, slot_mapping=slots)


@_write.register_fake
def _write_fake(keys, cache, slots, handle):
    pass


@torch.library.custom_op("flare::mxfp4_index", mutates_args=("output", "scratch", "candidates_out", "counts_out"))
def _select(query: torch.Tensor, scales: torch.Tensor, weights: torch.Tensor, cache: torch.Tensor,
            pages: torch.Tensor, lengths: torch.Tensor, active: torch.Tensor, output: torch.Tensor,
            scratch: torch.Tensor, candidates: torch.Tensor | None, counts: torch.Tensor | None,
            candidates_out: torch.Tensor | None, counts_out: torch.Tensor | None, handle: int) -> None:
    binding = dsa.bind(plan_from_handle(handle), scratch=scratch, q_mxfp4=query, q_scales=scales,
        query_weights=weights, index_k_cache=cache, page_table=pages, cache_lengths=lengths,
        active_width=active, output_indices=output, candidate_indices=candidates, candidate_lengths=counts,
        candidate_output=candidates_out, candidate_output_lengths=counts_out)
    dsa.run(binding)


@_select.register_fake
def _select_fake(query, scales, weights, cache, pages, lengths, active, output, scratch,
                 candidates, counts, candidates_out, counts_out, handle):
    pass


class PreparedMXFP4Indexer:
    def __init__(self, query, weights, cache, pages, lengths, active, output, *, page_size,
                 session, name, workspace, candidates=None, counts=None, candidates_out=None, counts_out=None,
                 buffers=None, mode="decode"):
        self.query, self.cache, self.output = query, cache, output
        # Each call quantizes the query into these before selecting, so a graph's `buffers` may hold them.
        packed, scales = (*query.shape[:-1], 64), (*query.shape[:-1], 4)
        if buffers is None:
            self.packed = torch.zeros(packed, dtype=torch.uint8, device=query.device)
            self.scales = torch.full(scales, 127, dtype=torch.uint8, device=query.device)
        else:
            self.packed = buffers.zeros(packed, torch.uint8)
            self.scales = buffers.full(scales, 127, torch.uint8)
        caps = dsa.Caps(device=query.device, num_q_heads=query.shape[1], max_q_rows=query.shape[0],
            max_page_table_width=pages.shape[1], topk=512, page_size=page_size, cache_format="mxfp4", mode=mode,
            max_candidates=0 if candidates is None else candidates.shape[1],
            candidate_topk_blocks=0 if candidates_out is None else candidates_out.shape[1] // 8)
        self.arguments = dict(q_mxfp4=self.packed, q_scales=self.scales, query_weights=weights,
            index_k_cache=cache, page_table=pages, cache_lengths=lengths, active_width=active,
            output_indices=output, candidate_indices=candidates, candidate_lengths=counts,
            candidate_output=candidates_out, candidate_output_lengths=counts_out)
        self.plan = dsa.plan(caps, invocation=dsa.invocation_from_tensors(caps, **self.arguments))
        self.workspace = workspace

        declaration = self.arguments

        def prepare(state):
            scratch = planned_scratch(state.layout.scratch_specs(), workspace)
            inputs = acquire_inputs(state, declaration, session)
            arguments = inputs.arguments
            binding = state.bind(scratch=scratch, **arguments)
            def run():
                state.quantize_query(inputs.query, q_mxfp4=arguments["q_mxfp4"], q_scales=arguments["q_scales"])
                state.run(binding)
            return PreparedCall(run=run, owners=prime_owners(workspace, inputs, scratch, binding))

        declare(session, (self.plan.request(name=name, prepare_call=prepare,
            benchmark_call=partial(benchmark_call, arguments=declaration, session=session, workspace=workspace)),))
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = planned_scratch(self.plan.scratch_specs(), self.workspace)

    def write_keys(self, keys, slots):
        _write(keys, self.cache, slots, self.plan.handle)

    def functional(self, *, session):
        self.layouts = {key: None if value is None else (tuple(value.shape), tuple(value.stride()), value.dtype)
                       for key, value in self.arguments.items()
                       if key in ("q_mxfp4", "q_scales", "output_indices", "candidate_output", "candidate_output_lengths")}
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.scratch_layout = tuple((tuple(spec.shape), spec.dtype) for spec in self.plan.scratch_specs())
        self.query = self.output = self.packed = self.scales = self.arguments = self.workspace = self.scratch = None

    def __call__(self, query=None, weights=None, pages=None, lengths=None, active=None, *, candidates=None, counts=None):
        if self.arguments is None:
            def allocate(name, fill):
                layout = self.layouts[name]
                return None if layout is None else torch.empty_strided(layout[0], layout[1], dtype=layout[2], device=query.device).fill_(fill)
            packed, scales = allocate("q_mxfp4", 0), allocate("q_scales", 127)
            output = allocate("output_indices", -1)
            candidates_out, counts_out = allocate("candidate_output", -1), allocate("candidate_output_lengths", 0)
            scratch, = (torch.empty(shape, dtype=dtype, device=query.device) for shape, dtype in self.scratch_layout)
            _quantize(query, packed, scales, self.plan.handle)
            _select(packed, scales, weights, self.cache, pages, lengths, active, output, scratch,
                    candidates, counts, candidates_out, counts_out, self.plan.handle)
            return output if candidates_out is None else (output, candidates_out, counts_out)
        _quantize(self.query, self.packed, self.scales, self.plan.handle)
        a = self.arguments
        _select(self.packed, self.scales, a["query_weights"], self.cache, a["page_table"],
            a["cache_lengths"], a["active_width"], self.output, self.scratch,
            a["candidate_indices"], a["candidate_lengths"], a["candidate_output"],
            a["candidate_output_lengths"], self.plan.handle)
        return self.output
