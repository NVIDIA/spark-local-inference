# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared FP8 sparse selection with shared scratch and caller-owned metadata."""

from functools import partial

import torch

from b12x.attention import dsa_indexer
from b12x.preparation import PreparedCall, plan_from_handle

from ignition.preparation.fixtures import indexer as fixtures
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer


@torch.library.custom_op("flare::sparse_indexer", mutates_args=("output", "scores", "scratch"))
def _select(query: torch.Tensor, weights: torch.Tensor, cache: torch.Tensor, pages: torch.Tensor,
            lengths: torch.Tensor, active: torch.Tensor | None, output: torch.Tensor, scores: torch.Tensor | None,
            scratch: torch.Tensor, handle: int) -> None:
    binding = dsa_indexer.bind(plan_from_handle(handle), scratch=scratch, q_fp8=query,
        query_weights=weights, index_k_cache=cache, page_table=pages, cache_lengths=lengths,
        active_width=active, output_indices=output, output_scores=scores)
    dsa_indexer.run(binding)


@_select.register_fake
def _fake_select(query, weights, cache, pages, lengths, active, output, scores, scratch, handle):
    pass


class PreparedSparseIndexer:
    """Select sparse cache records using the recipe's fixed geometry and index space.

    Prefill plans require one shared page table; decode plans accept a table per
    query row. Device lengths determine each row's visible candidate prefix.
    Inputs and output remain in caller-owned storage. Tuning uses private finite
    operands, then finalization binds the selected plan to the shared arena.
    """

    def __init__(self, query, weights, cache, page_table, lengths, active_width, output, *,
                 caps, session, name, workspace, scores=None):
        self.inputs = {"q_fp8": query, "query_weights": weights, "index_k_cache": cache,
            "page_table": page_table, "cache_lengths": lengths, "active_width": active_width,
            "output_indices": output, "output_scores": scores}
        self.output, self.scores = output, scores
        self.plan = dsa_indexer.plan(caps,
            invocation=dsa_indexer.invocation_from_tensors(caps, **self.inputs))

        declaration = self.inputs

        def prepare(state):
            storage = planned_scratch(state.layout.scratch_specs(), workspace)
            runtime = fixtures.bind(state, declaration, storage)
            return PreparedCall(run=lambda: fixtures.run(state, runtime, declaration),
                                owners=prime_owners(workspace, storage, runtime))

        declare(session, (self.plan.request(name=name, prepare_call=prepare,
            benchmark_call=partial(fixtures.benchmark_call, arguments=declaration,
                                   session=session, workspace=workspace)),))

        def finalize():
            if self.output is None:
                return
            self.binding = dsa_indexer.bind(self.plan,
                scratch=planned_scratch(self.plan.scratch_specs(), workspace), **declaration)

        self._finalize = finalize
        register_finalizer(session, finalize, workspace=workspace)

    def rebind_cache(self, cache):
        """Bind a replacement index cache before capture; its page count is a runtime bound."""
        self.inputs["index_k_cache"] = cache
        if self._finalize is not None:
            self._finalize()

    def functional(self, *, session):
        self.output_layout = tuple(self.output.shape), tuple(self.output.stride()), self.output.dtype
        self.score_layout = None if self.scores is None else (tuple(self.scores.shape), tuple(self.scores.stride()), self.scores.dtype)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.scratch_layout = tuple((tuple(spec.shape), spec.dtype) for spec in self.plan.scratch_specs())
        self.inputs = {"index_k_cache": self.inputs["index_k_cache"]}
        self.output = self.scores = self.binding = self._finalize = None

    def __call__(self, query=None, weights=None, page_table=None, lengths=None, active_width=None, *, cache=None):
        if self.output is not None:
            return dsa_indexer.run(self.binding)
        shape, stride, dtype = self.output_layout
        output = torch.empty_strided(shape, stride, dtype=dtype, device=query.device).fill_(-1)
        scores = None if self.score_layout is None else torch.empty_strided(self.score_layout[0], self.score_layout[1],
            dtype=self.score_layout[2], device=query.device).zero_()
        scratch, = (torch.empty(shape, dtype=dtype, device=query.device) for shape, dtype in self.scratch_layout)
        _select(query, weights, self.inputs["index_k_cache"] if cache is None else cache, page_table,
                lengths, active_width, output, scores, scratch, self.plan.handle)
        return output if scores is None else (output, scores)
