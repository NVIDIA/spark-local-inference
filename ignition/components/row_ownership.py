# SPDX-License-Identifier: Apache-2.0
"""Prepare exact candidate transport and byte-preserving row-owner broadcasts."""

import itertools
import weakref

import torch
import triton
import triton.language as tl
from b12x.comm.pcie import _owner_preparation

from ignition.components.collectives import PreparedAllReduce
from ignition.preparation.tuning import declare, register_finalizer

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


@torch.library.custom_op("flare::candidate_owner_stage", mutates_args=("indices", "scores"))
def _stage(local_indices: torch.Tensor, local_scores: torch.Tensor,
           indices: torch.Tensor, scores: torch.Tensor, dependencies: list[torch.Tensor], handle: int) -> None:
    operation = _OWNERS[handle]
    if (indices.data_ptr() != operation.indices.data_ptr()
            or scores.data_ptr() != operation.scores.data_ptr()):
        raise ValueError("candidate-stage outputs must alias the reserved channel views")
    with operation.runtime.capture(plan=operation.plan):
        operation.runtime.stage_candidates(local_indices, local_scores, plan=operation.plan)


@_stage.register_fake
def _fake_stage(local_indices, local_scores, indices, scores, dependencies, handle):
    pass


@torch.library.custom_op("flare::candidate_owner_stage_counted", mutates_args=("indices", "scores", "counts"))
def _stage_counted(local_indices: torch.Tensor, local_scores: torch.Tensor, local_counts: torch.Tensor,
                   indices: torch.Tensor, scores: torch.Tensor, counts: torch.Tensor,
                   dependencies: list[torch.Tensor], handle: int) -> None:
    operation = _OWNERS[handle]
    if (indices.data_ptr() != operation.indices.data_ptr()
            or scores.data_ptr() != operation.scores.data_ptr()
            or counts.data_ptr() != operation.counts.data_ptr()):
        raise ValueError("counted candidate-stage outputs must alias the reserved channel views")
    with operation.runtime.capture(plan=operation.plan):
        operation.runtime.stage_candidates(local_indices, local_scores, local_counts, plan=operation.plan)


@_stage_counted.register_fake
def _fake_stage_counted(local_indices, local_scores, local_counts, indices, scores, counts, dependencies, handle):
    pass


class PreparedCandidateOwnerStage:
    """Deliver rank-major score/ID rows to their contiguous row partitions.

    Every rank declares and executes the same call. Outputs alias fixed channel
    storage; their consumers must complete before another call reuses it.
    """

    def __init__(self, local_indices, local_scores, *, runtime, session, coordinator, name,
                 local_counts=None, dependencies=()):
        self.runtime = runtime
        self.local_indices, self.local_scores, self.local_counts = local_indices, local_scores, local_counts
        self.dependencies = list(dependencies)
        self.indices, self.scores = runtime.graph_candidate_views(local_indices.shape[0])
        self.counts = None if local_counts is None else runtime.graph_count_views(local_indices.shape[0])
        call = dict(local_indices=local_indices, local_scores=local_scores, local_counts=local_counts)
        self.plan = _owner_preparation.plan(
            _owner_preparation.query_from_runtime(runtime, call=call), runtime=runtime)
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: _owner_preparation.prepared_call(state, **call),
            collective=_owner_preparation.collective(runtime, key=name)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.local_indices = self.local_scores = self.local_counts = None
        self.dependencies = []

    def __call__(self, local_indices=None, local_scores=None, local_counts=None, *, dependencies=()):
        local_indices = self.local_indices if local_indices is None else local_indices
        local_scores = self.local_scores if local_scores is None else local_scores
        if self.counts is not None:
            local_counts = self.local_counts if local_counts is None else local_counts
            _stage_counted(local_indices, local_scores, local_counts,
                           self.indices, self.scores, self.counts,
                           [*self.dependencies, *dependencies], self.handle)
            return self.indices, self.scores, self.counts
        _stage(local_indices, local_scores, self.indices, self.scores,
               [*self.dependencies, *dependencies], self.handle)
        return self.indices, self.scores


@triton.jit
def _encode_owned_rows(Source, Rank, Send, ROWS: tl.constexpr, OWNER_ROWS: tl.constexpr,
                       ROW_BYTES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    local_row = row - tl.load(Rank) * OWNER_ROWS
    value = tl.full((BLOCK,), 0, tl.uint8)
    if (local_row >= 0) & (local_row < OWNER_ROWS):
        value = tl.load(Source + local_row * ROW_BYTES + column, column < ROW_BYTES, other=0)
    tl.store(Send + row * ROW_BYTES + column, value.to(tl.float32), column < ROW_BYTES)


@triton.jit
def _decode_owned_rows(Received, Output, ROW_BYTES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Received + row * ROW_BYTES + column, column < ROW_BYTES, other=0)
    tl.store(Output + row * ROW_BYTES + column, value.to(tl.uint8), column < ROW_BYTES)


class PreparedOwnedRowBroadcast:
    """Copy contiguous row partitions to every rank without changing payload bits.

    Source rows belong to the channel rank: rank r supplies rows starting at
    r*ceil(total_rows/world). The supplied int64 rank tensor must identify that
    rank on the device. Each byte is represented as a BF16 integer in [0,255];
    all other ranks contribute zero. Reduction is therefore exact, including
    payloads that represent negative integers, signed zero or NaNs.
    """

    def __init__(self, source, output, rank, *, runtime, session, coordinator, name, buffers=None):
        world = 1 if runtime is None else runtime.world_size
        if (source.ndim != 2 or output.ndim != 2 or min(output.shape) <= 0
                or source.shape != (-(-output.shape[0] // world), output.shape[1])
                or source.dtype != output.dtype or source.device != output.device
                or not source.is_contiguous() or not output.is_contiguous()
                or rank.shape != (1,) or rank.dtype != torch.int64 or rank.device != source.device):
            raise ValueError("owned-row broadcast requires contiguous matching row partitions and an int64 group rank")
        self.source, self.output, self.rank = source, output, rank
        self.rows, self.owner_rows = output.shape[0], source.shape[0]
        self.row_bytes = output.shape[1] * output.element_size()
        if self.row_bytes % 16:
            raise ValueError("owned-row payload bytes must be a multiple of 16")
        def allocate():
            shape = (self.rows, self.row_bytes)
            return (torch.empty(shape, dtype=torch.bfloat16, device=source.device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))
        self.send, self.received = allocate(), allocate()
        self.reduction = PreparedAllReduce(self.send, self.received, runtime=runtime,
            session=session, coordinator=coordinator, name=name)

    def functional(self, *, session):
        self.output_layout = tuple(self.output.shape), self.output.dtype
        self.source = self.output = self.send = self.received = None
        self.reduction.functional(session=session)
        return self

    def __call__(self, source=None):
        if source is None:
            source, output, send = self.source, self.output, self.send
        else:
            shape, dtype = self.output_layout
            output = torch.empty(shape, dtype=dtype, device=source.device)
            send = torch.empty((self.rows, self.row_bytes), dtype=torch.bfloat16, device=source.device)
        grid = (self.rows, triton.cdiv(self.row_bytes, 1024))
        torch.library.wrap_triton(_encode_owned_rows)[grid](
            source.view(torch.uint8), self.rank, send,
            ROWS=self.rows, OWNER_ROWS=self.owner_rows, ROW_BYTES=self.row_bytes, BLOCK=1024, num_warps=4)
        received = self.reduction(send)
        torch.library.wrap_triton(_decode_owned_rows)[grid](
            received, output.view(torch.uint8), ROW_BYTES=self.row_bytes, BLOCK=1024, num_warps=4)
        return output
