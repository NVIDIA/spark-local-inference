# SPDX-License-Identifier: Apache-2.0
"""Distribute exact candidate selection by query row and replicate final IDs."""

import torch
import triton
import triton.language as tl

from ignition.components.candidate_rows import PreparedCandidateRows
from ignition.components.ordered_row_topk import PreparedOrderedRowTopK
from ignition.components.row_ownership import PreparedCandidateOwnerStage, PreparedOwnedRowBroadcast


@triton.jit
def _owner_candidates(Indices, ScoreBits, Active, Rank, Packed, OwnerActive,
                      ROWS: tl.constexpr, OWNER_ROWS: tl.constexpr,
                      WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    global_row = tl.load(Rank) * OWNER_ROWS + row
    live = tl.load(Active + global_row, global_row < ROWS, other=0) != 0
    ids = tl.load(Indices + row * WIDTH + column, live & (column < WIDTH), other=-1)
    bits = tl.load(ScoreBits + row * WIDTH + column, live & (column < WIDTH), other=-8388608)
    tl.store(Packed + (row * WIDTH + column) * 2, bits, column < WIDTH)
    tl.store(Packed + (row * WIDTH + column) * 2 + 1, ids, column < WIDTH)
    if tl.program_id(1) == 0:
        tl.store(OwnerActive + row, live)


@triton.jit
def _owner_candidates_counted(Indices, ScoreBits, Counts, Active, Rank, Packed, OwnerActive,
                              ROWS: tl.constexpr, OWNER_ROWS: tl.constexpr, WORLD: tl.constexpr,
                              TOPK: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    global_row = tl.load(Rank) * OWNER_ROWS + row
    active = tl.load(Active + global_row, global_row < ROWS, other=0) != 0
    peer = column // TOPK
    offset = column % TOPK
    count = tl.load(Counts + row * WORLD + peer, mask=column < WIDTH, other=0)
    count = tl.maximum(0, tl.minimum(count, TOPK))
    live = active & (offset < count)
    ids = tl.load(Indices + row * WIDTH + column, live, other=-1)
    bits = tl.load(ScoreBits + row * WIDTH + column, live, other=-8388608)
    tl.store(Packed + (row * WIDTH + column) * 2, bits, column < WIDTH)
    tl.store(Packed + (row * WIDTH + column) * 2 + 1, ids, column < WIDTH)
    if tl.program_id(1) == 0:
        tl.store(OwnerActive + row, active)


def _candidate_transport_counts(indices, scores, output):
    """Find the last position that is not canonical padding on every row.

    A nonprefix valid ID, malformed entry or nonfinite score extends the
    transported prefix. Only a canonical (-1, -inf) suffix is elided.
    """
    columns = torch.arange(1, indices.shape[1] + 1, dtype=torch.int32, device=indices.device)
    noncanonical = (indices != -1) | (scores != -torch.inf)
    output.copy_(torch.where(noncanonical, columns, 0).amax(dim=1))


class PreparedOwnerTopK:
    """Select the same rank-major candidates on one owner per query.

    Input labels are already global int32 IDs, and score words are FP32.
    Candidate preprocessing and ordered cutoff selection match the replicated
    composition, including underfull rows, exceptional values and cutoff ties.
    Only selected IDs are broadcast; no scores are reduced or requantized.
    """

    def __init__(self, local_indices, local_scores, active, rank, output, *,
                 owner_runtime, reduction_runtime, session, coordinator, name, buffers,
                 dependencies=(), local_counts=None):
        rows, local_topk = local_indices.shape
        world = owner_runtime.world_size
        if (output.ndim != 2 or output.shape[0] != rows or output.dtype != torch.int32
                or active.shape != (rows,) or active.dtype != torch.bool
                or rank.shape != (1,) or rank.dtype != torch.int64
                or local_scores.shape != local_indices.shape
                or local_indices.dtype != torch.int32 or local_scores.dtype != torch.float32
                or reduction_runtime.world_size != world
                or reduction_runtime.rank != owner_runtime.rank
                or any(t.device != local_indices.device or not t.is_contiguous()
                       for t in (local_indices, local_scores, active, rank, output))):
            raise ValueError("owner selection requires matching candidate rows and communication ranks")
        if (local_counts is not None and (local_counts.shape != (rows,) or local_counts.dtype != torch.int32
                or local_counts.device != local_indices.device or not local_counts.is_contiguous())):
            raise ValueError("owner selection requires contiguous int32 candidate counts per row")
        if local_counts is not None:
            start, end = local_counts.data_ptr(), local_counts.data_ptr() + local_counts.nbytes
            if any(start < tensor.data_ptr() + tensor.nbytes and tensor.data_ptr() < end
                   for tensor in (local_indices, local_scores, active, rank, output)):
                raise ValueError("candidate count scratch must not overlap inputs or output")
        owner_rows, width, topk = -(-rows // world), local_topk * world, output.shape[1]
        self.rows, self.owner_rows, self.width = rows, owner_rows, width
        self.world, self.local_topk = world, local_topk
        self.rank, self.active, self.output = rank, active, output
        self.transport = PreparedCandidateOwnerStage(local_indices, local_scores,
            runtime=owner_runtime, session=session, coordinator=coordinator, name=name + ".stage",
            dependencies=dependencies, local_counts=local_counts)
        self.packed = buffers.zeros((owner_rows, width, 2), torch.float32)
        self.owner_active = buffers.zeros((owner_rows,), torch.bool)
        self.global_scores = buffers.zeros((owner_rows, width), torch.float32)
        self.global_ids = buffers.zeros((owner_rows, width), torch.int32)
        self.lengths = buffers.zeros((owner_rows,), torch.int32)
        self.selected_scores = buffers.zeros((owner_rows, topk), torch.float32)
        self.selected_ids = buffers.zeros((owner_rows, topk), torch.int32)
        self.candidates = PreparedCandidateRows(self.packed, self.owner_active,
            self.global_scores, self.global_ids, self.lengths, topk=topk, buffers=buffers)
        self.fold = PreparedOrderedRowTopK(self.global_scores, self.lengths, self.global_ids,
            self.selected_scores, self.selected_ids, session=session, name=name + ".fold", buffers=buffers)
        self.broadcast = PreparedOwnedRowBroadcast(self.selected_ids, output, rank,
            runtime=reduction_runtime, session=session, coordinator=coordinator,
            name=name + ".selected", buffers=buffers)

    def functional(self, *, session):
        self.transport.functional(session=session)
        self.candidates.functional()
        self.fold.functional(session=session)
        self.broadcast.functional(session=session)
        for name in ("active", "output", "packed", "owner_active", "global_scores", "global_ids",
                     "lengths", "selected_scores", "selected_ids"):
            setattr(self, name, None)
        return self

    def __call__(self, local_indices=None, local_scores=None, active=None, *, dependencies=()):
        local_indices = self.transport.local_indices if local_indices is None else local_indices
        local_scores = self.transport.local_scores if local_scores is None else local_scores
        if self.transport.counts is None:
            indices, scores = self.transport(local_indices, local_scores, dependencies=dependencies)
            counts = None
        else:
            local_counts = (self.transport.local_counts if active is None else
                            torch.empty((self.rows,), dtype=torch.int32, device=local_indices.device))
            _candidate_transport_counts(local_indices, local_scores, local_counts)
            indices, scores, counts = self.transport(local_indices, local_scores, local_counts,
                                                     dependencies=dependencies)
        if active is None:
            active, packed, owner_active = self.active, self.packed, self.owner_active
        else:
            packed = torch.empty((self.owner_rows, self.width, 2), dtype=torch.float32, device=indices.device)
            owner_active = torch.empty((self.owner_rows,), dtype=torch.bool, device=indices.device)
        if counts is None:
            torch.library.wrap_triton(_owner_candidates)[(self.owner_rows, triton.cdiv(self.width, 1024))](
                indices, scores.view(torch.int32), active, self.rank, packed.view(torch.int32), owner_active,
                ROWS=self.rows, OWNER_ROWS=self.owner_rows, WIDTH=self.width, BLOCK=1024, num_warps=4)
        else:
            torch.library.wrap_triton(_owner_candidates_counted)[(self.owner_rows, triton.cdiv(self.width, 1024))](
                indices, scores.view(torch.int32), counts, active, self.rank, packed.view(torch.int32), owner_active,
                ROWS=self.rows, OWNER_ROWS=self.owner_rows, WORLD=self.world,
                TOPK=self.local_topk, WIDTH=self.width, BLOCK=1024, num_warps=4)
        if self.packed is not None:
            self.candidates()
            self.fold()
            return self.broadcast()
        values, ids, lengths = self.candidates(packed, owner_active)
        selected = self.fold(values, lengths, ids)
        return self.broadcast(selected)
