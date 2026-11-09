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


class PreparedOwnerTopK:
    """Select the same rank-major candidates on one owner per query.

    Input labels are already global int32 IDs, and score words are FP32.
    Candidate preprocessing and ordered cutoff selection match the replicated
    composition, including underfull rows, exceptional values and cutoff ties.
    Only selected IDs are broadcast; no scores are reduced or requantized.
    """

    def __init__(self, local_indices, local_scores, active, rank, output, *,
                 owner_runtime, reduction_runtime, session, coordinator, name, buffers, dependencies=()):
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
        owner_rows, width, topk = -(-rows // world), local_topk * world, output.shape[1]
        self.rows, self.owner_rows, self.width = rows, owner_rows, width
        self.rank, self.active, self.output = rank, active, output
        self.transport = PreparedCandidateOwnerStage(local_indices, local_scores,
            runtime=owner_runtime, session=session, coordinator=coordinator, name=name + ".stage",
            dependencies=dependencies)
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

    def __call__(self):
        indices, scores = self.transport()
        torch.library.wrap_triton(_owner_candidates)[(self.owner_rows, triton.cdiv(self.width, 1024))](
            indices, scores.view(torch.int32), self.active, self.rank, self.packed.view(torch.int32), self.owner_active,
            ROWS=self.rows, OWNER_ROWS=self.owner_rows, WIDTH=self.width, BLOCK=1024, num_warps=4)
        self.candidates()
        self.fold()
        return self.broadcast()
