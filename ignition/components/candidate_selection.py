# SPDX-License-Identifier: Apache-2.0

"""Exact rank-local candidate merging with prepared communication alternatives."""

import torch

from ignition.preparation.tuning import register_finalizer

from .candidate_rows import PreparedCandidateRows
from .collectives import PreparedAllGather
from .ordered_row_topk import PreparedOrderedRowTopK
from .owner_topk import PreparedOwnerTopK


class PreparedCandidateSelection:
    """Normalize local IDs and select global rows without changing score bits.

    The replicated composition gathers every candidate on every rank. The
    owner composition selects each query on one rank and broadcasts final IDs.
    Both use the same candidate preprocessing, cutoff ties and emission order.
    Preparation binds one composition before model compilation; serving has
    no runtime choice. Serial calls may share storage and transport channels.
    """

    def __init__(self, indices, scores, positions, rank, output, *, world, gather_runtime,
                 owner_runtime, reduction_runtime, session, coordinator, name, buffers,
                 dependencies=(), alternatives=("replicated", "owner"), local_counts=None):
        rows, topk = indices.shape
        if (indices.dtype != torch.int32 or scores.shape != indices.shape or scores.dtype != torch.float32
                or positions.shape != (rows,) or positions.dtype != torch.int64
                or output.shape != (rows, topk) or output.dtype != torch.int32
                or (rank is None and world != 1)
                or (rank is not None and (rank.shape != (1,) or rank.dtype != torch.int64))
                or any(not value.is_contiguous() or value.device != indices.device
                       for value in (indices, scores, positions, rank, output) if value is not None)):
            raise ValueError("candidate merging requires matching contiguous local rows and global output IDs")
        if not alternatives or not set(alternatives) <= {"replicated", "owner"}:
            raise ValueError("candidate merging requires supported prepared alternatives")
        if (local_counts is not None and (local_counts.shape != (rows,) or local_counts.dtype != torch.int32
                or local_counts.device != indices.device or not local_counts.is_contiguous())):
            raise ValueError("candidate merging requires contiguous int32 counts per row")
        if local_counts is not None:
            start, end = local_counts.data_ptr(), local_counts.data_ptr() + local_counts.nbytes
            if any(start < tensor.data_ptr() + tensor.nbytes and tensor.data_ptr() < end
                   for tensor in (indices, scores, positions, rank, output) if tensor is not None):
                raise ValueError("candidate count scratch must not overlap inputs or output")
        self.indices, self.scores, self.positions, self.output = indices, scores, positions, output
        self.rank = 0 if rank is None else rank
        self.rows, self.world, self.topk = rows, world, topk
        self.dependencies = list(dependencies)
        self.alternatives = tuple(alternatives)
        self.implementation = None
        self.active = buffers.zeros((rows,), torch.bool)
        self.gather = self.owner = None
        if "replicated" in alternatives:
            self.packed = buffers.zeros((rows, topk, 2), torch.float32)
            self.gathered = buffers.zeros((rows, topk*world, 2), torch.float32)
            self.values = buffers.zeros((rows, topk*world), torch.float32)
            self.global_ids = buffers.zeros((rows, topk*world), torch.int32)
            self.lengths = buffers.zeros((rows,), torch.int32)
            self.selected_scores = buffers.zeros((rows, topk), torch.float32)
            self.gather = PreparedAllGather(self.packed.view(torch.bfloat16).view(rows, -1),
                self.gathered.view(torch.bfloat16).view(rows, -1), runtime=gather_runtime,
                session=session, coordinator=coordinator, name=name+".replicated.transport")
            self.gather.dependencies = self.dependencies
            self.candidates = PreparedCandidateRows(self.gathered, self.active, self.values,
                self.global_ids, self.lengths, topk=topk, buffers=buffers)
            self.fold = PreparedOrderedRowTopK(self.values, self.lengths, self.global_ids,
                self.selected_scores, output, session=session, name=name+".replicated.fold", buffers=buffers)
        if "owner" in alternatives:
            if owner_runtime is None or reduction_runtime is None or owner_runtime.world_size != world:
                raise ValueError("owner selection requires matching prepared transport owners")
            self.planar = buffers.zeros((2*rows, topk), torch.float32)
            self.owner_scores, self.owner_ids = self.planar[:rows], self.planar[rows:].view(torch.int32)
            self.owner = PreparedOwnerTopK(self.owner_ids, self.owner_scores, self.active, rank, output,
                owner_runtime=owner_runtime, reduction_runtime=reduction_runtime, session=session,
                coordinator=coordinator, name=name+".owner", buffers=buffers, local_counts=local_counts)
            self.owner.transport.dependencies = self.dependencies

    def bind(self, implementation):
        if implementation not in self.alternatives:
            raise ValueError("candidate selection was not prepared for this implementation")
        self.implementation = implementation

    def plans(self, implementation):
        if implementation == "replicated" and self.gather is not None:
            return (("transport", self.gather.plan), ("selection", self.fold.plan))
        if implementation == "owner" and self.owner is not None:
            return (("transport", self.owner.transport.plan), ("selection", self.owner.fold.plan),
                    ("replication", self.owner.broadcast.reduction.plan))
        raise ValueError("candidate selection implementation has no prepared plans")

    def functional(self, *, session):
        if self.gather is not None:
            self.gather.functional(session=session)
            self.candidates.functional()
            self.fold.functional(session=session)
        if self.owner is not None:
            self.owner.functional(session=session)
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        for name in ("indices", "scores", "positions", "output", "active", "packed", "gathered",
                     "values", "global_ids", "lengths", "selected_scores", "planar", "owner_scores", "owner_ids"):
            if hasattr(self, name):
                setattr(self, name, None)
        self.dependencies = []

    def pack(self, implementation=None, *, indices=None, scores=None):
        implementation = self.implementation if implementation is None else implementation
        dynamic = indices is not None
        indices = self.indices if indices is None else indices
        scores = self.scores if scores is None else scores
        live = indices >= 0
        if implementation == "owner":
            planar = (torch.empty((2 * self.rows, self.topk), dtype=torch.float32, device=scores.device)
                      if dynamic else self.planar)
            owner_scores, owner_ids = planar[:self.rows], planar[self.rows:].view(torch.int32)
            owner_ids.copy_(torch.where(live, indices * self.world + self.rank, -1).to(torch.int32))
            owner_scores.copy_(torch.where(live, scores, -torch.inf))
            return planar
        if implementation == "replicated":
            packed = (torch.empty((self.rows, self.topk, 2), dtype=torch.float32, device=scores.device)
                      if dynamic else self.packed)
            packed[:, :, 0].copy_(torch.where(live, scores, -torch.inf))
            packed.view(torch.int32)[:, :, 1].copy_(
                torch.where(live, indices * self.world + self.rank, -1).to(torch.int32))
            return packed
        raise RuntimeError("candidate selection must bind a composition before execution")

    def exchange(self, implementation=None, *, positions=None, packed=None, dependencies=()):
        implementation = self.implementation if implementation is None else implementation
        if packed is not None:
            active = positions >= 0
            if implementation == "owner":
                return self.owner(packed[self.rows:].view(torch.int32), packed[:self.rows],
                                  active, dependencies=dependencies)
            if implementation == "replicated":
                gathered = self.gather(packed.view(torch.bfloat16).view(self.rows, -1),
                                      dependencies=dependencies).view(torch.float32).view(self.rows, self.topk * self.world, 2)
                values, indices, lengths = self.candidates(gathered, active)
                return self.fold(values, lengths, indices)
        else:
            self.active.copy_((self.positions if positions is None else positions) >= 0)
            if implementation == "owner":
                return self.owner()
            if implementation == "replicated":
                self.gather(self.packed.view(torch.bfloat16).view(self.rows, -1))
                self.candidates()
                self.fold()
                return self.output
        raise RuntimeError("candidate selection must bind a composition before execution")


class CandidateSelectionForward:
    """One immutable composition used by preparation-time compiled trials."""

    def __init__(self, selection, implementation):
        self.selection, self.implementation = selection, implementation

    def __call__(self):
        self.selection.pack(self.implementation)
        return self.selection.exchange(self.implementation)
