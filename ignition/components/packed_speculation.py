# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One captured proposal, bounded causal allocation, and target commit.

The host supplies immutable per-request prefix ceilings and reserves their cache
extents before submission. Only the admitted lengths are device-owned; the
cohort, physical target capacity, page tables, and request epochs cannot change
inside the round.
"""

from dataclasses import replace

import torch

from .causal_prefix import PreparedCausalPrefixAllocation
from .collectives import PreparedAllReduce, PreparedBroadcast
from .split_draft import PreparedDraftConfidence


class PreparedPackedSpeculativeRound:
    """Compose proposal and target tensors within one captured round."""

    def __init__(self, source, target, prefix_limits, confidence_output, *, name):
        s, t = source, target
        if (s.batch != t.batch or s.owner.inputs is not t.owner.inputs
                or s.owner.requests is not t.owner.requests or not 1 < s.width <= t.width):
            raise ValueError("packed rounds require matching proposal and target inputs")
        o, b = s.owner, s.batch
        if (prefix_limits.shape != (o.requests.capacity,) or prefix_limits.dtype != torch.int32
                or prefix_limits.device != o.pool.backing.device or not prefix_limits.is_contiguous()):
            raise ValueError("prefix limits must be contiguous int32 values for every request lane")
        if (confidence_output.ndim != 3 or confidence_output.shape[0] != 2
                or confidence_output.shape[1] != o.requests.capacity or confidence_output.shape[2] < s.width - 1
                or confidence_output.dtype != torch.float32 or confidence_output.device != o.pool.backing.device
                or not confidence_output.is_contiguous()):
            raise ValueError("confidence output requires two contiguous FP32 request frames covering draft depth")
        query_lengths = o.inputs.query_lengths[:b]
        prefix_begin, query_begin = prefix_limits.data_ptr(), query_lengths.data_ptr()
        if (prefix_begin < query_begin + query_lengths.numel() * query_lengths.element_size()
                and query_begin < prefix_begin + prefix_limits.numel() * prefix_limits.element_size()):
            raise ValueError("host prefix limits must not overlap device-owned query lengths")
        self.source, self.target, self.prefix_limits = s, t, prefix_limits
        self.confidence_output = confidence_output
        device = o.pool.backing.device
        confidence = torch.empty((b, s.width - 1), dtype=torch.float32, device=device)
        self.confidence_head = PreparedDraftConfidence(s, o.weights.draft.confidence_head, confidence,
            name=name + ".confidence_head", bias=getattr(o.weights.draft, "confidence_bias", None))
        validation_input = torch.empty((2, o.requests.capacity), dtype=torch.bfloat16, device=device)
        self.validation_shape = tuple(validation_input.shape)
        self.validation = PreparedAllReduce(validation_input, torch.empty_like(validation_input),
            runtime=o.collectives["runtime"], coordinator=o.collectives["coordinator"],
            session=o.session, name=name + ".allocation_validation").functional(session=o.session)
        self.world = o.weights.shard.world_size
        slots = torch.empty((b,), dtype=torch.int64, device=device)
        submitted, source_valid, admitted = (torch.empty((b,), dtype=torch.bool, device=device) for _ in range(3))
        self.allocator = PreparedCausalPrefixAllocation(confidence, slots, submitted, source_valid,
            prefix_limits[:b], query_lengths, admitted, token_capacity=t.target_token_capacity,
            request_capacity=o.requests.capacity).functional()
        runtime = getattr(o, "result_runtime", None)
        if runtime is None:
            runtime = o.collectives["vocabulary_runtime"]
        self.broadcast = PreparedBroadcast((confidence,), runtime=runtime, session=o.session,
            coordinator=o.collectives["coordinator"], name=name + ".confidence").functional(session=o.session)

    def __call__(self):
        s, t = self.source, self.target
        o, batch = s.owner, s.batch
        slots = o.inputs.row_slot_ids[:batch].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        finished = ((slots >= 0) & (slots < o.requests.capacity)
            & (o.inputs.row_generations[:batch] == o.requests.generation[safe]) & (o.requests.status[safe] == 2))
        submitted = (torch.arange(batch, device=slots.device) < o.inputs.live_rows) & ~finished
        o.inputs.query_lengths[:batch].copy_(self.prefix_limits[:batch])
        probabilities = torch.empty(s.probability_shape, dtype=torch.float32, device=slots.device)
        proposal = s.propose(advance_rng=False, probabilities=probabilities)
        s.commit_draft_rng(proposal=proposal)
        confidence = self.confidence_head(proposal)
        source_valid = (proposal.active & torch.isfinite(confidence).all(1)
                        & (confidence >= 0).all(1) & (confidence <= 1).all(1))
        self.broadcast((confidence,))
        confidence = confidence.masked_fill(~submitted[:, None], 0.)
        self.confidence_output.index_copy_(0, o.frame.to(torch.int64),
            torch.nn.functional.pad(confidence, (0, self.confidence_output.shape[2] - confidence.shape[1],
                0, self.confidence_output.shape[1] - batch)).unsqueeze(0))
        admitted = self.allocator(confidence, slots, submitted, source_valid,
            self.prefix_limits[:batch], o.inputs.query_lengths[:batch])
        validation = torch.zeros(self.validation_shape, dtype=torch.bfloat16, device=slots.device)
        validation[0, :batch].copy_(admitted)
        agreement = self.validation(validation)
        admitted = admitted & (~submitted | (agreement[0, :batch] == self.world)).all()
        o.inputs.query_lengths[:batch].copy_(torch.where(admitted, o.inputs.query_lengths[:batch], 0))
        pad = t.width - s.width
        proposal = replace(proposal, active=proposal.active & admitted,
            candidates=torch.nn.functional.pad(proposal.candidates, (0, pad)),
            drafts=torch.nn.functional.pad(proposal.drafts, (0, pad)),
            valid=torch.nn.functional.pad(proposal.valid, (0, pad)))
        probabilities = torch.nn.functional.pad(probabilities, (0, 0, 0, pad))
        return t.verify(admitted, probabilities=probabilities, proposal=proposal)
