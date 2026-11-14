# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal MTP proposals, stochastic verification, and accepted-prefix state updates."""

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.paged_batch import Metadata, PreparedBatch
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
)
from ignition.components.state import save_hidden

class PreparedCausalDraftRound:
    def __init__(self, owner, batch, width, name, *, decoder_factory, context_factory,
                 proposal_factory, draft_context_lag=0):
        o, b, w = owner, batch, width
        self.batch, self.width = b, w
        self.blocks = o.inputs.page_tables.shape[1]
        self.draft_context_lag = draft_context_lag
        self.collectives, self.session, self.name = o.collectives, o.session, name
        self.inputs, self.requests, self.pool = o.inputs, o.requests, o.pool
        self.completion, self.frame, self.sequence = o.completion, o.frame, o.sequence
        self.rng, self.draft_offset, self.draft_generation = o.rng, o.offset, o.generation
        self.hidden = o.weights.config.hidden
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.rows = torch.arange(b, device=device)
        self.metadata = Metadata.allocate(b * w, b, blocks, device, samples=b * w)
        self.validator = PreparedBatch(o.inputs, o.requests, Metadata.allocate(b, b, blocks, device),
            mode="decode", shard=o.weights.shard, pool=o.pool, verification_width=w)
        self.validation = PreparedAllReduce(self.validator.validation_input, self.validator.validation_sum,
            runtime=o.collectives["runtime"], coordinator=o.collectives["coordinator"],
            session=o.session, name=name + ".validation").functional(session=o.session)
        common = dict(session=o.session, workspace=o.workspace, **o.collectives)
        scratch = o.scratch.rows(b * w, b * w)
        self.decoder = decoder_factory(o.weights, scratch, self.metadata, o.pool,
            mode="decode" if w == 1 else "verify", name=name + ".target", overlap=o.overlap, **common)
        self.context = context_factory(o.weights, self.metadata, o.pool, **common)
        self.draft = proposal_factory(o.weights, o.pool, o.draft_scratch, b, w - 1,
                                      name=name + ".draft", **common)
        def zeros(shape, dtype=torch.int64):
            return torch.zeros(shape, dtype=dtype, device=device)
        self.before = zeros((b,))
        self.active = zeros((b,), torch.bool)
        self.candidates = zeros((b, w))
        self.drafts = zeros((b, w - 1))
        self.probabilities = zeros((b, w - 1, o.requests.vocab_size), torch.float32)
        self.valid = zeros((b, w - 1), torch.bool)
        self.seed, self.offset, self.base_offset, self.sampled = (zeros((b,)) for _ in range(4))
        from ignition.preparation.scratch import bind_scratch
        layouts = {"target": PreparedSpeculativeDistributions.scratch_layout(b, w, o.requests.vocab_size),
                   "verify": PreparedStochasticVerification.scratch_layout(b, w, o.requests.vocab_size)}
        if w > 1:
            layouts["proposal"] = PreparedDraftDistribution.scratch_layout(b, 1, o.requests.vocab_size)
        self.sampling_scratch = {stage: bind_scratch(layout, device=device) for stage, layout in layouts.items()}
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        b, w = self.batch, self.width
        args = (self.requests, self.inputs.row_slot_ids[:b], self.inputs.row_generations[:b], self.inputs.live_rows)
        target = PreparedSpeculativeDistributions(self.decoder.scratch.logits.view(b, w, -1), *args,
            self.drafts, workspace=self.sampling_scratch["target"])
        draft_logits = torch.empty(self.draft.logit_shape, dtype=self.draft.logit_dtype, device=self.requests.generation.device)
        proposals = tuple(PreparedDraftDistribution(draft_logits.view(b, 1, -1), *args,
            self.drafts, prefix_offset=i, seed=self.seed, offset=self.offset, sampled=self.sampled,
            workspace=self.sampling_scratch["proposal"]) for i in range(w - 1))
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(self.probabilities, self.drafts, target.probabilities,
            *args, rng=self.rng, draft_valid=self.valid, target_valid=target.valid,
            token_limits=self.inputs.query_lengths[:b], workspace=self.sampling_scratch["verify"])
        verify.prepare()
        from ignition.components.parallel_draft import prepare_sampling_broadcasts
        prepare_sampling_broadcasts(proposals, verify, collectives=self.collectives, session=self.session,
                                    name=self.name)
        commit = PreparedSpeculativeCommit(self.requests, verify.output, self.candidates,
            self.inputs.row_slot_ids[:b], self.inputs.row_generations[:b], self.inputs.live_rows, self.completion, self.frame,
            self.sequence, logits=self.decoder.scratch.logits.view(b, w, -1), rng=self.rng, token_limits=self.inputs.query_lengths[:b], draft_context_lag=self.draft_context_lag)
        commit.prepare()
        self.sampling_layout = {stage: {name: (tuple(value.shape), value.dtype)
            for name, value in values.items()} for stage, values in self.sampling_scratch.items()}
        target.functional(session=self.session)
        for proposal in proposals:
            proposal.functional(session=self.session)
        verify.functional(session=self.session)
        commit.functional(session=self.session)
        self.validator.functional()
        self.context.functional()
        self.decoder.release_outputs()
        self.draft.release_outputs()
        self.metadata = self.rows = self.before = self.active = self.candidates = self.drafts = None
        self.probabilities = self.valid = self.seed = self.offset = self.base_offset = self.sampled = None
        self.sampling_scratch = None
        self.operations = target, proposals, verify, commit

    def __call__(self):
        b, w = self.batch, self.width
        target, proposals, verify, commit = self.operations
        validation, present = self.validator.validate()
        validated = self.validator.begin(self.validation(validation), present=present)
        slots = self.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, self.requests.capacity - 1)
        active = validated.lengths > 0
        before = self.requests.consumed_tokens[safe]
        limits = torch.where(active, self.inputs.query_lengths[:b], 0)
        states = self.inputs.request_state_rows[safe, 0]
        candidates = torch.zeros((b, w), dtype=torch.int64, device=slots.device)
        candidates[:, 0].copy_(self.requests.pending_token[safe])
        drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=slots.device)
        probabilities = torch.empty((b, w - 1, self.requests.vocab_size), dtype=torch.float32, device=slots.device)
        valid = torch.zeros((b, w - 1), dtype=torch.bool, device=slots.device)
        base_offset = torch.where(self.draft_generation[safe] == self.requests.generation[safe], self.draft_offset[safe], 0)
        seed = self.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN
        scratch = {stage: {name: torch.empty(shape, dtype=dtype, device=slots.device)
            for name, (shape, dtype) in values.items()} for stage, values in self.sampling_layout.items()}
        draft_state = self.draft.begin(torch.where(active, states, 0))
        for i, proposal in enumerate(proposals):
            tokens = candidates[:, 0] if i == 0 else drafts[:, i - 1]
            proposing = active & (i < limits - 1)
            logits, draft_state = self.draft.step(draft_state, i, tokens, before, states, active, proposing)
            sampled = torch.empty((b,), dtype=torch.int64, device=slots.device)
            proposal.run(scratch["proposal"], logits.view(b, 1, -1), candidates=drafts,
                seed=seed, offset=base_offset + 4 * i, sampled=sampled)
            drafts[:, i].copy_(sampled)
            probabilities[:, i].copy_(scratch["proposal"]["probabilities"][:, 0])
            valid[:, i].copy_(scratch["proposal"]["valid"][:, 0] & proposing)
        if w > 1:
            matches = active[None] & (slots[None] == torch.arange(self.requests.capacity, device=slots.device)[:, None])
            base = torch.where(matches, base_offset[None], 0).sum(1)
            self.draft_offset.copy_(torch.where(matches.any(1), base + 4 * (w - 1), self.draft_offset))
            self.draft_generation.copy_(torch.where(matches.any(1), self.requests.generation, self.draft_generation))
        self.draft.finish(draft_state, before, states, active)
        candidates[:, 1:].copy_(drafts)
        metadata = Metadata.allocate(b * w, b, self.blocks, slots.device, samples=b * w)
        metadata.set(before, limits, states, self.inputs.page_tables[safe], candidates, self.inputs.live_rows)
        hidden, logits, auxiliary, records = self.decoder(metadata)
        p = target.run(scratch["target"], logits.view(b, w, -1), candidates=drafts)
        target_valid = scratch["target"]["valid"]
        acceptance = verify.run(scratch["verify"], probabilities, p, target_valid,
            drafts=drafts, draft_valid=valid, disjoint=True)
        committed = commit(logits.view(b, w, -1), acceptance=acceptance, inputs=candidates)
        counts = committed.emitted_count
        if w > 1:
            self.decoder.commit(counts, records, metadata)
        rows = torch.arange(b, device=slots.device)
        hidden = hidden.view(b, w, self.hidden)[rows, (counts - 1).long().clamp_min(0)]
        self.context(auxiliary, counts, metadata)
        save_hidden(hidden, self.pool.last_hidden, metadata.state_rows, counts > 0)
        return metadata, acceptance, committed
