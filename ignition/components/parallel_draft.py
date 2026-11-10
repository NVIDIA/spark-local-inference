# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Parallel draft proposals, stochastic target verification and accepted-state commits."""

from dataclasses import dataclass

import torch

from .candidate_selector import PreparedCandidateSelector
from .collectives import PreparedAllReduce
from .draft_sampling import DRAFT_CANDIDATES, PreparedDraftDistribution, PreparedDraftBlockBroadcast
from .paged_batch import Metadata, PreparedBatch
from .speculation import (
    DRAFT_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
)
from .state import save_hidden


def prepare_sampling_scratch(batch, width, vocab, device, *, buffers=None, proposal=True, verification=True):
    """Share sequential sampling stages and retain target results through verification.

    Proposal probabilities must be copied to retained storage before the next
    proposal position. Target probabilities and validity stay disjoint from the
    verification workspace; filtering intermediates are dead before verification.
    """
    from ignition.preparation.scratch import bind_scratch

    layouts = {}
    if proposal and width > 1:
        layouts["proposal"] = PreparedDraftDistribution.scratch_layout(batch, 1, vocab)
    if verification:
        layouts["target"] = PreparedSpeculativeDistributions.scratch_layout(batch, width, vocab)
        layouts["verify"] = PreparedStochasticVerification.scratch_layout(batch, width, vocab)
    if buffers is None:
        return {name: bind_scratch(layout, device=device) for name, layout in layouts.items()}
    proposal_buffers = buffers.fork()
    retained = (bind_scratch({key: layouts["target"].pop(key) for key in ("probabilities", "valid")},
                            device=device, workspace=buffers) if verification else {})
    cursors = {name: proposal_buffers if name == "proposal" else buffers.fork() for name in layouts}
    result = {name: bind_scratch(layout, device=device, workspace=cursors[name])
              for name, layout in reversed(tuple(layouts.items()))}
    if verification:
        result["target"].update(retained)
    buffers.join(*cursors.values())
    return result


def prepare_sampling_broadcasts(proposals, verification, *, collectives, session, name, buffers=None,
                                result_runtime=None):
    """Synchronize proposals before dependent work and acceptance before state commits."""
    cursors = []
    operations = [(f"proposal{index}", proposal) for index, proposal in enumerate(proposals)]
    if verification is not None:
        operations.append(("acceptance", verification))
    runtime = collectives["vocabulary_runtime"] if result_runtime is None else result_runtime
    for suffix, operation in operations:
        cursor = None if buffers is None else buffers.fork()
        operation.prepare_broadcast(runtime=runtime, session=session,
            coordinator=collectives["coordinator"], name=f"{name}.{suffix}", buffers=cursor)
        if cursor is not None:
            cursors.append(cursor)
    if buffers is not None:
        buffers.join(*cursors)


@dataclass(frozen=True)
class DraftProposalState:
    before: torch.Tensor
    active: torch.Tensor
    candidates: torch.Tensor
    drafts: torch.Tensor
    valid: torch.Tensor
    base_offset: torch.Tensor
    metadata: Metadata | None


class PreparedParallelDraftRound:
    """Prepare verification and optionally its draft generator.

    A target-only forward receives proposals from its enclosing captured round;
    it allocates no draft decoder, Markov projection, or proposal samplers.
    """

    @staticmethod
    def storage_layout(batch, width, vocab):
        return dict(probabilities=((batch, width - 1, vocab), torch.float32),
                    logits=((batch, 1, vocab), torch.float32))

    @classmethod
    def allocate_buffers(cls, batch, width, vocab, device, *, buffers):
        return {}, {}

    def __init__(self, owner, batch, width, name, *, generate_proposals=True, buffers=None, proposal_only=False,
                 target_token_capacity=None, storage=None, forwards=None, verification_dummy=None):
        self.owner, self.batch, self.width = owner, batch, width
        self.local_metadata = getattr(owner, "local_metadata", False)
        self.vocabulary_parallel = getattr(owner, "vocabulary_parallel_sampling", False)
        self.name, self.buffers = name, buffers
        self.verification_dummy = verification_dummy
        tokens = batch * width if target_token_capacity is None else target_token_capacity
        if (type(tokens) is not int or not batch <= tokens <= batch * width
                or (proposal_only and target_token_capacity is not None)):
            raise ValueError("packed verification capacity must cover the request batch within its prefix bounds")
        self.target_token_capacity = tokens
        self.generate_proposals = generate_proposals
        self.proposal_only = proposal_only
        o, b, w = owner, batch, width
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        if storage is not None:
            self.sampling_scratch = storage[0]
        elif self.vocabulary_parallel:
            self.sampling_scratch = {}
        else:
            self.sampling_scratch = prepare_sampling_scratch(b, w, o.requests.vocab_size, device,
                buffers=buffers, proposal=generate_proposals, verification=not proposal_only)
        forwards = o.forwards if forwards is None else forwards
        self.rows = torch.arange(b, device=device)
        self.draft_positions = torch.arange(w - 1, device=device) if self.vocabulary_parallel else None
        self.metadata = Metadata.allocate(tokens, b, blocks, device, samples=b * w, max_query_tokens=w)
        self.validator = PreparedBatch(
            o.inputs,
            o.requests,
            Metadata.allocate(b, b, blocks, device),
            mode="decode",
            shard=o.weights.shard,
            pool=o.pool,
            verification_width=w,
            target_token_capacity=tokens,
            page_tokens=o.page_tokens,
        )
        self.validation = PreparedAllReduce(
            self.validator.validation_input,
            self.validator.validation_sum,
            runtime=o.collectives["runtime"],
            coordinator=o.collectives["coordinator"],
            session=o.session,
            name=name + ".validation",
        )
        self.decoder = None if proposal_only else forwards.decoder(
            o.weights,
            o.tables,
            o.scratch.rows(tokens, b * w),
            self.metadata,
            o.pool,
            mode="decode" if w == 1 else "verify",
            session=o.session,
            name=name + ".target",
            workspace=o.workspace,
            **o.collectives,
            **({"vocabulary_parallel": True} if self.vocabulary_parallel else {}),
        )
        self.context = None if proposal_only else forwards.context(
            o.weights,
            self.decoder.auxiliary,
            self.metadata,
            o.pool,
            session=o.session,
            name=name + ".context",
            workspace=o.workspace,
        )
        block = o.block_size
        self.draft_metadata = Metadata.allocate(b * block, b, blocks, device, samples=b * block)
        self.draft = (
            None
            if w == 1 or not generate_proposals
            else forwards.draft(
                o.weights,
                o.draft_scratch.rows(b * block, b * block),
                self.draft_metadata,
                o.pool,
                session=o.session,
                name=name + ".draft",
                workspace=o.workspace,
                **o.collectives,
                **({"vocabulary_parallel": True} if self.vocabulary_parallel else {}),
            )
        )
        self.selector = getattr(self.draft, "selector", None)
        self.draft_inputs = torch.full(
            (b, block), o.weights.config.draft.noise_token, dtype=torch.int64, device=device
        )
        self.markov_token = torch.zeros(b, dtype=torch.int64, device=device)
        self.markov = (
            None
            if self.draft is None or o.forwards.markov is None or self.vocabulary_parallel
            else o.forwards.markov(
                o.weights,
                self.markov_token,
                runtime=o.markov_runtime,
                coordinator=o.collectives["coordinator"],
                session=o.session,
                name=name + ".markov",
            )
        )
        if self.draft is not None and self.vocabulary_parallel:
            from .dspark import PreparedMarkovEmbedding

            self.markov = PreparedMarkovEmbedding(o.weights, self.markov_token,
                                                  session=o.session, name=name + ".markov")
        self.shortlisted_selector = (self.selector if self.markov is None
            and isinstance(self.selector, PreparedCandidateSelector) else None)
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        from ignition.preparation.scratch import bind_scratch

        local_vocab = o.scratch.local_logits.shape[-1] if self.vocabulary_parallel else o.requests.vocab_size
        layout = self.storage_layout(b, w, local_vocab)
        if not generate_proposals or self.vocabulary_parallel:
            del layout["logits"]
        retained = storage[1] if storage is not None else bind_scratch(layout, device=device, workspace=buffers)
        self.logits = retained.get("logits")
        if self.logits is None and "logits" in layout:
            shape, dtype = layout["logits"]
            self.logits = torch.empty(shape, dtype=dtype, device=device)
        self.probabilities = retained.get("probabilities")
        if self.probabilities is None:
            shape, dtype = layout["probabilities"]
            self.probabilities = torch.empty(shape, dtype=dtype, device=device)
        for tensor in retained.values():
            tensor.zero_()
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        self.seed, self.offset, self.base_offset, self.sampled = (
            torch.zeros_like(self.before) for _ in range(4)
        )
        self.operations = None
        self.proposal_block = None
        self.full_proposal = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        if self.vocabulary_parallel:
            self._prepare_vocabulary_parallel()
            return
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        target = None if self.proposal_only else PreparedSpeculativeDistributions(
            self.decoder.scratch.logits.view(b, w, -1), *args, self.drafts,
            workspace=self.sampling_scratch.get("target")
        )
        head = getattr(getattr(o.weights, "draft", None), "markov_head", None)
        from .markov_sampling import RowScaledFP8MarkovWeight

        # Full-vocabulary transition fusion improves one-row proposals. Wider
        # rows retain the bounded sampler because transition-weight reuse wins.
        fuse_markov = (b == 1 and w > 1 and self.generate_proposals
            and getattr(o, "block_proposal_broadcast", False) and o.markov_runtime is None
            and self.selector is None and self.markov is not None
            and ((isinstance(head, torch.Tensor) and head.dtype == torch.bfloat16 and head.is_contiguous())
                 or isinstance(head, RowScaledFP8MarkovWeight)))
        proposal_workspace = self.sampling_scratch.get("proposal")
        if proposal_workspace is None and "target" not in self.sampling_scratch and self.generate_proposals and w > 1:
            from ignition.preparation.scratch import bind_scratch

            proposal_workspace = bind_scratch(PreparedDraftDistribution.scratch_layout(b, 1, o.requests.vocab_size),
                                              device=self.logits.device)
        proposals = tuple(
            PreparedDraftDistribution(
                self.logits,
                *args,
                self.drafts,
                prefix_offset=i,
                seed=self.seed,
                offset=self.offset,
                sampled=self.sampled,
                workspace=proposal_workspace,
            )
            for i in range(w - 1 if self.generate_proposals and not fuse_markov else 0)
        )
        if target is not None:
            target.prepare()
        draft_logits_example = None
        shortlist_example = None
        if proposals and self.draft is not None:
            draft_logits_example = getattr(self.draft.scratch, "logits", None)
            if draft_logits_example is None:
                draft_logits_example = torch.empty(self.draft.scratch.logit_shape,
                                                   dtype=torch.float32, device=self.owner.requests.histogram.device)
            if self.shortlisted_selector is not None:
                shortlist_example = getattr(self.shortlisted_selector, "ids", None)
                if shortlist_example is None:
                    shortlist_example = torch.empty((b, self.draft_inputs.shape[1], self.shortlisted_selector.top_k),
                                                    dtype=torch.int64, device=draft_logits_example.device)
        for index, proposal in enumerate(proposals):
            bias = None if self.markov is None else getattr(self.markov, "logits", None)
            if self.shortlisted_selector is not None and self.shortlisted_selector.top_k <= DRAFT_CANDIDATES:
                proposal.bind_shortlist(shortlist_example[:, index + o.logit_offset])
            elif (self.draft is not None and self.selector is None
                    and (self.markov is None or bias is not None)):
                block = self.draft_inputs.shape[1]
                proposal.bind_logits(draft_logits_example.view(b, block, -1)[:, index + o.logit_offset],
                    bias)
            proposal.prepare()
        if fuse_markov:
            from .markov_sampling import PreparedMarkovProposals, PreparedDraftTokenBroadcast

            block = self.draft_inputs.shape[1]
            base = self.draft.scratch.logits.view(b, block, -1)[:, o.logit_offset:o.logit_offset + w - 1]
            self.full_proposal = PreparedMarkovProposals(base, o.weights.draft.markov_embedding, head,
                self.markov_token, self.drafts, self.probabilities, self.valid, o.requests,
                o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows,
                self.seed, self.base_offset, workspace=self.buffers)
            self.full_proposal.prepare()
            runtime = getattr(o, "result_runtime", None)
            if runtime is None:
                runtime = o.collectives["vocabulary_runtime"]
            self.proposal_block = PreparedDraftTokenBroadcast(self.drafts, self.valid, self.sampled,
                runtime=runtime, session=o.session, coordinator=o.collectives["coordinator"],
                name=self.name + ".proposal_tokens", buffers=self.buffers)
        broadcast_proposals = proposals
        if proposals and getattr(o, "block_proposal_broadcast", False):
            if o.markov_runtime is not None or self.selector is not None:
                raise ValueError("block proposal broadcast requires replicated Markov work without a selector")
            runtime = getattr(o, "result_runtime", None)
            if runtime is None:
                runtime = o.collectives["vocabulary_runtime"]
            self.proposal_block = PreparedDraftBlockBroadcast(
                proposals, self.drafts, self.probabilities, self.valid, self.sampled,
                runtime=runtime, session=o.session, coordinator=o.collectives["coordinator"],
                name=self.name + ".proposal_block", buffers=self.buffers)
            broadcast_proposals = ()
        if self.proposal_only:
            prepare_sampling_broadcasts(broadcast_proposals, None, collectives=o.collectives, session=o.session,
                                        name=self.name, buffers=self.buffers,
                                        result_runtime=getattr(o, "result_runtime", None))
            self.operations = None, proposals, None, None
            if self.local_metadata:
                self._release_local_metadata()
            return
        verify = PreparedStochasticVerification(
            self.probabilities,
            self.drafts,
            target.probabilities,
            *args,
            rng=o.rng,
            draft_valid=self.valid,
            target_valid=target.valid,
            token_limits=o.inputs.query_lengths[:b],
            workspace=self.sampling_scratch.get("verify"),
            dummy=self.verification_dummy,
        )
        verify.prepare()
        if "target" not in self.sampling_scratch:
            target.release_scratch()
            verify.release_scratch()
        prepare_sampling_broadcasts(broadcast_proposals, verify, collectives=o.collectives, session=o.session,
                                    name=self.name, buffers=self.buffers,
                                    result_runtime=getattr(o, "result_runtime", None))
        if "target" not in self.sampling_scratch:
            for proposal in proposals:
                proposal.release_scratch()
            if self.proposal_block is not None and not fuse_markov:
                self.proposal_block.probabilities = None
            self.probabilities = None
            self.logits = None
        from .speculation import PreparedSpeculativeStateCommit

        private = getattr(o, "study_width", None) is not None
        commit_type = PreparedSpeculativeStateCommit if private else PreparedSpeculativeCommit
        commit = commit_type(
            o.requests,
            verify.output,
            self.candidates,
            o.inputs.row_slot_ids[:b],
            o.inputs.row_generations[:b],
            o.inputs.live_rows,
            None if private else o.completion,
            o.frame,
            o.sequence,
            logits=None if private else self.decoder.scratch.logits.view(b, w, -1), rng=o.rng,
            draft_context_lag=0,
            token_limits=o.inputs.query_lengths[:b],
        )
        commit.prepare()
        if "target" not in self.sampling_scratch and commit.logprobs is not None:
            commit.logprobs.release_logits()
        release_outputs = getattr(self.decoder, "release_outputs", None)
        if release_outputs is not None:
            release_outputs()
        self.operations = target, proposals, verify, commit
        if self.local_metadata:
            self._release_local_metadata()

    def _release_local_metadata(self):
        if self.markov is not None or self.proposal_block is not None or self.full_proposal is not None:
            raise ValueError("local proposal metadata requires independently sampled draft positions")
        self.validator.functional()
        self.validation.functional(session=self.owner.session)
        if self.draft is not None:
            self.draft.functional()
        if self.context is not None:
            self.context.functional()
        target, proposals, verify, commit = self.operations
        for operation in (target, *proposals, verify, commit):
            if operation is not None:
                operation.functional(session=self.owner.session)
        self.metadata = self.draft_metadata = None
        self.rows = self.draft_positions = None
        for name in ("draft_inputs", "markov_token", "before", "active", "candidates", "drafts",
                     "valid", "seed", "offset", "base_offset", "sampled", "logits", "probabilities"):
            setattr(self, name, None)

    def _prepare_vocabulary_parallel(self):
        from .vocabulary_sampling import PreparedVocabularySoftmax, PreparedVocabularyBlockVerification

        o, b, w = self.owner, self.batch, self.width
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        start, _ = o.weights.shard.padded_partition(o.requests.vocab_size, alignment=64)
        options = dict(start=start, runtime=o.sampling_runtime, session=o.session,
                       coordinator=o.collectives["coordinator"])
        if self.draft is not None:
            local = self.draft.scratch.local_logits.view(b, o.block_size, -1)
            base = local[:, o.logit_offset:o.logit_offset + w - 1]
            self.full_proposal = PreparedVocabularySoftmax(base, self.probabilities, self.valid, *args,
                self.drafts, **options, name=self.name + ".proposal", workspace=self.buffers,
                embedding=o.weights.draft.markov_embedding, head=o.weights.draft.markov_head,
                pending=self.markov_token, seeds=self.seed, offsets=self.base_offset, tokens=self.drafts)
            self.full_proposal.prepare()
        if self.proposal_only:
            self.operations = None, (), None, None
            return
        logits = self.decoder.local_logits.view(b, w, -1)
        from ignition.preparation.scratch import bind_scratch

        retained = bind_scratch(dict(probabilities=(tuple(logits.shape), torch.float32),
            valid=((b, w), torch.bool)), device=logits.device, workspace=self.buffers)
        target = PreparedVocabularySoftmax(logits, retained["probabilities"], retained["valid"], *args,
            self.drafts, **options, name=self.name + ".target", workspace=self.buffers)
        target.prepare()
        verify = PreparedVocabularyBlockVerification(self.probabilities, self.drafts, target.probabilities, *args,
            **options, name=self.name + ".verification", rng=o.rng, draft_valid=self.valid,
            target_valid=target.valid, token_limits=o.inputs.query_lengths[:b], workspace=self.buffers)
        verify.prepare()
        from .vocabulary_logprobs import PreparedVocabularyParallelLogprobs

        logprobs = PreparedVocabularyParallelLogprobs(logits, o.completion,
            o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows, o.frame,
            vocabulary=o.requests.vocab_size, offset=start, runtime=o.logprob_runtime, session=o.session,
            coordinator=o.collectives["coordinator"], name=self.name + ".logprobs")
        commit = PreparedSpeculativeCommit(o.requests, verify.output, self.candidates,
            o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows,
            o.completion, o.frame, o.sequence, rng=o.rng, draft_context_lag=0,
            token_limits=o.inputs.query_lengths[:b], logprobs=logprobs)
        commit.prepare()
        self.operations = target, (), verify, commit

    def _round(self):
        probabilities = self.probabilities
        if probabilities is None:
            probabilities = torch.empty((self.batch, self.width - 1, self.owner.requests.vocab_size),
                                        dtype=torch.float32, device=self.owner.requests.histogram.device)
        proposal = self.propose(probabilities=probabilities)
        result = self.verify(probabilities=probabilities, proposal=proposal)
        return proposal, result

    def __call__(self):
        self._round()

    def observe(self):
        """Execute one round and return its metadata and acceptance for diagnostics."""
        proposal, result = self._round()
        if self.local_metadata:
            metadata, acceptance, committed = result
            draft_metadata = proposal.metadata
        else:
            metadata, draft_metadata = self.metadata, self.draft_metadata if self.draft is not None else None
            _, _, verification, committed = self.operations
            acceptance = verification.output
        values = {}
        for prefix, item in (("target", metadata), ("draft", draft_metadata)):
            if item is not None:
                for field in ("live_counts", "starts", "lengths", "query_start_loc"):
                    values[f"{prefix}.{field}"] = getattr(item, field)
        for field in ("offered_drafts", "accepted_drafts", "survival", "valid"):
            values[f"acceptance.{field}"] = getattr(acceptance, field)
        for field in ("accepted_drafts", "emitted_count", "emitted_ids"):
            values[f"commit.{field}"] = getattr(committed, field)
        return values

    def propose(self, advance_rng=True, observer=None, allowed=None, *, probabilities=None):
        if probabilities is None:
            probabilities = self.probabilities
        if not self.generate_proposals:
            raise RuntimeError("verification-only preparation cannot generate proposals")
        o, b, w = self.owner, self.batch, self.width
        target, proposals, verify, commit = self.operations
        if self.local_metadata:
            validation_input, present = self.validator.validate()
            validated = self.validator.begin(self.validation(validation_input), present=present)
            device = o.requests.histogram.device
            before = torch.empty(b, dtype=torch.int64, device=device)
            active = torch.empty(b, dtype=torch.bool, device=device)
            candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
            drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
            valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
            seed, offset, base_offset, sampled, markov_token = (
                torch.zeros(b, dtype=torch.int64, device=device) for _ in range(5))
            draft_inputs = torch.full((b, o.block_size), o.weights.config.draft.noise_token,
                                      dtype=torch.int64, device=device)
            draft_metadata = None
        else:
            self.validator.begin(self.validation(self.validator.validate()))
            before, active, candidates, drafts, valid = self.before, self.active, self.candidates, self.drafts, self.valid
            seed, offset, base_offset, sampled = self.seed, self.offset, self.base_offset, self.sampled
            markov_token, draft_inputs, draft_metadata = self.markov_token, self.draft_inputs, self.draft_metadata
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        active.copy_(validated.lengths > 0 if self.local_metadata else self.validator.token_state.active.bool())
        if allowed is not None:
            active.logical_and_(allowed)
        before.copy_(o.requests.consumed_tokens[safe])
        limits = torch.where(active, o.inputs.query_lengths[:b], 0)
        states = o.inputs.request_state_rows[safe, 0]
        candidates[:, 0].copy_(o.requests.pending_token[safe])
        base_offset.copy_(
            torch.where(o.generation[safe] == o.requests.generation[safe], o.offset[safe], 0)
        )
        seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        if self.draft is not None:
            block = draft_inputs.shape[1]
            draft_inputs[:, 0].copy_(candidates[:, 0])
            if self.local_metadata:
                draft_metadata = Metadata.pack(before, active.int() * block, states,
                    o.inputs.page_tables[safe], draft_inputs, o.inputs.live_rows,
                    token_capacity=b * block, sample_count=b * block)
                draft_result = self.draft(draft_metadata)
            else:
                draft_metadata.set(before, active.int() * block, states, o.inputs.page_tables[safe],
                                   draft_inputs, o.inputs.live_rows)
                draft_result = self.draft()
            if draft_result is None:
                draft_logits, selector_features = self.draft.scratch.logits, None
            else:
                draft_logits, selector_features = draft_result
            draft_logits = draft_logits.view(b, block, -1)
            unary = shortlist_ids = None
            if self.shortlisted_selector is not None:
                shortlist = self.shortlisted_selector.prepare_shortlists(draft_logits)
                if shortlist is None:
                    shortlist_ids = self.shortlisted_selector.ids
                else:
                    unary, shortlist_ids = shortlist
            markov_token.copy_(torch.where(active, candidates[:, 0], 0))
            if self.full_proposal is not None:
                self.full_proposal()
                if self.vocabulary_parallel:
                    positions = (torch.arange(w - 1, device=limits.device) if self.local_metadata
                                 else self.draft_positions)
                    valid.logical_and_(active[:, None] & (positions[None, :] < limits[:, None] - 1))
                    sampled.copy_(drafts[:, -1])
                    markov_token.copy_(sampled)
            for i, proposal in enumerate(proposals):
                position = i + o.logit_offset
                local_scratch = "proposal" not in self.sampling_scratch and not self.vocabulary_parallel
                if proposal.has_source_logits:
                    if self.markov is not None:
                        self.markov()
                    proposal_logits = draft_logits[:, position]
                else:
                    proposal_logits = draft_logits[:, position].float()
                    if not local_scratch:
                        self.logits[:, 0].copy_(proposal_logits)
                        proposal_logits = self.logits[:, 0]
                    if self.markov is not None:
                        proposal_logits = proposal_logits + self.markov() if local_scratch else proposal_logits.add_(self.markov())
                    if self.shortlisted_selector is not None:
                        if selector_features is None:
                            if local_scratch:
                                proposal_logits = proposal_logits.clone()
                            self.shortlisted_selector.apply_shortlist(proposal_logits, markov_token, position)
                        else:
                            proposal_logits = self.shortlisted_selector.apply_shortlist(
                                proposal_logits, markov_token, position, selector_features, unary, shortlist_ids)
                    elif self.selector is not None:
                        if selector_features is not None:
                            proposal_logits = self.selector.apply(proposal_logits, markov_token,
                                                                  position, selector_features)
                        else:
                            if local_scratch:
                                proposal_logits = proposal_logits.clone()
                            self.selector(proposal_logits, markov_token, position)
                if observer is not None:
                    observer.record_embedding(i, self.markov.embedding)
                offset.copy_(base_offset + 4 * i)
                if not local_scratch:
                    proposal_probabilities = proposal()
                else:
                    scratch = {name: torch.empty(shape, dtype=dtype, device=proposal_logits.device)
                               for name, (shape, dtype) in proposal.layout.items() if name != "valid"}
                    if self.local_metadata:
                        shape, dtype = proposal.layout["valid"]
                        scratch["valid"] = torch.empty(shape, dtype=dtype, device=proposal_logits.device)
                    else:
                        scratch["valid"] = proposal.valid
                    proposal_probabilities = proposal.run(scratch, proposal_logits.view(b, 1, -1),
                        source_logits=proposal_logits if proposal.has_source_logits else None,
                        shortlist_ids=None if shortlist_ids is None else shortlist_ids[:, position],
                        **(dict(candidates=drafts, seed=seed, offset=offset, sampled=sampled)
                           if self.local_metadata else {}))
                drafts[:, i].copy_(sampled)
                if self.proposal_block is None:
                    probabilities[:, i].copy_(proposal_probabilities[:, 0])
                    valid[:, i].copy_((scratch["valid"] if self.local_metadata else proposal.valid)[:, 0]
                                      & active & (i < limits - 1))
                else:
                    self.proposal_block.retain(i, proposal)
                markov_token.copy_(sampled)
            if self.proposal_block is not None:
                if self.probabilities is None:
                    self.proposal_block.finish(active, limits, probabilities)
                else:
                    self.proposal_block.finish(active, limits)
                markov_token.copy_(sampled)
            if advance_rng:
                self.commit_draft_rng(active=active, base_offset=base_offset)
        candidates[:, 1:].copy_(drafts)

        if self.local_metadata:
            return DraftProposalState(before, active, candidates, drafts, valid, base_offset, draft_metadata)

    def commit_draft_rng(self, drafts=None, *, active=None, base_offset=None):
        o, b, w = self.owner, self.batch, self.width
        drafts = w - 1 if drafts is None else drafts
        slots = o.inputs.row_slot_ids[:b].long()
        slot_ids = torch.arange(o.requests.capacity, device=slots.device)
        active = self.active if active is None else active
        base_offset = self.base_offset if base_offset is None else base_offset
        matches = active[None, :] & (slots[None, :] == slot_ids[:, None])
        base = torch.where(matches, base_offset[None, :], 0).sum(1)
        o.offset.copy_(torch.where(matches.any(1), base + 4 * drafts, o.offset))
        o.generation.copy_(torch.where(matches.any(1), o.requests.generation, o.generation))

    def verify(self, proposal_valid=None, *, probabilities=None, proposal=None):
        if probabilities is None:
            probabilities = self.probabilities
        o, b, w = self.owner, self.batch, self.width
        target, _, verify, commit = self.operations
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        state = proposal if self.local_metadata else self
        active = state.active if proposal_valid is None else state.active & proposal_valid
        limits = torch.where(active, o.inputs.query_lengths[:b], 0)
        states = o.inputs.request_state_rows[safe, 0]
        if self.local_metadata:
            metadata = Metadata.pack(state.before, limits, states, o.inputs.page_tables[safe],
                state.candidates, o.inputs.live_rows, token_capacity=self.target_token_capacity,
                sample_count=b * w, max_query_tokens=w)
            result = self.decoder(metadata)
        else:
            metadata = self.metadata
            metadata.set(state.before, limits, states, o.inputs.page_tables[safe], state.candidates, o.inputs.live_rows)
            result = self.decoder()
        if result is None:
            hidden, logits, auxiliary = self.decoder.scratch.hidden, self.decoder.scratch.logits, self.decoder.auxiliary
            records = None
        else:
            hidden, logits, auxiliary = result[:3]
            records = result[3] if len(result) > 3 else None
        if self.vocabulary_parallel or "target" in self.sampling_scratch:
            target()
            verify()
            acceptance = verify.output
        else:
            target_scratch = {name: torch.empty(shape, dtype=dtype, device=logits.device)
                              for name, (shape, dtype) in target.layout.items()}
            verify_scratch = {name: torch.empty(shape, dtype=dtype, device=logits.device)
                              for name, (shape, dtype) in verify.layout.items()}
            target.run(target_scratch, logits.view(b, w, -1),
                       **(dict(candidates=state.drafts) if self.local_metadata else {}))
            acceptance = verify.run(verify_scratch, probabilities, target_scratch["probabilities"],
                target_scratch["valid"], disjoint=True,
                **(dict(drafts=state.drafts, draft_valid=state.valid) if self.local_metadata else {}))
            if not self.local_metadata:
                acceptance = verify.output
        if proposal_valid is not None:
            acceptance.valid.logical_and_(proposal_valid)
        if self.local_metadata:
            committed = commit(logits.view(b, w, -1), acceptance=acceptance, inputs=state.candidates)
        else:
            if self.vocabulary_parallel or "target" in self.sampling_scratch:
                commit()
            else:
                commit(logits.view(b, w, -1))
            committed = commit
        counts = committed.emitted_count
        if w > 1:
            if self.local_metadata:
                self.decoder.commit(counts, records, metadata)
            elif result is None or len(result) <= 3:
                self.decoder.commit(counts)
            else:
                self.decoder.commit(counts, records)
        rows = torch.arange(b, device=hidden.device) if self.local_metadata else self.rows
        hidden = hidden.view(b, w, o.weights.config.text["hidden_size"])[
            rows, (counts - 1).long().clamp_min(0)
        ]
        save_hidden(hidden, o.pool.last_hidden, metadata.state_rows, counts > 0)
        if self.local_metadata:
            self.context(auxiliary, counts, metadata)
            return metadata, acceptance, committed
        self.context(auxiliary, counts)
