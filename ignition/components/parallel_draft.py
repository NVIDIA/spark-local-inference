# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Parallel draft proposals, stochastic target verification and accepted-state commits."""

from dataclasses import dataclass

import torch

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


def prepare_sampling_broadcasts(proposals, verification, *, collectives, session, name, result_runtime=None):
    """Synchronize proposals before dependent work and acceptance before state commits."""
    operations = [(f"proposal{index}", proposal) for index, proposal in enumerate(proposals)]
    if verification is not None:
        operations.append(("acceptance", verification))
    runtime = collectives["vocabulary_runtime"] if result_runtime is None else result_runtime
    for suffix, operation in operations:
        operation.prepare_broadcast(runtime=runtime, session=session,
            coordinator=collectives["coordinator"], name=f"{name}.{suffix}")


@dataclass(frozen=True)
class DraftProposalState:
    before: torch.Tensor
    active: torch.Tensor
    candidates: torch.Tensor
    drafts: torch.Tensor
    valid: torch.Tensor
    base_offset: torch.Tensor
    metadata: Metadata | None
    features: dict[str, torch.Tensor] | None = None


class PreparedParallelDraftRound:
    """Prepare verification and optionally its draft generator.

    A target-only forward receives proposals from its enclosing captured round;
    it allocates no draft decoder, Markov projection, or proposal samplers.
    """

    def __init__(self, owner, batch, width, name, *, generate_proposals=True, proposal_only=False,
                 target_token_capacity=None, forwards=None, verification_dummy=None):
        self.owner, self.batch, self.width = owner, batch, width
        self.vocabulary_parallel = getattr(owner, "vocabulary_parallel_sampling", False)
        self.name = name
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
            and callable(getattr(self.selector, "prepare_shortlists", None)) else None)
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)

        local_vocab = o.scratch.local_logits.shape[-1] if self.vocabulary_parallel else o.requests.vocab_size
        self.probability_shape = (b, w - 1, local_vocab)
        self.logits = (torch.zeros((b, 1, local_vocab), dtype=torch.float32, device=device)
                       if generate_proposals and not self.vocabulary_parallel else None)
        self.probabilities = torch.zeros(self.probability_shape, dtype=torch.float32, device=device)
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
            workspace=None
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
        proposals = tuple(
            PreparedDraftDistribution(
                self.logits,
                *args,
                self.drafts,
                prefix_offset=i,
                seed=self.seed,
                offset=self.offset,
                sampled=self.sampled,
                workspace=None,
            )
            for i in range(w - 1 if self.generate_proposals and not fuse_markov else 0)
        )
        if target is not None:
            target.prepare()
        draft_logits_example = None
        shortlist_example = None
        if self.draft is not None and (proposals or fuse_markov):
            draft_logits_example = torch.empty(self.draft.logit_shape, dtype=self.draft.logit_dtype,
                                               device=self.owner.requests.histogram.device)
            if self.shortlisted_selector is not None:
                shortlist_example = torch.empty((b, self.draft_inputs.shape[1], self.shortlisted_selector.top_k),
                                                dtype=torch.int64, device=draft_logits_example.device)
        for index, proposal in enumerate(proposals):
            bias = (None if self.markov is None else torch.empty((b, o.requests.vocab_size),
                dtype=torch.float32, device=self.owner.requests.histogram.device))
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
            base = draft_logits_example.view(b, block, -1)[:, o.logit_offset:o.logit_offset + w - 1]
            self.full_proposal = PreparedMarkovProposals(base, o.weights.draft.markov_embedding, head,
                self.markov_token, self.drafts, self.probabilities, self.valid, o.requests,
                o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows,
                self.seed, self.base_offset, workspace=None)
            self.full_proposal.prepare()
            runtime = getattr(o, "result_runtime", None)
            if runtime is None:
                runtime = o.collectives["vocabulary_runtime"]
            self.proposal_block = PreparedDraftTokenBroadcast(self.drafts, self.valid, self.sampled,
                runtime=runtime, session=o.session, coordinator=o.collectives["coordinator"],
                name=self.name + ".proposal_tokens")
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
                name=self.name + ".proposal_block")
            broadcast_proposals = ()
        if self.proposal_only:
            prepare_sampling_broadcasts(broadcast_proposals, None, collectives=o.collectives, session=o.session,
                                        name=self.name,
                                        result_runtime=getattr(o, "result_runtime", None))
            self.operations = None, proposals, None, None
            self._release_examples()
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
            workspace=None,
            dummy=self.verification_dummy,
        )
        verify.prepare()
        prepare_sampling_broadcasts(broadcast_proposals, verify, collectives=o.collectives, session=o.session,
            name=self.name, result_runtime=getattr(o, "result_runtime", None))
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
        if commit.logprobs is not None:
            commit.logprobs.release_logits()
        self.decoder.release_outputs()
        self.operations = target, proposals, verify, commit
        self._release_examples()

    def _release_examples(self):
        self.validator.functional()
        self.validation.functional(session=self.owner.session)
        if self.draft is not None:
            self.draft.release_outputs()
        if self.context is not None:
            self.context.functional()
        target, proposals, verify, commit = self.operations
        for operation in (target, *proposals, verify, commit):
            if operation is not None:
                operation.functional(session=self.owner.session)
        if self.full_proposal is not None:
            self.full_proposal.functional(session=self.owner.session)
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
            local = torch.empty(self.draft.local_logit_shape, dtype=self.draft.local_logit_dtype,
                                device=o.requests.histogram.device).view(b, o.block_size, -1)
            base = local[:, o.logit_offset:o.logit_offset + w - 1]
            self.full_proposal = PreparedVocabularySoftmax(base, self.probabilities, self.valid, *args,
                self.drafts, **options, name=self.name + ".proposal", workspace=None,
                embedding=o.weights.draft.markov_embedding, head=o.weights.draft.markov_head,
                pending=self.markov_token, seeds=self.seed, offsets=self.base_offset, tokens=self.drafts)
            self.full_proposal.prepare()
        if self.proposal_only:
            self.operations = None, (), None, None
            self._release_examples()
            return
        logits = self.decoder.local_logits.view(b, w, -1)
        from ignition.preparation.scratch import bind_scratch

        retained = bind_scratch(dict(probabilities=(tuple(logits.shape), torch.float32),
            valid=((b, w), torch.bool)), device=logits.device, workspace=None)
        target = PreparedVocabularySoftmax(logits, retained["probabilities"], retained["valid"], *args,
            self.drafts, **options, name=self.name + ".target", workspace=None)
        target.prepare()
        verify = PreparedVocabularyBlockVerification(self.probabilities, self.drafts, target.probabilities, *args,
            **options, name=self.name + ".verification", rng=o.rng, draft_valid=self.valid,
            target_valid=target.valid, token_limits=o.inputs.query_lengths[:b], workspace=None)
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
        self.decoder.release_outputs()
        self._release_examples()

    def _round(self):
        probabilities = torch.empty(self.probability_shape, dtype=torch.float32,
                                    device=self.owner.requests.histogram.device)
        proposal = self.propose(probabilities=probabilities)
        result = self.verify(probabilities=probabilities, proposal=proposal)
        return proposal, result

    def __call__(self):
        self._round()

    def observe(self):
        """Execute one round and return its metadata and acceptance for diagnostics."""
        proposal, result = self._round()
        metadata, acceptance, committed = result
        draft_metadata = proposal.metadata
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

    def propose(self, advance_rng=True, allowed=None, *, probabilities):
        if not self.generate_proposals:
            raise RuntimeError("verification-only preparation cannot generate proposals")
        o, b, w = self.owner, self.batch, self.width
        _, proposals, _, _ = self.operations
        validation_input, present = self.validator.validate()
        validated = self.validator.begin(self.validation(validation_input), present=present)
        device = o.requests.histogram.device
        active = validated.lengths > 0
        if allowed is not None:
            active = active & allowed
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        before = o.requests.consumed_tokens[safe]
        limits = torch.where(active, o.inputs.query_lengths[:b], 0)
        states = o.inputs.request_state_rows[safe, 0]
        candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        candidates[:, 0].copy_(o.requests.pending_token[safe])
        base_offset = torch.where(o.generation[safe] == o.requests.generation[safe], o.offset[safe], 0)
        seed = o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN
        sampled = torch.zeros(b, dtype=torch.int64, device=device)
        draft_metadata = features = None
        if self.draft is not None:
            block = o.block_size
            draft_inputs = torch.full((b, block), o.weights.config.draft.noise_token, dtype=torch.int64, device=device)
            draft_inputs[:, 0].copy_(candidates[:, 0])
            draft_metadata = Metadata.pack(before, active.int() * block, states,
                o.inputs.page_tables[safe], draft_inputs, o.inputs.live_rows,
                token_capacity=b * block, sample_count=b * block)
            result = self.draft(draft_metadata)
            draft_logits, selector_features = result[:2]
            features = result[2] if len(result) == 3 else None
            draft_logits = draft_logits.view(b, block, -1)
            unary = shortlist_ids = None
            if self.shortlisted_selector is not None:
                unary, shortlist_ids = self.shortlisted_selector.prepare_shortlists(draft_logits, features=selector_features)
            markov_token = torch.where(active, candidates[:, 0], 0)
            block_state = None
            if self.full_proposal is not None:
                base = draft_logits[:, o.logit_offset:o.logit_offset + w - 1]
                if self.vocabulary_parallel:
                    self.full_proposal.run(base, drafts, pending=markov_token, seeds=seed, offsets=base_offset,
                        probabilities=probabilities, valid=valid, tokens=drafts)
                else:
                    self.full_proposal.run(base, markov_token, drafts, probabilities, valid, seed, base_offset)
                positions = torch.arange(w - 1, device=device)
                valid = valid & active[:, None] & (positions[None, :] < limits[:, None] - 1)
                sampled = drafts[:, -1]
                markov_token = sampled
            elif self.proposal_block is not None:
                block_state = self.proposal_block.begin(device)
            for i, sampler in enumerate(proposals):
                position = i + o.logit_offset
                bias = self.markov(markov_token) if self.markov is not None else None
                source_logits = draft_logits[:, position]
                if sampler.has_source_logits:
                    proposal_logits = torch.empty((b, 1, o.requests.vocab_size), dtype=torch.float32, device=device)
                else:
                    proposal_logits = source_logits.float()
                    if bias is not None:
                        proposal_logits = proposal_logits + bias
                    if self.shortlisted_selector is not None:
                        proposal_logits = self.shortlisted_selector.apply_shortlist(
                            proposal_logits, markov_token, position, selector_features, unary, shortlist_ids)
                    elif self.selector is not None:
                        proposal_logits = self.selector.apply(proposal_logits, markov_token, position, selector_features)
                    proposal_logits = proposal_logits.contiguous().view(b, 1, -1)
                scratch = {name: torch.empty(shape, dtype=dtype, device=device)
                           for name, (shape, dtype) in sampler.layout.items()}
                if block_state is not None:
                    scratch["support_ids"] = block_state["support_ids"][i]
                    scratch["support_probabilities"] = block_state["support_probabilities"][i]
                proposal_probabilities = sampler.run(scratch, proposal_logits,
                    source_logits=source_logits if sampler.has_source_logits else None,
                    shortlist_ids=None if shortlist_ids is None else shortlist_ids[:, position],
                    candidates=drafts, seed=seed, offset=base_offset + 4 * i, sampled=sampled,
                    **({"source_bias": bias} if sampler.has_source_logits and bias is not None else {}))
                drafts[:, i].copy_(sampled)
                if block_state is None:
                    probabilities[:, i].copy_(proposal_probabilities[:, 0])
                    valid[:, i].copy_(scratch["valid"][:, 0] & active & (i < limits - 1))
                else:
                    self.proposal_block.retain(block_state, i, scratch, sampled)
                markov_token = sampled.clone()
            if self.proposal_block is not None:
                if self.full_proposal is not None:
                    drafts, valid, sampled = self.proposal_block.finish(drafts, valid, active, limits)
                else:
                    self.proposal_block.finish(block_state, drafts, probabilities, valid, sampled, active, limits)
        candidates[:, 1:].copy_(drafts)
        proposal = DraftProposalState(before, active, candidates, drafts, valid, base_offset, draft_metadata, features)
        if advance_rng and self.draft is not None:
            self.commit_draft_rng(proposal=proposal)
        return proposal

    def commit_draft_rng(self, drafts=None, *, proposal):
        o, b, w = self.owner, self.batch, self.width
        drafts = w - 1 if drafts is None else drafts
        slots = o.inputs.row_slot_ids[:b].long()
        slot_ids = torch.arange(o.requests.capacity, device=slots.device)
        matches = proposal.active[None, :] & (slots[None, :] == slot_ids[:, None])
        base = torch.where(matches, proposal.base_offset[None, :], 0).sum(1)
        o.offset.copy_(torch.where(matches.any(1), base + 4 * drafts, o.offset))
        o.generation.copy_(torch.where(matches.any(1), o.requests.generation, o.generation))

    def verify(self, proposal_valid=None, *, probabilities, proposal):
        o, b, w = self.owner, self.batch, self.width
        target, _, verify, commit = self.operations
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        active = proposal.active if proposal_valid is None else proposal.active & proposal_valid
        limits = torch.where(active, o.inputs.query_lengths[:b], 0)
        states = o.inputs.request_state_rows[safe, 0]
        metadata = Metadata.pack(proposal.before, limits, states, o.inputs.page_tables[safe],
            proposal.candidates, o.inputs.live_rows, token_capacity=self.target_token_capacity,
            sample_count=b * w, max_query_tokens=w)
        hidden, logits, auxiliary, records = self.decoder(metadata)
        if self.vocabulary_parallel:
            target_probabilities, target_valid, _ = target.run(logits.view(b, w, -1), proposal.drafts)
            acceptance = verify.run(probabilities, proposal.drafts, target_probabilities, proposal.valid, target_valid,
                token_limits=o.inputs.query_lengths[:b])
        else:
            target_scratch = {name: torch.empty(shape, dtype=dtype, device=logits.device)
                              for name, (shape, dtype) in target.layout.items()}
            verify_scratch = {name: torch.empty(shape, dtype=dtype, device=logits.device)
                              for name, (shape, dtype) in verify.layout.items()}
            target.run(target_scratch, logits.view(b, w, -1), candidates=proposal.drafts)
            acceptance = verify.run(verify_scratch, probabilities, target_scratch["probabilities"],
                target_scratch["valid"], disjoint=True, drafts=proposal.drafts, draft_valid=proposal.valid)
        if proposal_valid is not None:
            acceptance.valid.logical_and_(proposal_valid)
        committed = commit(logits.view(b, w, -1), acceptance=acceptance, inputs=proposal.candidates)
        counts = committed.emitted_count
        if w > 1:
            self.decoder.commit(counts, records, metadata)
        rows = torch.arange(b, device=hidden.device)
        hidden = hidden.view(b, w, o.weights.config.text["hidden_size"])[rows, (counts - 1).long().clamp_min(0)]
        save_hidden(hidden, o.pool.last_hidden, metadata.state_rows, counts > 0)
        if self.context is not None:
            self.context(auxiliary, counts, metadata)
        return metadata, acceptance, committed
