# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured MTP proposals with optional history augmentation, target verification, and commit rounds."""

from __future__ import annotations

import torch

from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.state import gather_rows

from .decoder import PreparedDecoder
from .draft_state import DRAFT_RAW_FIELDS, DraftMetadata
from ignition.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit
from .mtp import MTPScratch, PreparedMTP
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN, PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions, PreparedStochasticVerification, SpeculativeRngState,
)
from .speculative_state import PreparedVerificationState


class SpeculativePreparation:
    """Shared persistent owners for all prepared speculative graph capacities."""

    def __init__(self, *, weights, mtp_weights, pool, requests, inputs, scratch,
                 workspaces, rotary, cache, table, session, completion, frame, sequence,
                 context, prefill_capacity, max_cache_pages,
                 runtime=None, vocabulary_runtime=None, coordinator=None, verification_width=8,
                 sampling_workspace=None):
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.weights, self.mtp_weights, self.pool, self.requests = weights, mtp_weights, pool, requests
        self.inputs, self.scratch, self.workspaces, self.rotary = inputs, scratch, workspaces, rotary
        self.cache, self.table, self.session = cache, table, session
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.context, self.max_cache_pages = context, max_cache_pages
        device = pool.backing.device
        if type(verification_width) is not int or not 1 <= verification_width <= 8:
            raise ValueError("verification width must be between one and eight")
        self.verification_width = verification_width
        self.sampling_workspace = sampling_workspace
        b, w = requests.capacity, verification_width
        self.verification_residual = torch.zeros((b * w, 10240), dtype=torch.bfloat16, device=device)
        self.mtp_scratch = MTPScratch.allocate(prefill_capacity, requests.capacity, device, world_size=weights.shard.world_size,
            hyperconnection_lowrank=mtp_weights.module.hyper_connection_mixer.input_mix_weight_down.weight.shape[0])
        # Feedback is consumed before the MTP mixer writes normalized activations.
        self.feedback_hidden = self.mtp_scratch.hyperconnection.normalized
        self.prefill_residual = scratch.residual
        self.raw_backup = {name: torch.empty_like(getattr(cache[48], name)) for name in DRAFT_RAW_FIELDS}
        self.previous_accepted = torch.ones(requests.capacity, dtype=torch.int32, device=device)
        self.candidates = torch.zeros((requests.capacity, 8), dtype=torch.int64, device=device)
        self.draft_tables = torch.full((requests.capacity, context // 64), -1, dtype=torch.int32, device=device)
        self.copy_ids = torch.full((requests.capacity, 7), -1, dtype=torch.int64, device=device)
        self.copy_count = torch.zeros(requests.capacity, dtype=torch.int32, device=device)
        self.copy_expected = torch.full((requests.capacity,), -1, dtype=torch.int64, device=device)
        self.copy_generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.rng = SpeculativeRngState.allocate(device, capacity=requests.capacity)
        self.draft_rng_generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.draft_rng_offset = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        # Verification records of every target GDN layer, indexed by model slot.
        heads = next(iter(pool.gdn.values())).recurrent.shape[1] if pool.gdn else 0
        self.gdn_records = (GDNLayerRecords.allocate(pool.gdn, requests.capacity, device,
                                                     key_heads=heads // 3, value_heads=heads)
                            if pool.gdn else None)
        self.owners, self.decoders, self.mtp_modules = [], [], []

    def draft_metadata(self, tokens, rows):
        return DraftMetadata(tokens=tokens, rows=rows, context=self.context, device=self.pool.backing.device,
                             hidden=self.feedback_hidden[:tokens])

    def buffers(self):
        buffers = {"copy_token_ids": self.copy_ids, "copy_token_count": self.copy_count,
                   "copy_expected_consumed": self.copy_expected, "copy_generation": self.copy_generation,
                   "draft_page_tables": self.draft_tables}
        return buffers

    def mtp(self, metadata, name, *, cache_only=False):
        module = PreparedMTP(self.mtp_weights, self.mtp_scratch.rows(metadata.tokens, metadata.rows),
            metadata.token_ids, metadata.hidden, self.cache[48], metadata.qsa, self.rotary,
            self.workspaces, output_indices=metadata.output_indices, session=self.session,
            name=name, draft_tokens=7, max_cache_pages=self.max_cache_pages, cache_only=cache_only, **self.collectives)
        self.mtp_modules.append(module)
        return module

    def prefill(self, decoder, name):
        operation = PreparedPrefillAbsorption(self, decoder, name)
        self.owners.append(operation)
        return operation

    def resume(self, resume):
        return resume

    def round(self, batch, width, name):
        operation = PreparedSpeculativeRound(self, batch, width, name, decoder_factory=PreparedDecoder)
        self.owners.append(operation)
        return operation

    def rebind_pages_for_preparation(self, cache):
        self.cache = cache
        for module in self.mtp_modules:
            module.rebind_pages_for_preparation(cache[48])

    def release_examples(self):
        self.scratch = self.mtp_scratch = self.feedback_hidden = self.prefill_residual = None
        self.verification_residual = self.raw_backup = self.gdn_records = self.candidates = None
        self.sampling_workspace = None


class PreparedPrefillAbsorption:
    """Absorb exactly P-1 target-derived pairs before publishing a P-token prefix."""

    def __init__(self, owner, decoder, name, *, residual_output=None):
        self.owner, self.decoder = owner, decoder
        tokens, device = decoder.state.tokens, owner.pool.backing.device
        self.metadata = owner.draft_metadata(tokens, 1)
        self.mtp = owner.mtp(self.metadata, f"{name}.mtp_absorb", cache_only=True)
        self.target_residual = (owner.prefill_residual[:tokens]
                                if residual_output is None else residual_output)
        decoder.residual_output = self.target_residual
        self.previous = torch.zeros((1, 10240), dtype=torch.bfloat16, device=device)
        self.previous_rope = torch.zeros((1, 3), dtype=torch.int64, device=device)
        self.before = torch.zeros(1, dtype=torch.int64, device=device)
        self.indices = torch.arange(tokens, device=device)

    def functional(self, *, session):
        self.decoder.functional(session=session)
        self.mtp.functional(session=session)
        self.metadata.functional()
        self.before = self.previous = self.previous_rope = self.indices = self.target_residual = None
        self.decoder.release_outputs()
        self.mtp.release_outputs()
        return self

    def __call__(self):
        o, decoder = self.owner, self.decoder
        slots = o.inputs.row_slot_ids[:1]
        safe = slots.long().clamp(0, o.requests.capacity - 1)
        before = o.requests.consumed_tokens[safe]
        previous = torch.empty((1, 10240), dtype=torch.bfloat16, device=slots.device)
        previous_rope = torch.empty((1, 3), dtype=torch.int64, device=slots.device)
        state_rows = o.inputs.request_state_rows[safe, 0]
        gather_rows(o.pool.region("target_residual"), state_rows, previous)
        gather_rows(o.pool.region("target_residual_position"), state_rows, previous_rope)
        errors = None if decoder.embedding_injection is None else decoder.embedding_injection.validate()
        runtime = decoder.state.begin_runtime()
        runtime.injection_errors = errors
        _, _, auxiliary, _ = decoder(runtime)
        self.finish_runtime(o.inputs, runtime, auxiliary.residual, before, previous, previous_rope)

    def finish_runtime(self, inputs, runtime, residual, before, previous, previous_rope):
        o = self.owner
        slots = inputs.row_slot_ids[:1]
        active = runtime.values["active"].bool()
        length = (runtime.gdn.query_start_loc[1:] - runtime.gdn.query_start_loc[:-1]).long()
        indices = torch.arange(runtime.ple.token_ids.shape[0], device=slots.device)
        shift = (before == 0).long()
        indices = (indices + shift[0]).clamp_max(indices.numel() - 1)
        hidden = residual[(indices - 1).clamp_min(0)]
        rope = runtime.qsa.rope_positions[(indices - 1).clamp_min(0)]
        first = (torch.arange(runtime.ple.token_ids.shape[0], device=slots.device) == 0) & (before[0] > 0)
        hidden = torch.where(first[:, None], previous, hidden)
        rope = torch.where(first[:, None], previous_rope, rope)
        lengths = torch.where(active, (length - shift).clamp_min(0), 0)
        metadata = self.metadata.set(slots, lengths, (before - 1).clamp_min(0),
            runtime.ple.token_ids[indices], hidden, rope, o.draft_tables, capacity=indices.numel())
        self.mtp(metadata.token_ids, metadata.hidden, metadata.qsa, metadata.output_indices)



class PreparedSpeculativeRound:
    """One graph contains every draft, all target rows, acceptance and state commit."""

    def __init__(self, owner, batch, width, name, *, decoder_factory, residual_output=None):
        self.owner, self.batch, self.width = owner, batch, width
        if not 1 <= width <= owner.verification_width:
            raise ValueError("round exceeds prepared verification scratch capacity")
        device = owner.pool.backing.device
        self.token_limits = owner.inputs.query_lengths[:batch]
        self.state = PreparedVerificationState(owner.inputs, owner.requests, owner.pool,
            owner.candidates, owner.previous_accepted, width=width, rows=batch, token_limits=self.token_limits)
        scratch = owner.scratch.rows(batch * width, batch * width)
        self.target_residual = (owner.verification_residual[:batch * width]
                                if residual_output is None else residual_output)
        self.decoder = decoder_factory(owner.weights, scratch, self.state, owner.cache, owner.rotary,
            owner.table, owner.workspaces, owner.completion, owner.frame, owner.sequence,
            session=owner.session, name=f"{name}.target", max_cache_pages=owner.max_cache_pages,
            residual_output=self.target_residual, verification=True,
            gdn_records=owner.gdn_records if width > 1 else None,
            gdn_in_place=width == 1 and owner.gdn_records is not None, **owner.collectives)
        owner.decoders.append(getattr(self.decoder, "forward", self.decoder))
        self.draft_metadata = owner.draft_metadata(batch, batch)
        self.draft = owner.mtp(self.draft_metadata, f"{name}.draft") if width > 1 else None
        self.absorb_metadata = owner.draft_metadata(batch * width, batch)
        self.absorb = owner.mtp(self.absorb_metadata, f"{name}.absorb", cache_only=True)
        self.before = torch.zeros(batch, dtype=torch.int64, device=device)
        self.pending = torch.zeros(batch, dtype=torch.int64, device=device)
        self.previous = torch.zeros((batch, 10240), dtype=torch.bfloat16, device=device)
        self.previous_rope = torch.zeros((batch, 3), dtype=torch.int64, device=device)
        self.rope_delta = torch.zeros(batch, dtype=torch.int64, device=device)
        self.active = torch.zeros(batch, dtype=torch.bool, device=device)
        self.drafts = torch.zeros((batch, width - 1), dtype=torch.int64, device=device)
        self.draft_probabilities = torch.zeros((batch, width - 1, owner.requests.vocab_size),
                                               dtype=torch.float32, device=device)
        self.draft_valid = torch.zeros((batch, width - 1), dtype=torch.bool, device=device)
        self.offered = torch.full((batch,), width - 1, dtype=torch.int32, device=device)
        self.copy_mask = torch.zeros(batch, dtype=torch.int64, device=device)
        self.raw_backup = owner.raw_backup
        self.rows = torch.arange(batch, device=device)
        self.columns = torch.arange(width, device=device)
        self.seed = torch.zeros(batch, dtype=torch.int64, device=device)
        self.offset = torch.zeros(batch, dtype=torch.int64, device=device)
        self.base_offset = torch.zeros(batch, dtype=torch.int64, device=device)
        self.sampled = torch.zeros(batch, dtype=torch.int64, device=device)
        self.name = name
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        slots, epochs = o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b]
        target = PreparedSpeculativeDistributions(self.decoder.scratch.logits.view(b, w, -1),
            o.requests, slots, epochs, o.inputs.live_rows, self.drafts)
        proposals = tuple(PreparedDraftDistribution(self.draft.scratch.logits.view(b, 1, -1),
            o.requests, slots, epochs, o.inputs.live_rows, self.drafts, prefix_offset=index,
            seed=self.seed, offset=self.offset, sampled=self.sampled) for index in range(w - 1))
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(self.draft_probabilities, self.drafts, target.probabilities,
            o.requests, slots, epochs, o.inputs.live_rows, rng=o.rng,
            draft_valid=self.draft_valid, target_valid=target.valid, token_limits=self.token_limits)
        verify.prepare()
        from ignition.components.parallel_draft import prepare_sampling_broadcasts
        prepare_sampling_broadcasts(proposals, verify, collectives=o.collectives, session=o.session,
                                    name=self.name)
        commit = PreparedSpeculativeCommit(o.requests, verify.output,
            self.state.token_ids.view(b, w), slots, epochs, o.inputs.live_rows,
            o.completion, o.frame, o.sequence, logits=self.decoder.scratch.logits.view(b, w, -1), rng=o.rng, copy_offered_mask=self.copy_mask,
            draft_context_lag=1, token_limits=self.token_limits)
        commit.prepare()
        self.state.prepare_commit(commit.emitted_count, commit.consumed_before,
                                  self.target_residual, self.decoder.scratch.sample_hidden)
        model_commit = tuple(self.state._commit)
        if o.gdn_records is not None and w > 1:
            heads = o.gdn_records.delta.shape[2]
            model_commit += (PreparedGDNCommit(o.pool.backing,
                {layer: state.recurrent for layer, state in o.pool.gdn.items()}, o.gdn_records, self.state.gdn,
                commit.emitted_count, key_heads=heads // 3, value_heads=heads),)
        self.operations = (target, proposals, verify, commit, model_commit)

    def functional(self, *, session, dynamic_inputs=False):
        if self.operations is None:
            raise RuntimeError("speculative round prepare() must complete before functional conversion")
        target, proposals, verify, commit, model_commit = self.operations
        target.functional(session=session, dynamic_inputs=dynamic_inputs)
        for proposal in proposals:
            proposal.functional(session=session, dynamic_inputs=dynamic_inputs)
        verify.functional(session=session, dynamic_inputs=dynamic_inputs)
        commit.functional(session=session, dynamic_inputs=dynamic_inputs)
        self.decoder.functional(session=session)
        if self.draft is not None:
            self.draft.functional(session=session)
        self.absorb.functional(session=session)
        self.draft_metadata.functional()
        self.absorb_metadata.functional()
        self.gdn_commit = None
        if len(model_commit) > len(self.state._commit):
            self.gdn_commit = model_commit[-1].functional(session=session, dynamic_metadata=True)
        self.state.functional(dynamic_inputs=dynamic_inputs)
        self.decoder.scratch = self.decoder.residual_output = None
        if self.draft is not None:
            self.draft.release_outputs()
        self.absorb.release_outputs()
        self.operations = (target, proposals, verify, commit)
        self.draft_probabilities = None
        self.before = self.pending = self.previous = self.previous_rope = self.rope_delta = self.active = None
        self.drafts = self.draft_valid = self.offered = self.copy_mask = self.raw_backup = None
        self.rows = self.columns = self.seed = self.offset = self.base_offset = self.sampled = None
        self.target_residual = None
        if dynamic_inputs:
            self.token_limits = None
        return self

    def run(self, inputs=None, packing=None):
        from ignition.preparation.scratch import bind_scratch

        o, b, w = self.owner, self.batch, self.width
        inputs = o.inputs if inputs is None else inputs
        token_limits = inputs.query_lengths[:b]
        target, proposals, verify, commit = self.operations
        slots = inputs.row_slot_ids[:b]
        safe = slots.long().clamp(0, o.requests.capacity - 1)
        rows, columns = torch.arange(b, device=slots.device), torch.arange(w, device=slots.device)
        active = ((rows < inputs.live_rows[0]) & (slots >= 0) & (slots < o.requests.capacity)
            & (o.requests.generation[safe] == inputs.row_generations[:b]) & (o.requests.status[safe] == 1))
        before, pending = o.requests.consumed_tokens[safe], o.requests.pending_token[safe]
        previous = torch.empty((b, 10240), dtype=torch.bfloat16, device=slots.device)
        previous_rope = torch.empty((b, 3), dtype=torch.int64, device=slots.device)
        rope_delta = torch.empty((b,), dtype=torch.int64, device=slots.device)
        gather_rows(o.pool.region("target_residual"), inputs.request_state_rows[safe, 0], previous)
        gather_rows(o.pool.region("target_residual_position"), inputs.request_state_rows[safe, 0], previous_rope)
        delta = gather_rows(o.pool.rope_delta, inputs.request_state_rows[safe, 0], rope_delta)
        raw_backup = {name: getattr(o.cache[48], name).clone() for name in DRAFT_RAW_FIELDS}
        drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=slots.device)
        draft_valid = torch.zeros((b, w - 1), dtype=torch.bool, device=slots.device)
        draft_probabilities = torch.empty((b, w - 1, o.requests.vocab_size), dtype=torch.float32, device=slots.device)
        copy_mask = torch.zeros(b, dtype=torch.int64, device=slots.device)
        base_offset = torch.where(o.draft_rng_generation[safe] == o.requests.generation[safe],
                                  o.draft_rng_offset[safe], 0)
        seed = o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN
        draft_result = draft_metadata = None
        for index, proposal in enumerate(proposals):
            hidden = previous if index == 0 else draft_result.residual[draft_metadata.output_indices]
            tokens = pending if index == 0 else drafts[:, index - 1]
            rope = previous_rope if index == 0 else (before + index - 1 + delta)[:, None].expand(-1, 3)
            draft_active = active if token_limits is None else active & (index < token_limits)
            draft_metadata = self.draft_metadata.set(slots, draft_active.long(), before - 1 + index,
                tokens, hidden, rope, o.draft_tables, capacity=1)
            draft_result = self.draft(draft_metadata.token_ids, draft_metadata.hidden,
                                      draft_metadata.qsa, draft_metadata.output_indices)
            sampled = torch.empty((b,), dtype=torch.int64, device=slots.device)
            proposal_scratch = bind_scratch(proposal.layout, device=slots.device)
            probabilities = proposal.run(proposal_scratch, draft_result.logits.view(b, 1, -1),
                candidates=drafts, seed=seed, offset=base_offset + 4 * index, sampled=sampled,
                slots=slots, generations=inputs.row_generations[:b], live_rows=inputs.live_rows)
            copy = (active & (o.copy_generation[safe] == o.requests.generation[safe])
                    & (o.copy_expected[safe] == before) & (index < o.copy_count[safe])
                    & (o.copy_ids[safe, index] >= 0) & (o.copy_ids[safe, index] < o.requests.vocab_size))
            point_mass = (target.vocab_ids == o.copy_ids[safe, index, None]).float()
            draft_probabilities[:, index].copy_(torch.where(copy[:, None], point_mass, probabilities[:, 0]))
            sampled = torch.where(copy, o.copy_ids[safe, index], sampled)
            copy_mask = copy_mask | (copy.long() * (1 << index))
            drafts[:, index].copy_(sampled)
            draft_valid[:, index].copy_(copy | proposal_scratch["valid"][:, 0])
        if w > 1:
            slot_ids = torch.arange(o.requests.capacity, device=slots.device)
            match = active[None, :] & (slots[None, :] == slot_ids[:, None])
            matched = match.any(1)
            base = torch.where(match, base_offset[None, :], 0).sum(1)
            o.draft_rng_offset.copy_(torch.where(matched, base + 4 * (w - 1), o.draft_rng_offset))
            o.draft_rng_generation.copy_(torch.where(matched, o.requests.generation, o.draft_rng_generation))
        if token_limits is not None:
            drafts.masked_fill_(columns[None, :-1] >= token_limits[:, None] - 1, 0)
        candidates = torch.zeros((b, 8), dtype=torch.int64, device=slots.device)
        candidates[:, 0].copy_(pending)
        candidates[:, 1:w].copy_(drafts)
        runtime = self.state.begin_runtime(candidates, inputs)
        hidden, logits, auxiliary, records = (self.decoder(runtime) if packing is None
                                               else self.decoder(runtime, packing))
        target_scratch = bind_scratch(target.layout, device=slots.device)
        target_probs = target.run(target_scratch, logits.view(b, w, -1), candidates=drafts, slots=slots, generations=inputs.row_generations[:b], live_rows=inputs.live_rows)
        verify_scratch = bind_scratch(verify.layout, device=slots.device)
        acceptance = verify.run(verify_scratch, draft_probabilities, target_probs, target_scratch["valid"],
                                drafts=drafts, draft_valid=draft_valid, slots=slots,
                                generations=inputs.row_generations[:b], live_rows=inputs.live_rows, token_limits=token_limits)
        committed = commit(logits.view(b, w, -1), acceptance=acceptance,
                           inputs=runtime.ple.token_ids.view(b, w), copy_offered_mask=copy_mask,
                           slots=slots, generations=inputs.row_generations[:b],
                           live_rows=inputs.live_rows, token_limits=token_limits)
        self.state.commit_runtime(runtime, hidden, auxiliary.residual, committed.emitted_count)
        if self.gdn_commit is not None:
            self.gdn_commit(committed.emitted_count, records=records, metadata=runtime.gdn)
        for name, backup in raw_backup.items():
            getattr(o.cache[48], name).copy_(backup)
        residual = auxiliary.residual.view(b, w, 10240)
        hidden = torch.cat((previous[:, None, :], residual[:, :-1]), dim=1)
        rope = torch.cat((previous_rope[:, None, :], runtime.qsa.rope_positions.view(b, w, 3)[:, :-1]), dim=1)
        metadata = self.absorb_metadata.set(slots, committed.emitted_count.long(), before - 1,
            runtime.ple.token_ids, hidden, rope, o.draft_tables, capacity=w)
        self.absorb(metadata.token_ids, metadata.hidden, metadata.qsa, metadata.output_indices)
        if packing is not None:
            return auxiliary.mixed

    def __call__(self):
        return self.run()
