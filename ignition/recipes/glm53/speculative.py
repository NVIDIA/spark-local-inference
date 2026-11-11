# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured GLM proposals, packed target verification and accepted state commits."""

import torch
import triton
import triton.language as tl
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ignition.components.collectives import PreparedAllReduce
from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.recurrent import DeltaMetadata, DeltaState
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN, PreparedSpeculativeCommit, PreparedSpeculativeDistributions,
    PreparedStochasticVerification, SpeculativeRngState,
)
from ignition.components.state import gather_rows, save_hidden
from ignition.preparation.persistent import persistent_copy
from ignition.components.kda_speculation import PreparedKDAReplay as PreparedRecordedKDAReplay

from .batch import PreparedBatch
from .dcp import ContextShards
from .kda import KDARecords
from .mtp import DraftMetadata, MTPScratch, PreparedMTP
from .stack import PreparedDecoder
from .state import DecoderMetadata


@triton.jit
def _shift_convolution(Halves, Slots, Counts, StateRows, Convolution, row_halves,
                       CHANNELS: tl.constexpr, COLUMNS: tl.constexpr, CHANNEL_BLOCK: tl.constexpr):
    # One program shifts a slice of one layer's convolution history in the live row past the emitted tokens.
    # Rows that kept at most one token already hold their history.
    layer, row, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    count = tl.load(Counts + row).to(tl.int64)
    if count > 1:
        slot = tl.load(Slots + row).to(tl.int64)
        live = tl.load(StateRows + slot * 8)
        channels = chunk * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
        present = channels < CHANNELS
        history = Halves + live * row_halves + tl.load(Convolution + layer) + channels * COLUMNS
        first = tl.load(history + count - 1, present)
        second = tl.load(history + count, present)
        third = tl.load(history + count + 1, present)
        tl.store(history, first, present)
        tl.store(history + 1, second, present)
        tl.store(history + 2, third, present)


def delta_layers(pool):
    """Decoder indices of the KDA layers, whose states live in the hybrid pool."""
    return [i for i, layer in enumerate(pool.decoder.layers) if isinstance(layer, DeltaState)]


def convolution_regions(pool):
    """Half offsets of every KDA convolution history region and its channel and column extents."""
    convolution = [pool.layout.region(f"layer{i}.convolution") for i in delta_layers(pool)]
    if len({(r.shape, r.dtype) for r in convolution}) != 1 or convolution[0].dtype != torch.bfloat16:
        raise ValueError("KDA commits require uniform BF16 convolution regions")
    return (persistent_copy(torch.tensor([r.offset // 2 for r in convolution], dtype=torch.int64, device=pool.backing.device)),
            *convolution[0].shape)


class PreparedKDACommit:
    """Commit accepted KDA records to every live recurrent row."""

    def __init__(self, pool, records, plan, state_indices, num_seqs, *, session, workspace):
        layers = delta_layers(pool)
        if tuple(layers) != records.layers:
            raise ValueError("KDA commit requires records for every KDA layer")
        states = [pool.decoder.layers[i].recurrent for i in layers]
        self.plan = plan
        if plan is None:
            from .kda_recovery import PreparedKDAReplay

            lengths = torch.zeros(state_indices.shape[0], dtype=torch.int32, device=pool.backing.device)
            self.replay = PreparedKDAReplay(pool.backing, states, records, state_indices, lengths, num_seqs)
        else:
            self.replay = PreparedRecordedKDAReplay(pool.backing, states, records, plan, session=session)
        self.replay.functional()

    def __call__(self, counts, records, metadata):
        self.replay(counts, records, metadata)


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, scratch, prefill_scratch, *, session,
                 workspace, completion, frame, sequence, runtime, vocabulary_runtime, coordinator, max_cache_pages=None,
                 prefill_tokens=None, decode_tokens=None, context=None):
        self.weights, self.pool, self.requests, self.inputs = weights, pool, requests, inputs
        self.max_cache_pages, self.context = max_cache_pages, context
        self.shards = ContextShards() if context is None else context.shards
        blocks, device = inputs.page_tables.shape[1], pool.backing.device
        self.draft_scratch = {}
        if prefill_tokens:
            self.draft_scratch["prefill"] = MTPScratch.absorbing(
                prefill_scratch.rows(prefill_tokens, "prefill", sample_rows=1))
        if decode_tokens:
            self.draft_scratch["decode"] = MTPScratch.allocate(decode_tokens, weights.shard.world_size, blocks,
                "decode", device, context_ranks=self.shards.world, remote=context is not None and context.remote)
        self.scratch, self.session, self.workspace = scratch, session, workspace
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.draft_tables = torch.full_like(inputs.page_tables, -1)
        self.draft_generation = torch.zeros_like(requests.generation)
        self.draft_offset = torch.zeros_like(requests.generation)
        # Verification records every KDA layer's tokens, so each request keeps one recurrent row.
        layers = delta_layers(pool)
        recurrent = pool.decoder.layers[layers[0]].recurrent
        self.kda_records = KDARecords.allocate({i: weights.layers[i].attention for i in layers},
            recurrent.shape[0], recurrent.shape[1], device)
        self.record_layouts = tuple((tuple(t.shape), t.dtype)
                                   for t in (self.kda_records.corrections, self.kda_records.keys_and_gates))
        self.record_weights = self.kda_records.A_log, self.kda_records.dt_bias
        self.record_layers = tuple(layers)
        self.convolution = convolution_regions(pool)

    def release_examples(self):
        self.scratch = self.draft_scratch = self.kda_records = None

    def records(self):
        device = self.pool.backing.device
        planes = [torch.zeros(shape, dtype=dtype, device=device) for shape, dtype in self.record_layouts]
        return KDARecords(self.record_layers, *planes, *self.record_weights)

    def draft(self, metadata, name, *, cache_only=False, mode="decode"):
        shared = self.draft_scratch.get(mode)
        scratch = None if shared is None else shared.rows(metadata.token_ids.numel(), mode)
        return PreparedMTP(self.weights, self.pool.draft, metadata, mode=mode,
            session=self.session, name=name, workspace=self.workspace, cache_only=cache_only,
            max_cache_pages=self.max_cache_pages, scratch=scratch, context=self.context, **self.collectives)


class PreparedPrefill:
    """Absorb every prompt pair, retaining normalized target feedback at its endpoint."""

    def __init__(self, owner, generation, name):
        self.owner, self.generation = owner, generation
        self.tokens = generation.batch.tokens
        metadata = DraftMetadata(self.tokens, 1, owner.inputs.page_tables.shape[1],
                                 owner.inputs.token_ids.device, shards=owner.shards)
        self.absorb = owner.draft(metadata, f"{name}.absorb", cache_only=True, mode="prefill")
        self.absorb.functional(session=owner.session)

    def rebind_pages(self, pages):
        self.absorb.rebind_pages()
        self.generation.rebind_pages(pages)

    def __call__(self):
        o, g = self.owner, self.generation
        slot = o.inputs.row_slot_ids[:1].long().clamp(0, o.requests.capacity - 1)
        before = o.requests.consumed_tokens[slot]
        previous = torch.empty((1, 4096), dtype=torch.bfloat16, device=slot.device)
        gather_rows(o.pool.last_hidden, o.inputs.request_state_rows[slot, 0], previous)
        target = g.forward()
        target_metadata, feedback = target.metadata, target.feedback
        length = target_metadata.recurrent.num_tokens.long()
        skip = (before == 0).long()
        rows = torch.arange(self.tokens, device=slot.device)
        indices = (rows + skip[0]).clamp_max(self.tokens - 1)
        hidden = feedback[(indices - 1).clamp_min(0)]
        hidden = torch.where(((rows == 0) & (before[0] > 0))[:, None], previous, hidden)
        metadata = DraftMetadata.pack(o.inputs.row_slot_ids[:1], (length - skip).clamp_min(0),
            (before - 1).clamp_min(0), target_metadata.token_ids[indices], hidden,
            o.draft_tables, state_rows=o.inputs.request_state_rows, capacity=self.tokens, shards=o.shards)
        self.absorb(metadata)


class PreparedRound:
    """Rank zero's proposal and acceptance decisions are broadcast before dependent work."""

    def __init__(self, owner, batch, width, name):
        self.owner, self.batch, self.width = owner, batch, width
        o, b, w = owner, batch, width
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.sampling_layouts = {
            "target": PreparedSpeculativeDistributions.scratch_layout(b, w, o.requests.vocab_size),
            "verify": PreparedStochasticVerification.scratch_layout(b, w, o.requests.vocab_size),
        }
        if w > 1:
            self.sampling_layouts["proposal"] = PreparedDraftDistribution.scratch_layout(b, 1, o.requests.vocab_size)
        self.sampling_scratch = self._sampling_scratch(device)
        self.convolution_regions = o.convolution
        self.metadata = DecoderMetadata.allocate(b * w, b, blocks, device)
        self.metadata = DecoderMetadata(self.metadata.token_ids, torch.arange(b * w, device=device),
            DeltaMetadata.allocate(b, device, verification_columns=8), self.metadata.indexed)
        self.validator = PreparedBatch(o.inputs, o.requests, DecoderMetadata.allocate(b, b, blocks, device),
            mode="decode", shard=o.weights.shard, pages=o.pool.memory_capacity[1:], shards=o.shards)
        validation = torch.empty((1, max(8, 2 * o.requests.capacity)), dtype=torch.bfloat16, device=device)
        self.validation = PreparedAllReduce(validation, torch.empty_like(validation),
            runtime=o.collectives['runtime'], coordinator=o.collectives['coordinator'],
            session=o.session, name=f"{name}.validation")
        self.decoder = PreparedDecoder(o.weights, o.scratch.rows(b * w, "decode", sample_rows=b * w),
            o.pool.decoder, self.metadata, mode="decode", session=o.session, name=f"{name}.target",
            workspace=o.workspace, max_cache_pages=o.max_cache_pages, context=o.context,
            kda_records=o.kda_records, **o.collectives)
        first = self.decoder.layers[o.kda_records.layers[0]].attention
        self.kda_commit = PreparedKDACommit(o.pool, o.kda_records, first.recurrent_plan,
            self.metadata.recurrent.state_indices[:, 0], self.metadata.recurrent.num_seqs,
            session=o.session, workspace=o.workspace)
        self.draft_metadata = DraftMetadata(b, b, blocks, device, shards=o.shards)
        self.draft = o.draft(self.draft_metadata, f"{name}.draft") if w > 1 else None
        self.draft_logits = (torch.zeros(self.draft.logits.shape, dtype=self.draft.logits.dtype, device=device)
                             if self.draft is not None else None)
        self.absorb_metadata = DraftMetadata(b * w, b, blocks, device, shards=o.shards)
        self.absorb = o.draft(self.absorb_metadata, f"{name}.absorb", cache_only=True)
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.pending = torch.zeros_like(self.before)
        self.previous = torch.zeros((b, 4096), dtype=torch.bfloat16, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        self.probabilities = torch.zeros((b, w - 1, o.requests.vocab_size), dtype=torch.float32, device=device)
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        self.seed, self.offset, self.base_offset, self.sampled = (torch.zeros_like(self.before) for _ in range(4))
        self.tail_backup = torch.empty_like(o.pool.draft.index.tail[:b])
        self.name = name
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        target = PreparedSpeculativeDistributions(self.decoder.scratch.logits.view(b, w, -1), *args, self.drafts,
                                                 workspace=self.sampling_scratch["target"])
        proposals = tuple(PreparedDraftDistribution(self.draft_logits.view(b, 1, -1), *args,
            self.drafts, prefix_offset=i, seed=self.seed, offset=self.offset, sampled=self.sampled,
            workspace=self.sampling_scratch["proposal"])
            for i in range(w - 1))
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(self.probabilities, self.drafts, target.probabilities,
            *args, rng=o.rng, draft_valid=self.valid, target_valid=target.valid,
            token_limits=o.inputs.query_lengths[:b], workspace=self.sampling_scratch["verify"])
        verify.prepare()
        from ignition.components.parallel_draft import prepare_sampling_broadcasts
        prepare_sampling_broadcasts(proposals, verify, collectives=o.collectives, session=o.session,
                                    name=self.name)
        commit = PreparedSpeculativeCommit(o.requests, verify.output, self.candidates,
            o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows,
            o.completion, o.frame, o.sequence, logits=self.decoder.scratch.logits.view(b, w, -1), rng=o.rng, token_limits=o.inputs.query_lengths[:b])
        commit.prepare()
        self.operations = target, proposals, verify, commit
        target.functional(session=o.session)
        for proposal in proposals:
            proposal.functional(session=o.session)
        verify.functional(session=o.session)
        commit.functional(session=o.session)
        self.validation.functional(session=o.session)
        self.validator.functional()
        self.decoder.functional(session=o.session)
        if self.draft is not None:
            self.draft.functional(session=o.session)
        self.absorb.functional(session=o.session)
        for field in ("sampling_scratch", "metadata", "draft_metadata", "absorb_metadata", "draft_logits",
                      "before", "pending", "previous", "active", "candidates", "drafts", "probabilities",
                      "valid", "seed", "offset", "base_offset", "sampled", "tail_backup"):
            setattr(self, field, None)

    def rebind_pages(self, pages):
        self.decoder.rebind_pages()
        for draft in (self.draft, self.absorb):
            if draft is not None:
                draft.rebind_pages()
        self.validator.pages = self.owner.pool.memory_capacity[1:]

    def _sampling_scratch(self, device):
        return {role: {name: torch.empty(shape, dtype=dtype, device=device)
                       for name, (shape, dtype) in layout.items()}
                for role, layout in self.sampling_layouts.items()}

    def target_metadata(self, slots, limits, active, before, candidates):
        o, b, w = self.owner, self.batch, self.width
        device = slots.device
        m = DecoderMetadata.allocate(b * w, b, o.inputs.page_tables.shape[1], device,
                                     samples=b * w, verification_columns=8)
        g, index = m.recurrent, m.indexed
        lengths = torch.where(active, limits, 0)
        ends = lengths.cumsum(0)
        starts = ends - lengths
        g.query_start_loc[1:].copy_(ends)
        states = o.inputs.request_state_rows[slots.clamp(0, o.requests.capacity - 1)]
        g.convolution_slots.copy_(torch.where(active, states[:, 0], -1))
        g.accepted_tokens.fill_(1)
        state_columns = torch.arange(8, device=device)
        states = torch.where(state_columns < 1, states, 0)
        g.state_indices.copy_(torch.where(active[:, None], states, 0))
        g.num_seqs.copy_(o.inputs.live_rows.clamp(0, b))
        g.num_tokens.copy_(ends[-1:])
        index.query_start_loc.copy_(g.query_start_loc)
        index.state_slots.copy_(g.convolution_slots)
        index.block_table.copy_(o.inputs.page_tables[slots.clamp(0, o.requests.capacity - 1)])
        offsets = torch.arange(b * w, device=device)
        row = (offsets[:, None] >= ends[None]).sum(1).clamp_max(b - 1)
        column = offsets - starts[row]
        live = offsets < ends[-1]
        pos = before[row] + column
        index.requests.copy_(row)
        index.positions.copy_(torch.where(live, pos, -1))
        index.token_slots.copy_(torch.where(live, o.shards.token_slots(pos, index.block_table[row]), -1))
        source = (row * w + column).clamp(0, b * w - 1)
        m.token_ids.copy_(torch.where(live, candidates.reshape(-1)[source], 0))
        columns = torch.arange(w, device=device)
        m.sample_indices.copy_((starts[:, None] + columns).clamp(0, b * w - 1).reshape(-1))
        return m

    def commit_state(self, slots, counts, records, metadata, hidden):
        """Replay accepted KDA records, shift convolution history, and retain endpoint feedback."""
        o, w = self.owner, self.width
        states = o.inputs.request_state_rows[slots.clamp(0, o.requests.capacity - 1)]
        live = torch.where(counts > 0, states[:, 0], 0).long()
        self.kda_commit(counts, records, metadata)
        backing = o.pool.backing
        halves, channels, columns = self.convolution_regions
        args = (backing.view(torch.bfloat16), slots, counts, o.inputs.request_state_rows, halves,
                backing.shape[1] // 2, channels, columns, 128)
        grid = (halves.numel(), self.batch, triton.cdiv(channels, 128))
        if torch.compiler.is_compiling() or get_proxy_mode() is not None:
            torch.library.wrap_triton(_shift_convolution)[grid](*args, num_warps=4)
        else:
            _shift_convolution[grid](*args, num_warps=4)
        rows = torch.arange(self.batch, device=slots.device)
        selected_hidden = hidden.view(self.batch, w, 4096)[rows, (counts - 1).clamp_min(0).long()]
        save_hidden(selected_hidden, o.pool.last_hidden, live, (counts > 0).int())

    def __call__(self):
        o, b, w = self.owner, self.batch, self.width
        target, proposals, verify, commit = self.operations
        device = o.inputs.token_ids.device
        scratch = self._sampling_scratch(device)
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        limits = o.inputs.query_lengths[:b]
        validation, present = self.validator.validate()
        end = o.requests.consumed_tokens[safe] + limits
        page_tokens = o.shards.page_tokens
        bad = (limits < 1) | (limits > w) | (end > o.inputs.page_tables.shape[1] * page_tokens)
        pages = o.inputs.page_tables[safe]
        block_ids = torch.arange(pages.shape[1], device=device)
        required = block_ids < (end[:, None] + page_tokens - 1) // page_tokens
        bad |= (required & ((pages < 0) | (pages >= self.validator.pages))).any(1)
        slot_ids = torch.arange(o.requests.capacity, device=device)
        bad_slots = ((slot_ids[:, None] == slots[None]) & bad[None]).any(1)
        validation[0, :o.requests.capacity].add_(bad_slots)
        validated = self.validator.begin(self.validation(validation), present)
        active = validated.recurrent.convolution_slots >= 0
        before = o.requests.consumed_tokens[safe]
        pending = o.requests.pending_token[safe]
        previous = torch.empty((b, 4096), dtype=torch.bfloat16, device=device)
        tail_backup = torch.empty_like(o.pool.draft.index.tail[:b])
        gather_rows(o.pool.last_hidden, o.inputs.request_state_rows[safe, 0], previous)
        gather_rows(o.pool.draft.index.tail, o.inputs.request_state_rows[safe, 0], tail_backup)
        base_offset = torch.where(o.draft_generation[safe] == o.requests.generation[safe], o.draft_offset[safe], 0)
        seed = o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN
        drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        probabilities = torch.zeros((b, w - 1, o.requests.vocab_size), dtype=torch.float32, device=device)
        valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        sampled = torch.zeros_like(before)
        draft_hidden = previous
        for i, proposal in enumerate(proposals):
            token = pending if i == 0 else drafts[:, i - 1]
            proposing = active & (i < limits - 1)
            draft_metadata = DraftMetadata.pack(slots, proposing.long(), before - 1 + i,
                token, draft_hidden, o.draft_tables, state_rows=o.inputs.request_state_rows,
                capacity=1, shards=o.shards)
            hidden, logits = self.draft(draft_metadata)
            draft_hidden = hidden[draft_metadata.output_indices]
            draft_logits = logits[draft_metadata.output_indices].view(b, 1, -1)
            proposal.run(scratch['proposal'], draft_logits, candidates=drafts,
                         seed=seed, offset=base_offset + 4 * i, sampled=sampled)
            probabilities[:, i].copy_(scratch['proposal']['probabilities'][:, 0])
            drafts[:, i].copy_(sampled)
            valid[:, i].copy_(scratch['proposal']['valid'][:, 0] & proposing)
        match = active[None, :] & (slots[None, :] == slot_ids[:, None])
        matched = match.any(1)
        base = torch.where(match, base_offset[None, :], 0).sum(1)
        o.draft_offset.copy_(torch.where(matched, base + 4 * (w - 1), o.draft_offset))
        o.draft_generation.copy_(torch.where(matched, o.requests.generation, o.draft_generation))
        candidates = torch.cat((pending[:, None], drafts), dim=1)
        metadata = self.target_metadata(slots, limits, active, before, candidates)
        records = o.records()
        hidden, logits, _, _ = self.decoder(metadata, records=records)
        logits = logits.view(b, w, -1)
        target.run(scratch['target'], logits, candidates=drafts)
        acceptance = verify.run(scratch['verify'], probabilities, scratch['target']['probabilities'],
            scratch['target']['valid'], drafts=drafts, draft_valid=valid)
        committed = commit(logits, acceptance=acceptance, inputs=candidates)
        self.commit_state(slots, committed.emitted_count, records, metadata.recurrent, hidden)
        tail = o.pool.draft.index.tail
        save_hidden(tail_backup.view(b, -1), tail.view(tail.shape[0], -1),
                    o.inputs.request_state_rows[safe, 0], active)
        context = torch.cat((previous[:, None], hidden.view(b, w, 4096)[:, :-1]), dim=1)
        absorb_metadata = DraftMetadata.pack(slots, committed.emitted_count.long(), before - 1,
            candidates, context, o.draft_tables, state_rows=o.inputs.request_state_rows, capacity=w,
            shards=o.shards)
        self.absorb(absorb_metadata)
