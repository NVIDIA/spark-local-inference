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
from ignition.preparation.scratch import GraphBufferPool
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
    return (torch.tensor([r.offset // 2 for r in convolution], dtype=torch.int64, device=pool.backing.device),
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

            self.lengths = torch.zeros(state_indices.shape[0], dtype=torch.int32, device=pool.backing.device)
            self.replay = PreparedKDAReplay(pool.backing, states, records, state_indices, self.lengths, num_seqs)
        else:
            self.replay = PreparedRecordedKDAReplay(pool.backing, states, records, plan, state_indices,
                session=session, workspace=workspace)

    def __call__(self, counts):
        if self.plan is None:
            self.lengths.copy_(counts)
            self.replay()
        else:
            self.replay(counts)


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, scratch, prefill_scratch, *, session,
                 workspace, completion, frame, sequence, runtime, vocabulary_runtime, coordinator, max_cache_pages=None,
                 prefill_tokens=None, decode_tokens=None, context=None):
        self.weights, self.pool, self.requests, self.inputs = weights, pool, requests, inputs
        self.max_cache_pages, self.context = max_cache_pages, context
        self.shards = ContextShards() if context is None else context.shards
        blocks, device = inputs.page_tables.shape[1], pool.backing.device
        self.sampling_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
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
        # A checkpoint declaring index_share_for_mtp_iteration attends every draft pass of a round through
        # its first pass's sparse selection.
        self.share_draft_selection = bool(getattr(weights.tree.config.text_config,
                                                  "index_share_for_mtp_iteration", False))
        # Verification records every KDA layer's tokens, so each request keeps one recurrent row.
        layers = delta_layers(pool)
        recurrent = pool.decoder.layers[layers[0]].recurrent
        self.kda_records = KDARecords.allocate({i: weights.layers[i].attention for i in layers},
            recurrent.shape[0], recurrent.shape[1], device)

    def draft(self, metadata, name, *, cache_only=False, mode="decode"):
        shared = self.draft_scratch.get(mode)
        scratch = None if shared is None else shared.rows(metadata.token_ids.numel(), mode)
        return PreparedMTP(self.weights, self.pool.draft, metadata, mode=mode,
            session=self.session, name=name, workspace=self.workspace, cache_only=cache_only,
            max_cache_pages=self.max_cache_pages, scratch=scratch, context=self.context, **self.collectives)


class PreparedPrefill:
    """Absorb every prompt pair, retaining normalized target feedback at its endpoint."""

    def __init__(self, owner, generation, feedback, name):
        self.owner, self.generation, self.feedback = owner, generation, feedback
        tokens = feedback.shape[0]
        device = feedback.device
        self.metadata = DraftMetadata(tokens, 1, owner.inputs.page_tables.shape[1], device, shards=owner.shards)
        self.absorb = owner.draft(self.metadata, f"{name}.absorb", cache_only=True, mode="prefill")
        self.previous = torch.zeros((1, 4096), dtype=torch.bfloat16, device=device)
        self.indices = torch.arange(tokens, device=device)
        self.before = torch.zeros(1, dtype=torch.int64, device=device)

    def rebind_pages(self, pages):
        self.absorb.rebind_pages()
        self.generation.rebind_pages(pages)

    def __call__(self):
        o, g = self.owner, self.generation
        slot = o.inputs.row_slot_ids[:1].long().clamp(0, o.requests.capacity - 1)
        self.before.copy_(o.requests.consumed_tokens[slot])
        gather_rows(o.pool.last_hidden, o.inputs.request_state_rows[slot, 0], self.previous)
        g()
        length = g.batch.metadata.recurrent.num_tokens.long()
        skip = (self.before == 0).long()
        indices = (self.indices + skip[0]).clamp_max(self.indices.numel() - 1)
        hidden = self.feedback[(indices - 1).clamp_min(0)]
        hidden = torch.where(((self.indices == 0) & (self.before[0] > 0))[:, None], self.previous, hidden)
        self.metadata.set(o.inputs.row_slot_ids[:1], (length - skip).clamp_min(0),
            (self.before - 1).clamp_min(0), g.batch.metadata.token_ids[indices], hidden,
            o.draft_tables, state_rows=o.inputs.request_state_rows, capacity=self.indices.numel())
        self.absorb()


class PreparedRound:
    """Rank zero's proposal and acceptance decisions are broadcast before dependent work."""

    def __init__(self, owner, batch, width, name):
        self.owner, self.batch, self.width = owner, batch, width
        o, b, w = owner, batch, width
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        from ignition.components.parallel_draft import prepare_sampling_scratch

        buffers = o.sampling_buffers.cursor()
        self.sampling_scratch = prepare_sampling_scratch(b, w, o.requests.vocab_size, device, buffers=buffers)
        self.rows, self.columns = torch.arange(b, device=device), torch.arange(w, device=device)
        self.offsets = torch.arange(b * w, device=device)
        self.state_columns = torch.arange(8, device=device)
        self.convolution_regions = convolution_regions(o.pool)
        self.metadata = DecoderMetadata.allocate(b * w, b, blocks, device)
        self.metadata = DecoderMetadata(self.metadata.token_ids, torch.arange(b * w, device=device),
            DeltaMetadata.allocate(b, device, verification_columns=8), self.metadata.indexed)
        self.validator = PreparedBatch(o.inputs, o.requests, DecoderMetadata.allocate(b, b, blocks, device),
            mode="decode", shard=o.weights.shard, pages=o.pool.memory_capacity[1:], shards=o.shards)
        self.validation = PreparedAllReduce(self.validator.validation_input, self.validator.validation_sum,
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
        self.draft_logits = (buffers.zeros(self.draft.logits.shape, self.draft.logits.dtype)
                             if self.draft is not None else None)
        self.selection = None
        if o.share_draft_selection and w > 2:
            index = self.draft.attention.scratch.index
            self.selection = (buffers.zeros(index.physical.shape, index.physical.dtype),
                              buffers.zeros(index.selected_lengths.shape, index.selected_lengths.dtype))
        self.absorb_metadata = DraftMetadata(b * w, b, blocks, device, shards=o.shards)
        self.absorb = o.draft(self.absorb_metadata, f"{name}.absorb", cache_only=True)
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.pending = torch.zeros_like(self.before)
        self.previous = torch.zeros((b, 4096), dtype=torch.bfloat16, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        self.probabilities = buffers.zeros((b, w - 1, o.requests.vocab_size), torch.float32)
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        self.seed, self.offset, self.base_offset, self.sampled = (torch.zeros_like(self.before) for _ in range(4))
        self.tail_backup = torch.empty_like(o.pool.draft.index.tail[:b])
        self.broadcast_buffers, self.name = buffers, name
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
                                    name=self.name, buffers=self.broadcast_buffers)
        commit = PreparedSpeculativeCommit(o.requests, verify.output, self.candidates,
            o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows,
            o.completion, o.frame, o.sequence, logits=self.decoder.scratch.logits.view(b, w, -1), rng=o.rng, token_limits=o.inputs.query_lengths[:b])
        commit.prepare()
        self.operations = target, proposals, verify, commit

    def rebind_pages(self, pages):
        self.decoder.rebind_pages()
        for draft in (self.draft, self.absorb):
            if draft is not None:
                draft.rebind_pages()
        self.validator.pages = self.owner.pool.memory_capacity[1:]

    def retain_selection(self):
        """Keep each request's first-pass selection from its output row.

        Requests that propose a later draft proposed the first one; other requests' copies are never read.
        """
        index, rows = self.draft.attention.scratch.index, self.draft_metadata.output_indices
        physical, lengths = self.selection
        torch.index_select(index.physical, 0, rows, out=physical)
        torch.index_select(index.selected_lengths, 0, rows, out=lengths)

    def reuse_selection(self):
        """Place each proposing request's first-pass selection in its packed draft row; idle rows select nothing."""
        index, m = self.draft.attention.scratch.index, self.draft_metadata.indexed
        physical, lengths = self.selection
        requests = m.requests.long()
        torch.index_select(physical, 0, requests, out=index.physical)
        index.selected_lengths.copy_(torch.where(m.positions >= 0, lengths[requests], 0))

    def target_metadata(self, slots, limits):
        o, b, w, m = self.owner, self.batch, self.width, self.metadata
        g, index = m.recurrent, m.indexed
        lengths = torch.where(self.active, limits, 0)
        ends = lengths.cumsum(0)
        starts = ends - lengths
        g.query_start_loc[0].zero_()
        g.query_start_loc[1:].copy_(ends)
        states = o.inputs.request_state_rows[slots.clamp(0, o.requests.capacity - 1)]
        g.convolution_slots.copy_(torch.where(self.active, states[:, 0], -1))
        g.fresh.zero_()
        g.accepted_tokens.fill_(1)
        # Verification reads each request's live row in column zero and records its tokens.
        states = torch.where(self.state_columns < 1, states, 0)
        g.state_indices.copy_(torch.where(self.active[:, None], states, 0))
        g.num_seqs.copy_(o.inputs.live_rows.clamp(0, b))
        g.num_tokens.copy_(ends[-1:])
        index.query_start_loc.copy_(g.query_start_loc)
        index.state_slots.copy_(g.convolution_slots)
        index.block_table.copy_(o.inputs.page_tables[slots.clamp(0, o.requests.capacity - 1)])
        row = (self.offsets[:, None] >= ends[None]).sum(1).clamp_max(b - 1)
        column = self.offsets - starts[row]
        active = self.offsets < ends[-1]
        pos = self.before[row] + column
        index.requests.copy_(row)
        index.positions.copy_(torch.where(active, pos, -1))
        index.token_slots.copy_(torch.where(active, o.shards.token_slots(pos, index.block_table[row]), -1))
        source = (row * w + column).clamp(0, b * w - 1)
        m.token_ids.copy_(torch.where(active, self.candidates.reshape(-1)[source], 0))
        m.sample_indices.copy_((starts[:, None] + self.columns).clamp(0, b * w - 1).reshape(-1))

    def commit_state(self, slots, counts):
        """Replay each live KDA row's accepted records for every layer, then shift its convolution history."""
        o, w = self.owner, self.width
        states = o.inputs.request_state_rows[slots.clamp(0, o.requests.capacity - 1)]
        live = torch.where(counts > 0, states[:, 0], 0).long()
        self.kda_commit(counts)
        backing = o.pool.backing
        halves, channels, columns = self.convolution_regions
        args = (backing.view(torch.bfloat16), slots, counts, o.inputs.request_state_rows, halves,
                backing.shape[1] // 2, channels, columns, 128)
        grid = (halves.numel(), self.batch, triton.cdiv(channels, 128))
        if torch.compiler.is_compiling() or get_proxy_mode() is not None:
            torch.library.wrap_triton(_shift_convolution)[grid](*args, num_warps=4)
        else:
            _shift_convolution[grid](*args, num_warps=4)
        hidden = self.decoder.scratch.hidden.view(self.batch, w, 4096)
        selected_hidden = hidden[self.rows, (counts - 1).clamp_min(0).long()]
        save_hidden(selected_hidden, o.pool.last_hidden,
                    live, (counts > 0).int())

    def __call__(self):
        o, b, w = self.owner, self.batch, self.width
        target, proposals, verify, commit = self.operations
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        limits = o.inputs.query_lengths[:b]
        self.validator.validate()
        end = o.requests.consumed_tokens[safe] + limits
        page_tokens = o.shards.page_tokens
        bad = (limits < 1) | (limits > w) | (end > o.inputs.page_tables.shape[1] * page_tokens)
        pages = o.inputs.page_tables[safe]
        required = self.validator.block_ids < (end[:, None] + page_tokens - 1) // page_tokens
        bad |= (required & ((pages < 0) | (pages >= self.validator.pages))).any(1)
        bad_slots = ((self.validator.slot_ids[:, None] == slots[None]) & bad[None]).any(1)
        self.validator.validation_input[0, :o.requests.capacity].add_(bad_slots)
        self.validator.begin(self.validation(self.validator.validation_input))
        self.active.copy_(self.validator.token_state.active.bool())
        self.before.copy_(o.requests.consumed_tokens[safe])
        self.pending.copy_(o.requests.pending_token[safe])
        gather_rows(o.pool.last_hidden, o.inputs.request_state_rows[safe, 0], self.previous)
        gather_rows(o.pool.draft.index.tail, o.inputs.request_state_rows[safe, 0], self.tail_backup)
        self.base_offset.copy_(torch.where(o.draft_generation[safe] == o.requests.generation[safe], o.draft_offset[safe], 0))
        self.seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        for i, proposal in enumerate(proposals):
            hidden = self.previous if i == 0 else self.draft.hidden[self.draft_metadata.output_indices]
            token = self.pending if i == 0 else self.drafts[:, i - 1]
            proposing = self.active & (i < limits - 1)
            self.draft_metadata.set(slots, proposing.long(), self.before - 1 + i,
                token, hidden, o.draft_tables, state_rows=o.inputs.request_state_rows, capacity=1)
            reuse = self.selection is not None and i > 0
            if reuse:
                self.reuse_selection()
            self.draft(select=not reuse)
            if self.selection is not None and i == 0:
                self.retain_selection()
            torch.index_select(self.draft.logits, 0, self.draft_metadata.output_indices, out=self.draft_logits)
            self.offset.copy_(self.base_offset + 4 * i)
            proposal()
            self.probabilities[:, i].copy_(proposal.probabilities[:, 0])
            self.drafts[:, i].copy_(self.sampled)
            self.valid[:, i].copy_(proposal.valid[:, 0] & proposing)
        # One masked update per tensor instead of one storage write per request slot.
        slot_ids = torch.arange(o.requests.capacity, device=slots.device)
        match = self.active[None, :] & (slots[None, :] == slot_ids[:, None])
        matched = match.any(1)
        base = torch.where(match, self.base_offset[None, :], 0).sum(1)
        o.draft_offset.copy_(torch.where(matched, base + 4 * (w - 1), o.draft_offset))
        o.draft_generation.copy_(torch.where(matched, o.requests.generation, o.draft_generation))
        self.candidates[:, 0].copy_(self.pending)
        self.candidates[:, 1:].copy_(self.drafts)
        self.target_metadata(slots, limits)
        self.decoder()
        target()
        verify()
        commit()
        self.commit_state(slots, commit.emitted_count)
        tail = o.pool.draft.index.tail
        save_hidden(self.tail_backup.view(b, -1), tail.view(tail.shape[0], -1),
                    o.inputs.request_state_rows[safe, 0], self.active)
        hidden = torch.cat((self.previous[:, None], self.decoder.scratch.hidden.view(b, w, 4096)[:, :-1]), dim=1)
        self.absorb_metadata.set(slots, commit.emitted_count.long(), self.before - 1,
            self.candidates, hidden, o.draft_tables, state_rows=o.inputs.request_state_rows, capacity=w)
        self.absorb()
