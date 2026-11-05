# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen hyperconnection features and target state for shared DFlash2 rounds."""

from types import SimpleNamespace

import torch

from ignition.components.dflash2 import PreparedContext
from ignition.components.dflash2_preparation import DFlash2Preparation, PreparedContextAbsorption
from ignition.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit
from ignition.components.parallel_draft import PreparedParallelDraftRound
from ignition.components.recurrent import DeltaMetadata
from ignition.components.token_state import PreparedTokenState
from ignition.preparation.scratch import GraphBufferPool

from .attention import QSAMetadata
from .decoder import PreparedDecoder
from .ple import PLEMetadata


class PreparedTargetState:
    """Translate packed verification rows to QSA, GDN and PLE metadata."""

    def __init__(self, owner, metadata):
        self.owner, self.metadata = owner, metadata
        self.inputs, self.requests, self.pool = owner.inputs, owner.requests, owner.pool
        self.tokens, self.rows = metadata.token_ids.numel(), metadata.starts.numel()
        self.mode = "decode"
        device = metadata.token_ids.device
        self.gdn = DeltaMetadata.allocate(self.rows, device,
            verification_columns=metadata.max_query_tokens)
        self.qsa = QSAMetadata.allocate(tokens=self.tokens, batch=self.rows,
            context=owner.context, device=device)
        self.token_state = PreparedTokenState(owner.requests, metadata.token_ids,
            metadata.query_start_loc, owner.inputs.row_slot_ids[:self.rows],
            owner.inputs.row_generations[:self.rows], owner.inputs.live_rows,
            history=owner.requests.ple_history, history_pad_id=248044)
        self.token_state.prepare()
        self.ple = PLEMetadata(metadata.token_ids, self.gdn.query_start_loc,
            self.token_state.history, self.gdn.convolution_slots, self.gdn.fresh,
            self.gdn.accepted_tokens, self.gdn.num_seqs, self.gdn.num_tokens)
        self.slot_ids = torch.arange(owner.requests.capacity, device=device)

    def prepare_hidden(self, hidden, sample):
        if sample.shape != (self.metadata.sample_indices.numel(), 2560):
            raise ValueError("Qwen DFlash2 sampling rows must match logical verification rows")
        self.hidden, self.sample = hidden, sample

    def prepare_residual(self, residual):
        pass

    def begin(self):
        m, g, q, o = self.metadata, self.gdn, self.qsa, self.owner
        self.token_state.gather()
        active = m.lengths > 0
        slots = o.inputs.row_slot_ids[:self.rows].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        g.query_start_loc.copy_(m.query_start_loc)
        g.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        g.fresh.zero_()
        g.accepted_tokens.copy_(torch.where(active, o.previous_accepted[safe], 1))
        g.state_indices.zero_()
        g.state_indices[:, 0].copy_(m.state_rows)
        g.initial_state_indices.copy_(m.state_rows)
        g.final_state_indices.copy_(m.state_rows)
        g.num_seqs.copy_(m.live_counts[1:])
        g.num_tokens.copy_(m.live_counts[:1])
        q.query_start_loc.copy_(m.query_start_loc)
        q.raw_state_slot_ids.copy_(torch.where(active, slots + 1, -1))
        q.num_accepted_tokens.copy_(g.accepted_tokens)
        q.is_prefilling.zero_()
        q.sequence_lengths.copy_(torch.where(active, m.starts + m.lengths, 0))
        q.main_block_table.copy_(m.page_table)
        q.compressed_block_table.copy_(m.page_table)
        valid = m.positions >= 0
        q.request_ids.copy_(torch.where(valid, m.requests, -1))
        q.query_positions.copy_(m.positions)
        delta = o.pool.rope_delta[m.state_rows[m.requests], 0]
        rope = torch.where(valid, m.positions + delta, -1)
        q.rope_positions.copy_(rope[:, None].expand(-1, 3))
        page = m.page_table[m.requests, (m.positions // 64).clamp(0, m.page_table.shape[1] - 1)]
        q.slot_mapping.copy_(torch.where(valid, page.long() * 64 + m.positions.remainder(64), -1))

    def retain_accepted(self, counts):
        slots = self.inputs.row_slot_ids[:self.rows]
        match = (self.slot_ids[:, None] == slots[None, :]) & (counts[None, :] > 0)
        accepted = torch.where(match, counts[None, :], 0).sum(1).to(torch.int32)
        self.owner.previous_accepted.copy_(torch.where(match.any(1), accepted,
            self.owner.previous_accepted))

    def finish(self):
        self.sample.copy_(self.hidden[self.metadata.sample_indices])
        if self.metadata.max_query_tokens == 1:
            self.retain_accepted(self.metadata.lengths)


class PreparedTarget:
    """Expose Qwen's target forward to the shared speculative round."""

    def __init__(self, owner, scratch, metadata, *, session, name, workspace, mode, **collectives):
        self.state = PreparedTargetState(owner, metadata)
        wide = metadata.max_query_tokens > 1
        self.forward = PreparedDecoder(owner.target_weights, scratch, self.state,
            owner.cache, owner.rotary, owner.table, workspace,
            owner.completion, owner.frame, owner.sequence, session=session, name=name,
            max_cache_pages=owner.max_cache_pages, verification=True,
            gdn_records=owner.records if wide else None, gdn_in_place=not wide, **collectives)
        self.forward.capture_context(metadata, owner.weights.config.draft,
            buffers=owner.target_buffers.cursor())
        self.auxiliary = self.forward.auxiliary
        self.scratch = SimpleNamespace(logits=self.forward.scratch.logits,
            hidden=self.forward.scratch.sample_hidden)
        self.commit_counts = torch.zeros_like(metadata.lengths)
        self.replay = None
        if wide:
            heads = owner.records.delta.shape[2]
            self.replay = PreparedGDNCommit(owner.pool.backing,
                {layer: state.recurrent for layer, state in owner.pool.gdn.items()},
                owner.records, self.state.gdn, self.commit_counts,
                key_heads=heads // 3, value_heads=heads)
        owner.decoders.append(self.forward)

    def __call__(self):
        self.forward()

    def commit(self, counts):
        self.commit_counts.copy_(counts)
        if self.replay is not None:
            self.replay()
        self.state.retain_accepted(counts)


class SpeculativePreparation(DFlash2Preparation):
    def __init__(self, *, weights, draft_weights, draft_config, pool, requests, inputs, scratch,
                 workspaces, rotary, cache, table, session, completion, frame, sequence,
                 context, prefill_capacity, max_cache_pages, draft_head=None,
                 runtime=None, vocabulary_runtime=None, coordinator=None, verification_width=8,
                 sampling_workspace=None, graph_buffers=None, gather_runtime=None):
        self.target_weights, self.cache, self.rotary, self.table = weights, cache, rotary, table
        self.context, self.max_cache_pages = context, max_cache_pages
        device = pool.backing.device
        self.target_buffers, self.context_buffers, self.draft_buffers, self.round_buffers = (
            GraphBufferPool(device) for _ in range(4))
        self.decoders, self.owners = [], []
        self.previous_accepted = torch.ones(requests.capacity, dtype=torch.int32, device=device)
        self.draft_tables = None
        heads = next(iter(pool.gdn.values())).recurrent.shape[1]
        self.records = GDNLayerRecords.allocate(pool.gdn, requests.capacity, device,
            key_heads=heads // 3, value_heads=heads, width=verification_width)
        facade = SimpleNamespace(config=SimpleNamespace(
            text={"hidden_size": 2560, "vocab_size": 248320}, draft=draft_config),
            draft=draft_weights, shard=weights.shard)

        def decoder(weights, tables, scratch, metadata, pool, **kwargs):
            return PreparedTarget(self, scratch, metadata, **kwargs)

        first, last = weights.shard.partition(2560, alignment=64)
        super().__init__(facade, pool, requests, inputs, scratch, decoder=decoder,
            embedding=weights.tree.model.language_model.embed_tokens.weight[:, first:last],
            head=weights.tree.lm_head.weight if draft_head is None else draft_head,
            page_tokens=64, completion=completion, frame=frame, sequence=sequence,
            session=session, workspace=workspaces, buffers=self.draft_buffers.cursor(),
            context_buffers=self.context_buffers, verification_width=verification_width,
            runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator,
            block_size=draft_config.block_size, gather_runtime=gather_runtime)

    def buffers(self):
        return {}

    def round(self, batch, width, name):
        operation = PreparedParallelDraftRound(self, batch, width, name,
            buffers=self.round_buffers.cursor())
        operation.prepare()
        self.owners.append(operation)
        return operation

    def prefill(self, decoder, name):
        s = decoder.state
        metadata = SimpleNamespace(positions=s.qsa.query_positions,
            starts=s.token_state.positions, lengths=s.inputs.query_lengths[:s.rows],
            requests=s.qsa.request_ids, state_rows=s.gdn.convolution_slots,
            query_start_loc=s.gdn.query_start_loc)
        decoder.capture_context(metadata, self.weights.config.draft,
            buffers=self.target_buffers.cursor(), compact_prefill=True)
        context = PreparedContext(self.weights, decoder.auxiliary, metadata, self.pool,
            session=self.session, name=name + ".context", workspace=self.workspace,
            runtime=self.gather_runtime, coordinator=self.collectives["coordinator"],
            buffers=self.context_buffers.cursor(), compact_prefill=True)
        operation = PreparedContextAbsorption(decoder, context, metadata)
        self.owners.append(operation)
        return operation

    def resume(self, resume):
        return resume

    def rebind_pages_for_preparation(self, cache):
        self.cache = cache

    def scratch_tensors(self):
        values = [super().scratch_tensors(), self.records, self.target_buffers,
                  self.context_buffers, self.draft_buffers, self.round_buffers]
        for operation in self.owners:
            if isinstance(operation, PreparedContextAbsorption):
                continue
            state = operation.decoder.state
            values.extend((state.gdn, state.qsa, operation.decoder.commit_counts,
                state.token_state.pending_tokens, state.token_state.positions,
                state.token_state.history, state.token_state.active))
        return tuple(values)
