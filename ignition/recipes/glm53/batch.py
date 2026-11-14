# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed native uploads and captured request metadata for GLM text forwards."""

from dataclasses import dataclass

import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import ERROR, RUNNING, DeviceRequestState
from ignition.components.token_state import PreparedTokenState

from .state import DecoderMetadata


@dataclass(frozen=True)
class BatchInputs:
    request_state_rows: torch.Tensor
    row_slot_ids: torch.Tensor
    row_generations: torch.Tensor
    live_rows: torch.Tensor
    query_lengths: torch.Tensor
    token_ids: torch.Tensor
    page_tables: torch.Tensor
    sample_enabled: torch.Tensor

    @classmethod
    def allocate(cls, tokens, blocks, device, *, capacity=4):
        if any(type(value) is not int or value <= 0 for value in (tokens, blocks)):
            raise ValueError("GLM upload capacities require positive token and page counts")
        return cls(torch.zeros((capacity, 8), dtype=torch.int64, device=device),torch.full((capacity,), -1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int64, device=device), torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device), torch.zeros(tokens, dtype=torch.int64, device=device),
            torch.full((capacity, blocks), -1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class PreparedBatch:
    """Map request slots to packed KDA and sparse-MLA metadata inside a graph.

    Rust uploads prefill spans and request identities to every rank. Decode
    gathers pending tokens and positions from replicated GPU state. Validation precedes persistent
    model-state writes. Invalid or stale rows contribute no tokens or cache
    addresses. Recipes compile these tensor operations together with forward.
    """

    def __init__(self, inputs: BatchInputs, requests: DeviceRequestState, metadata: DecoderMetadata,
                 *, mode, shard: TensorParallelShard, pages, shards=None, validate_prefill=None):
        rows, tokens = metadata.sample_indices.numel(), metadata.token_ids.numel()
        self.capacity = requests.capacity
        if mode not in ("prefill", "decode") or not 1 <= rows <= self.capacity:
            raise ValueError("GLM batches require prefill/decode and rows within request capacity")
        if (mode == "prefill" and rows != 1) or (mode == "decode" and tokens != rows):
            raise ValueError("GLM prefill has one sequence; decode has one token per row")
        if tokens > inputs.token_ids.numel() or inputs.page_tables.shape[1] != metadata.indexed.block_table.shape[1]:
            raise ValueError("GLM upload and metadata capacities differ")
        if not isinstance(pages, torch.Tensor) or pages.shape != (1,) or pages.dtype != torch.int64 or requests.vocab_size != 154880:
            raise ValueError("GLM batches require positive cache capacity and a 154880-token vocabulary")
        self.inputs, self.requests, self.metadata = inputs, requests, metadata
        self.rows, self.tokens, self.pages, self.mode, self.shard = rows, tokens, pages, mode, shard
        self.blocks = inputs.page_tables.shape[1]
        self.sample_count = metadata.sample_indices.numel()
        self.verification_columns = metadata.recurrent.state_indices.shape[1]
        from .dcp import ContextShards

        self.shards = ContextShards() if shards is None else shards
        self.validate_prefill = validate_prefill
        self.token_state = PreparedTokenState(requests, metadata.token_ids, metadata.recurrent.query_start_loc,
            inputs.row_slot_ids[:rows], inputs.row_generations[:rows], inputs.live_rows)
        self.token_state.prepare()

    def functional(self):
        self.token_state.functional()
        self.metadata = None
        return self

    def validate(self):
        b, r = self.inputs, self.requests
        slots = b.row_slot_ids[:self.rows].long()
        safe_slots = slots.clamp(0, self.capacity - 1)
        row_ids = torch.arange(self.rows, device=r.generation.device)
        slot_ids = torch.arange(self.capacity, device=r.generation.device)
        block_ids = torch.arange(b.page_tables.shape[1], device=r.generation.device)
        token_offsets = torch.arange(self.tokens, device=r.generation.device)
        resident = (row_ids < b.live_rows) & (slots >= 0) & (slots < self.capacity)
        resident &= b.row_generations[:self.rows] == r.generation[safe_slots]
        duplicate = ((slots[:, None] == slots[None, :]) & resident[:, None] & resident[None, :]).sum(1) > 1
        lengths = b.query_lengths[:self.rows] if self.mode == "prefill" else torch.ones_like(safe_slots)
        positions = r.consumed_tokens[safe_slots]
        ends = positions + lengths
        valid = ~duplicate & (b.live_rows <= self.rows) & (lengths > 0) & (lengths <= self.tokens)
        page_tokens = self.shards.page_tokens
        valid &= (positions >= 0) & (ends <= b.page_tables.shape[1] * page_tokens)
        pages = b.page_tables[safe_slots]
        valid &= (((pages >= 0) & (pages < self.pages))
                  | (block_ids >= (ends[:, None] + page_tokens - 1) // page_tokens)).all(1)
        pending = r.pending_token[safe_slots]
        uploaded = b.token_ids[:self.tokens]
        if self.mode == "prefill":
            invalid_tokens = ((uploaded < 0) | (uploaded >= r.vocab_size)) & (token_offsets < lengths[0])
            valid &= (pending < 0) & ~invalid_tokens.any()
            if self.validate_prefill is not None:
                valid &= self.validate_prefill(uploaded, lengths)
        else:
            valid &= (pending >= 0) & (pending < r.vocab_size)
        present = resident & (r.status[safe_slots] == RUNNING)
        bad = present & ~valid
        bad_slots = ((slot_ids[:, None] == slots[None, :]) & bad[None, :]).any(1)
        present_slots = ((slot_ids[:, None] == slots[None, :]) & present[None, :]).any(1)
        validation_input = torch.zeros((1, max(8, 2 * self.capacity)), dtype=torch.bfloat16,
                                       device=r.generation.device)
        validation_input[0, :self.capacity].copy_(bad_slots)
        validation_input[0, self.capacity:2 * self.capacity].copy_(present_slots)
        return validation_input, present_slots

    def begin(self, validation_sum, present):
        participants = validation_sum[0, self.capacity:2 * self.capacity]
        mismatch = (participants > 0) & (participants < self.shard.world_size)
        bad_slots = present & ((validation_sum[0, :self.capacity] > 0) | mismatch)
        self.requests.status.copy_(torch.where(bad_slots, ERROR, self.requests.status))
        self.requests.error_code.copy_(torch.where(bad_slots, 32, self.requests.error_code))
        pending_tokens, positions, _, active_rows = self.token_state.gather()
        b = self.inputs
        slots = b.row_slot_ids[:self.rows].long()
        active = active_rows.bool()
        lengths = b.query_lengths[:self.rows] if self.mode == "prefill" else torch.ones_like(b.query_lengths[:self.rows])
        lengths = torch.where(active, lengths, 0)
        ends = lengths.cumsum(0, dtype=torch.int32)
        starts = ends - lengths
        token_offsets = torch.arange(self.tokens, device=ends.device)
        request = (token_offsets[:, None] >= ends[None, :]).sum(1).clamp(max=self.rows - 1)
        live_token = token_offsets < ends[-1]
        position = positions[request] + token_offsets - starts[request]
        tokens = b.token_ids[:self.tokens] if self.mode == "prefill" else pending_tokens[request]
        metadata = DecoderMetadata.allocate(self.tokens, self.rows, self.blocks, self.inputs.token_ids.device,
            samples=self.sample_count, verification_columns=self.verification_columns)
        g, index = metadata.recurrent, metadata.indexed
        query_start_loc = torch.cat((torch.zeros((1,), dtype=torch.int32, device=ends.device), ends))
        state = b.request_state_rows[slots.clamp(0, self.capacity - 1), 0]
        pages = torch.where(active[:, None], b.page_tables[slots.clamp(0, self.capacity - 1)], -1)
        g.query_start_loc.copy_(query_start_loc)
        g.convolution_slots.copy_(torch.where(active, state, -1))
        g.fresh.copy_(active & (positions == 0))
        g.accepted_tokens.fill_(1)
        state_columns = torch.zeros_like(g.state_indices)
        state_columns[:, 0] = torch.where(active, state, 0)
        g.state_indices.copy_(state_columns)
        g.initial_state_indices.copy_(torch.where(active & (positions > 0), state, 0))
        g.final_state_indices.copy_(torch.where(active, state, 0))
        g.checkpoint_state_indices.zero_()
        g.checkpoint_offsets.zero_()
        g.num_seqs.copy_(b.live_rows.clamp(0, self.rows))
        g.num_tokens.copy_(ends[-1:])
        metadata.sample_indices.copy_((ends - 1).clamp(min=0))
        index.query_start_loc.copy_(query_start_loc)
        index.state_slots.copy_(g.convolution_slots)
        index.block_table.copy_(pages)
        index.requests.copy_(torch.where(live_token, request, 0))
        index.positions.copy_(torch.where(live_token, position, -1))
        index.token_slots.copy_(torch.where(live_token, self.shards.token_slots(position, pages[request]), -1))
        metadata.token_ids.copy_(torch.where(live_token, tokens, 0))
        return metadata

    def finish(self, metadata):
        self.token_state.consume(metadata.token_ids, metadata.recurrent.query_start_loc)
