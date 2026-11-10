# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM packed verification and shared commit semantics."""

from types import SimpleNamespace

import pytest
import torch

from tests.components.completion_support import RecordOnlyLogprobs, completion_outputs

from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import PreparedSpeculativeCommit, SpeculativeAcceptance
from ignition.recipes.glm53.batch import BatchInputs
from ignition.recipes.glm53.dcp import ContextShards
from ignition.recipes.glm53.speculative import PreparedRound
from ignition.recipes.glm53.mtp import DraftMetadata
from ignition.recipes.glm53.state import DecoderMetadata
from ignition.components.recurrent import DeltaMetadata


def test_shared_commit_without_previous_token_embeddings():
    state = DeviceRequestState.allocate("cpu", vocab_size=32, capacity=2)
    state.admit(1, RequestAdmission(generation=7, seed=19, temperature=0.7, max_new_tokens=100), [1, 2], consumed_tokens=2)
    state.pending_token[1] = 3
    state.generated_tokens[1] = 1
    acceptance = SpeculativeAcceptance.allocate(1, 6, "cpu")
    acceptance.token_ids[0, :3] = torch.tensor([4, 5, 17])
    acceptance.proposed_count.fill_(3)
    acceptance.accepted_drafts.fill_(2)
    acceptance.offered_drafts.fill_(5)
    acceptance.valid.fill_(True)
    records = torch.zeros((2, 2, 32), dtype=torch.int64)
    commit = PreparedSpeculativeCommit(state, acceptance, torch.tensor([[3, 4, 5, 6, 7, 8]]),
        torch.tensor([1], dtype=torch.int32), torch.tensor([7]), torch.tensor([1], dtype=torch.int32),
        completion_outputs(records), torch.tensor([0], dtype=torch.int32), torch.tensor([1]), logits=torch.zeros((1, 6, state.vocab_size)))
    commit()
    assert state.consumed_tokens.tolist() == [0, 5]
    assert state.pending_token[1] == 17
    assert records[0, 1, 13] == 4
    assert records[0, 1, 16:19].tolist() == [4, 5, 17]


@pytest.mark.parametrize("width,limits,positions,token_slots,state_indices", [
    (1, [1, 1], [255, 510], [10 * 256 + 255, 21 * 256 + 254],
     [[1, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]]),
    (6, [6, 2], [255, 256, 257, 258, 259, 260, 510, 511, -1, -1, -1, -1],
     [10 * 256 + 255, *range(11 * 256, 11 * 256 + 5), 21 * 256 + 254, 21 * 256 + 255],
     # Verification records tokens against each request's live row only.
     [[1, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]]),
])
def test_verification_packs_unequal_lengths_and_retains_request_output_rows(
        width, limits, positions, token_slots, state_indices):
    op = PreparedRound.__new__(PreparedRound)
    op.batch, op.width = 2, width
    inputs = BatchInputs.allocate(128, 2, "cpu", capacity=2)
    inputs.page_tables.copy_(torch.tensor([[10, 11], [20, 21]]))
    inputs.live_rows.fill_(2)
    inputs.request_state_rows.copy_(torch.tensor([[1, 8, 9, 10, 11, 12, 13, 14], [2, 15, 16, 17, 18, 19, 20, 21]]))
    op.owner = SimpleNamespace(inputs=inputs, requests=SimpleNamespace(capacity=2), shards=ContextShards())
    m = DecoderMetadata.allocate(2 * width, 2, 2, "cpu")
    op.metadata = DecoderMetadata(m.token_ids, torch.zeros(2 * width, dtype=torch.int64),
        DeltaMetadata.allocate(2, "cpu", verification_columns=8), m.indexed)
    op.active = torch.tensor([True, True])
    op.before = torch.tensor([255, 510])
    op.offsets, op.columns, op.state_columns = torch.arange(2 * width), torch.arange(width), torch.arange(8)
    op.candidates = torch.arange(2 * width).view(2, width)
    op.target_metadata(torch.tensor([0, 1]), torch.tensor(limits, dtype=torch.int32))
    assert op.metadata.recurrent.query_start_loc.tolist() == [0, limits[0], sum(limits)]
    assert op.metadata.recurrent.num_tokens.item() == sum(limits)
    assert op.metadata.indexed.positions.tolist() == positions
    assert op.metadata.indexed.token_slots.tolist()[:sum(limits)] == token_slots
    # Columns past the round's width stay on the null row instead of reaching the next slot's rows.
    assert op.metadata.recurrent.state_indices.tolist() == state_indices


def test_draft_rows_exclude_exhausted_requests_and_restore_output_order():
    metadata = DraftMetadata(3, 3, 2, "cpu")
    slots = torch.tensor([2, 0, 1])
    state_rows = torch.arange(1, 4)[:, None].expand(3, 8)
    tables = torch.tensor([[10, 11], [20, 21], [30, 31]])
    hidden = torch.arange(3, dtype=torch.bfloat16)[:, None].expand(3, 4096)
    metadata.set(slots, torch.tensor([0, 1, 1]), torch.tensor([512, 255, 510]),
                 torch.tensor([7, 8, 9]), hidden, tables, capacity=1, state_rows=state_rows)
    assert metadata.indexed.query_start_loc.tolist() == [0, 0, 1, 2]
    assert metadata.indexed.positions.tolist() == [255, 510, -1]
    assert metadata.indexed.token_slots.tolist() == [10 * 256 + 255, 21 * 256 + 254, -1]
    assert metadata.token_ids[:2].tolist() == [8, 9]
    assert metadata.hidden[metadata.output_indices][1:, 0].tolist() == [1, 2]
    metadata.set(slots, torch.zeros(3, dtype=torch.int64), torch.full((3,), 512),
                 torch.tensor([7, 8, 9]), hidden, tables, capacity=1, state_rows=state_rows)
    assert metadata.num_tokens.item() == 0
    assert metadata.indexed.positions.tolist() == [-1, -1, -1]
    assert metadata.indexed.token_slots.tolist() == [-1, -1, -1]


def test_later_draft_passes_reuse_each_requests_first_selection():
    op = PreparedRound.__new__(PreparedRound)
    op.draft_metadata = DraftMetadata(3, 3, 2, "cpu")
    index = SimpleNamespace(physical=torch.zeros((3, 2051), dtype=torch.int32),
                            selected_lengths=torch.zeros(3, dtype=torch.int32))
    op.draft = SimpleNamespace(attention=SimpleNamespace(scratch=SimpleNamespace(index=index)))
    op.selection = (torch.zeros((3, 2051), dtype=torch.int32), torch.zeros(3, dtype=torch.int32))
    slots, starts = torch.tensor([2, 0, 1]), torch.tensor([255, 510, 300])
    state_rows = torch.arange(1, 4)[:, None].expand(3, 8)
    tables = torch.tensor([[10, 11], [20, 21], [30, 31]])
    hidden = torch.zeros((3, 4096), dtype=torch.bfloat16)
    tokens = torch.tensor([7, 8, 9])
    # The first pass packs requests one and two; request zero has no draft to propose.
    op.draft_metadata.set(slots, torch.tensor([0, 1, 1]), starts, tokens, hidden, tables,
                          capacity=1, state_rows=state_rows)
    first = {1: torch.tensor([3, 5, 9]), 2: torch.tensor([4, 6, 8, 12])}
    index.physical.fill_(-1)
    for row, request in enumerate((1, 2)):
        index.physical[row, :first[request].numel()] = first[request]
        index.selected_lengths[row] = first[request].numel()
    op.retain_selection()
    # The second pass keeps only request two, which now occupies packed row zero.
    op.draft_metadata.set(slots, torch.tensor([0, 0, 1]), starts + 1, tokens, hidden, tables,
                          capacity=1, state_rows=state_rows)
    index.physical.fill_(-2)
    index.selected_lengths.fill_(-2)
    op.reuse_selection()
    assert index.selected_lengths.tolist() == [4, 0, 0]
    assert index.physical[0, :4].tolist() == first[2].tolist()


@pytest.fixture(autouse=True)
def records_only_cpu_commit(monkeypatch):
    # These state-machine tests run without CUDA; reduction and graph ordering
    # have independent CUDA coverage.
    monkeypatch.setattr("ignition.components.speculation.PreparedLogprobs", RecordOnlyLogprobs)
