# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for causal-round operand wiring with substituted GPU operators."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import causal_draft


@pytest.mark.parametrize("batch", (1, 2))
@pytest.mark.parametrize("width", (1, 3))
def test_verification_receives_complete_probabilities_and_validity(monkeypatch, batch, width):
    vocabulary, hidden, capacity = 7, 4, batch + 2
    slots = torch.arange(batch - 1, -1, -1, dtype=torch.int32)
    operation = causal_draft.PreparedCausalDraftRound.__new__(causal_draft.PreparedCausalDraftRound)
    operation.batch, operation.width, operation.blocks, operation.hidden = batch, width, 1, hidden
    state_rows = torch.zeros((capacity, 8), dtype=torch.int64)
    state_rows[:, 0] = torch.arange(1, capacity + 1)
    operation.inputs = SimpleNamespace(row_slot_ids=slots, query_lengths=torch.full((batch,), width, dtype=torch.int32),
        request_state_rows=state_rows, page_tables=torch.ones((capacity, 1), dtype=torch.int32),
        live_rows=torch.tensor([batch], dtype=torch.int32))
    operation.requests = SimpleNamespace(capacity=capacity, vocab_size=vocabulary,
        consumed_tokens=torch.arange(capacity) + 10, pending_token=torch.arange(capacity),
        generation=torch.ones(capacity, dtype=torch.int64), seed=torch.arange(capacity))
    operation.draft_offset = torch.zeros(capacity, dtype=torch.int64)
    operation.draft_generation = torch.zeros_like(operation.draft_offset)
    operation.pool = SimpleNamespace(last_hidden=torch.zeros((capacity + 1, hidden)))
    operation.validator = SimpleNamespace(validate=lambda: (torch.zeros(1), torch.ones(capacity)),
        begin=lambda value, *, present: SimpleNamespace(lengths=torch.ones(batch, dtype=torch.int32)))
    operation.validation = lambda value: value
    operation.sampling_layout = {
        "target": {"probabilities": ((batch, width, vocabulary), torch.float32),
                   "valid": ((batch, width), torch.bool)},
        "proposal": {"probabilities": ((batch, 1, vocabulary), torch.float32),
                     "valid": ((batch, 1), torch.bool)},
        "verify": {},
    }
    observed = {}
    auxiliary, records, acceptance = object(), object(), object()
    target_hidden = torch.arange(batch * width * hidden).reshape(batch * width, hidden).float()
    target_logits = torch.arange(batch * width * vocabulary).reshape(batch * width, vocabulary).float()

    class Decoder:
        def __call__(self, metadata):
            observed["metadata"] = metadata
            return target_hidden, target_logits, auxiliary, records

        def commit(self, counts, passed_records, metadata):
            assert passed_records is records and metadata is observed["metadata"]
            observed["decoder_commit"] = counts

    operation.decoder = Decoder()
    operation.draft = SimpleNamespace(begin=lambda states: states,
        step=lambda state, index, *args: (torch.zeros((batch, vocabulary)), state),
        finish=lambda *args: None)

    def propose(scratch, logits, *, sampled, **kwargs):
        sampled.copy_(torch.arange(batch) + 3)
        scratch["probabilities"].fill_(1 / vocabulary)
        scratch["valid"].fill_(True)

    def target(scratch, logits, *, candidates):
        scratch["probabilities"].copy_(logits.softmax(-1))
        scratch["valid"].fill_(True)
        scratch["valid"][-1, -1] = False
        observed["probabilities"], observed["valid"] = scratch["probabilities"], scratch["valid"]
        return scratch["probabilities"]

    def verify(scratch, draft_probabilities, target_probabilities, target_valid, **kwargs):
        assert target_probabilities is observed["probabilities"]
        assert target_probabilities.shape == (batch, width, vocabulary)
        assert target_valid is observed["valid"] and target_valid.dtype == torch.bool
        assert target_valid.shape == (batch, width)
        torch.testing.assert_close(target_probabilities, target_logits.view(batch, width, -1).softmax(-1))
        assert not target_valid[-1, -1]
        return acceptance

    def commit(logits, *, acceptance, inputs):
        assert acceptance is observed["acceptance"]
        torch.testing.assert_close(inputs[:, 0], operation.requests.pending_token[slots.long()])
        return SimpleNamespace(emitted_count=torch.ones(batch, dtype=torch.int64))

    observed["acceptance"] = acceptance
    operation.operations = (SimpleNamespace(run=target),
        tuple(SimpleNamespace(run=propose) for _ in range(width - 1)), SimpleNamespace(run=verify), commit)

    def context(passed_auxiliary, counts, metadata):
        assert passed_auxiliary is auxiliary and metadata is observed["metadata"]
        observed["context_counts"] = counts

    operation.context = context

    def save_hidden(value, destination, rows, active):
        destination.index_copy_(0, rows[active], value[active])
        observed["saved_hidden"] = value

    monkeypatch.setattr(causal_draft, "save_hidden", save_hidden)
    metadata, passed_acceptance, committed = operation()
    assert metadata is observed["metadata"] and passed_acceptance is acceptance
    assert observed["context_counts"] is committed.emitted_count
    torch.testing.assert_close(observed["saved_hidden"], target_hidden.view(batch, width, hidden)[:, 0])
    if width > 1:
        assert observed["decoder_commit"] is committed.emitted_count
