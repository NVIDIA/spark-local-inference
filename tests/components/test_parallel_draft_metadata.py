# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Proposal composition keeps metadata and sampling intermediates local to each call."""

from types import SimpleNamespace

import torch

from ignition.components.parallel_draft import PreparedParallelDraftRound
from ignition.components.speculation import DRAFT_RNG_DOMAIN


class Proposal:
    has_source_logits = True
    layout = {"probabilities": ((2, 1, 8), torch.float32), "valid": ((2, 1), torch.bool)}

    def __init__(self, token, prefix):
        self.token, self.prefix = token, prefix

    def run(self, scratch, logits, *, source_logits, shortlist_ids, candidates, seed, offset, sampled):
        assert source_logits.shape == (2, 8)
        assert shortlist_ids is None
        assert candidates[0, :len(self.prefix)].tolist() == self.prefix
        assert seed.tolist() == [103 ^ DRAFT_RNG_DOMAIN, 101 ^ DRAFT_RNG_DOMAIN]
        sampled.copy_(torch.tensor([self.token, 0]))
        scratch["valid"].fill_(True)
        scratch["probabilities"].zero_()
        scratch["probabilities"].scatter_(2, sampled[:, None, None], 1.)
        return scratch["probabilities"]


def test_proposal_metadata_and_rng_use_runtime_rows_without_retaining_destinations():
    requests = SimpleNamespace(capacity=3, vocab_size=8, histogram=torch.zeros(3, 8, dtype=torch.int32),
        consumed_tokens=torch.tensor([3, 7, 9]), pending_token=torch.tensor([1, 2, 3]),
        generation=torch.tensor([4, 5, 6]), seed=torch.tensor([101, 102, 103], dtype=torch.int64))
    inputs = SimpleNamespace(row_slot_ids=torch.tensor([2, 0], dtype=torch.int32),
        query_lengths=torch.tensor([3, 1], dtype=torch.int32), request_state_rows=torch.tensor([[1], [2], [3]]),
        page_tables=torch.tensor([[0], [1], [2]], dtype=torch.int32), live_rows=torch.tensor([2], dtype=torch.int32))
    owner = SimpleNamespace(requests=requests, inputs=inputs, block_size=3, logit_offset=1,
        weights=SimpleNamespace(config=SimpleNamespace(draft=SimpleNamespace(noise_token=7))),
        generation=torch.tensor([4, 5, 0]), offset=torch.tensor([40, 50, 60]))
    operation = object.__new__(PreparedParallelDraftRound)
    operation.owner, operation.batch, operation.width = owner, 2, 3
    operation.local_metadata, operation.generate_proposals = True, True
    operation.vocabulary_parallel = False
    operation.operations = None, (Proposal(6, []), Proposal(5, [6])), None, None
    operation.validator = SimpleNamespace(validate=lambda: (torch.zeros(1, 8), torch.tensor([True, False, True])),
        begin=lambda totals, present: SimpleNamespace(lengths=torch.tensor([1, 0])))
    operation.validation = lambda values: values
    observed_metadata = []

    def draft(metadata):
        observed_metadata.append(metadata)
        return torch.zeros(6, 8), None

    operation.draft = draft
    operation.shortlisted_selector = operation.selector = operation.markov = None
    operation.full_proposal = operation.proposal_block = None
    operation.sampling_scratch = {}
    for name in ("metadata", "draft_metadata", "draft_inputs", "markov_token", "before", "active", "candidates",
                 "drafts", "valid", "seed", "offset", "base_offset", "sampled", "logits", "probabilities"):
        setattr(operation, name, None)
    probabilities = torch.empty(2, 2, 8)
    first = operation.propose(probabilities=probabilities)
    assert first.candidates.tolist() == [[3, 6, 5], [1, 0, 0]]
    assert first.valid.tolist() == [[True, True], [False, False]]
    assert first.before.tolist() == [9, 3]
    assert first.base_offset.tolist() == [0, 40]
    assert first.metadata.state_rows.tolist() == [3, 0]
    assert first.metadata.lengths.tolist() == [3, 0]
    assert first.metadata.token_ids.tolist() == [3, 7, 7, 0, 0, 0]
    assert first.metadata.live_counts.tolist() == [3, 2]
    assert owner.offset.tolist() == [40, 50, 8]
    assert owner.generation.tolist() == [4, 5, 6]
    assert torch.equal(probabilities.sum(-1), torch.ones(2, 2))
    requests.pending_token[2] = 4
    requests.consumed_tokens[2] = 12
    second = operation.propose(advance_rng=False, probabilities=probabilities)
    assert second.candidates[0, 0] == 4 and second.before[0] == 12
    assert first.candidates[0, 0] == 3 and first.before[0] == 9
    assert second.base_offset.tolist() == [8, 40]
    assert second.metadata is not first.metadata
    for name in ("metadata", "draft_metadata", "draft_inputs", "before", "active", "candidates", "drafts",
                 "valid", "seed", "offset", "base_offset", "sampled", "markov_token", "probabilities"):
        assert getattr(operation, name) is None


def test_verification_passes_local_metadata_through_commit_and_preserves_inactive_hidden_rows():
    from ignition.components.parallel_draft import DraftProposalState

    proposal = DraftProposalState(torch.tensor([9, 3]), torch.tensor([True, False]),
        torch.tensor([[3, 6, 5], [1, 0, 0]]), torch.tensor([[6, 5], [0, 0]]),
        torch.tensor([[True, True], [False, False]]), torch.tensor([0, 40]), None)
    inputs = SimpleNamespace(row_slot_ids=torch.tensor([2, 0]), query_lengths=torch.tensor([3, 1]),
        request_state_rows=torch.tensor([[1], [2], [3]]), page_tables=torch.tensor([[0], [1], [2]]),
        live_rows=torch.tensor([2], dtype=torch.int32))
    hidden = torch.arange(12).view(6, 2).float()
    pool = SimpleNamespace(last_hidden=torch.full((4, 2), -3.))
    calls = []

    class Decoder:
        def __call__(self, metadata):
            calls.append(("decode", metadata))
            return hidden, torch.zeros(6, 8), torch.ones(4, 2), {"records": torch.ones(1)}

        def commit(self, counts, records, metadata):
            assert counts.tolist() == [2, 0] and records["records"].item() == 1
            calls.append(("commit", metadata))

    class Target:
        layout = {"probabilities": ((2, 3, 8), torch.float32), "valid": ((2, 3), torch.bool)}

        def run(self, scratch, logits, *, candidates):
            assert candidates is proposal.drafts and logits.shape == (2, 3, 8)
            scratch["probabilities"].fill_(.125)
            scratch["valid"].fill_(True)

    acceptance = SimpleNamespace(valid=torch.tensor([True, False]))

    class Verification:
        layout = {}

        def run(self, scratch, probabilities, target, valid, *, drafts, draft_valid, disjoint):
            assert drafts is proposal.drafts and draft_valid is proposal.valid and disjoint
            return acceptance

    committed = SimpleNamespace(emitted_count=torch.tensor([2, 0], dtype=torch.int32))

    def commit(logits, *, acceptance, inputs):
        assert inputs is proposal.candidates
        assert acceptance.valid.tolist() == [True, False]
        return committed

    operation = object.__new__(PreparedParallelDraftRound)
    operation.owner = SimpleNamespace(inputs=inputs, requests=SimpleNamespace(capacity=3),
        weights=SimpleNamespace(config=SimpleNamespace(text={"hidden_size": 2})), pool=pool)
    operation.batch, operation.width, operation.target_token_capacity = 2, 3, 4
    operation.rows = None
    operation.local_metadata, operation.vocabulary_parallel = True, False
    operation.sampling_scratch = {}
    operation.operations = Target(), (), Verification(), commit
    operation.decoder = Decoder()
    operation.context = lambda auxiliary, counts, metadata: calls.append(("context", metadata))
    operation.metadata = operation.active = operation.before = operation.candidates = None
    metadata, actual_acceptance, actual_commit = operation.verify(
        probabilities=torch.full((2, 2, 8), .125), proposal=proposal)
    assert actual_acceptance is acceptance and actual_commit is committed
    assert [kind for kind, _ in calls] == ["decode", "commit", "context"]
    assert all(value is metadata for _, value in calls)
    assert metadata.starts.tolist() == [9, 3] and metadata.lengths.tolist() == [3, 0]
    assert metadata.token_ids.tolist() == [3, 6, 5, 0]
    assert torch.equal(pool.last_hidden[3], hidden[1])
    assert torch.equal(pool.last_hidden[:3], torch.full((3, 2), -3.))
    assert operation.metadata is None and operation.candidates is None
