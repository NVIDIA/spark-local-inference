# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from tests.components.completion_support import RecordOnlyLogprobs, completion_outputs

from ignition.components.sampling import ERROR, FINISHED, RUNNING
from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN,
    INVALID_SPECULATION,
    TARGET_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
    SpeculativeAcceptance,
    SpeculativeRngState,
)


def example(*, batch=1, width=4, consumed=2, budget=20, eos=(), slot=2, vocab=16, completion_capacity=16):
    state = DeviceRequestState.allocate("cpu", vocab_size=vocab)
    state.admit(slot, RequestAdmission(generation=7, seed=2**64 - 1, max_new_tokens=budget,
        temperature=1, top_k=0, top_p=1, eos_token_ids=eos), [1, 2], consumed_tokens=2, ple_history=(1, 2))
    state.consumed_tokens[slot] = consumed
    state.generated_tokens[slot] = 1
    state.pending_token[slot] = 3
    state.histogram[slot, 3] = 1
    state.rng_counter[slot] = 1
    slots = torch.full((batch,), -1, dtype=torch.int32)
    slots[0] = slot
    epochs = torch.full((batch,), 7, dtype=torch.int64)
    live = torch.tensor([1], dtype=torch.int32)
    inputs = torch.arange(3, 3 + width).expand(batch, -1).clone()
    acceptance = SpeculativeAcceptance.allocate(batch, width, "cpu")
    emitted = min(width, 3)
    acceptance.token_ids[:, :emitted] = inputs[:, :emitted] + 1
    if width > 2:
        acceptance.token_ids[:, emitted - 1] = 9
    acceptance.proposed_count.fill_(emitted)
    acceptance.accepted_drafts.fill_(emitted - 1)
    acceptance.offered_drafts.fill_(width - 1)
    acceptance.valid.fill_(True)
    records = torch.full((2, 4, 16 + completion_capacity), -99, dtype=torch.int64)
    frame, sequence = torch.tensor([1], dtype=torch.int32), torch.tensor([37], dtype=torch.int64)
    logits = torch.zeros((*inputs.shape, state.vocab_size), device=inputs.device)
    commit = PreparedSpeculativeCommit(state, acceptance, inputs, slots, epochs, live,
        completion_outputs(records), frame, sequence, logits=logits)
    return state, acceptance, commit, records


@pytest.mark.parametrize("width", range(1, 17))
def test_commit_accepts_each_explicit_verified_prefix(width):
    for emitted in range(1, width + 1):
        state, acceptance, commit, records = example(width=width, vocab=32)
        acceptance.token_ids.fill_(-1)
        acceptance.token_ids[:, :emitted] = commit.inputs[:, :emitted] + 1
        acceptance.proposed_count.fill_(emitted)
        acceptance.accepted_drafts.fill_(emitted - 1)
        commit()
        assert commit.emitted_count.item() == emitted
        assert commit.commit_column.item() == emitted - 1
        assert state.consumed_tokens[2] == 2 + emitted
        assert state.pending_token[2] == 3 + emitted
        assert records[1, 2, 16:16 + emitted].tolist() == list(range(4, 4 + emitted))
        assert records[1, 2, 16 + emitted:].eq(-1).all()


@pytest.mark.parametrize("field,value", [("proposed_count", 9), ("accepted_drafts", 4),
    ("offered_drafts", 8), ("offered_drafts", 0)])
def test_commit_rejects_inconsistent_acceptance_contract(field, value):
    state, acceptance, commit, records = example(width=8)
    getattr(acceptance, field).fill_(value)
    commit()
    assert commit.emitted_count.item() == 0
    assert state.consumed_tokens[2] == 2
    assert state.status[2] == ERROR
    assert records[1, 2, 6] == INVALID_SPECULATION


def test_commit_advances_input_and_output_histories_once():
    state, acceptance, commit, records = example()
    commit()
    assert commit.consumed_before.tolist() == [2]
    assert commit.generated_before.tolist() == [1]
    assert commit.history_before.tolist() == [[1, 2]]
    assert commit.emitted_ids.tolist() == [[4, 5, 9, -1]]
    assert commit.committed_input_ids.tolist() == [[3, 4, 5, -1]]
    assert commit.accepted_count.tolist() == [3]
    assert commit.commit_column.tolist() == [2]
    assert state.consumed_tokens[2] == 5 and state.generated_tokens[2] == 4
    assert state.pending_token[2] == 9 and state.ple_history[2].tolist() == [4, 5]
    assert state.histogram[2, 3] == 1 and state.histogram[2, 9] == 1
    assert state.rng_counter[2].item() == 4
    assert records[1, 2, :11].tolist() == [7, 37, 5, 4, 9, RUNNING, 0, 0, 3, 3, 2]
    assert records[1, 2, 12:].tolist() == [-1, 4, 0, 4, 4, 5, 9] + [-1] * 13
    assert records[0].eq(-99).all()
    assert records[1, 0, 16:].eq(-1).all()


@pytest.mark.parametrize("eos,budget", [((5,), 20), ((), 3)])
def test_clip_inside_drafts_keeps_final_draft_pending(eos, budget):
    state, acceptance, commit, records = example(eos=eos, budget=budget)
    commit()
    assert commit.emitted_ids.tolist() == [[4, 5, -1, -1]]
    assert commit.accepted_drafts.item() == 2
    assert commit.commit_column.item() == 1
    assert state.consumed_tokens[2] == 4 and state.generated_tokens[2] == 3
    assert state.pending_token[2] == 5
    assert state.ple_history[2].tolist() == [3, 4]
    assert state.status[2] == FINISHED
    assert records[1, 2, 10] == 2
    frozen = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
    commit()
    for name, expected in frozen.items():
        assert torch.equal(getattr(state, name), expected), name
    assert records[1, 2, 8] == 0


def test_live_history_uses_accepted_input_not_verification_endpoint():
    state, acceptance, commit, records = example(consumed=63)
    commit()
    assert state.ple_history[2].tolist() == [4, 5]
    assert records[1, 2, 12] == -1
    assert state.consumed_tokens[2] == 66


@pytest.mark.parametrize("fault", ["stale", "inactive", "bad_slot", "duplicate", "pending", "invalid_distribution"])
def test_bad_or_inactive_rows_do_not_advance_state(fault):
    state, acceptance, commit, _ = example(batch=2)
    if fault == "stale": commit.generations[0] = 6
    elif fault == "inactive": commit.live.zero_()
    elif fault == "bad_slot": commit.slots[0] = 99
    elif fault == "duplicate": commit.slots[1] = 2; commit.live.fill_(2)
    elif fault == "pending": commit.inputs[0, 0] = 0
    elif fault == "invalid_distribution": acceptance.valid[0] = False
    counts = state.histogram.clone()
    commit()
    assert commit.emitted_count.eq(0).all()
    assert state.consumed_tokens[2] == 2 and state.generated_tokens[2] == 1
    assert state.pending_token[2] == 3 and torch.equal(state.histogram, counts)
    assert state.status[2] == (ERROR if fault in ("duplicate", "pending", "invalid_distribution") else RUNNING)


def test_rng_domains_epoch_reuse_and_reservation():
    state, acceptance, commit, _ = example()
    rng = SpeculativeRngState.allocate("cpu")
    commit.rng = rng
    rng.generation[2], rng.target_offset[2] = 6, 1000
    acceptance.rng_increment.fill_(8); commit()
    assert rng.generation[2] == 7 and rng.target_offset[2] == 8
    assert TARGET_RNG_DOMAIN != DRAFT_RNG_DOMAIN
    for width, reservation in ((1, 4), (2, 4), (4, 8), (8, 12)):
        sample = PreparedStochasticVerification(torch.empty(1, width - 1, 16), torch.zeros((1, width - 1), dtype=torch.int64),
            torch.empty(1, width, 16), state, commit.slots, commit.generations, commit.live, rng=rng)
        assert sample.draw_reservation == reservation
        with pytest.raises(ValueError, match="requires CUDA"):
            sample.prepare()


def test_conditional_penalties_include_only_prior_drafts():
    state, _, commit, _ = example()
    state.parameters[2] = torch.tensor([1., 1., 0., 2., 3., .5])
    logits = torch.ones((1, 4, 16)) * 8
    candidates = torch.tensor([[4, 4, 5]])
    histogram = state.histogram.clone()
    op = PreparedSpeculativeDistributions(logits, state, commit.slots, commit.generations, commit.live, candidates)
    op.prepare_scores()
    assert op.scores[0, :, 3].tolist() == [.5, .5, .5, .5]
    assert op.scores[0, :, 4].tolist() == [8., .5, 0., 0.]
    assert op.scores[0, :, 5].tolist() == [8., 8., 8., .5]
    assert torch.equal(state.histogram, histogram)
    state.grammar_enabled[2] = 1
    op.prepare_scores()
    assert op.valid.tolist() == [[True, False, False, False]]


def test_invalid_conditional_id_does_not_index_histogram_out_of_bounds():
    initialized = torch.cuda.is_initialized()
    state, _, commit, _ = example()
    op = PreparedSpeculativeDistributions(torch.zeros(1, 4, 16), state,
        commit.slots, commit.generations, commit.live, torch.tensor([[4, 99, 5]]))
    op.prepare_scores()
    assert op.valid.tolist() == [[True, True, False, False]]
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("prefix_offset", [0, 1, 2, 3])
def test_draft_penalties_ignore_unoffered_candidates(prefix_offset):
    state, _, commit, _ = example()
    state.parameters[2] = torch.tensor([1., 1., 0., 2., 3., .5])
    candidates = torch.tensor([[4, 4, 5, -1, -1, -1, -1]])
    histogram = state.histogram.clone()
    op = PreparedSpeculativeDistributions(torch.full((1, 1, 16), 8.), state,
        commit.slots, commit.generations, commit.live, candidates, prefix_offset=prefix_offset)
    op.prepare_scores()
    assert op.valid.item()
    assert op.scores[0, 0, 4].item() == (8., .5, 0., 0.)[prefix_offset]
    assert op.scores[0, 0, 5].item() == (.5 if prefix_offset == 3 else 8.)
    assert torch.equal(state.histogram, histogram)
    scores = op.scores.clone()
    candidates[:, prefix_offset:] = 99
    op.prepare_scores()
    assert op.valid.item()
    assert torch.equal(state.histogram, histogram) and torch.equal(op.scores, scores)
    if prefix_offset:
        candidates[:, prefix_offset - 1] = 99
        op.prepare_scores()
        assert not op.valid.item()


@pytest.fixture(autouse=True)
def records_only_cpu_commit(monkeypatch):
    # These state-machine tests run without CUDA; reduction and graph ordering
    # have independent CUDA coverage.
    monkeypatch.setattr("ignition.components.speculation.PreparedLogprobs", RecordOnlyLogprobs)


@pytest.mark.parametrize("clipping", ["none", "budget", "eos", "inactive"])
def test_sixteen_token_commit_preserves_records_histogram_rng_history_and_survival(clipping):
    state, acceptance, original, records = example(width=16, vocab=32, budget=40)
    acceptance.token_ids.copy_(original.inputs + 1)
    acceptance.proposed_count.fill_(16)
    acceptance.accepted_drafts.fill_(15)
    acceptance.rng_increment.fill_(31)
    acceptance.survival.copy_(torch.linspace(.95, .25, 15)[None])
    outputs = completion_outputs(records)
    rng = SpeculativeRngState.allocate("cpu")
    rng.generation[2], rng.target_offset[2] = 7, 99
    commit = PreparedSpeculativeCommit(state, acceptance, original.inputs, original.slots,
        original.generations, original.live, outputs, original.frame, original.sequence,
        logits=torch.zeros((1, 16, 32)), rng=rng)
    if clipping == "budget":
        state.max_new_tokens[2] = 15
    elif clipping == "eos":
        state.eos_count[2], state.eos_token_ids[2, 0] = 1, 17
    elif clipping == "inactive":
        original.live.zero_()
    before = {name: tensor.clone() for name, tensor in state.__dict__.items()
              if isinstance(tensor, torch.Tensor)}
    commit()
    count = 0 if clipping == "inactive" else 14 if clipping in ("budget", "eos") else 16
    assert commit.emitted_count.item() == count
    assert state.consumed_tokens[2] == 2 + count
    assert state.generated_tokens[2] == 1 + count
    assert state.pending_token[2] == 3 + count
    assert state.rng_counter[2] == 1 + count
    assert rng.target_offset[2] == (130 if count else 99)
    if count:
        record = records[1, 2]
        assert record[8:11].tolist() == [count, 15, min(15, count)]
        assert record[15] == 16
        assert record[16:16 + count].tolist() == list(range(4, 4 + count))
        assert record[16 + count:].eq(-1).all()
        assert state.ple_history[2].tolist() == [count + 1, count + 2]
        assert state.histogram[2, 4:4 + count].eq(1).all()
        extent = 13 if clipping == "budget" else min(count, 15)
        assert torch.equal(outputs.speculative_survival[1, 2, :extent], acceptance.survival[0, :extent])
        assert outputs.speculative_survival[1, 2, extent:].eq(-1).all()
    else:
        for name, tensor in before.items():
            assert torch.equal(getattr(state, name), tensor), name
        assert records[1, :, 5].eq(0).all()
        assert outputs.speculative_survival[1].eq(-1).all()
    assert records[0].eq(-99).all()
    assert records[1, [0, 1, 3], 16:].eq(-1).all()
    assert outputs.speculative_survival[1, [0, 1, 3]].eq(-1).all()


@pytest.mark.parametrize("capacity", [1, 8, 13, 16, 32])
@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("clipping", ["none", "budget", "eos", "inactive"])
def test_core_capacity_bounds_publication_without_truncating_verified_prefix(capacity, short, clipping):
    width = min(capacity, 4) if short else capacity
    state, acceptance, commit, records = example(width=width, vocab=64, budget=80, completion_capacity=capacity)
    acceptance.token_ids.copy_(commit.inputs + 1)
    acceptance.proposed_count.fill_(width)
    acceptance.accepted_drafts.fill_(width - 1)
    acceptance.survival.fill_(.5)
    if clipping == "budget":
        state.max_new_tokens[2] = 10
    elif clipping == "eos":
        state.eos_count[2], state.eos_token_ids[2, 0] = 1, 3 + min(width, 10)
    elif clipping == "inactive":
        commit.live.zero_()
    commit()
    emitted = 0 if clipping == "inactive" else min(width, 9 if clipping == "budget" else 10 if clipping == "eos" else width)
    assert state.consumed_tokens[2] == 2 + emitted
    assert state.generated_tokens[2] == 1 + emitted
    assert state.pending_token[2] == 3 + emitted
    assert state.rng_counter[2] == 1 + emitted
    assert records.shape == (2, 4, 16 + capacity)
    assert records[1, 2, 8].item() == emitted
    assert records[1, 2, 16:16 + emitted].tolist() == list(range(4, 4 + emitted))
    assert records[1, 2, 16 + emitted:].eq(-1).all()
    assert records[0].eq(-99).all()
    if capacity > 1:
        assert commit.survival.shape == (2, 4, capacity - 1)
        logical = min(width - 1, 8 if clipping == "budget" else width - 1)
        extent = 0 if not emitted else min(logical, emitted)
        assert commit.survival[1, 2, :extent].eq(.5).all()
        assert commit.survival[1, 2, extent:].eq(-1).all()
        assert commit.survival[1, [0, 1, 3]].eq(-1).all()
    else:
        assert commit.survival is None


def test_functional_commit_uses_runtime_rows_and_token_limit():
    state, acceptance, commit, records = example(width=4)
    inputs = commit.inputs.clone()
    class NoLogprobs:
        def functional(self, **kwargs):
            pass

        def __call__(self):
            pass

    commit.logprobs = NoLogprobs()
    commit.functional(dynamic_inputs=True)
    result = commit(acceptance=acceptance, inputs=inputs,
        slots=torch.tensor([2], dtype=torch.int32), generations=torch.tensor([7], dtype=torch.int64),
        live_rows=torch.tensor([1], dtype=torch.int32), token_limits=torch.tensor([2], dtype=torch.int32))
    assert result.emitted_count.tolist() == [2]
    assert state.consumed_tokens[2] == 4
    assert records[1, 2, 8] == 2
