# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Committed conditional sequences exercise request state, RNG and raw logprobs."""

import itertools
import json
import math
import os

import pytest
import torch
import triton
import triton.language as tl

from ignition.components.sampling import FINISHED
from tests.components.causal_prefix_oracle import allocate
from tests.components.test_generation_logprobs_cuda import _assert_entry
from tests.components.vocabulary_commit_support import ToyCommittedModel

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def start(model, *, generation=1, budget=20, eos=(), rows=None):
    model.stage(generation=generation, max_new_tokens=budget, eos=eos, rows=rows)
    model.replay("admit")
    model.replay("prefill")


def assert_committed(model, checkpoint):
    before = {id(tensor): value for tensor, value in checkpoint}
    state = model.requests
    slots = model.slots.cpu().tolist()
    for row, slot in enumerate(slots):
        count = int(model.committed.emitted_count[row])
        if row >= int(model.live_rows) or int(model.epochs[row]) != int(state.generation[slot]):
            assert count == 0
            continue
        assert int(state.consumed_tokens[slot]) == int(before[id(state.consumed_tokens)][slot]) + count
        assert int(state.generated_tokens[slot]) == int(before[id(state.generated_tokens)][slot]) + count
        assert int(state.rng_counter[slot]) == int(before[id(state.rng_counter)][slot]) + count
        previous_generation = int(before[id(model.rng.generation)][slot])
        base = int(before[id(model.rng.target_offset)][slot]) if previous_generation == int(state.generation[slot]) else 0
        assert int(model.rng.target_offset[slot]) == (base + model.verify.draw_reservation if count else
                                                     int(before[id(model.rng.target_offset)][slot]))
        if count:
            inputs = model.candidates[row, :count]
            history = torch.cat((before[id(state.ple_history)][slot], inputs))[-2:]
            torch.testing.assert_close(state.ple_history[slot], history)
            assert state.pending_token[slot] == model.committed.emitted_ids[row, count - 1]
            expected_histogram = before[id(state.histogram)][slot] + torch.bincount(
                model.committed.emitted_ids[row, :count], minlength=model.vocabulary).to(torch.int32)
            torch.testing.assert_close(state.histogram[slot], expected_histogram)
            record = model.completion.records[int(model.frame), slot]
            assert int(record[8]) == count
            torch.testing.assert_close(record[16:16 + count], model.committed.emitted_ids[row, :count])
            if int(model.completion.requested[slot]):
                queried = model.completion.token_ids[slot, :int(model.completion.token_id_counts[slot])].tolist()
                for col in range(count):
                    _assert_entry(model.completion.logprobs[int(model.frame), slot, col], model.logits[row, col],
                                  int(model.committed.emitted_ids[row, col]), ids=queried or None)


def test_causal_prefix_commit_offsets_rollback_stale_rows_and_slot_reuse():
    model = ToyCommittedModel(token_capacity=5)
    try:
        model.completion.requested.fill_(1)
        model.completion.token_id_counts[0] = 21
        model.completion.token_ids[0, :21] = torch.arange(21, device="cuda", dtype=torch.int32)
        model.capture()
        start(model)
        before = model.checkpoint()
        model.replay("round")
        assert_committed(model, before)
        expected_lengths, expected_valid = allocate(model.confidence.cpu().tolist(), model.slots.cpu().tolist(),
            [True] * 4, [True] * 4, [3] * 4, 5, 4)
        assert model.limits.tolist() == expected_lengths
        assert model.admitted.tolist() == expected_valid
        assert (model.owner.offset == 8).all()
        assert (model.rng.target_offset == model.verify.draw_reservation).all()
        previous = model.candidates[:, 0]
        for step in range(model.width - 1):
            scores = model.proposal_base[:, step].double() + model.embedding[previous].double() @ model.head.double().T
            torch.testing.assert_close(model.q[:, step].double(), scores.softmax(-1), rtol=3e-6, atol=1e-9)
            previous = model.drafts[:, step]
        first_tokens = model.committed.emitted_ids.clone()
        first_q = model.q.clone()
        first_logprobs = model.completion.logprobs.clone()
        # The complete checkpoint includes all physical RNG domains and consumed
        # predecessor state. Rejected suffixes and dead graph scratch are absent.
        model.restore(before)
        model.q.fill_(float("nan"))
        model.p.fill_(float("nan"))
        model.replay("round")
        assert_committed(model, before)
        torch.testing.assert_close(model.committed.emitted_ids, first_tokens)
        torch.testing.assert_close(model.q, first_q)
        torch.testing.assert_close(model.completion.logprobs, first_logprobs)
        model.live_rows.fill_(1)
        before = model.checkpoint()
        model.replay("round")
        assert_committed(model, before)
        inactive = [slot for slot in range(4) if slot != int(model.slots[0])]
        for value, saved in before:
            actual_bits = value.view(torch.int64) if value.dtype == torch.uint64 else value
            saved_bits = saved.view(torch.int64) if saved.dtype == torch.uint64 else saved
            torch.testing.assert_close(actual_bits[inactive], saved_bits[inactive])
        # A stale graph lane cannot advance request state or either RNG domain.
        model.epochs[0] += 1
        before = model.checkpoint()
        model.replay("round")
        for value, saved in before:
            torch.testing.assert_close(value, saved)
        model.prefix_limits[0] = expected_lengths[0]
        model.stage(generation=2, rows=[0])
        model.replay("admit")
        model.replay("prefill")
        before = model.checkpoint()
        model.replay("round")
        assert_committed(model, before)
        reused = int(model.slots[0])
        assert int(model.owner.generation[reused]) == 2
        assert int(model.owner.offset[reused]) == 8
        assert int(model.rng.generation[reused]) == 2
        assert int(model.rng.target_offset[reused]) == model.verify.draw_reservation
        # Reusing the seed and complete initial state restarts both streams.
        torch.testing.assert_close(model.committed.emitted_ids[0], first_tokens[0])
    finally:
        model.close()


@pytest.mark.parametrize("stop", ["budget", "eos"])
def test_budget_and_natural_stop_clip_commit_and_leave_rng_terminal(stop):
    model = ToyCommittedModel()
    try:
        model.completion.requested.fill_(1)
        if stop == "eos":
            model.target_table.fill_(-float("inf"))
            model.target_table[:, 2] = 0.
        else:
            model.prefix_limits.fill_(1)
        model.capture()
        start(model, budget=2 if stop == "budget" else 20, eos=(2,) if stop == "eos" else ())
        before = model.checkpoint()
        model.replay("round")
        assert_committed(model, before)
        assert (model.committed.emitted_count == 1).all()
        assert (model.requests.status == FINISHED).all()
        assert (model.requests.generated_tokens == 2).all()
        if stop == "eos":
            assert (model.requests.pending_token == 2).all()
        before = model.checkpoint()
        model.replay("round")
        for value, saved in before:
            torch.testing.assert_close(value, saved)
        assert (model.committed.emitted_count == 0).all()
    finally:
        model.close()


@triton.jit
def _observe(Emitted, Lengths, Sequence, Counts, Rounds, B: tl.constexpr):
    row = tl.program_id(0)
    count = tl.load(Counts + row)
    emitted = tl.load(Lengths + row)
    col = tl.arange(0, 4)
    value = tl.load(Emitted + row * 3 + col, col < 3, other=-1)
    tl.store(Sequence + row * 3 + count + col, value, (col < emitted) & (count + col < 3))
    tl.store(Counts + row, count + emitted)
    tl.store(Rounds + row, tl.load(Rounds + row) + (emitted > 0).to(tl.int32))


class ConditionalTrial:
    """Generate exactly three tokens through admission and actual commits."""

    def __init__(self, model):
        from ignition.preparation.triton import prepare_triton
        self.model = model
        self.sequence = torch.full((model.batch, 3), -1, device="cuda", dtype=torch.int64)
        self.counts = torch.zeros(model.batch, device="cuda", dtype=torch.int32)
        self.rounds = torch.zeros_like(self.counts)
        self.observe = prepare_triton(_observe, (model.committed.emitted_ids, model.committed.emitted_count,
            self.sequence, self.counts, self.rounds), (model.batch, 1, 1), B=model.batch, num_warps=4)

    def __call__(self):
        m = self.model
        m.admission_buffers.admission_generations.add_(1)
        m.admission_buffers.admission_seed_bits.add_(m.batch)
        m.epochs.copy_(m.admission_buffers.admission_generations)
        m.initialize()
        m.prefill()
        self.sequence.fill_(-1)
        self.counts.zero_()
        self.rounds.zero_()
        safe = m.slots.long()
        for _ in range(3):
            m.prefix_limits.copy_((m.requests.max_new_tokens[safe] - m.requests.generated_tokens[safe]).clamp(1, 3).int())
            m()
            self.observe()


def test_fixed_length_conditional_law_through_admission_prefix_allocation_and_commit():
    model = ToyCommittedModel(batch=64)
    try:
        model.stage(generation=1, max_new_tokens=4)
        trial = ConditionalTrial(model)
        model.capture(extra={"trial": trial})
        sequences = []
        for _ in range(2048):
            model.replay("trial")
            sequences.append(trial.sequence.clone())
        observed_sequences = torch.cat(sequences)
        assert (observed_sequences >= 0).all() and (observed_sequences < 3).all()
        assert (trial.counts == 3).all()
        assert (model.requests.generated_tokens == 4).all()
        assert (model.requests.consumed_tokens == 4).all()
        assert (model.requests.rng_counter == 4).all()
        assert (model.requests.histogram.sum(-1) == 5).all()
        safe = model.slots.long()
        torch.testing.assert_close(model.requests.pending_token[safe], trial.sequence[:, -1])
        torch.testing.assert_close(model.requests.ple_history[safe], trial.sequence[:, :2])
        torch.testing.assert_close(model.owner.offset[safe], trial.rounds.long() * 8)
        torch.testing.assert_close(model.rng.target_offset[safe], trial.rounds.long() * model.verify.draw_reservation)
        matrix = model.target_table[:3, :3].double().softmax(-1).cpu()
        measurements = []
        for width in (2, 3):
            codes = (observed_sequences[:, :width] * 3 ** torch.arange(width - 1, -1, -1, device="cuda")).sum(-1)
            observed = codes.bincount(minlength=3**width).cpu().double() / codes.numel()
            expected = []
            for ids in itertools.product(range(3), repeat=width):
                probability, previous = 1., 0
                for token in ids:
                    probability *= float(matrix[previous, token])
                    previous = token
                expected.append(probability)
            error = float((observed - torch.tensor(expected, dtype=torch.float64)).abs().max())
            bound = math.sqrt(math.log(2 * (9 + 27) / 1e-7) / (2 * codes.numel()))
            assert error < bound
            measurements.append(dict(width=width, sequences=codes.numel(), max_joint_error=error,
                                     simultaneous_hoeffding_bound=bound))
        print(json.dumps(dict(committed_conditional_joint_law=measurements)))
    finally:
        model.close()
