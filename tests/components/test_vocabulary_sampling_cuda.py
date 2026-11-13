# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP32 vocabulary-parallel proposal consistency and block-rejection laws."""

import os
import math
import json

import pytest
import torch
import triton
import triton.language as tl

from ignition.components.sampling import DeviceRequestState, RUNNING
from ignition.components.speculation import DRAFT_RNG_DOMAIN, SpeculativeRngState
from ignition.components.vocabulary_sampling import PreparedVocabularySoftmax, PreparedVocabularyBlockVerification

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def fixture(batch=4, depth=3, vocab=8, rank=64, fp8=False):
    state = DeviceRequestState.allocate("cuda", vocab_size=vocab, capacity=batch)
    state.status.fill_(RUNNING)
    state.generation.fill_(7)
    state.seed.copy_(torch.arange(batch, device="cuda", dtype=torch.int64) + 7351)
    state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device="cuda"))
    state.top_k.fill_(-1)
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    epochs = torch.full((batch,), 7, device="cuda", dtype=torch.int64)
    live = torch.tensor([batch], device="cuda", dtype=torch.int32)
    embedding = torch.randn(vocab, rank, device="cuda", dtype=torch.bfloat16) / 8
    head = torch.randn_like(embedding) / 8
    if fp8:
        from ignition.components.markov_sampling import fp8_markov_head

        head = fp8_markov_head(head)
    logits = torch.randn(batch, depth, vocab, device="cuda", dtype=torch.bfloat16)
    tokens = torch.zeros(batch, depth, device="cuda", dtype=torch.int64)
    probabilities = torch.empty(batch, depth, vocab, device="cuda")
    valid = torch.empty(batch, depth, device="cuda", dtype=torch.bool)
    seeds = state.seed.view(torch.int64) ^ DRAFT_RNG_DOMAIN
    offsets = torch.full((batch,), 2**32 + 28, device="cuda", dtype=torch.int64)
    proposal = PreparedVocabularySoftmax(logits, probabilities, valid, state, slots, epochs, live, tokens,
        start=0, runtime=None, session=None, coordinator=None, name="proposal", embedding=embedding, head=head,
        pending=torch.zeros(batch, device="cuda", dtype=torch.int64), seeds=seeds, offsets=offsets, tokens=tokens)
    proposal.prepare()
    return proposal


@pytest.mark.parametrize("batch,depth,vocab", [(1, 1, 8), (4, 3, 1024), (1, 7, 129280)])
@pytest.mark.parametrize("fp8", [False, True])
def test_markov_scores_and_retained_q_share_fp32_math(batch, depth, vocab, fp8):
    op = fixture(batch, depth, vocab, fp8=fp8)
    op()
    previous = op.pending
    for step in range(depth):
        weight = op.head.double()
        if fp8:
            weight *= op.scale.double()[:, None]
        exact = op.logits[:, step].double() + op.embedding[previous].double() @ weight.T
        torch.testing.assert_close(op.probabilities[:, step].double(), exact.softmax(-1),
                                   rtol=3e-6, atol=1e-9)
        previous = op.tokens[:, step]
    assert op.valid.all()
    assert (op.probabilities.gather(-1, op.tokens[:, :, None]) > 0).all()


@pytest.mark.parametrize("case", ["inactive", "duplicate", "generation", "nan", "grammar"])
def test_proposal_row_and_grammar_contract(case):
    op = fixture()
    if case == "inactive":
        op.state.status.zero_()
    elif case == "duplicate":
        op.slots.zero_()
    elif case == "generation":
        op.generations.zero_()
    elif case == "nan":
        op.logits[:, :, 7] = float("nan")
    else:
        op.state.grammar_enabled.fill_(True)
        op.state.grammar_mask.zero_()
        op.state.grammar_mask[:, 0] = 1 << 3
    op()
    if case == "grammar":
        assert op.valid.all() and (op.tokens == 3).all()
    else:
        assert not op.valid.any()
        assert (op.tokens == 0).all()
        assert (op.probabilities[:, :, 0] == 1).all()
        assert (op.probabilities[:, :, 1:] == 0).all()


def verifier(op, p, *, limits=None):
    valid = torch.ones(p.shape[:2], device="cuda", dtype=torch.bool)
    rng = SpeculativeRngState.allocate("cuda", capacity=op.state.capacity)
    rng.generation.fill_(7)
    result = PreparedVocabularyBlockVerification(op.probabilities, op.tokens, p,
        op.state, op.slots, op.generations, op.live_rows, start=0, runtime=None, session=None,
        coordinator=None, name="verify", rng=rng, draft_valid=op.valid, target_valid=valid, token_limits=limits)
    result.prepare()
    return result


@pytest.mark.parametrize("depth,limit", [(3, 1), (3, 2), (3, 4), (7, 8), (15, 9), (15, 16)])
def test_identical_distributions_accept_every_offered_token(depth, limit):
    op = fixture(depth=depth)
    op()
    p = torch.cat((op.probabilities, op.probabilities[:, -1:]), 1)
    limits = torch.full((op.batch,), limit, device="cuda", dtype=torch.int32)
    verify = verifier(op, p, limits=limits)
    verify()
    assert verify.output.valid.all()
    assert (verify.output.accepted_drafts == limit - 1).all()
    assert (verify.output.proposed_count == limit).all()
    torch.testing.assert_close(verify.output.token_ids[:, :limit - 1], op.tokens[:, :limit - 1])
    assert (verify.output.token_ids[:, limit:] == -1).all()
    assert verify.output.survival[:, :limit - 1].eq(1).all()
    assert verify.output.survival[:, limit - 1:].eq(-1).all()


@pytest.mark.parametrize("depth", [1, 3, 7])
def test_block_rejection_first_token_target_law(depth):
    op = fixture(batch=64, depth=depth)
    op.embedding.zero_()
    op.head.zero_()
    op.logits.copy_(torch.tensor([2., 1., 0., -1., -.25, .75, -.5, -1.5], device="cuda"))
    expected = torch.tensor([.4, 1.2, -.8, .5, .2, -1.5, 0., -.3], device="cuda").softmax(0)
    p = expected.expand(op.batch, depth + 1, -1).contiguous()
    verify = verifier(op, p)
    op()
    verify()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.offsets.add_(4 * depth)
        verify.rng.target_offset.add_(verify.draw_reservation)
        op()
        verify()
    samples = []
    for _ in range(1024):
        graph.replay()
        samples.append(verify.output.token_ids.clone())
    torch.cuda.synchronize()
    tokens = torch.cat(samples)
    observed = tokens[:, 0].bincount(minlength=8).double() / (1024 * op.batch)
    bound = math.sqrt(math.log(16 / 1e-7) / (2 * 1024 * op.batch))
    first_error = float((observed - expected.double()).abs().max())
    assert first_error < bound
    emitted = tokens[tokens >= 0]
    emitted_observed = emitted.bincount(minlength=8).double() / emitted.numel()
    emitted_error = float((emitted_observed - expected.double()).abs().max())
    assert emitted_error < bound
    print(json.dumps(dict(depth=depth, first_samples=1024 * op.batch, emitted_samples=emitted.numel(),
                         simultaneous_hoeffding_bound=bound, first_error=first_error, emitted_error=emitted_error)))


@pytest.mark.parametrize("filter_name", ["top_k", "top_p", "min_p"])
def test_target_rejects_unprepared_filter_policy(filter_name):
    op = fixture()
    op()
    logits = torch.randn(op.batch, op.width + 1, op.local, device="cuda")
    p = torch.empty_like(logits)
    valid = torch.empty(logits.shape[:2], device="cuda", dtype=torch.bool)
    if filter_name == "top_k":
        op.state.top_k.fill_(2)
    elif filter_name == "top_p":
        op.state.parameters[:, 1] = .9
    else:
        op.state.parameters[:, 2] = .1
    target = PreparedVocabularySoftmax(logits, p, valid, op.state, op.slots, op.generations,
        op.live_rows, op.tokens, start=0, runtime=None, session=None, coordinator=None, name="target")
    target.prepare()
    target()
    assert not valid.any()


def test_seed_offsets_and_batch_repacking_preserve_request_draws():
    op = fixture(batch=8)
    op()
    tokens, q = op.tokens.clone(), op.probabilities.clone()
    # Repack the same requests into different graph lanes. Every request-local
    # operand follows its slot, while request state remains in physical slots.
    order = torch.tensor([3, 0, 7, 2, 6, 1, 5, 4], device="cuda")
    for tensor in (op.logits, op.pending, op.slots, op.generations, op.seeds, op.offsets):
        tensor.copy_(tensor[order])
    op()
    torch.testing.assert_close(op.tokens, tokens[order])
    torch.testing.assert_close(op.probabilities, q[order])
    op.offsets.add_(2**32)
    op()
    assert (op.tokens != tokens[order]).any()


def test_compiled_native_replay_mutable_inputs_and_storage_lifetimes():
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle

    op = fixture(batch=4, depth=3, vocab=1024)
    p = torch.full((4, 4, 1024), 1 / 1024, device="cuda")
    verify = verifier(op, p)

    class Pipeline:
        def __call__(self):
            op()
            verify()

    pipeline = Pipeline()
    compiled = compile_forward(pipeline)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    bundle = PreparedGraphBundle(device=0, graphs={"sampling": graph}, buffers={}, resources=(op, verify, compiled))
    worker = bundle.start_worker()
    try:
        for iteration in range(8):
            op.offsets.add_(4 * op.width)
            verify.rng.target_offset.add_(verify.draw_reservation)
            op.logits.add_(.125)
            for tensor in op.scratch.values():
                tensor.fill_(float("nan"))
            op.probabilities.fill_(float("nan"))
            verify.scratch["partials"].fill_(float("nan"))
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            ticket = worker.submit(0)
            result = worker.poll(10000)
            assert result is not None and result.error is None and result.ticket == ticket
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert verify.output.valid.all()
            # Every reduction buffer is dead after normalization. Rejection
            # consumes only the retained probabilities and request state.
            tokens = verify.output.token_ids.clone()
            for tensor in op.scratch.values():
                tensor.fill_(float("nan"))
            verify()
            torch.testing.assert_close(verify.output.token_ids, tokens)
    finally:
        worker.close()
        graph.reset()


def test_target_penalties_temperature_and_grammar_follow_global_token_ids():
    op = fixture(batch=4)
    op()
    logits = torch.randn(4, 4, 8, device="cuda")
    p = torch.empty_like(logits)
    valid = torch.empty(4, 4, device="cuda", dtype=torch.bool)
    op.state.parameters.copy_(torch.tensor([.7, 1., 0., 1.2, .3, .1], device="cuda"))
    op.state.histogram[:, 3] = 2
    op.state.grammar_enabled.fill_(True)
    op.state.grammar_mask.zero_()
    op.state.grammar_mask[:, 0] = (1 << 3) | (1 << 5)
    target = PreparedVocabularySoftmax(logits, p, valid, op.state, op.slots, op.generations,
        op.live_rows, op.tokens, start=0, runtime=None, session=None, coordinator=None, name="target")
    target.prepare()
    target()
    assert valid.all()
    for step in range(4):
        count = op.state.histogram[:4].clone()
        for prior in range(step):
            count.scatter_add_(1, op.tokens[:, prior:prior + 1], torch.ones(4, 1, device="cuda", dtype=count.dtype))
        x = logits[:, step]
        x = torch.where(count > 0, torch.where(x > 0, x / 1.2, x * 1.2), x)
        x = (x - (count > 0).float() * .3 - count.float() * .1) / .7
        x[:, [0, 1, 2, 4, 6, 7]] = -float("inf")
        torch.testing.assert_close(p[:, step], x.softmax(-1), atol=1e-7, rtol=3e-6)



def test_zero_prefix_and_extreme_later_ratio_keep_finite_residuals():
    op = fixture(batch=1, depth=3)
    op.tokens.fill_(0)
    op.valid.fill_(True)
    op.probabilities.zero_()
    op.probabilities[:, :, 0] = 1e-40
    op.probabilities[:, :, 1] = 1.
    p = torch.zeros(1, 4, 8, device="cuda")
    p[:, :, 0] = 1.
    p[:, 0, 0] = 0.
    p[:, 0, 2] = 1.
    verify = verifier(op, p)
    verify()
    assert verify.output.valid.all()
    assert verify.output.token_ids[0, 0] == 2
    assert verify.output.accepted_drafts[0] == 0


@triton.jit
def _append_fixed_sequence(Tokens, Counts, Pending, Sequence, Proposed, Witness,
                            Selected, Residual, Seeds, Offsets,
                            DOMAIN: tl.constexpr):
    row = tl.program_id(0)
    count = tl.load(Counts + row)
    proposed = tl.load(Proposed + row)
    column = tl.arange(0, 4)
    token = tl.load(Tokens + row * 3 + column, column < 3, other=-1)
    tl.store(Sequence + row * 3 + count + column, token, (column < proposed) & (count + column < 3))
    if proposed > 0:
        tl.store(Pending + row, tl.load(Tokens + row * 3 + proposed - 1))
    tl.store(Counts + row, count + proposed)
    if count == 0:
        # Independently recover the first block decision from the compact
        # statistics, then count rounds that emit both drafts after rejecting it.
        p = tl.load(Selected + row * 32)
        q = tl.load(Selected + row * 32 + 8)
        prefix = tl.exp(tl.minimum(tl.log(p) - tl.log(q), 0.0))
        weight = tl.load(Residual + (row * 3 + 1) * 32)
        denominator = weight + 1.0 - prefix
        h = tl.where(denominator > 0, weight / denominator, 1.0)
        seed = tl.load(Seeds + row).to(tl.int64) ^ DOMAIN
        draw = tl.load(Offsets + row).to(tl.uint64)
        bits, _, _, _ = tl.philox(seed, draw.to(tl.uint32), (draw >> 32).to(tl.uint32), 0, 0)
        u = (bits.to(tl.float32) + .5) * 2.3283064365386963e-10
        tl.atomic_add(Witness, ((u > h) & (proposed == 3)).to(tl.int32))


def test_conditional_block_rejection_joint_law_with_truncated_rounds():
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.triton import prepare_triton
    from ignition.components.speculation import TARGET_RNG_DOMAIN

    op = fixture(batch=64, depth=2, vocab=3)
    op.embedding.zero_()
    op.embedding[:, :3] = torch.eye(3, device="cuda", dtype=torch.bfloat16)
    # Every proposal row strongly favors a different successor. Markov features
    # select the predecessor row, so q changes after every accepted or corrected token.
    q = torch.tensor([[.05, .85, .10], [.75, .15, .10], [.10, .15, .75]], device="cuda")
    op.head.zero_()
    op.head[:, :3] = q.log().T.bfloat16()
    op.logits.zero_()
    target = torch.tensor([[.70, .20, .10], [.10, .25, .65], [.35, .60, .05]],
                          device="cuda", dtype=torch.float32)
    p = torch.empty(64, 3, 3, device="cuda")
    limits = torch.full((64,), 3, device="cuda", dtype=torch.int32)
    verify = verifier(op, p, limits=limits)
    counts = torch.zeros(64, device="cuda", dtype=torch.int32)
    sequence = torch.full((64, 3), -1, device="cuda", dtype=torch.int64)
    witness = torch.zeros(1, device="cuda", dtype=torch.int32)
    append = prepare_triton(_append_fixed_sequence, (verify.output.token_ids, counts, op.pending,
        sequence, verify.output.proposed_count, witness, verify.scratch["selected"],
        verify.scratch["residual"], op.state.seed, verify.rng.target_offset), (64, 1, 1),
        DOMAIN=TARGET_RNG_DOMAIN, num_warps=4, enable_fp_fusion=False)

    class ConditionalRounds:
        def __call__(self):
            counts.zero_()
            sequence.fill_(-1)
            op.pending.zero_()
            for _ in range(3):
                limits.copy_((3 - counts).clamp(0, 3))
                op()
                p[:, 0].copy_(target[op.pending])
                p[:, 1:].copy_(target[op.tokens])
                verify()
                append()
                op.offsets.add_((limits > 0).long() * 8)
                verify.rng.target_offset.add_(verify.output.rng_increment)

    compiled = compile_forward(ConditionalRounds())
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    observed_sequences = []
    for _ in range(2048):
        graph.replay()
        observed_sequences.append(sequence.clone())
    torch.cuda.synchronize()
    observed_sequences = torch.cat(observed_sequences)
    assert (observed_sequences >= 0).all()
    assert (counts == 3).all()
    assert int(witness) > 0
    matrix = target.cpu().double()
    matrix /= matrix.sum(-1, keepdim=True)
    records = []
    for width in (2, 3):
        powers = 3 ** torch.arange(width - 1, -1, -1, device="cuda")
        code = (observed_sequences[:, :width] * powers).sum(-1)
        observed = code.bincount(minlength=3**width).cpu().double() / code.numel()
        expected = []
        import itertools
        for sequence_ids in itertools.product(range(3), repeat=width):
            probability, previous = 1., 0
            for token in sequence_ids:
                probability *= float(matrix[previous, token])
                previous = token
            expected.append(probability)
        error = float((observed - torch.tensor(expected, dtype=torch.float64)).abs().max())
        # A union bound covers all 9 two-token and 27 three-token events.
        bound = math.sqrt(math.log(2 * (9 + 27) / 1e-7) / (2 * code.numel()))
        assert error < bound
        records.append(dict(width=width, sequences=code.numel(), simultaneous_hoeffding_bound=bound,
                            max_joint_error=error))
    print(json.dumps(dict(conditional_joint_law=records, rejected_then_later_accepted_rounds=int(witness))))
    graph.reset()


def test_width_one_replays_target_only_with_empty_draft_storage():
    op = fixture(batch=4, depth=1, vocab=8)
    p = torch.zeros(4, 1, 8, device="cuda")
    p[:, :, 3] = 1
    rng = SpeculativeRngState.allocate("cuda", capacity=4)
    rng.generation.fill_(7)
    verification = PreparedVocabularyBlockVerification(
        torch.empty(4, 0, 8, device="cuda"), torch.empty(4, 0, dtype=torch.int64, device="cuda"), p,
        op.state, op.slots, op.generations, op.live_rows, start=0, runtime=None, session=None,
        coordinator=None, name="target-only", rng=rng,
        draft_valid=torch.empty(4, 0, dtype=torch.bool, device="cuda"),
        target_valid=torch.ones(4, 1, dtype=torch.bool, device="cuda"))
    verification.prepare()
    verification()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        verification()
    try:
        for token in (3, 1, 7, 0):
            p.zero_()
            p[:, :, token] = 1
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            assert verification.output.valid.all()
            assert (verification.output.token_ids == token).all()
            assert (verification.output.proposed_count == 1).all()
            assert (verification.output.accepted_drafts == 0).all()
            assert (verification.output.offered_drafts == 0).all()
            assert (verification.output.rng_increment == verification.draw_reservation).all()
            assert (verification.output.survival == -1).all()
    finally:
        graph.reset()


@pytest.mark.parametrize('fp8', [False, True])
def test_functional_distributions_and_verification_return_local_replay_values(fp8):
    from ignition.preparation.compilation import compile_forward

    proposal = fixture(batch=4, depth=3, vocab=1024, fp8=fp8)
    proposal()
    p = torch.full((4, 4, 1024), 1 / 1024, device='cuda')
    verification = verifier(proposal, p)
    verification()
    expected_q = proposal.probabilities.clone()
    expected_valid = proposal.valid.clone()
    expected_tokens = proposal.tokens.clone()
    expected = {name: getattr(verification.output, name).clone()
                for name in verification.output.__dataclass_fields__}
    logits, candidates, pending = proposal.logits, proposal.candidates, proposal.pending
    seeds, offsets = proposal.seeds, proposal.offsets
    target_valid = verification.target_valid
    proposal.functional(session=None)
    verification.functional(session=None)
    output_q = torch.empty_like(expected_q)
    output_valid = torch.empty_like(expected_valid)
    output_tokens = torch.empty_like(expected_tokens)
    actual = {name: torch.empty_like(value) for name, value in expected.items()}

    def forward():
        q, valid, tokens = proposal.run(logits, candidates, pending=pending, seeds=seeds, offsets=offsets)
        accepted = verification.run(q, tokens, p, valid, target_valid)
        output_q.copy_(q)
        output_valid.copy_(valid)
        output_tokens.copy_(tokens)
        for name in actual:
            actual[name].copy_(getattr(accepted, name))

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for _ in range(3):
        output_q.fill_(float('nan'))
        output_tokens.fill_(-99)
        before = torch.cuda.memory_stats()
        graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for key in ('allocation.all.allocated', 'allocation.all.freed',
                    'segment.all.allocated', 'segment.all.freed'):
            assert before[key] == after[key]
        torch.testing.assert_close(output_q, expected_q, rtol=0, atol=0)
        assert torch.equal(output_valid, expected_valid)
        assert torch.equal(output_tokens, expected_tokens)
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
    assert proposal.scratch is None and verification.scratch is None
    graph.reset()
