# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""FP32 full-vocabulary proposal probabilities and independent Philox counters."""

import os

import pytest
import torch
import triton
import triton.language as tl

from ignition.components.markov_sampling import (
    PreparedMarkovProposals, _proposal_random_bits, fp8_markov_head,
)
from ignition.components.sampling import DeviceRequestState, RUNNING
from ignition.components.speculation import DRAFT_RNG_DOMAIN

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an explicitly assigned CUDA device"
)


def philox(seed, counter):
    """Random123 Philox4x32-10, including its published zero key/counter vector."""
    c0, c1, c2, c3 = counter
    k0, k1 = seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF
    for _ in range(10):
        p0, p1 = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (
            ((p1 >> 32) ^ c1 ^ k0) & 0xFFFFFFFF,
            p1 & 0xFFFFFFFF,
            ((p0 >> 32) ^ c3 ^ k1) & 0xFFFFFFFF,
            p0 & 0xFFFFFFFF,
        )
        k0, k1 = (k0 + 0x9E3779B9) & 0xFFFFFFFF, (k1 + 0xBB67AE85) & 0xFFFFFFFF
    return c0, c1, c2, c3


@triton.jit
def rng_probe(Seeds, Offsets, Tokens, Output, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    seed = tl.load(Seeds + row, row < N, other=0)
    draw = tl.load(Offsets + row, row < N, other=0)
    token = tl.load(Tokens + row, row < N, other=0)
    tl.store(Output + row, _proposal_random_bits(seed, draw, token), row < N)


def test_philox_counter_coordinates():
    assert philox(0, (0, 0, 0, 0)) == (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    coordinates = [
        (token, draw & 0xFFFFFFFF, draw >> 32, 0x47554D42)
        for draw in (0, 4, 24, 28, 2**32, 2**32 + 4, 2**63 - 4)
        for token in (0, 1, 64, 129279)
    ]
    assert len(set(coordinates)) == len(coordinates)
    # A separate request key and the complete 64-bit draw counter are retained.
    seeds = torch.tensor([DRAFT_RNG_DOMAIN + i % 3 for i in range(len(coordinates))], device="cuda", dtype=torch.int64)
    draws = torch.tensor([c[1] | c[2] << 32 for c in coordinates], device="cuda", dtype=torch.int64)
    tokens = torch.tensor([c[0] for c in coordinates], device="cuda", dtype=torch.int32)
    output = torch.empty_like(tokens, dtype=torch.uint32)
    rng_probe[(1,)](seeds, draws, tokens, output, len(coordinates), triton.next_power_of_2(len(coordinates)))
    expected = [philox(int(seed), counter)[0] for seed, counter in zip(seeds.cpu(), coordinates)]
    assert output.cpu().tolist() == expected


def fixture(batch=1, depth=7, vocab=1024, rank=256, random=True, fp8=False):
    gen = torch.Generator().manual_seed(714)
    embedding = torch.randn((vocab, rank), generator=gen) / 8 if random else torch.zeros((vocab, rank))
    head = torch.randn((vocab, rank), generator=gen) / 8 if random else torch.zeros((vocab, rank))
    embedding, head = (embedding.to("cuda", torch.bfloat16), head.to("cuda", torch.bfloat16))
    if fp8:
        head = fp8_markov_head(head)
    base = torch.randn((batch, depth, vocab), generator=gen).to("cuda", torch.bfloat16)
    pending = torch.arange(batch, device="cuda", dtype=torch.int64) % vocab
    tokens = torch.empty((batch, depth), device="cuda", dtype=torch.int64)
    q = torch.empty((batch, depth, vocab), device="cuda")
    valid = torch.empty((batch, depth), device="cuda", dtype=torch.bool)
    state = DeviceRequestState.allocate("cuda", vocab_size=vocab, capacity=max(4, batch))
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    state.parameters.copy_(torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0], device="cuda"))
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    epochs = torch.full((batch,), 7, device="cuda", dtype=torch.int64)
    live = torch.tensor([batch], device="cuda", dtype=torch.int32)
    seeds = torch.arange(batch, device="cuda", dtype=torch.int64) ^ DRAFT_RNG_DOMAIN
    offsets = torch.full((batch,), 2**32 + 28, device="cuda", dtype=torch.int64)
    op = PreparedMarkovProposals(
        base, embedding, head, pending, tokens, q, valid, state, slots, epochs, live, seeds, offsets
    )
    op.prepare()
    return op


@pytest.mark.parametrize("batch,depth,vocab", [(1, 2, 1024), (8, 7, 1024), (1, 1, 129280)])
def test_fp8_markov_transition_uses_reconstructed_weights_and_fp32_q(batch, depth, vocab):
    op = fixture(batch, depth, vocab, fp8=True)
    values = op.head.values.double()
    scales = op.head.scale.double()
    u = 2.0**-24
    gamma = (values.shape[1] - 1) * u / (1 - (values.shape[1] - 1) * u)
    for call in op.kernels[:-1]:
        call()
    raw = op.probabilities.clone().double()
    op.kernels[-1]()
    previous = op.pending
    for step in range(depth):
        embedding = op.embedding[previous].double()
        dot = embedding @ values.T
        exact = op.base[:, step].double() + dot * scales
        absolute = (embedding.abs() @ values.abs().T) * scales.abs()
        bound = gamma * absolute + 2 * u * (dot.abs() * scales.abs() + op.base[:, step].double().abs())
        assert ((raw[:, step] - exact).abs() <= bound + 2.0**-149).all()
        # The probability oracle uses the quantized values and FP32 row scales,
        # rather than the checkpoint BF16 weights.
        expected_q = raw[:, step].softmax(-1)
        torch.testing.assert_close(op.probabilities[:, step].double(), expected_q, atol=1e-10, rtol=3e-6)
        previous = op.tokens[:, step]
    assert op.valid.all()
    assert op.probabilities.dtype == torch.float32


@pytest.mark.parametrize("kind", ["single_token", "inactive", "nan"])
def test_fp8_markov_preserves_grammar_and_invalid_row_contract(kind):
    op = fixture(batch=1, depth=2, random=False, fp8=True)
    op.base.zero_()
    if kind == "single_token":
        op.state.grammar_enabled.fill_(True)
        op.state.grammar_mask.zero_()
        op.state.grammar_mask[0, 0] = 1 << 3
    elif kind == "inactive":
        op.state.status.zero_()
    else:
        op.base[:, :, 7] = float("nan")
    op()
    assert torch.isfinite(op.probabilities).all()
    if kind == "single_token":
        assert op.valid.all() and (op.tokens == 3).all()
    else:
        assert not op.valid.any() and (op.tokens == 0).all()
        assert (op.probabilities[:, :, 0] == 1).all()
        assert (op.probabilities[:, :, 1:] == 0).all()


@pytest.mark.parametrize(
    "batch,depth,vocab", [(1, 1, 8), (1, 2, 1024), (1, 6, 1024), (1, 7, 1024), (8, 7, 1024), (1, 7, 129280)]
)
def test_fp32_scores_against_exact_bf16_products(batch, depth, vocab):
    op = fixture(batch, depth, vocab)
    for call in op.kernels[:-1]:
        call()
    raw = op.probabilities.clone()
    op.kernels[-1]()
    # Every BF16 product is exactly representable in FP64. A standard FP32
    # accumulation bound covers reduction order without assuming BF16 output.
    u = 2.0**-24
    gamma = (op.head.shape[1] - 1) * u / (1 - (op.head.shape[1] - 1) * u)
    previous = op.pending
    for step in range(depth):
        e = op.embedding[previous].double()
        w = op.head.double()
        bias = e @ w.T
        exact = op.base[:, step].double() + bias
        absolute_products = e.abs() @ w.abs().T
        bound = gamma * absolute_products + u * (bias.abs() + op.base[:, step].double().abs())
        assert ((raw[:, step].double() - exact).abs() <= bound + 2.0**-149).all()
        expected = raw[:, step].double().softmax(-1)
        # Triton's approximate FP32 exponentials and sum reduction are checked
        # separately from transition accumulation; accumulated mass is tighter
        # than verification's existing 1e-4 probability-row requirement.
        torch.testing.assert_close(op.probabilities[:, step].double(), expected, atol=1e-10, rtol=3e-6)
        assert (op.probabilities[:, step].gather(1, op.tokens[:, step, None]) > 0).all()
        previous = op.tokens[:, step]
    assert op.valid.all()
    assert (op.probabilities.sum(-1) - 1).abs().max() < 1e-5


@pytest.mark.parametrize(
    "kind",
    [
        "uniform",
        "all_negative",
        "extreme",
        "nan",
        "positive_inf",
        "all_masked",
        "single_token",
        "inactive",
        "duplicate",
    ],
)
def test_masks_nonfinite_and_denominator_boundaries(kind):
    op = fixture(batch=8, random=False)
    op.base.zero_()
    if kind == "all_negative":
        op.base.fill_(-10000)
    elif kind == "extreme":
        op.base.fill_(-10000)
        op.base[:, :, 1] = 10000
    elif kind == "nan":
        op.base[:, :, 7] = float("nan")
    elif kind == "positive_inf":
        op.base[:, :, 7] = float("inf")
    elif kind in ("all_masked", "single_token"):
        op.state.grammar_enabled.fill_(True)
        op.state.grammar_mask.zero_()
        if kind == "single_token":
            op.state.grammar_mask[:, 0] = 1 << 3
    elif kind == "inactive":
        op.state.status.zero_()
    elif kind == "duplicate":
        op.slots.zero_()
    op()
    assert torch.isfinite(op.probabilities).all()
    invalid = kind in ("nan", "positive_inf", "all_masked", "inactive", "duplicate")
    assert (op.valid == (not invalid)).all()
    if invalid:
        assert (op.tokens == 0).all()
        assert (op.probabilities[:, :, 0] == 1).all() and (op.probabilities[:, :, 1:] == 0).all()
    elif kind == "single_token":
        assert (op.tokens == 3).all()
    elif kind == "extreme":
        assert (op.tokens == 1).all()
    elif kind in ("uniform", "all_negative"):
        assert (op.probabilities == 1 / op.vocab).all()


@pytest.mark.parametrize("fp8", [False, True])
def test_counter_advance_rollback_and_batch_permutation(fp8):
    op = fixture(batch=8, fp8=fp8)
    op()
    tokens = op.tokens.clone()
    q = op.probabilities.clone()
    op.offsets.add_(28)
    op()
    assert not torch.equal(tokens, op.tokens)
    op.offsets.sub_(28)
    op()
    assert torch.equal(tokens, op.tokens) and torch.equal(q.view(torch.int32), op.probabilities.view(torch.int32))
    op.offsets.add_(2**32)
    op()
    assert not torch.equal(tokens, op.tokens)
    op.offsets.sub_(2**32)
    permutation = torch.tensor([4, 1, 7, 0, 5, 2, 6, 3], device="cuda")
    for value in (op.base, op.pending, op.slots, op.generations, op.seeds, op.offsets):
        value.copy_(value[permutation])
    op()
    assert torch.equal(tokens[permutation], op.tokens)
    assert torch.equal(q[permutation].view(torch.int32), op.probabilities.view(torch.int32))


@torch.no_grad()
@pytest.mark.parametrize("fp8", [False, True])
def test_uniform_endpoint_keeps_single_token_grammar_support_under_native_replay(fp8):
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle

    allowed = 106475
    seed = 213 ^ DRAFT_RNG_DOMAIN
    assert philox(seed, (allowed, 0, 0, 0x47554D42))[0] == 0xFFFFFFFF
    operation = fixture(batch=1, depth=1, vocab=129280, random=False, fp8=fp8)
    operation.base.zero_()
    operation.seeds.fill_(seed)
    operation.offsets.zero_()
    operation.state.grammar_enabled.fill_(True)
    operation.state.grammar_mask.zero_()
    operation.state.grammar_mask[0, allowed // 32] = 1 << (allowed % 32)
    call = compile_forward(operation)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    bundle = PreparedGraphBundle(device=0, graphs={"endpoint": graph}, buffers={},
        resources=(operation, call))
    worker = bundle.start_worker()
    try:
        for _ in range(4):
            operation.probabilities.fill_(float("nan"))
            operation.tokens.zero_()
            operation.valid.fill_(False)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(0)
            result = worker.poll(timeout_ms=30000)
            assert result is not None and result.ticket == ticket and result.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            assert operation.valid.all()
            assert operation.tokens.item() == allowed
            assert operation.probabilities[0, 0, allowed].item() == 1.0
            assert operation.probabilities.sum().item() == 1.0
            assert operation.probabilities[0, 0, 0].item() == 0.0
    finally:
        worker.close()
        graph.reset()
