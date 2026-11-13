# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared conditional scores match compiled unfused scores and native replay."""

import os

import pytest
import torch

from ignition.components.sampling import DeviceRequestState, RUNNING
from ignition.components.speculation import PreparedSpeculativeDistributions
from ignition.preparation.compilation import compile_forward
from .target_score_reference import UnfusedSpeculativeDistributions

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


def _fixture(batch, width, dtype, *, vocab=10000, offset=0, prefix_masks=False):
    generator = torch.Generator().manual_seed(932 + batch * 10 + width)
    capacity = max(4, batch)
    state = DeviceRequestState.allocate("cuda:0", vocab_size=vocab, capacity=capacity)
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    state.histogram.copy_(torch.randint(0, 3, state.histogram.shape, generator=generator, dtype=torch.int32))
    state.parameters.copy_(torch.tensor([[0., .95, 0., .8, -.1, .3], [.7, .9, .1, 1.3, .2, -.05],
        [1.2, 1., .05, 1., -.2, .1], [1., .8, 0., 1.1, .3, -.1]] * (capacity // 4)))
    state.top_k.copy_(torch.tensor([0, 17, 32, 100] * (capacity // 4), dtype=torch.int32))
    state.grammar_enabled.copy_(torch.tensor([0, 1, 0, 1] * (capacity // 4), dtype=torch.int32))
    state.grammar_mask.fill_(-1)
    slots = torch.tensor(([3, 0, 2, 0, *range(4, batch)] if batch >= 4 else list(range(batch))),
        dtype=torch.int32, device="cuda:0")
    epochs = torch.full((batch,), 7, dtype=torch.int64, device="cuda:0")
    if batch >= 4:
        epochs[2] = 99
    live = torch.tensor([batch], dtype=torch.int32, device="cuda:0")
    candidates = torch.tensor([[7, 7, -1, 9, vocab, 3, 4, 5]] * batch, dtype=torch.int64, device="cuda:0")
    logits = torch.randn((batch, width, vocab), generator=generator).to(dtype).cuda()
    masks = state.grammar_mask[:, None].expand(-1, 8, -1).clone() if prefix_masks else None
    counts = torch.full((capacity,), 2, dtype=torch.int32, device="cuda:0") if prefix_masks else None
    options = dict(prefix_offset=offset, grammar_masks=masks, grammar_mask_count=counts)
    reference = UnfusedSpeculativeDistributions(logits, state, slots, epochs, live, candidates, **options)
    candidate = PreparedSpeculativeDistributions(logits, state, slots, epochs, live, candidates, **options)
    return reference, candidate


def _exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.is_floating_point():
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        finite_or_infinite = ~torch.isnan(expected)
        assert torch.equal(actual.view(torch.int32)[finite_or_infinite],
                           expected.view(torch.int32)[finite_or_infinite])
    else:
        assert torch.equal(actual, expected)


def _request_snapshot(state):
    return {name: getattr(state, name).view(torch.int64).clone()
            if getattr(state, name).dtype == torch.uint64 else getattr(state, name).clone()
            for name in state.__dataclass_fields__}


def _assert_request_unchanged(state, before):
    for name, value in before.items():
        actual = getattr(state, name)
        actual = actual.view(torch.int64) if actual.dtype == torch.uint64 else actual
        assert torch.equal(actual, value), name


def _check_scores(reference, candidate, *, mutations=True):
    baseline_call = compile_forward(reference.prepare_scores)
    candidate_call = compile_forward(candidate.prepare_scores)
    baseline_call()
    candidate_call()
    before = _request_snapshot(candidate.state)
    for phase in range(4 if mutations else 1):
        if phase == 1:
            candidate.live_rows.fill_(max(0, candidate.slots.numel() - 1))
            if candidate.grammar_mask_count is not None:
                candidate.grammar_mask_count[0] = 0
                candidate.grammar_mask_count[1] = 9
        elif phase == 2:
            candidate.live_rows.fill_(candidate.slots.numel())
            candidate.logits[0, 0, 5] = float("nan")
            candidate.logits[0, 0, 6] = float("inf")
            candidate.logits[0, 0, 7] = -float("inf")
            candidate.logits[0, 0, 0] = 0.
            candidate.logits[0, 0, 1] = -0.
        elif phase == 3:
            candidate.live_rows.fill_(-1)
        eager_safe, eager_params = reference.prepare_scores()
        safe, params = candidate.prepare_scores()
        _exact(safe, eager_safe)
        _exact(params, eager_params)
        for name in ("scores", "maximum", "valid", "packed_top_k", "packed_top_p"):
            _exact(getattr(candidate, name), getattr(reference, name))
        _assert_request_unchanged(candidate.state, before)
        baseline_safe, baseline_params = baseline_call()
        safe, params = candidate_call()
        torch.cuda.synchronize()
        _exact(safe, baseline_safe)
        _exact(params, baseline_params)
        assert params.shape == (candidate.slots.numel(), 6)
        for name in ("scores", "maximum", "valid", "packed_top_k", "packed_top_p"):
            _exact(getattr(candidate, name), getattr(reference, name))
        _assert_request_unchanged(candidate.state, before)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("width", range(1, 7))
def test_compiled_conditional_score_catalog(batch, width):
    dtype = torch.float32 if width % 2 else torch.bfloat16
    reference, candidate = _fixture(batch, width, dtype, offset=1 if width < 6 else 0,
                                    prefix_masks=width % 2 == 0)
    _check_scores(reference, candidate)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_conditional_score_exact_order_and_prefix_validity(dtype):
    reference, candidate = _fixture(2, 6, dtype)
    candidate.state.grammar_enabled.zero_()
    candidate.state.histogram[:, 7] = 2
    _check_scores(reference, candidate)


@pytest.mark.parametrize("width,offset", [(1, 3), (3, 0), (6, 0)])
@pytest.mark.parametrize("prefix_masks", [False, True])
def test_conditional_score_grammar_and_nonfinite_policy(width, offset, prefix_masks):
    reference, candidate = _fixture(4, width, torch.float32, offset=offset, prefix_masks=prefix_masks)
    candidate.candidates.fill_(7)
    candidate.slots.copy_(torch.arange(4, dtype=torch.int32, device="cuda:0"))
    candidate.generations.fill_(7)
    candidate.state.grammar_enabled.fill_(1)
    candidate.logits.fill_(-float("inf"))
    candidate.logits[:, :, 4] = 1.
    candidate.logits[:, :, 5] = float("nan")
    masks = candidate.state.grammar_mask if candidate.grammar_masks is None else candidate.grammar_masks
    masks.zero_()
    masks[..., 0] = 1 << 4
    _check_scores(reference, candidate)


@pytest.mark.parametrize("batch,width", [(1, 1), (8, 6)])
def test_conditional_score_full_vocabulary(batch, width):
    _check_scores(*_fixture(batch, width, torch.bfloat16, vocab=129280), mutations=False)


@pytest.mark.parametrize("mode", ["signed_zero", "all_masked", "invalid_slots"])
def test_conditional_score_empty_and_slot_boundaries(mode):
    reference, candidate = _fixture(4, 3, torch.float32)
    candidate.candidates.fill_(7)
    candidate.generations.fill_(7)
    candidate.slots.copy_(torch.arange(4, dtype=torch.int32, device="cuda:0"))
    if mode == "signed_zero":
        candidate.state.parameters.copy_(torch.tensor([[1, 1, 0, 1, 0, 0]] * 4, device="cuda:0"))
        candidate.state.grammar_enabled.zero_()
        candidate.logits.zero_()
        candidate.logits[..., -1] = -0.
    elif mode == "all_masked":
        candidate.state.grammar_enabled.fill_(1)
        candidate.state.grammar_mask.zero_()
    else:
        candidate.slots[0] = -1
        candidate.slots[1] = candidate.state.capacity
        candidate.live_rows.fill_(5)
    _check_scores(reference, candidate, mutations=False)


def test_native_filtered_target_probabilities_match_with_reusable_scratch():
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.scratch import bind_scratch
    from ignition.components.speculation import PreparedStochasticVerification, SpeculativeRngState

    reference, candidate = _fixture(4, 6, torch.float32, prefix_masks=True)
    candidate.candidates.fill_(7)
    candidate.grammar_mask_count.fill_(8)
    candidate.state.seed.copy_(torch.tensor([131, 271, 419, 587], dtype=torch.uint64, device="cuda:0"))
    # Independent filter storage feeds the same request-owned verification policy.
    reference.prepare()
    work = bind_scratch(candidate.scratch_layout(4, 6, 10000), device="cuda:0", workspace=None)
    for name, tensor in work.items():
        setattr(candidate, name, tensor)
    from ignition.components.probability_normalization import PreparedProbabilityNormalization
    candidate.normalization = PreparedProbabilityNormalization(candidate.probabilities, candidate.valid)
    candidate.prepare()
    generator = torch.Generator().manual_seed(177)
    proposal_probs = torch.softmax(torch.randn((4, 5, 10000), generator=generator).cuda(), -1)
    ids = candidate.candidates[:, :5].contiguous()
    owners, graphs, verifiers = [], {}, []
    for name, distribution in (("unfused", reference), ("fused", candidate)):
        rng = SpeculativeRngState.allocate("cuda:0", capacity=4)
        rng.generation.fill_(7)
        rng.target_offset.copy_(torch.tensor([11, 22, 33, 44], device="cuda:0"))
        verify = PreparedStochasticVerification(proposal_probs, ids, distribution.probabilities,
            candidate.state, candidate.slots, candidate.generations, candidate.live_rows,
            rng=rng, target_valid=distribution.valid)
        verify.prepare()
        def run(distribution=distribution, verify=verify):
            distribution()
            verify()
        call = compile_forward(run)
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        owners.append((distribution, call))
        verifiers.append(verify)
        graphs[name] = graph
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={}, resources=(owners, work, verifiers))
    worker = bundle.start_worker()
    try:
        for live in (4, 3, 4, 1, 0, 4):
            candidate.live_rows.fill_(live)
            results = []
            for index, distribution in enumerate((reference, candidate)):
                if index == 1:
                    for tensor in work.values():
                        tensor.fill_(float("nan") if tensor.is_floating_point() else 1)
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(index)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                verify = verifiers[index]
                results.append(tuple(value.clone() for value in (distribution.probabilities, distribution.valid,
                    verify.output.token_ids, verify.output.accepted_drafts, verify.output.proposed_count,
                    verify.output.rng_increment, verify.offset)))
            for actual, expected in zip(results[1], results[0], strict=True):
                _exact(actual, expected)
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()


def test_native_c32_full_vocabulary_filter_and_verification():
    from ignition.components.probability_normalization import PreparedProbabilityNormalization
    from ignition.components.speculation import PreparedStochasticVerification, SpeculativeRngState
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.scratch import bind_scratch

    batch, width, vocabulary = 32, 6, 129280
    reference, candidate = _fixture(batch, width, torch.float32,
                                    vocab=vocabulary, prefix_masks=True)
    candidate.slots.copy_(torch.arange(batch, dtype=torch.int32, device='cuda'))
    candidate.generations.fill_(7)
    candidate.candidates.fill_(7)
    candidate.grammar_mask_count.fill_(8)
    state = candidate.state
    state.status.fill_(RUNNING)
    state.seed.copy_(torch.tensor([131 + 1409 * index for index in range(batch)],
                                 dtype=torch.uint64, device='cuda'))
    state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device='cuda'))
    state.top_k.fill_(-1)
    state.grammar_enabled.zero_()
    reference.prepare()
    work = bind_scratch(candidate.scratch_layout(batch, width, vocabulary),
                        device='cuda:0', workspace=None)
    for name, tensor in work.items():
        setattr(candidate, name, tensor)
    candidate.normalization = PreparedProbabilityNormalization(candidate.probabilities, candidate.valid)
    candidate.prepare()
    generator = torch.Generator(device='cuda').manual_seed(177)
    proposal_probs = torch.softmax(torch.randn((batch, width - 1, vocabulary),
        generator=generator, device='cuda'), -1)
    ids = candidate.candidates[:, :width - 1].contiguous()
    owners, graphs, verifiers = [], {}, []
    for name, distribution in (('reference', reference), ('candidate', candidate)):
        rng = SpeculativeRngState.allocate('cuda:0', capacity=batch)
        rng.generation.fill_(7)
        rng.target_offset.copy_(torch.arange(batch, device='cuda') * 11 + 11)
        verify = PreparedStochasticVerification(proposal_probs, ids, distribution.probabilities,
            state, candidate.slots, candidate.generations, candidate.live_rows,
            rng=rng, target_valid=distribution.valid)
        verify.prepare()
        def run(distribution=distribution, verify=verify):
            distribution()
            verify()
        call = compile_forward(run)
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        owners.append((distribution, verify, call))
        verifiers.append(verify)
        graphs[name] = graph
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={}, resources=(owners, work, verifiers))
    worker = bundle.start_worker()
    try:
        request_counter = state.rng_counter.clone()
        odd = torch.arange(batch, device='cuda').remainder(2) == 1
        for mixed in (False, True, False, True, False):
            state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device='cuda'))
            state.top_k.fill_(-1)
            state.grammar_enabled.zero_()
            candidate.grammar_masks.fill_(-1)
            if mixed:
                state.parameters[odd] = torch.tensor([.7, .95, .05, 1.3, .2, -.05], device='cuda')
                state.top_k[odd] = 20
                state.grammar_enabled[odd] = 1
                candidate.grammar_masks[odd] = 0
                candidate.grammar_masks[odd, :, 0] = (1 << 7) | (1 << 11) | (1 << 13)
            for index, distribution in enumerate((reference, candidate)):
                if index == 1:
                    for tensor in (*work.values(), candidate.top_p_workspace,
                                   candidate.normalization.maxima, candidate.normalization.masses,
                                   candidate.normalization.totals):
                        tensor.fill_(float('nan') if tensor.is_floating_point() else 255)
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()['allocation.all.allocated']
                ticket = worker.submit(index, repetitions=3)
                completion = worker.poll(timeout_ms=60000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
            _exact(candidate.probabilities, reference.probabilities)
            _exact(candidate.valid, reference.valid)
            for name in ('token_ids', 'accepted_drafts', 'offered_drafts', 'proposed_count',
                         'valid', 'rng_increment'):
                _exact(getattr(verifiers[1].output, name), getattr(verifiers[0].output, name))
            for name in ('offset', 'seed_bits'):
                _exact(getattr(verifiers[1], name), getattr(verifiers[0], name))
            _exact(state.rng_counter, request_counter)
            for verify in verifiers:
                verify.rng.target_offset.add_(verify.output.rng_increment)
            _exact(verifiers[1].rng.target_offset, verifiers[0].rng.target_offset)
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()


def test_target_score_aot_storage_binding_and_serialization(tmp_path):
    from ignition.preparation.aot import aot_forward

    reference, candidate = _fixture(2, 3, torch.float32, prefix_masks=True)
    candidate.candidates.fill_(7)
    candidate.grammar_mask_count.fill_(8)
    reference.prepare()
    candidate.prepare()
    reference()
    candidate()
    _exact(candidate.probabilities, reference.probabilities)
    reference_compiled = aot_forward(reference)
    reference_compiled(reference)
    compiled = aot_forward(candidate)
    compiled(candidate)
    _exact(candidate.probabilities, reference.probabilities)
    saved = tmp_path / "target-filter.bin"
    compiled.save_compiled_function(str(saved))
    with saved.open("rb") as source:
        restored = torch.compiler.load_compiled_function(source)
    assert restored.guard_check(candidate)
    candidate.probabilities.fill_(float("nan"))
    restored(candidate)
    _exact(candidate.probabilities, reference.probabilities)


@pytest.mark.parametrize("batch", [16, 32, 64])
@pytest.mark.parametrize("width", [7, 8])
def test_conditional_score_request_and_width_boundaries(batch, width):
    dtype = torch.float32 if width == 7 else torch.bfloat16
    reference, candidate = _fixture(batch, width, dtype, vocab=256,
        offset=1 if width == 7 else 0, prefix_masks=width == 8)
    _check_scores(reference, candidate)
