# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Quantized proposal heads preserve target weights and replay with bounded storage."""

import os

import pytest
import torch

from ignition.components.vocabulary import PreparedVocabularyProjection, mxfp8_draft_head, nvfp4_draft_head
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 5, 20, 40, 160])
@pytest.mark.parametrize("hidden", [5120, 6144])
@torch.no_grad()
def test_nvfp4_vocabulary_shard_keeps_target_weights(rows, hidden):
    device = torch.device("cuda:0")
    torch.manual_seed(1017)
    # A TP4 DeepSeek vocabulary shard has 64-aligned rows, requiring scale padding.
    target = torch.randn((32320, hidden), device=device, dtype=torch.bfloat16) * .02
    original = target.clone()
    weight = nvfp4_draft_head(target)
    assert torch.equal(target, original)
    source = torch.randn((rows, hidden), device=device, dtype=torch.bfloat16) * .1
    local = torch.empty((rows, 32320), device=device, dtype=torch.bfloat16)
    logits = torch.empty_like(local)
    with RecordedPreparationSession(device=device, checkpoint="draft-vocabulary-shard",
            workload_schema="nvfp4-a16", numerical_policy="nvfp4", compile_workers=4) as session:
        operation = PreparedVocabularyProjection(source, weight, local, logits,
            runtime=None, coordinator=None, session=session, name="draft-head")
        session.tune(show_progress=False)
        compiled = compile_forward(lambda: operation(source))
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for factor in (1., -.5):
            source.mul_(factor)
            expected = torch.nn.functional.linear(source.float(), original.float())
            operation(source)
            eager = logits.clone()
            logits.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(logits, eager, atol=0, rtol=0)
            assert torch.isfinite(logits).all()
            assert (logits.float() - expected).norm() / expected.norm() < .15
            assert torch.equal(target, original)
        graph.reset()


@pytest.mark.parametrize("rows", [1, 7, 8, 32, 4096])
@torch.no_grad()
def test_mxfp8_vocabulary_shard_replays_changed_inputs_without_target_mutation(rows):
    from ignition.preparation.graphs import PreparedGraphBundle

    torch.manual_seed(41017)
    target = torch.randn((192, 512), device="cuda", dtype=torch.bfloat16) * .02
    target[:32].zero_()
    original = target.clone()
    weight = mxfp8_draft_head(target, rows_per_chunk=71)
    scales = weight.weight.scale_rows.view(torch.float8_e8m0fnu).reshape(192, 16)
    restored = (weight.weight.values.float().reshape(192, 16, 32) * scales.float()[..., None]).reshape(192, 512)
    # Compute the conversion independently, including all-zero groups and the chunk tail.
    blocks = original.float().reshape(192, 16, 32)
    maxima = blocks.abs().amax(-1)
    exponents = torch.ceil(torch.log2(torch.where(maxima > 0, maxima / 448., 1.)))
    expected_scales = torch.exp2(exponents)
    expected_values = (blocks / expected_scales[..., None]).to(torch.float8_e4m3fn)
    torch.testing.assert_close(scales.float(), expected_scales, rtol=0, atol=0)
    assert torch.equal(weight.weight.values.view(torch.uint8), expected_values.reshape(192, 512).view(torch.uint8))
    assert torch.equal(target, original)
    source = torch.randn((rows, 512), device="cuda", dtype=torch.bfloat16)
    local, logits = (torch.empty((rows, 192), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    with RecordedPreparationSession(device="cuda:0", checkpoint="mxfp8-proposal-head",
            workload_schema="a16-vocabulary", numerical_policy="mxfp8-weight-bf16-source",
            autotune_effort="default", compile_workers=4) as session:
        operation = PreparedVocabularyProjection(source, weight, local, logits,
            runtime=None, coordinator=None, session=session, name="draft-head")
        session.tune(show_progress=False)
        assert operation.projection.plan.prepared.selection.config.mode == "a16"
        compiled = compile_forward(lambda: operation(source))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs={"head": graph}, buffers={},
            resources=(operation, compiled, source, weight, local, logits, session))
        worker = bundle.start_worker()
        try:
            for factor in (1., -.5, 0.):
                source.mul_(factor)
                expected = source.double() @ restored.double().T
                logits.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                for _ in range(11):
                    ticket = worker.submit(0)
                    completion = worker.poll(timeout_ms=30000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                bound = expected.abs() * 2**-7 + expected.abs().max() * 1e-6
                assert bool(((logits.double() - expected).abs() <= bound).all())
                assert torch.equal(target, original)
        finally:
            worker.close()
            graph.reset()


@torch.no_grad()
def test_mxfp8_head_retains_the_actual_bounded_proposal_distribution():
    from ignition.components.draft_sampling import PreparedDraftProposal
    from ignition.components.linear import UnquantizedProjection
    from ignition.preparation.graphs import PreparedGraphBundle

    torch.manual_seed(41018)
    rows, vocab = 7, 1024
    target = torch.randn((vocab, 512), device="cuda", dtype=torch.bfloat16) * .02
    saved_target = target.clone()
    weight = mxfp8_draft_head(target)
    source = torch.randn((rows, 512), device="cuda", dtype=torch.bfloat16)
    local, base = (torch.empty((rows, vocab), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    markov_head = torch.randn((vocab, 256), device="cuda", dtype=torch.bfloat16) * .02
    saved_markov = markov_head.clone()
    markov_inputs = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
    bias = torch.empty((rows, vocab), device="cuda", dtype=torch.float32)
    scores, probabilities = (torch.empty_like(bias) for _ in range(2))
    valid = torch.ones(rows, device="cuda", dtype=torch.bool)
    top_k = torch.full((rows,), 64, device="cuda", dtype=torch.int32)
    top_p = torch.ones(rows, device="cuda")
    parameters = torch.tensor([[1., 1., 0., 1., 0., 0.]], device="cuda").repeat(rows, 1)
    slots = torch.arange(rows, device="cuda", dtype=torch.int32)
    seeds = torch.arange(17, 17 + rows, device="cuda", dtype=torch.int64)
    offsets = torch.zeros(rows, device="cuda", dtype=torch.int64)
    sampled = torch.empty(rows, device="cuda", dtype=torch.int64)
    proposal = PreparedDraftProposal(scores, probabilities, valid, top_k, top_p,
        parameters, slots, seeds, offsets, sampled, capacity=rows)
    with RecordedPreparationSession(device="cuda:0", checkpoint="mxfp8-draft-proposal",
            workload_schema="bounded-q-checkpoint-markov", numerical_policy="mxfp8-draft-checkpoint-target",
            autotune_effort="default", compile_workers=4) as session:
        head = PreparedVocabularyProjection(source, weight, local, base,
            runtime=None, coordinator=None, session=session, name="draft-head")
        markov = UnquantizedProjection(markov_inputs, markov_head, bias,
            session=session, name="markov")
        session.tune(show_progress=False)
        head(source)
        markov(markov_inputs)
        scores.copy_(base.float() + bias)
        proposal.prepare()

        def run():
            head(source)
            markov(markov_inputs)
            scores.copy_(base.float() + bias)
            valid.fill_(True)
            proposal()

        compiled = compile_forward(run)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs={"proposal": graph}, buffers={},
            resources=(compiled, head, markov, proposal, source, weight, local, base, session))
        worker = bundle.start_worker()
        try:
            for factor in (1., -.5, 0.):
                source.mul_(factor)
                offsets.add_(28)
                base.fill_(float("nan"))
                probabilities.fill_(float("nan"))
                sampled.fill_(-1)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                for _ in range(11):
                    ticket = worker.submit(0)
                    completion = worker.poll(timeout_ms=30000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                assert valid.all() and torch.isfinite(probabilities).all()
                assert (probabilities.gather(1, sampled[:, None]) > 0).all()
                # The ordinary sampler retains its bounded q from the emitted
                # quantized-head scores, rather than checkpoint-head logits.
                values, ids = scores.double().topk(64, dim=-1)
                expected = torch.zeros_like(scores, dtype=torch.float64)
                expected.scatter_(1, ids, values.softmax(-1))
                torch.testing.assert_close(probabilities.double(), expected, atol=1e-9, rtol=3e-6)
                torch.testing.assert_close(bias, torch.nn.functional.linear(markov_inputs.float(), markov_head.float()),
                                           atol=2e-5, rtol=2e-5)
                assert torch.equal(markov_head, saved_markov)
                assert torch.equal(target, saved_target)
        finally:
            worker.close()
            graph.reset()
