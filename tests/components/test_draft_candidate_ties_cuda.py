# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded proposals retain strictly better scores when cutoff ties fill a block."""

from contextlib import contextmanager
import os

import pytest
import torch

from ignition.components.draft_sampling import PreparedDraftProposal
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@contextmanager
def _captured_proposal(scores, top_k):
    device = scores.device
    vocab = scores.shape[1]
    probabilities = torch.zeros_like(scores)
    valid = torch.ones(1, dtype=torch.bool, device=device)
    packed_k = torch.tensor([top_k or vocab], dtype=torch.int32, device=device)
    packed_p = torch.tensor([.95], device=device)
    parameters = torch.tensor([[1., .95, 0., 1., 0., 0.]], device=device)
    slots = torch.zeros(1, dtype=torch.int32, device=device)
    seed = torch.tensor([17], dtype=torch.int64, device=device)
    offset = torch.zeros(1, dtype=torch.int64, device=device)
    sampled = torch.zeros(1, dtype=torch.int64, device=device)
    operation = PreparedDraftProposal(scores, probabilities, valid, packed_k, packed_p,
                                     parameters, slots, seed, offset, sampled, capacity=1)
    operation.prepare()
    compiled = compile_forward(operation)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    bundle = PreparedGraphBundle(device=0, graphs={"proposal": graph}, buffers={"scores": scores},
        resources=(operation, compiled, probabilities, valid, packed_k, packed_p,
                   parameters, slots, seed, offset, sampled))
    worker = bundle.start_worker()
    def replay():
        torch.cuda.synchronize()
        ticket = worker.submit(bundle.graph_indices["proposal"], repetitions=2)
        completion = worker.poll(timeout_ms=120000)
        assert completion is not None and completion.ticket == ticket and completion.error is None
    try:
        yield probabilities, valid, sampled, replay
    finally:
        worker.close()
        graph.reset()


@pytest.mark.parametrize("higher_indices", [(4095,), (4096,), (127, 4095, 4096, 8191)])
@pytest.mark.parametrize("top_k", [0, 3, 64])
@torch.no_grad()
def test_higher_score_survives_local_and_global_cutoff_ties(higher_indices, top_k):
    device = torch.device("cuda", 0)
    vocab = 8192
    scores = torch.full((1, vocab), -torch.inf, device=device)
    scores[0, :64] = 1.
    scores[0, list(higher_indices)] = 2.
    with _captured_proposal(scores, top_k) as (probabilities, valid, sampled, replay):
        replay()
        # Top-k below the number of higher scores keeps their boundary tie.
        # Otherwise top-p reaches the lower tied group and retains all 64 candidates.
        low_count = 0 if 0 < top_k <= len(higher_indices) else 64 - len(higher_indices)
        expected_ids = torch.cat((torch.arange(low_count, device=device),
                                  torch.tensor(higher_indices, device=device)))
        actual_ids = torch.nonzero(probabilities[0] > 0).flatten()
        assert torch.equal(actual_ids, expected_ids)
        expected = scores[0, expected_ids].softmax(0)
        torch.testing.assert_close(probabilities[0, expected_ids], expected, rtol=2e-6, atol=1e-8)
        assert (sampled[:, None] == expected_ids).any()
        assert valid.all()


@pytest.mark.parametrize("vocab", [17, 8192])
@pytest.mark.parametrize("top_k", [0, 7, 64])
@torch.no_grad()
def test_signed_zero_scores_preserve_boundary_ties_and_draw_order(vocab, top_k):
    scores = torch.zeros((1, vocab), device="cuda")
    with _captured_proposal(scores, top_k) as (probabilities, valid, sampled, replay):
        replay()
        expected_probabilities, expected_sampled = probabilities.clone(), sampled.clone()
        scores[:, 1::2] = -0.
        replay()
        # Signed zeros are equal scores, so their token-index tie order and the
        # bounded support are identical to those of a row of positive zeros.
        assert torch.equal(probabilities, expected_probabilities)
        assert torch.equal(sampled, expected_sampled)
        assert valid.all()
