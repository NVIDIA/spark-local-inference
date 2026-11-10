"""Kimi transition logits feed stochastic proposals under captured replay."""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="CUDA opt-in")


@torch.no_grad()
def test_markov_bias_and_sampled_distribution_replay():
    from b12x.preparation import PreparationSession
    from ignition.recipes.kimi3.dspark import PreparedMarkov
    from ignition.components.draft_sampling import PreparedDraftDistribution
    from ignition.components.sampling import DeviceRequestState, RUNNING
    from ignition.preparation.compilation import compile_forward

    batch, vocab, features = 4, 128, 64
    device = torch.device("cuda:0")
    weights = SimpleNamespace(config=SimpleNamespace(vocabulary=vocab, draft=SimpleNamespace(markov_rank=features)),
        shard=SimpleNamespace(world_size=1), draft=SimpleNamespace(
            markov_embedding=torch.randn(vocab, features, device=device, dtype=torch.bfloat16) / 8,
            markov_head=torch.randn(vocab, features, device=device, dtype=torch.bfloat16) / 8))
    pending = torch.arange(batch, device=device, dtype=torch.int64)
    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=batch)
    state.status.fill_(RUNNING)
    state.generation.fill_(1)
    state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device=device))
    state.top_k.fill_(-1)
    slots = torch.arange(batch, device=device, dtype=torch.int32)
    epochs = torch.ones(batch, device=device, dtype=torch.int64)
    live = torch.tensor([batch], device=device, dtype=torch.int32)
    prefix = torch.zeros(batch, 1, device=device, dtype=torch.int64)
    seeds = torch.arange(batch, device=device, dtype=torch.int64) + 123
    offsets = torch.zeros_like(seeds)
    head = torch.randn(batch, vocab, device=device, dtype=torch.bfloat16) / 8
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        markov = PreparedMarkov(weights, pending, runtime=None, coordinator=None,
                                session=session, name="kimi.markov")
        proposal = PreparedDraftDistribution(torch.zeros(batch, 1, vocab, device=device), state,
            slots, epochs, live, prefix, seed=seeds, offset=offsets, sampled=torch.zeros_like(seeds))
        proposal.bind_logits(head, torch.empty(batch, vocab, dtype=torch.float32, device=device))
        proposal.prepare()
        proposal.functional(session=session, dynamic_inputs=True)
        observed_logits = torch.empty(batch, vocab, dtype=torch.float32, device=device)
        observed_probabilities = torch.empty(batch, 1, vocab, device=device)
        observed_valid = torch.empty(batch, 1, dtype=torch.bool, device=device)
        observed_sampled = torch.empty_like(seeds)

        def forward():
            transition = markov(pending)
            scratch = {name: torch.empty(shape, dtype=dtype, device=device)
                       for name, (shape, dtype) in proposal.layout.items()}
            logits = torch.empty(batch, 1, vocab, device=device)
            sampled = torch.empty_like(seeds)
            probabilities = proposal.run(scratch, logits, source_logits=head, source_bias=transition,
                candidates=prefix, seed=seeds, offset=offsets, sampled=sampled,
                slots=slots, generations=epochs, live_rows=live)
            observed_logits.copy_(transition)
            observed_probabilities.copy_(probabilities)
            observed_valid.copy_(scratch["valid"])
            observed_sampled.copy_(sampled)
        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        observed = []
        try:
            for iteration in range(12):
                pending.copy_((torch.arange(batch, device=device) + iteration) % vocab)
                offsets.fill_(4 * iteration)
                graph.replay()
                expected = weights.draft.markov_embedding[pending].float() @ weights.draft.markov_head.float().T
                torch.testing.assert_close(observed_logits, expected, rtol=1e-5, atol=1e-6)
                assert observed_valid.all()
                assert (observed_probabilities[:, 0].count_nonzero(-1) > 1).all()
                torch.testing.assert_close(observed_probabilities.sum(-1), torch.ones(batch, 1, device=device))
                observed.append(observed_sampled.clone())
            assert torch.stack(observed).unique().numel() > batch
        finally:
            graph.reset()
