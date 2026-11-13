# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sparse broadcast support preserves dense proposal bytes and clears divergent local support."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not (os.environ.get('FLARE_RUN_CUDA_TESTS') and torch.cuda.is_available()),
                                reason='set FLARE_RUN_CUDA_TESTS=1 on a CUDA host')


@pytest.mark.parametrize('batch,vocab', [(1, 257), (4, 129280)])
@pytest.mark.parametrize("shortlist", [False, True])
@torch.no_grad()
def test_support_emission_and_sparse_restore_native_replay(batch, vocab, shortlist):
    from ignition.components.draft_sampling import PreparedDraftDistribution, DRAFT_CANDIDATES
    from ignition.components.sampling import DeviceRequestState, RUNNING
    from ignition.preparation.graphs import PreparedGraphBundle

    state = DeviceRequestState.allocate('cuda:0', vocab_size=vocab, capacity=4)
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device='cuda'))
    slots = torch.arange(batch, dtype=torch.int32, device='cuda')
    epochs = torch.full((batch,), 7, dtype=torch.int64, device='cuda')
    live = torch.tensor([batch], dtype=torch.int32, device='cuda')
    candidates = torch.zeros((batch, 7), dtype=torch.int64, device='cuda')
    logits = torch.randn((batch, 1, vocab), device='cuda')
    seed = torch.arange(batch, dtype=torch.int64, device='cuda')
    offset = torch.zeros(batch, dtype=torch.int64, device='cuda')
    op = PreparedDraftDistribution(logits, state, slots, epochs, live, candidates,
        seed=seed, offset=offset,
        sampled=torch.zeros(batch, dtype=torch.int64, device='cuda'))
    if shortlist:
        support = torch.arange(16, dtype=torch.int64, device='cuda').expand(batch, -1).contiguous()
        logits[:, :, 16:] = -torch.inf
        op.bind_shortlist(support)
    op.prepare()
    reference = PreparedDraftDistribution(logits, state, slots, epochs, live, candidates,
        seed=seed, offset=offset, sampled=torch.zeros_like(seed))
    reference.prepare()
    op.support_ids = torch.zeros((batch, DRAFT_CANDIDATES), dtype=torch.int64, device='cuda')
    op.support_probabilities = torch.zeros((batch, DRAFT_CANDIDATES), device='cuda')
    op.proposal.bind_support(op.support_ids, op.support_probabilities)
    op._prepare_support_restore()
    for mode in ('ordinary', 'ties', 'all_invalid', 'filtered', 'token_zero_padding'):
        logits.normal_()
        state.top_k.zero_()
        state.parameters[:, 1] = 1.
        if mode == 'ties':
            logits.zero_()
        if mode == 'all_invalid':
            logits.fill_(float('nan'))
        if mode == 'filtered':
            state.top_k.fill_(3)
            state.parameters[:, 1] = .7
        if mode == 'token_zero_padding':
            state.top_k.fill_(3)
            logits.fill_(-10.)
            logits[:, :, 0] = 2.
            logits[:, :, 4] = 1.
            logits[:, :, 7] = .5
        if shortlist:
            logits[:, :, 16:] = -torch.inf
        reference()
        op()
        assert torch.equal(op.probabilities.view(torch.int32), reference.probabilities.view(torch.int32))
        assert torch.equal(op.sampled, reference.sampled) and torch.equal(op.valid, reference.valid)
        keys = op.proposal_selected
        expected_ids = torch.where(keys != -9223372036854775808, 2147483647-(keys & 2147483647), 0)
        expected_probs = op.probabilities[:, 0].gather(1, expected_ids)
        expected_probs = torch.where(op.valid, expected_probs, 1.)
        expected_ids = expected_ids.masked_fill(~op.valid, 0)
        assert torch.equal(op.support_ids, expected_ids)
        assert torch.equal(op.support_probabilities.view(torch.int32), expected_probs.view(torch.int32))

    root_ids = torch.zeros_like(op.support_ids)
    root_probabilities = torch.zeros_like(op.support_probabilities)
    expected = torch.zeros_like(op.probabilities)
    def receive_and_restore():
        op()
        # A delayed, divergent canonical payload arrives after local sampling has written q.
        torch.cuda._sleep(10000)
        op.support_ids.copy_(root_ids)
        op.support_probabilities.copy_(root_probabilities)
        op._restore_support()
    receive_and_restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        receive_and_restore()
    bundle = PreparedGraphBundle(device=0, graphs={'support': graph}, buffers={}, resources=(op, state, root_ids, root_probabilities))
    worker = bundle.start_worker()
    try:
        for phase, mode in enumerate(('ordinary', 'local_invalid', 'root_invalid', 'signed_zero', 'subnormal', 'ordinary')):
            logits.normal_()
            state.top_k.zero_()
            state.parameters[:, 1] = 1.
            if mode == 'local_invalid':
                logits.fill_(float('nan'))
            if shortlist:
                logits[:, :, 16:] = -torch.inf
            root_ids.zero_()
            root_ids[:, :4] = torch.tensor([0, 4, 7, 11], device='cuda')
            root_probabilities.zero_()
            root_probabilities[:, :4] = torch.tensor([.1, .2, .3, .4], device='cuda')
            if mode == 'root_invalid':
                root_ids.zero_()
                root_probabilities.fill_(1.)
            else:
                if mode == 'signed_zero':
                    root_probabilities[:, 0] = -0.
                if mode == 'subnormal':
                    root_probabilities[:, 0].view(torch.int32).fill_(1)
                root_probabilities[:, 4:] = root_probabilities[:, :1]
            expected.zero_()
            expected[:, 0].scatter_(1, root_ids, root_probabilities)
            op.probabilities.fill_(float('nan'))
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()['allocation.all.allocated']
            ticket = worker.submit(0, repetitions=3)
            completion = worker.poll(timeout_ms=30000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
            assert torch.equal(op.probabilities.view(torch.int32), expected.view(torch.int32)), mode
    finally:
        worker.close()
        graph.reset()
