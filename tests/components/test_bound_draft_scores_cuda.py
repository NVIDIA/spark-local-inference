# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Direct draft-head/bias inputs preserve the canonical proposal and RNG outputs."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not (os.environ.get('FLARE_RUN_CUDA_TESTS') and torch.cuda.is_available()),
                                reason='set FLARE_RUN_CUDA_TESTS=1 on a CUDA host')


@pytest.mark.parametrize('batch,vocab,dtype,compiled,prefix', [
    (1, 17, torch.float32, False, 0),
    (4, 10000, torch.bfloat16, True, 3),
    (32, 129280, torch.bfloat16, True, 7),
    (64, 257, torch.float32, False, 7),
])
@pytest.mark.parametrize('bias_dtype', [None, torch.bfloat16, torch.float32])
@torch.no_grad()
def test_bound_inputs_match_canonical_native_replay(batch, vocab, dtype, compiled, prefix, bias_dtype):
    from ignition.components.draft_sampling import PreparedDraftDistribution
    from ignition.components.sampling import DeviceRequestState, RUNNING
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle

    generator = torch.Generator().manual_seed(881 + prefix)
    capacity = max(4, batch)
    state = DeviceRequestState.allocate('cuda:0', vocab_size=vocab, capacity=capacity)
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    state.histogram.copy_(torch.randint(0, 4, state.histogram.shape, generator=generator, dtype=torch.int32))
    state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device='cuda'))
    state.top_k.zero_()
    slots = torch.arange(batch, dtype=torch.int32, device='cuda')
    epochs = torch.full((batch,), 7, dtype=torch.int64, device='cuda')
    live = torch.tensor([batch], dtype=torch.int32, device='cuda')
    candidates = torch.tensor([[7, 7, 0, 3, 4, 5, 6]] * batch, device='cuda')
    head = torch.randn((batch, 8, vocab), generator=generator).to(device='cuda', dtype=dtype)
    source = head[:, 2]
    bias = (torch.randn((batch, vocab), generator=generator).to(device='cuda', dtype=bias_dtype)
            if bias_dtype is not None else None)
    head_input = head.clone()
    bias_input = None if bias is None else bias.clone()
    source_input = head_input[:, 2]
    saved_head, saved_bias = head_input.clone(), None if bias_input is None else bias_input.clone()
    seed = torch.arange(batch, dtype=torch.int64, device='cuda') + 931
    offset = torch.zeros_like(seed)
    operations, calls, graphs = [], [], {}
    for bound in (False, True):
        logits = torch.zeros((batch, 1, vocab), device='cuda')
        op = PreparedDraftDistribution(logits, state, slots, epochs, live, candidates,
            seed=seed, offset=offset, sampled=torch.zeros_like(seed), prefix_offset=prefix)
        if bound:
            op.bind_logits(source, bias)
        op.prepare()
        def run(op=op, bound=bound):
            # The producer writes the full tensor; the fused reader holds its pre-created strided view.
            head.copy_(head_input)
            if bias is not None:
                bias.copy_(bias_input)
            if not bound:
                op.logits[:, 0].copy_(source.float())
                if bias is not None:
                    op.logits[:, 0].add_(bias)
            op()
        call = compile_forward(run) if compiled else run
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        operations.append(op)
        calls.append(call)
        graphs[str(bound)] = graph
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={}, resources=(operations, calls, state, head, bias, head_input, bias_input))
    worker = bundle.start_worker()
    outputs = lambda op: (op.probabilities, op.sampled, op.valid, op.maximum,
                           op.packed_top_k, op.packed_top_p, op.proposal_candidates, op.proposal_selected)
    expected = [torch.empty_like(tensor) for tensor in outputs(operations[0])]
    modes = ('neutral', 'penalties', 'grammar', 'ties', 'signed_zero', 'tiny_temperature',
             'nonfinite', 'invalid_prefix', 'inactive', 'duplicate_or_stale', 'rng_wrap', 'neutral')
    try:
        for phase, mode in enumerate(modes):
            head_input.copy_(saved_head)
            if bias is not None:
                bias_input.copy_(saved_bias)
            state.status.fill_(RUNNING)
            state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device='cuda'))
            state.top_k.copy_(torch.tensor([(-1, 0, 1, min(20, vocab), min(64, vocab))[i % 5]
                                           for i in range(capacity)], dtype=torch.int32, device='cuda'))
            state.grammar_enabled.zero_()
            slots.copy_(torch.arange(batch, dtype=torch.int32, device='cuda'))
            epochs.fill_(7)
            live.fill_(batch)
            candidates.copy_(torch.tensor([[7, 7, 0, 3, 4, 5, 6]] * batch, device='cuda'))
            expected_offset = phase * 4 + (2**32 if mode == 'rng_wrap' else 0)
            offset.fill_(expected_offset)
            if mode == 'penalties':
                state.parameters.copy_(torch.tensor([.7, .8, .1, 1.3, -.2, -.1], device='cuda'))
            if mode == 'grammar':
                state.grammar_enabled.fill_(1)
                state.grammar_mask.copy_(torch.randint(-2**31, 2**31-1, state.grammar_mask.shape,
                    generator=generator, dtype=torch.int64).to(torch.int32))
            if mode in ('ties', 'signed_zero'):
                source_input.zero_()
                source_input[:, 1::2] = -0. if mode == 'signed_zero' else 1.
                if bias is not None:
                    bias_input.zero_()
            if mode == 'signed_zero':
                state.parameters[:, 4] = -0.
                state.parameters[:, 5] = -0.
            if mode == 'tiny_temperature':
                state.parameters[:, 0] = torch.finfo(torch.float32).tiny
            if mode == 'nonfinite':
                source_input[:, :3] = torch.tensor([float('nan'), float('inf'), -float('inf')], device='cuda')
            if mode == 'invalid_prefix':
                candidates[:, 0] = -1
            if mode == 'inactive':
                live.fill_(batch // 2)
            if mode == 'duplicate_or_stale':
                epochs[-1] = 99
                if batch > 1:
                    slots[-1] = slots[0]
            for index, op in enumerate(operations):
                op.probabilities.fill_(float('nan'))
                op.block_max.fill_(float('nan'))
                op.proposal_candidates.fill_(71)
                op.proposal_selected.fill_(73)
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()['allocation.all.allocated']
                ticket = worker.submit(index)
                completed = worker.poll(timeout_ms=30000)
                assert completed is not None and completed.ticket == ticket and completed.error is None
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
                if index == 0:
                    for destination, tensor in zip(expected, outputs(op), strict=True):
                        destination.copy_(tensor)
                else:
                    for actual, reference in zip(outputs(op), expected, strict=True):
                        if actual is op.maximum:
                            torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
                        else:
                            assert torch.equal(actual, reference), (mode, actual.shape)
            # The explicit score-inspection API remains materialized and bit exact.
            for op in operations:
                op.prepare_scores()
            valid = operations[0].valid[:, 0]
            assert torch.equal(operations[0].scores[valid].view(torch.int32),
                               operations[1].scores[valid].view(torch.int32))
            assert torch.equal(offset, torch.full_like(offset, expected_offset))
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()


@pytest.mark.parametrize('bias_dtype', [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('selector', [False, True])
@torch.no_grad()
def test_parallel_draft_preparation_preserves_bias_and_selector_contract(bias_dtype, selector):
    from types import SimpleNamespace
    from ignition.components.parallel_draft import PreparedParallelDraftRound
    from ignition.components.sampling import DeviceRequestState, RUNNING

    batch, width, vocab, block = 2, 3, 257, 4
    state = DeviceRequestState.allocate('cuda:0', vocab_size=vocab, capacity=4)
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device='cuda'))
    inputs = SimpleNamespace(row_slot_ids=torch.arange(batch, dtype=torch.int32, device='cuda'),
        row_generations=torch.full((batch,), 7, dtype=torch.int64, device='cuda'),
        live_rows=torch.tensor([batch], dtype=torch.int32, device='cuda'))
    operation = PreparedParallelDraftRound.__new__(PreparedParallelDraftRound)
    operation.owner = SimpleNamespace(requests=state, inputs=inputs, logit_offset=1, session=None,
        weights=SimpleNamespace(draft=None),
        collectives=dict(vocabulary_runtime=None, coordinator=None))
    operation.batch, operation.width, operation.name = batch, width, 'bias-preparation'
    operation.proposal_only, operation.generate_proposals, operation.argmax = True, True, None
    operation.vocabulary_parallel = False
    operation.operations, operation.buffers, operation.sampling_scratch = None, None, {}
    operation.draft_inputs = torch.zeros((batch, block), dtype=torch.int64, device='cuda')
    operation.draft = SimpleNamespace(scratch=SimpleNamespace(
        logits=torch.zeros((batch * block, vocab), dtype=torch.bfloat16, device='cuda')))
    operation.markov = None if bias_dtype is None else SimpleNamespace(
        logits=torch.zeros((batch, vocab), dtype=bias_dtype, device='cuda'))
    operation.selector = object() if selector else None
    operation.shortlisted_selector = None
    operation.logits = torch.zeros((batch, 1, vocab), device='cuda')
    operation.drafts = torch.zeros((batch, width - 1), dtype=torch.int64, device='cuda')
    operation.seed = torch.zeros(batch, dtype=torch.int64, device='cuda')
    operation.offset, operation.sampled = torch.zeros_like(operation.seed), torch.zeros_like(operation.seed)
    operation.prepare()
    for index, proposal in enumerate(operation.operations[1]):
        if selector:
            assert proposal.source_logits is None
        else:
            assert proposal.source_logits.data_ptr() == operation.draft.scratch.logits.view(batch, block, vocab)[:, index + 1].data_ptr()
            assert proposal.source_logits.stride(0) == block * vocab
            assert proposal.source_bias is (None if operation.markov is None else operation.markov.logits)
        proposal()
        assert torch.isfinite(proposal.probabilities).all()
