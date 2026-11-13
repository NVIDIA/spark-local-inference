# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Dynamic score branches preserve exact arithmetic through compiled native replay."""

import os

import pytest
import torch

from ignition.components.sampling import DeviceRequestState, RUNNING, _compile, _scores
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from .sampling_score_reference import scores_reference


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
    reason='requires an explicitly assigned CUDA device')


class ScoreForward:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self):
        self.kernel()


def fixture(batch, vocabulary, dtype):
    state = DeviceRequestState.allocate('cuda:0', vocab_size=vocabulary, capacity=batch)
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    generator = torch.Generator(device='cuda').manual_seed(7139)
    state.histogram.copy_(torch.randint(0, 10000, state.histogram.shape,
        generator=generator, device='cuda', dtype=torch.int32))
    state.grammar_mask.random_(generator=generator)
    slots = torch.arange(batch, dtype=torch.int32, device='cuda')
    epochs = torch.full((batch,), 7, dtype=torch.int64, device='cuda')
    live = torch.tensor([batch], dtype=torch.int32, device='cuda')
    logits = torch.randn((batch, vocabulary), generator=generator, device='cuda').to(dtype)
    output = [torch.empty_like(logits, dtype=torch.float32) for _ in range(2)]
    arguments = [(logits, scores, slots, epochs, live, state.generation, state.status,
        state.histogram, state.parameters, state.grammar_mask, state.grammar_enabled,
        vocabulary, state.grammar_mask.shape[1], 1024, batch) for scores in output]
    grid = (batch, (vocabulary + 1023) // 1024, 1)
    forwards = [ScoreForward(_compile(kernel, args, grid))
        for kernel, args in zip((scores_reference, _scores), arguments, strict=True)]
    return state, slots, epochs, live, logits, output, forwards


def assert_exact(actual, expected):
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize('batch,vocabulary', [(1, 17), (8, 129280), (32, 129280)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('compiled', [False, True])
def test_neutral_and_dynamic_scores_native_replay(batch, vocabulary, dtype, compiled):
    state, slots, epochs, live, logits, outputs, forwards = fixture(batch, vocabulary, dtype)
    calls = [compile_forward(forward) if compiled else forward for forward in forwards]
    graphs = {}
    for label, call in zip(('reference', 'candidate'), calls, strict=True):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[label] = graph
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={},
        resources=(state, forwards, calls, outputs))
    worker = bundle.start_worker()
    special_bits = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x7f800000,
        0xff800000, 0x7fc00001, 0x7f800001]
    special = torch.tensor([value if value < 2**31 else value - 2**32
        for value in special_bits], dtype=torch.int32, device='cuda').view(torch.float32).to(dtype)
    try:
        for mode in ('neutral', 'temperature_only', 'tiny_temperature', 'mixed', 'nondefault',
                     'negative_zero', 'extremes', 'invalid', 'neutral'):
            state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device='cuda'))
            state.grammar_enabled.zero_()
            state.status.fill_(RUNNING)
            slots.copy_(torch.arange(batch, dtype=torch.int32, device='cuda'))
            epochs.fill_(7)
            live.fill_(batch)
            if mode in ('mixed', 'nondefault'):
                selected = torch.arange(batch, device='cuda').remainder(2) == 1
                if mode == 'nondefault':
                    selected.fill_(True)
                state.parameters[selected] = torch.tensor([.7, .95, .1, 1.3, -.2, .1], device='cuda')
                state.grammar_enabled[selected] = 1
            if mode == 'negative_zero':
                state.parameters[:, 4:6] = -0.
            if mode == 'temperature_only':
                state.parameters[:, 0] = .7
            if mode == 'tiny_temperature':
                state.parameters[:, 0] = torch.finfo(torch.float32).tiny
            if mode == 'extremes':
                state.parameters[:, 0] = torch.finfo(torch.float32).tiny
                state.parameters[:, 3] = torch.finfo(torch.float32).tiny
                state.parameters[:, 4] = torch.finfo(torch.float32).max
                state.parameters[:, 5] = -torch.finfo(torch.float32).max
            if mode == 'invalid':
                slots[0] = -1
                if batch > 1:
                    slots[1] = batch
                    epochs[-1] = 99
                    state.status[-2] = 0
            logits[:, :special.numel()].copy_(special)
            for output in outputs:
                output.fill_(float('nan'))
            before = {name:getattr(state, name).clone() for name in
                ('histogram', 'parameters', 'grammar_mask', 'grammar_enabled', 'seed', 'rng_counter')}
            torch.cuda.synchronize()
            allocation_count = torch.cuda.memory_stats()['allocation.all.allocated']
            for index in (1, 0, 1):
                ticket = worker.submit(index, repetitions=3)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocation_count
            assert_exact(outputs[1], outputs[0])
            for name, expected in before.items():
                actual = getattr(state, name)
                if actual.dtype == torch.uint64:
                    actual, expected = actual.view(torch.int64), expected.view(torch.int64)
                assert torch.equal(actual, expected), name
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()
