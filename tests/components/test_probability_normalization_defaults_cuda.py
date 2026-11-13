# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Normalization preserves exact values and validity with aliased native replay."""

import io
import os

import pytest
import torch

from ignition.components.probability_normalization import PreparedProbabilityNormalization
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from .probability_normalization_reference import UnconditionalProbabilityNormalization


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
    reason='requires an explicitly assigned CUDA device')


class RefilledNormalization:
    def __init__(self, operation, template, incoming_valid, probabilities, params):
        self.operation, self.template, self.incoming_valid = operation, template, incoming_valid
        self.probabilities, self.params = probabilities, params

    def __call__(self):
        self.probabilities.copy_(self.template)
        self.operation.valid.copy_(self.incoming_valid)
        return self.operation(self.probabilities, self.params)


def inputs(vocabulary):
    generator = torch.Generator(device='cuda').manual_seed(7193)
    template = torch.rand((8, 4, vocabulary), generator=generator, device='cuda')
    flat = template.flatten(0, 1)
    flat[2, vocabulary // 2] = float('nan')
    flat[3, vocabulary // 2] = float('inf')
    flat[4, vocabulary // 2] = -float('inf')
    flat[5].zero_()
    flat[5, 0] = -0.
    flat[6].fill_(-1.)
    flat[7, :4] = torch.tensor([-1., -0., 0., 1.], device='cuda')
    flat[8].fill_(1e-40)
    flat[9].fill_(-float('inf'))
    flat[10].fill_(float('nan'))
    flat[11, 0] = torch.finfo(torch.float32).max
    flat[12, 0] = torch.tensor([0x7f800001], dtype=torch.int32, device='cuda').view(torch.float32)[0]
    flat[13].fill_(1.)
    flat[14, :4] = torch.tensor([-float('inf'), -0., 0., .25], device='cuda')
    flat[15].fill_(.5)
    flat[16].fill_(float('nan'))
    flat[17].fill_(float('inf'))
    flat[18].fill_(-0.)
    flat[19].zero_()
    flat[20, :2] = torch.finfo(torch.float32).max
    flat[21].fill_(-1.)
    flat[21, 0] = float('nan')
    flat[22].fill_(-float('inf'))
    flat[22, 0] = float('nan')
    flat[23].fill_(float('nan'))
    flat[23, 0] = .5
    flat[24, :4] = torch.tensor([-1e-40, -0., 0., 1e-40], device='cuda')
    flat[25].fill_(.25)
    flat[26].fill_(float('inf'))
    flat[27, :2] = torch.tensor([float('nan'), float('inf')], device='cuda')
    flat[28].fill_(.1)
    flat[29].zero_()
    flat[29, 0] = 1e-40
    flat[30].fill_(float('nan'))
    flat[30, -1] = float('inf')
    flat[31, :4] = torch.tensor([-1., -0., 0., .5], device='cuda')
    incoming = torch.ones((8, 4), dtype=torch.bool, device='cuda')
    incoming.flatten()[1] = False
    incoming.flatten()[10] = False
    incoming.flatten()[26] = False
    params = torch.zeros((8, 12), device='cuda')[:, :6]
    params[:, 0] = 1.
    return template, incoming, params


def compiled_call(owner, mode):
    if mode == 'compiled':
        return compile_forward(owner)
    if mode == 'aot':
        from ignition.preparation.aot import aot_forward
        artifact = aot_forward(owner)
        artifact(owner)
        restored = torch.compiler.load_compiled_function(
            io.BytesIO(type(artifact).serialize(artifact).serialized_data))
        return lambda: restored(owner)
    return owner


@pytest.mark.parametrize('vocabulary', [17, 65537, 129280])
@pytest.mark.parametrize('alias', ['separate', 'same_tensor', 'same_storage_view'])
@pytest.mark.parametrize('mode', ['captured', 'compiled', 'aot'])
def test_dynamic_normalization_matches_unconditional_native_replay(vocabulary, alias, mode):
    template, incoming, params = inputs(vocabulary)
    outputs = [torch.empty_like(template), torch.empty_like(template)]
    valid = [torch.empty_like(incoming), torch.empty_like(incoming)]
    reference_input = torch.empty_like(template)
    candidate_input = (torch.empty_like(template) if alias == 'separate'
        else outputs[1] if alias == 'same_tensor' else outputs[1].view_as(outputs[1]))
    operations = [UnconditionalProbabilityNormalization(outputs[0], valid[0]),
                  PreparedProbabilityNormalization(outputs[1], valid[1])]
    owners = [RefilledNormalization(operation, template, incoming, probabilities, params)
        for operation, probabilities in zip(operations, (reference_input, candidate_input), strict=True)]
    calls = [compiled_call(owner, mode) for owner in owners]
    graphs = {}
    for label, call in zip(('reference', 'candidate'), calls, strict=True):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[label] = graph
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={},
        resources=(operations, owners, calls, outputs, valid, template, incoming, params))
    worker = bundle.start_worker()
    try:
        template_before, incoming_before = template.clone(), incoming.clone()
        for minimum in ((0., .05, 0., float('nan')), (-0., 1.1, .5, 0.),
                        (0., 0., 0., 0.), (.01, .9, 1., -.5), (0., 0., 0., 0.)):
            params[:, 2].copy_(torch.tensor(minimum * 2, device='cuda'))
            parameter_bits = params.view(torch.int32).clone()
            for operation in operations:
                for scratch in (operation.maxima, operation.masses, operation.totals, operation.output):
                    scratch.fill_(float('nan'))
                operation.valid.zero_()
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            rng_before = torch.cuda.get_rng_state()
            for index in (1, 0, 1):
                ticket = worker.submit(index, repetitions=3)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
            after = torch.cuda.memory_stats()
            for name in ('allocation.all.allocated', 'allocation.all.freed',
                         'segment.all.allocated', 'segment.all.freed'):
                assert before[name] == after[name]
            assert torch.equal(outputs[1].view(torch.int32), outputs[0].view(torch.int32))
            assert torch.equal(valid[1], valid[0])
            assert torch.equal(torch.cuda.get_rng_state(), rng_before)
            assert torch.equal(template.view(torch.int32), template_before.view(torch.int32))
            assert torch.equal(incoming, incoming_before)
            assert torch.equal(params.view(torch.int32), parameter_bits)
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()
