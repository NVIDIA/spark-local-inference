# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Row-strided normalization preserves staged arithmetic and native mutations."""
import os

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.rms_norm import PreparedRMSNorm
from ignition.preparation.graphs import PreparedGraphBundle

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
    reason='requires an explicitly assigned CUDA device')


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class StagedNormalization:
    """Reference computation with explicit contiguous input materialization."""
    def __init__(self, source, weight, output, *, session, name, residual=None, zero_centered=False):
        self.staging = torch.empty_like(source, memory_format=torch.contiguous_format)
        self.staging.copy_(source)
        self.residual = residual is not None
        self.operation = PreparedRMSNorm(self.staging, weight, output, eps=1e-6,
            session=session, name=name, residual=residual, zero_centered=zero_centered)

    def __call__(self, source, residual=None):
        self.staging.copy_(source)
        result = self.operation(self.staging, residual)
        if self.residual:
            source.copy_(self.staging)
        return result


class NormPair:
    def __init__(self, parent, weights, outputs, session, *, staged, name):
        self.parent = parent
        cls = StagedNormalization if staged else PreparedRMSNorm
        self.operations = []
        for index, source in enumerate((parent[:, :1280], parent[:, 1280:1792])):
            kwargs = dict(session=session, name=f'{name}.{index}')
            if not staged:
                kwargs['eps'] = 1e-6
            self.operations.append(cls(source, weights[index], outputs[index], **kwargs))

    def __call__(self):
        return (self.operations[0](self.parent[:, :1280]),
                self.operations[1](self.parent[:, 1280:1792]))


def inputs(tokens, source_dtype, weight_dtype):
    parent = torch.randn(tokens, 1808, device='cuda:0', dtype=source_dtype)
    parent[:, 1792:].fill_(float('nan'))
    weights = [torch.randn(w, device='cuda:0', dtype=weight_dtype) for w in (1280, 512)]
    outputs = lambda: [torch.empty(tokens, w, device='cuda:0', dtype=torch.bfloat16) for w in (1280, 512)]
    return parent, weights, outputs


@pytest.mark.parametrize('tokens', [1, 8, 24, 128, 4096])
@pytest.mark.parametrize('source_dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('weight_dtype', [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_strided_pair_exact_staged_arithmetic(tokens, source_dtype, weight_dtype):
    parent, weights, outputs = inputs(tokens, source_dtype, weight_dtype)
    actual, expected = outputs(), outputs()
    with PreparationSession(device='cuda:0', autotune=False, compile_workers=2) as session:
        candidate = NormPair(parent, weights, actual, session, staged=False, name='strided')
        reference = NormPair(parent, weights, expected, session, staged=True, name='staged')
        candidate(); reference()
        for a, e in zip(actual, expected, strict=True):
            assert torch.equal(a, e)
        calls = [torch.compile(x, fullgraph=True) for x in (candidate, reference)]
        for call in calls:
            call()
        graphs = []
        for call in calls:
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                call()
            graphs.append(graph)
        for gain in (1., -.5, 2.):
            parent[:, :1792].mul_(gain)
            parent[:, 1792:].fill_(float('nan'))
            source_before = parent[:, :1792].clone()
            for out in actual + expected:
                out.fill_(float('nan'))
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            for a, e in zip(actual, expected, strict=True):
                assert torch.equal(a, e)
            assert torch.equal(parent[:, :1792], source_before)
            assert parent[:, 1792:].isnan().all()
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize('source_dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('residual_dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('zero_centered', [False, True])
@torch.no_grad()
def test_strided_residual_exact_mutation(source_dtype, residual_dtype, zero_centered):
    parent = torch.randn(24, 1344, device='cuda:0', dtype=source_dtype)
    parent[:, 1280:].fill_(float('nan'))
    seed = parent[:, :1280].clone()
    reference_parent = parent.clone()
    source, reference_source = parent[:, :1280], reference_parent[:, :1280]
    residual = torch.randn(24, 1344, device='cuda:0', dtype=residual_dtype)[:, :1280]
    residual_before = residual.clone()
    weight = torch.randn(1280, device='cuda:0')
    actual, expected = [torch.empty_like(source, dtype=torch.bfloat16) for _ in range(2)]
    with PreparationSession(device='cuda:0', autotune=False, compile_workers=2) as session:
        candidate = PreparedRMSNorm(source, weight, actual, eps=1e-6, session=session,
            name='residual.strided', residual=residual, zero_centered=zero_centered)
        reference = StagedNormalization(reference_source, weight, expected, session=session,
            name='residual.staged', residual=residual, zero_centered=zero_centered)
        calls = [lambda: candidate(source, residual), lambda: reference(reference_source, residual)]
        for compiled in (False, True):
            source.copy_(seed)
            reference_source.copy_(seed)
            a, b = [torch.compile(x, fullgraph=True) for x in calls] if compiled else calls
            a(); b()
            torch.cuda.synchronize()
            assert torch.equal(actual, expected)
            assert torch.equal(source, reference_source)
            assert torch.equal(residual, residual_before)
            assert parent[:, 1280:].isnan().all()


@torch.no_grad()
def test_aot_native_strided_pair(tmp_path):
    from ignition.preparation.compile_pool import CompilePool
    device = torch.device('cuda:0')
    owners, graphs, results = [], {}, []
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        for tokens in (8, 24, 128):
            parent, weights, outputs = inputs(tokens, torch.bfloat16, torch.float32)
            actual, expected = outputs(), outputs()
            candidate = NormPair(parent, weights, actual, session, staged=False, name=f'native.{tokens}')
            reference = NormPair(parent, weights, expected, session, staged=True, name=f'oracle.{tokens}')
            path = tmp_path / f'norm-{tokens}.bin'
            with CompilePool(device) as pool:
                pool.submit(candidate, path).result()
            with path.open('rb') as file:
                compiled = torch.compiler.load_compiled_function(file)
            compiled(candidate)
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled(candidate)
            graphs[str(tokens)] = graph
            owners.append((candidate, compiled, reference, parent, actual, expected))
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={}, resources=(owners,))
        worker = bundle.start_worker()
        try:
            for index in (0, 1, 2, 1, 0, 2):
                candidate, compiled, reference, parent, actual, expected = owners[index]
                parent[:, :1792].mul_(-.75)
                parent[:, 1792:].fill_(float('nan'))
                reference()
                for out in actual:
                    out.fill_(float('nan'))
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()['allocation.all.allocated']
                ticket = worker.submit(index)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
                for a, e in zip(actual, expected, strict=True):
                    assert torch.equal(a, e)
                assert parent[:, 1792:].isnan().all()
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()


@pytest.mark.parametrize('residual_alias', [False, True])
@torch.no_grad()
def test_strided_alias_preserves_staged_behavior(residual_alias):
    seed = torch.randn(8, 1792, device='cuda:0', dtype=torch.bfloat16)
    parents = [seed.clone(), seed.clone()]
    sources = [p[:, :1280] for p in parents]
    outputs = ([torch.empty(8, 1280, device='cuda:0', dtype=torch.bfloat16) for _ in parents]
               if residual_alias else [p.view(-1)[:8 * 1280].view(8, 1280) for p in parents])
    weight = torch.randn(1280, device='cuda:0')
    with PreparationSession(device='cuda:0', autotune=False, compile_workers=2) as session:
        candidate = PreparedRMSNorm(sources[0], weight, outputs[0], eps=1e-6, session=session,
            name='alias.strided', residual=sources[0] if residual_alias else None)
        reference = StagedNormalization(sources[1], weight, outputs[1], session=session,
            name='alias.staged', residual=sources[1] if residual_alias else None)
        calls = [lambda: candidate(sources[0], sources[0] if residual_alias else None),
                 lambda: reference(sources[1], sources[1] if residual_alias else None)]
        for compiled in (False, True):
            for parent in parents:
                parent.copy_(seed)
            run = [torch.compile(call, fullgraph=True) for call in calls] if compiled else calls
            for call in run:
                call()
            torch.cuda.synchronize()
            assert torch.equal(outputs[0], outputs[1])
            assert torch.equal(parents[0], parents[1])


@torch.no_grad()
def test_disjoint_pool_views_do_not_require_staging():
    storage = torch.randn(8 * (1792 + 1280), device='cuda:0', dtype=torch.bfloat16)
    source = storage[:8 * 1792].view(8, 1792)[:, :1280]
    output = storage[8 * 1792:].view(8, 1280)
    weight = torch.ones(1280, device='cuda:0')
    with PreparationSession(device='cuda:0', autotune=False, compile_workers=2) as session:
        operation = PreparedRMSNorm(source, weight, output, eps=1e-6, session=session, name='pool')
        assert operation.staging is None
        operation(source)
        assert torch.isfinite(output).all()


class ProjectedNormForward:
    def __init__(self, source, projected, projection, candidate, reference):
        self.source, self.projected, self.projection = source, projected, projection
        self.candidate, self.reference = candidate, reference

    def __call__(self):
        self.projection(self.source)
        self.candidate()
        self.reference()
        # The projection storage is reusable after both normalization consumers.
        self.projected.fill_(float('nan'))


@pytest.mark.parametrize('tokens', [24, 4096])
@torch.no_grad()
def test_projection_to_strided_norm_native_order(tokens, tmp_path):
    from b12x.gemm.blockscaled import pack_weight
    from ignition.components.linear import QuantizedProjection
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device('cuda:0')
    source = torch.randn(tokens, 5120, device=device, dtype=torch.bfloat16) * .125
    projected = torch.empty(tokens, 1792, device=device, dtype=torch.bfloat16)
    packed = pack_weight((torch.randn(1792, 5120, device=device) * .125).to(torch.float8_e4m3fn),
        torch.full((1792, 160), 127, device=device, dtype=torch.uint8), recipe='mxfp8')
    weights = [torch.randn(w, device=device, dtype=torch.bfloat16) for w in (1280, 512)]
    actual, expected = [[torch.empty(tokens, w, device=device, dtype=torch.bfloat16)
                         for w in (1280, 512)] for _ in range(2)]
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint='projected-norm-order',
            workload_schema='projected-norm-order', numerical_policy='default') as session:
        session.register_scratch(arena)
        projection = QuantizedProjection(source, packed, projected, session=session,
            name='qkv_a', workspace=arena, activation_mode='auto')
        candidate = NormPair(projected, weights, actual, session, staged=False, name='direct')
        reference = NormPair(projected, weights, expected, session, staged=True, name='staged')
        session.tune()
        forward = ProjectedNormForward(source, projected, projection, candidate, reference)
        forward()
        for a, e in zip(actual, expected, strict=True):
            assert torch.equal(a, e) and torch.isfinite(a).all()
        path = tmp_path / 'projected-norm.bin'
        with CompilePool(device) as pool:
            pool.submit(forward, path).result()
        with path.open('rb') as file:
            compiled = torch.compiler.load_compiled_function(file)
        compiled(forward)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(forward)
        bundle = PreparedGraphBundle(device=0, graphs={'norm': graph}, buffers={},
            resources=(forward, compiled, arena, packed, weights))
        worker = bundle.start_worker()
        try:
            for factor in (1., -.5, 2.):
                source.mul_(factor)
                projected.fill_(float('nan'))
                for output in actual + expected:
                    output.fill_(float('nan'))
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()['allocation.all.allocated']
                ticket = worker.submit(0)
                result = worker.poll(timeout_ms=30000)
                assert result is not None and result.ticket == ticket and result.error is None
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
                for a, e in zip(actual, expected, strict=True):
                    assert torch.equal(a, e) and torch.isfinite(a).all()
                assert projected.isnan().all()
        finally:
            worker.close()
            graph.reset()
