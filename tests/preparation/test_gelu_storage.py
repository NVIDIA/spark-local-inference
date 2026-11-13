# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact ATen GELU writes into compiler-planned temporary storage."""

import os
import pickle

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from ignition.preparation import compilation
from ignition import runtime_patches


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('transposed', [False, True])
def test_exact_gelu_uses_explicit_destination(dtype, transposed):
    source = torch.linspace(-5, 5, 128, dtype=dtype).reshape(8, 16)
    if transposed:
        source = source.t()
    expected = torch.nn.functional.gelu(source)
    module = make_fx(torch.nn.functional.gelu)(source)
    calls = compilation._lower_storage(module)
    module.recompile()

    assert len(calls) == 1
    assert calls[0].target is torch.ops.aten.gelu.out
    allocations = [node for node in module.graph.nodes
                   if node.target is torch.ops.aten.empty_like.default]
    assert len(allocations) == 1
    writes = next(node for node in module.graph.nodes
                  if node.target is torch.ops.flare.storage_call.default).args[1]
    assert writes == allocations
    token = compilation._INVOCATIONS.set(calls)
    try:
        actual = module(source)
    finally:
        compilation._INVOCATIONS.reset(token)
    torch.testing.assert_close(actual, expected)
    assert actual.stride() == expected.stride()
    assert actual.untyped_storage()._cdata != source.untyped_storage()._cdata


@torch.library.custom_op('flare_test_gelu::consume', mutates_args=('destination',))
def _consume(source: torch.Tensor, destination: torch.Tensor) -> None:
    destination.copy_(source)


@_consume.register_fake
def _fake_consume(source, destination):
    pass


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires CUDA test opt-in'))])
def test_compiled_gelu_reuses_storage_after_opaque_consumer(device):
    from torch._inductor import standalone_compile
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    runtime_patches.apply()
    left = torch.randn(64, 32, dtype=torch.bfloat16, device=device)
    right = torch.randn_like(left)
    output = torch.empty((2, *left.shape), dtype=left.dtype, device=device)

    def forward(left, right, output):
        _consume(torch.nn.functional.gelu(left), output[0])
        _consume(torch.nn.functional.gelu(right), output[1])
        return (output,)

    module = make_fx(forward)(left, right, output)
    calls = compilation._lower_storage(module)
    module.recompile()
    addresses = []
    for call in calls:
        if call.target is torch.ops.aten.gelu.out:
            def record(source, *, out):
                addresses.append(out.data_ptr())
                return torch.ops.aten.gelu.out(source, out=out)
            call.target = record
    compiled = standalone_compile(module, [left, right, output],
        dynamic_shapes='from_example_inputs', aot=True,
        options={'config_patches': {'memory_planning': True, 'memory_pool': 'combined',
                                    'compile_threads': 1}})
    token = compilation._INVOCATIONS.set(calls)
    try:
        serialized = compiled.serialize()
        entry, _, _ = pickle.loads(serialized)
        assert entry.compiled_fw.result._flare_pool_sizes == ((device, left.numel() * left.element_size()),)
        compiled = AOTCompiledArtifact.deserialize(serialized)
        compiled(left, right, output)
        if device == 'cuda':
            torch.cuda.synchronize()
            addresses.clear()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compiled(left, right, output)
        assert len(addresses) == 2
        assert addresses[0] == addresses[1]
        for scale in (1., -2.):
            left.mul_(scale)
            right.mul_(scale)
            if device == 'cuda':
                graph.replay()
            else:
                compiled(left, right, output)
            expected = torch.stack((torch.nn.functional.gelu(left), torch.nn.functional.gelu(right)))
            torch.testing.assert_close(output, expected)
    finally:
        compilation._INVOCATIONS.reset(token)
