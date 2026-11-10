# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lagged coefficient propagation through prepared mHC and CUDA replay."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession
from b12x.testing.mhc import make_inputs, pre_reference, post_reference

from ignition.components.mhc import MHCWeights, MHCScratch, PreparedMHC, PreparedMHCPost
from ignition.preparation.compilation import compile_forward
from tests.tensor_examples import TensorExamples

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("operation", ["pre", "post_pre"])
@torch.no_grad()
def test_prepared_lagged_coefficients_and_rounded_normalization(tokens, operation):
    device = torch.device("cuda:0")
    residual, branch, fn, scale, bias = make_inputs(tokens=tokens, hidden_size=5120, seed=4102, device=device)
    norm = torch.linspace(.5, 1.5, 5120, device=device).bfloat16()
    incoming = torch.full((tokens, 4), .25, device=device)
    predicted = torch.empty_like(incoming)
    previous_post = torch.ones_like(incoming)
    previous_comb = torch.eye(4, device=device).expand(tokens, -1, -1).contiguous()
    buffers = TensorExamples(device)
    def declaration(value):
        return buffers.zeros(value.shape, value.dtype).copy_(value)
    residual, branch, incoming, predicted, previous_post, previous_comb = map(
        declaration, (residual, branch, incoming, predicted, previous_post, previous_comb))
    args = (residual,) if operation == "pre" else (branch, residual, previous_post, previous_comb)
    scratch = MHCScratch.allocate(tokens, 5120, device)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        prepared = PreparedMHC(args, MHCWeights(fn, scale, bias, norm), scratch,
            session=session, name="lagged", workspace=None, rms_eps=1e-20, norm_eps=1e-20,
            hc_eps=1e-6, sinkhorn_iters=20, pre_mix=incoming, pre_out=predicted).functional()
        def run():
            result = prepared(*args, pre_mix=incoming, pre_out=predicted)
            for destination, value in zip(scratch.bind_args().values(), result, strict=True):
                destination.copy_(value)
        forward = torch.compile(run, fullgraph=True)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        for basis in (0, 3):
            incoming.zero_()
            incoming[:, basis] = 1
            predicted.fill_(float("nan"))
            graph.replay()
            current = residual if operation == "pre" else post_reference(branch, residual, previous_post, previous_comb)
            flat = current.flatten(1).float()
            projected = torch.nn.functional.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-20)
            expected_pre = torch.sigmoid(projected[:, :4] * scale[0] + bias[:4]) + 1e-6
            _, expected_post, expected_comb = pre_reference(current, fn, scale, bias,
                rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20)
            collapsed = (current.float() * incoming[:, :, None]).sum(1).bfloat16().float()
            normalized = (collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-20) * norm.float()).bfloat16()
            torch.testing.assert_close(scratch.normalized, normalized, rtol=2e-5, atol=.008)
            torch.testing.assert_close(predicted, expected_pre, rtol=2e-5, atol=4e-5)
            torch.testing.assert_close(scratch.post, expected_post, rtol=2e-5, atol=4e-5)
            torch.testing.assert_close(scratch.comb, expected_comb, rtol=2e-5, atol=4e-5)
        graph.reset()


@pytest.mark.parametrize("operation", ["pre", "post_pre"])
@torch.no_grad()
def test_independent_prefill_declarations_replay_exactly(operation):
    device, tokens = torch.device("cuda:0"), 4096
    residual, branch, fn, scale, bias = make_inputs(tokens=tokens, hidden_size=5120, seed=4102, device=device)
    weights = MHCWeights(fn, scale, bias, torch.linspace(.5, 1.5, 5120, device=device).bfloat16())
    previous_post = torch.ones((tokens, 4), device=device)
    previous_comb = torch.eye(4, device=device).expand(tokens, -1, -1).contiguous()
    sources = (residual,) if operation == "pre" else (branch, residual, previous_post, previous_comb)
    graphs, operands, results, owners = [], [], [], []
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        for allocated_examples in (False, True):
            buffers = TensorExamples(device) if allocated_examples else None
            def allocate(shape, dtype):
                return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                        else buffers.zeros(shape, dtype))
            args = tuple(allocate(value.shape, value.dtype).copy_(value) for value in sources)
            incoming, predicted = (allocate((tokens, 4), torch.float32) for _ in range(2))
            scratch = MHCScratch.allocate(tokens, 5120, device)
            prepared = PreparedMHC(args, weights, scratch, session=session, name=f"lagged{allocated_examples}",
                workspace=None, rms_eps=1e-20, norm_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
                pre_mix=incoming, pre_out=predicted).functional()
            def run(prepared=prepared, args=args, incoming=incoming, predicted=predicted, scratch=scratch):
                result = prepared(*args, pre_mix=incoming, pre_out=predicted)
                for destination, value in zip(scratch.bind_args().values(), result, strict=True):
                    destination.copy_(value)
            forward = compile_forward(run)
            forward()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                forward()
            graphs.append(graph)
            operands.append(incoming)
            results.append((*scratch.bind_args().values(), predicted))
            owners.append((buffers, args, prepared, forward))
        for basis in (0, 3, 1):
            for incoming, outputs in zip(operands, results, strict=True):
                incoming.zero_()
                incoming[:, basis] = 1
                for output in outputs:
                    output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            for independent, declared in zip(*results, strict=True):
                assert torch.isfinite(declared).all()
                torch.testing.assert_close(declared, independent, rtol=0, atol=0)
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("tokens", [1, 4096])
@torch.no_grad()
def test_post_uses_prepared_candidate_and_local_output(tokens):
    device = torch.device("cuda:0")
    residual, branch, *_ = make_inputs(tokens=tokens, hidden_size=5120, seed=4102, device=device)
    post = torch.rand((tokens, 4), device=device)
    comb = torch.rand((tokens, 4, 4), device=device)
    output = torch.empty_like(residual)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        prepared = PreparedMHCPost(branch, residual, post, comb, output,
                                   session=session, name="post").functional()
        forward = compile_forward(lambda: output.copy_(prepared(branch, residual, post, comb)))
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        try:
            for gain in (1., -.5):
                branch.mul_(gain)
                output.fill_(float("nan"))
                expected = post_reference(branch, residual, post, comb)
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for field in ("allocation.all.allocated", "allocation.all.freed",
                              "segment.all.allocated", "segment.all.freed"):
                    assert before[field] == after[field]
                torch.testing.assert_close(output, expected, rtol=2e-5, atol=.008)
        finally:
            graph.reset()
