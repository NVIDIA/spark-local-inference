# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention outputs remain separate from scratch through compilation and replay."""

import os

import pytest
import torch
from b12x.attention import varlen
from b12x.preparation import PreparationSession

from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.preparation.compilation import compile_forward
from tests.tensor_examples import TensorExamples
from ignition.preparation.scratch import ScratchArena

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned CUDA device")


@pytest.mark.parametrize("rows", [129, 65536])
@torch.no_grad()
def test_attention_tuning_reuses_output_without_mutating_serving_inputs(rows):
    from ignition.preparation.fixtures import varlen_attention as fixtures

    device = torch.device("cuda", 0)
    q, k, v = [torch.zeros((rows, 12, 128), dtype=torch.bfloat16, device=device) for _ in range(3)]
    out = torch.full_like(q, float("nan"))
    cu = torch.zeros(2, dtype=torch.int32, device=device)
    arena = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        operation = PreparedVarlenAttention(q, k, v, cu, out, session=session,
            workspace=arena, name="attention")
        arena.finalize()
        arena.freeze()
        arguments = dict(q=q, k=k, v=v, out=out, cu_seqlens_q=cu,
                         max_seqlen_q=rows, max_seqlen_k=rows, causal=False)
        acquire = fixtures.input_factory(arguments)
        inputs = acquire()
        allocated = torch.cuda.memory_allocated()
        calls = [fixtures.benchmark_call(operation.plan.prepared.state, arguments=arguments,
                 inputs=acquire, workspace=arena) for _ in range(2)]
        assert torch.cuda.memory_allocated() == allocated
        selected = torch.tensor([0, rows // 2, rows - 1], device=device)
        scores = (inputs.seed_q[selected.cpu()].to(device).float().transpose(0, 1)
                  @ inputs.seed_k.to(device).float().permute(1, 2, 0)) * 128 ** -.5
        expected = (scores.softmax(-1) @ inputs.seed_v.to(device).float().transpose(0, 1)).transpose(0, 1)
        for call in calls:
            call.produce()
            call.invoke()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            for call in calls:
                call.produce()
                call.invoke()
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out[selected].float(), expected, rtol=.02, atol=.004)
        assert all(not value.count_nonzero() for value in (q, k, v, cu))
        graph.reset()


@pytest.mark.parametrize("width", [64, 72])
@pytest.mark.parametrize("math_mode", ["flash", "eager_bf16"])
@pytest.mark.parametrize("compiler", ["fx", "aot"])
@torch.no_grad()
def test_provided_attention_outputs_survive_scratch_reuse(width, math_mode, compiler):
    device = torch.device("cuda", 0)
    pool = TensorExamples(device)
    buffers = pool
    q, k, v, out = [buffers.zeros((32, 16, width), torch.bfloat16) for _ in range(4)]
    cu = torch.tensor([0, 16, 32], dtype=torch.int32, device=device)
    arena = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        operation = PreparedVarlenAttention(q, k, v, cu, out, session=session,
            workspace=arena, name="attention", math_mode=math_mode)
        arena.finalize()
        arena.freeze()
        binding = varlen.bind(operation.plan, q=q, k=k, v=v, cu_seqlens_q=cu,
            out=out, scratch=operation.scratch, max_seqlen_q=32, max_seqlen_k=32)
        result, lse = varlen.run(binding)
        assert result.data_ptr() == out.data_ptr()
        assert result.shape == out.shape and result.stride() == out.stride()
        assert lse is None
        for provided_lse in (None, torch.empty((16, 32), device=device)):
            torch.library.opcheck(torch.ops.b12x.varlen_attention_out.default,
                (q, k, v, cu, cu, out, provided_lse, operation.scratch, None, None,
                 None, None, 32, 32, width ** -0.5, operation.plan.handle),
                test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"))
        if compiler == "aot":
            from ignition.preparation.aot import aot_forward

            artifact = aot_forward(operation)
            forward = lambda: artifact(operation)
        else:
            forward = compile_forward(operation)
        forward()
        assert arena.nbytes <= 3 * 32 * 16 * 4 + 2048
        assert out.untyped_storage().data_ptr() != operation.scratch.untyped_storage().data_ptr()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        for middle in (7, 25, 16):
            cu[1] = middle
            for value in (q, k, v):
                value.normal_()
            inputs = [value.clone() for value in (q, k, v)]
            operation()
            expected = out.clone()
            out.fill_(float("nan"))
            arena.tensor.fill_(173)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            for actual, saved in zip((q, k, v), inputs, strict=True):
                torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        graph.reset()
