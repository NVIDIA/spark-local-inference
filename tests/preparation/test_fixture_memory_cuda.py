# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Projection trials release completed expert fixtures before allocating their inputs."""

import os
import weakref

import pytest
import torch

from ignition.components.moe import RouterScratch
from ignition.preparation.fixtures.fused_moe import ExpertBenchmarkInputs
from ignition.preparation.fixtures.projection import input_factory
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned, otherwise unused GPU")


@torch.no_grad()
def test_projection_can_follow_expert_tuning_under_memory_pressure():
    device = torch.device("cuda:0")
    source = torch.empty((4096, 7168), dtype=torch.bfloat16, device=device)
    output = torch.empty((4096, 768), dtype=torch.bfloat16, device=device)
    expert_source = torch.empty((4096, 3584), dtype=torch.bfloat16, device=device)
    with RecordedPreparationSession(device=device, checkpoint="fixture-memory",
            workload_schema="expert-then-projection", numerical_policy="bf16") as session:
        # Isolate the real fixture lifecycle without loading model weights or racing kernels.
        session._tune_status = "tuning"
        previous = weakref.ref(session.benchmark_resource("expert", lambda: ExpertBenchmarkInputs(
            expert_source, experts=896, top_k=16, allocate_router=RouterScratch.allocate)))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(device)[0]
        assert free > 512 << 20, "assign a GPU with at least 512 MiB free"
        reserve = torch.empty(free - (40 << 20), dtype=torch.uint8, device=device)
        try:
            # The 56 MiB input cannot coexist with the completed expert's cached fixture.
            with pytest.raises(torch.OutOfMemoryError):
                input_factory(source, output)()
            operands = input_factory(source, output, session=session)()
            assert previous() is None
            operands.produce()
            torch.cuda.synchronize()
        finally:
            del reserve
            torch.cuda.empty_cache()
        torch.testing.assert_close(operands.source, operands.seed.to(operands.source.device), rtol=0, atol=0)


@torch.no_grad()
def test_attention_trials_share_output_and_restore_host_seeds_without_device_temporaries():
    from b12x.preparation import PreparationSession
    from ignition.components.varlen_attention import PreparedVarlenAttention
    from ignition.preparation.fixtures import varlen_attention
    from ignition.preparation.scratch import ScratchArena

    device = torch.device("cuda:0")
    rows = 65536
    tensors = [torch.zeros(rows, 12, 128, device=device, dtype=torch.bfloat16) for _ in range(4)]
    q, k, v, output = tensors
    cu = torch.tensor([0, rows], dtype=torch.int32, device=device)
    arena = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        operation = PreparedVarlenAttention(q, k, v, cu, output, session=session,
                                           workspace=arena, name="vision-attention")
        arena.finalize()
        arena.freeze()
        arguments = dict(q=q, k=k, v=v, out=output, cu_seqlens_q=cu,
                         max_seqlen_q=rows, max_seqlen_k=rows, causal=False, softmax_scale=128**-.5)
        acquire = varlen_attention.input_factory(arguments)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(device)[0]
        assert free > 1 << 30, "assign a GPU with at least 1 GiB free"
        reserve = torch.empty(free - (900 << 20), dtype=torch.uint8, device=device)
        baseline = torch.cuda.memory_allocated()
        graph = None
        try:
            first, second = [varlen_attention.benchmark_call(operation.plan.prepared.state,
                arguments=arguments, inputs=acquire, workspace=arena) for _ in range(2)]
            fixture = first.owners[0]
            assert fixture is second.owners[0]
            assert torch.cuda.memory_allocated() - baseline < 770 << 20
            first.produce()
            first.invoke()
            expected = output.cpu()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                first.produce()
                first.invoke()
            for value in (fixture.q, fixture.k, fixture.v, output):
                value.fill_(float("nan"))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        finally:
            if graph is not None:
                graph.reset()
            del reserve
            torch.cuda.empty_cache()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        generator = torch.Generator(device=device).manual_seed(89017)
        for name, template in zip(("q", "k", "v"), (q, k, v), strict=True):
            reference = torch.empty_strided(template.shape, template.stride(), dtype=template.dtype,
                                            device=device).normal_(generator=generator)
            assert getattr(fixture, f"seed_{name}").is_pinned()
            assert torch.equal(getattr(fixture, name), reference)
        assert all(not value.count_nonzero() for value in (q, k, v))


@torch.no_grad()
def test_strided_attention_fixture_restoration_needs_no_device_temporary():
    from ignition.preparation.fixtures.varlen_attention import attention_benchmark_inputs

    source = torch.zeros(64, 24, 128, dtype=torch.bfloat16, device="cuda:0")[:, :12]
    inputs = attention_benchmark_inputs(dict(q=source, k=source, v=source,
        cu_seqlens_q=torch.zeros(2, dtype=torch.int32, device=source.device),
        max_seqlen_q=64, max_seqlen_k=64))
    inputs.produce()
    torch.cuda.synchronize()
    for value in (inputs.q, inputs.k, inputs.v):
        value.fill_(float("nan"))
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    inputs.produce()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    for name in ("q", "k", "v"):
        assert torch.equal(getattr(inputs, name).cpu(), getattr(inputs, f"seed_{name}"))
    assert not source.count_nonzero()


@pytest.mark.parametrize("rows,columns,outputs", [
    (65536, 1024, 4608), (65536, 1024, 4096), (65536, 4096, 1024),
    (16384, 4096, 7168), (4096, 35840, 448),
])
@torch.no_grad()
def test_projection_trials_fit_bounded_memory_at_encoder_and_draft_capacities(rows, columns, outputs):
    from b12x.preparation import PreparationSession
    from ignition.components.linear import UnquantizedProjection
    from ignition.preparation.fixtures.bf16_gemv import benchmark_call

    device = torch.device("cuda:0")
    source = torch.zeros(rows, columns, dtype=torch.bfloat16, device=device)
    output = torch.zeros(rows, outputs, dtype=torch.bfloat16, device=device)
    weight = torch.randn(outputs, columns, dtype=torch.bfloat16, device=device).mul_(.01)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        operation = UnquantizedProjection(source, weight, output, session=session, name="projection")
        acquire = input_factory(source, output)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(device)[0]
        assert free > 1 << 30, "assign a GPU with at least 1 GiB free"
        reserve = torch.empty(free - (900 << 20), dtype=torch.uint8, device=device)
        baseline = torch.cuda.memory_allocated()
        graph = None
        try:
            first, second = [benchmark_call(operation.plan.prepared.state,
                inputs=acquire, weight=weight, bias=None) for _ in range(2)]
            fixture = first.owners[0]
            assert fixture is second.owners[0]
            assert torch.cuda.memory_allocated() - baseline < source.nbytes + output.nbytes + (2 << 20)
            first.produce()
            first.invoke()
            expected = fixture.output.cpu()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                second.produce()
                second.invoke()
            fixture.source.fill_(float("nan"))
            fixture.output.fill_(float("nan"))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        finally:
            if graph is not None:
                graph.reset()
            del reserve
            torch.cuda.empty_cache()
        torch.testing.assert_close(fixture.output.cpu(), expected, rtol=0, atol=0)
        generator = torch.Generator(device=device).manual_seed(0xB12)
        reference = torch.randn(source.shape, device=device, dtype=source.dtype, generator=generator).mul_(.125)
        assert fixture.seed.is_pinned()
        assert torch.equal(fixture.source, reference)
        assert not source.count_nonzero() and not output.count_nonzero()
