# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed-precision collective normalization through compiled Flare forwards."""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires explicitly assigned GPUs")


def _rank(rank, world, rendezvous):
    from ignition.components.collectives import PreparedAllReduceRMSNorm
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=10))
    try:
        with CoreStorage(rank=rank, world=world, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=36,
                hidden_size=4096, element_bytes=4, ipc=storage.peer_storage)
            try:
                with RecordedPreparationSession(device=device, checkpoint="collective-rmsnorm",
                        workload_schema="fp32-residual-bf16-output", numerical_policy="zero-centered",
                        coordinator=coordinator) as session:
                    calls = []
                    for rows in (1, 4, 16, 36):
                        source = torch.zeros((rows, 4096), device=device)
                        residual = torch.zeros_like(source)
                        weight = torch.linspace(-.5, .5, 4096, device=device)
                        output = torch.empty_like(source, dtype=torch.bfloat16)
                        for rounding in (None, torch.bfloat16):
                            operation = PreparedAllReduceRMSNorm(source, residual, weight, output,
                                eps=1e-6, runtime=runtime, session=session, coordinator=coordinator,
                                name=f"rmsnorm:{rows}:{rounding}", zero_centered=True, reduction_dtype=rounding)
                            calls.append((source, residual, weight, output, operation, rounding))
                    session.tune(show_progress=False)
                    stream = torch.cuda.Stream(device=device)
                    graphs = []
                    for source, residual, weight, output, operation, rounding in calls:
                        compiled = compile_forward(lambda: operation(source, residual))
                        compiled()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        stream.wait_stream(torch.cuda.current_stream())
                        with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                            compiled()
                        graphs.append((graph, compiled))
                    session.freeze()
                    for iteration in range(3):
                        for (source, residual, weight, output, operation, rounding), (graph, _) in zip(calls, graphs):
                            base = torch.arange(source.numel(), dtype=torch.float32).view(source.shape) / 8192
                            values = [(base + peer * .07 + iteration * .11).sin() * .25 for peer in range(world)]
                            initial = (base * .4 + iteration).cos()
                            source.copy_(values[rank])
                            residual.copy_(initial)
                            output.fill_(float("nan"))
                            reduced = torch.stack(values).sum(0)
                            if rounding is not None:
                                reduced = reduced.to(rounding).float()
                            expected_residual = initial + reduced
                            expected = (expected_residual * torch.rsqrt(
                                expected_residual.square().mean(-1, keepdim=True) + 1e-6)) * (1 + weight.cpu())
                            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                            graph.replay()
                            torch.cuda.synchronize()
                            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                            torch.testing.assert_close(residual.cpu(), expected_residual, rtol=2e-5, atol=2e-5)
                            torch.testing.assert_close(output.cpu(), expected.bfloat16(), rtol=.008, atol=.008)
                    for graph, _ in graphs:
                        graph.reset()
            finally:
                runtime.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4, 8])
def test_compiled_collective_rmsnorm(tmp_path, world):
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} assigned GPUs")
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)


def _projection_transport_rank(rank, world, rendezvous):
    from ignition.components.collectives import PreparedAllReduceRMSNorm
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    try:
        with CoreStorage(rank=rank, world=world, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=36,
                hidden_size=4096, element_bytes=4, ipc=storage.peer_storage)
            graph = torch.cuda.CUDAGraph()
            try:
                with RecordedPreparationSession(device=device, checkpoint="collective-projection-input",
                        workload_schema="bf16-transport-fp32-residual", numerical_policy="zero-centered",
                        coordinator=coordinator) as session:
                    pairs = []
                    weight = torch.linspace(-.5, .5, 4096, device=device)
                    for rows in (1, 4, 7, 36):
                        source = torch.zeros((rows, 4096), device=device, dtype=torch.bfloat16)
                        widened = torch.zeros_like(source, dtype=torch.float32)
                        residuals = tuple(torch.zeros_like(widened) for _ in range(2))
                        outputs = tuple(torch.empty_like(source) for _ in range(2))
                        operations = tuple(PreparedAllReduceRMSNorm(value, residual, weight, output,
                            eps=1e-6, runtime=runtime, session=session, coordinator=coordinator,
                            name=f"projection:{rows}:{value.dtype}", zero_centered=True,
                            reduction_dtype=torch.bfloat16)
                            for value, residual, output in zip((source, widened), residuals, outputs))
                        pairs.append((source, widened, residuals, outputs, operations))
                    session.tune(show_progress=False)

                    def forward():
                        for source, widened, residuals, _, operations in pairs:
                            widened.copy_(source)
                            operations[0](source, residuals[0])
                            operations[1](widened, residuals[1])

                    compiled = compile_forward(forward)
                    compiled()
                    torch.cuda.synchronize()
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream())
                    with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                        compiled()
                    session.freeze()
                    failures = []
                    for iteration in range(3):
                        torch.manual_seed(511 + 17 * rank + iteration)
                        for source, _, residuals, outputs, _ in pairs:
                            source.copy_(torch.randint(-256, 257, source.shape, device=device).float() / 512)
                            initial = torch.randn_like(residuals[0]) * .3
                            for residual in residuals:
                                residual.copy_(initial)
                            for output in outputs:
                                output.fill_(float("nan"))
                        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                        graph.replay()
                        torch.cuda.synchronize()
                        try:
                            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                            for _, _, residuals, outputs, _ in pairs:
                                torch.testing.assert_close(*residuals, rtol=0, atol=0)
                                torch.testing.assert_close(*outputs, rtol=.008, atol=.008)
                        except AssertionError as exc:
                            failures.append(str(exc))
                    assert not failures, "\n".join(failures)
            finally:
                graph.reset()
                runtime.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4, 8])
def test_bf16_projection_transport_preserves_fp32_residuals(tmp_path, world):
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} assigned GPUs")
    mp.spawn(_projection_transport_rank, args=(world, (tmp_path / "transport").as_uri()),
             nprocs=world, join=True)
