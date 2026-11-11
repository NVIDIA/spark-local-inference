# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual GLM MLP weights in compiled TP graphs driven by native workers."""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four explicitly assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1")


def _rank(rank, world_size, rendezvous, snapshot, rows):
    from b12x.comm.pcie import AllReduce

    from ignition.components.collectives import PreparedAllReduce
    from ignition.components.mlp import PreparedMLP
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.weights import MLPWeights
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    shard = TensorParallelShard(rank, world_size)
    manifest = inspect_indexed_checkpoint(snapshot)
    prefixes = (f"{TEXT_PREFIX}.layers.0.mlp.", f"{TEXT_PREFIX}.layers.3.mlp.shared_experts.")
    with GlmWeightLoader(manifest, device, shard=shard, prefixes=prefixes) as loader:
        dense_weights = loader.load_mlp(0)
        shared_prefix = prefixes[1].rstrip(".")
        extent = shard.partition(2048, alignment=128)
        shared_weights = MLPWeights(
            loader.load_linear_group(tuple(f"{shared_prefix}.{name}" for name in ("gate_proj", "up_proj")),
                                     row_ranges=(extent, extent)),
            loader.load_linear_group((shared_prefix + ".down_proj",), column_range=extent),
        )
        assert loader._session.torch_copy_bytes == 0
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=world_size, timeout=timedelta(minutes=2))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    runtime = AllReduce.from_exchange_group(exchange_group=group, device=device, single_channel=True)
    graph = torch.cuda.CUDAGraph()
    worker = None
    try:
        source = torch.randn((rows, 4096), generator=torch.Generator().manual_seed(53), dtype=torch.bfloat16).to(device)
        seed = source.clone()
        partial_output = torch.empty_like(source)
        dense_output, output = torch.empty_like(source), torch.empty_like(source)
        gate_up = torch.empty((rows, 2 * 12288 // world_size), device=device, dtype=torch.bfloat16)
        intermediate = torch.empty((rows, 12288 // world_size), device=device, dtype=torch.bfloat16)
        counter = torch.zeros(1, device=device, dtype=torch.int32)
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
                                        workload_schema="glm-mlp-tp-v1", numerical_policy="default",
                                        coordinator=coordinator) as session:
            session.register_scratch(arena)
            dense = PreparedMLP(source, dense_weights, gate_up, intermediate, partial_output,
                                session=session, name="dense", workspace=arena, swiglu_limit=10.0)
            dense_reduce = PreparedAllReduce(partial_output, dense_output, runtime=runtime, session=session,
                                             coordinator=coordinator, name="dense.reduce")
            shared_gate_up = torch.empty((rows, 2 * 2048 // world_size), device=device, dtype=torch.bfloat16)
            shared_intermediate = torch.empty((rows, 2048 // world_size), device=device, dtype=torch.bfloat16)
            shared = PreparedMLP(dense_output, shared_weights, shared_gate_up, shared_intermediate, partial_output,
                                 session=session, name="shared", workspace=arena, swiglu_limit=10.0)
            shared_reduce = PreparedAllReduce(partial_output, output, runtime=runtime, session=session,
                                              coordinator=coordinator, name="shared.reduce")
            for operation in (dense, shared, dense_reduce, shared_reduce):
                operation.functional(session=session)
            session.tune()
            arena.release_preparation()

            def forward():
                dense_output.copy_(dense_reduce(dense(source)))
                output.copy_(shared_reduce(shared(dense_output)))
                counter.add_(1)

            dist.barrier(group=group)
            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                compiled()
            torch.cuda.synchronize(device)
            session.freeze()
            bundle = PreparedGraphBundle(device=rank, graphs={"mlp": graph},
                buffers={"source": source, "output": output, "counter": counter},
                resources=(compiled, dense, shared, dense_reduce, shared_reduce, runtime, session, arena, stream))
            worker = bundle.start_worker()
            for repetitions, factor in ((8, 1.), (17, -.5)):
                source.copy_(seed * factor)
                forward()
                expected = output.clone()
                assert bool(torch.isfinite(expected).all())
                counter.zero_()
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                dist.barrier(group=group)
                ticket = worker.submit(0, repetitions=repetitions)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                assert counter.item() == repetitions
                gathered = [None] * world_size
                dist.all_gather_object(gathered, output.cpu(), group=group)
                for other in gathered:
                    torch.testing.assert_close(other, expected.cpu(), rtol=0, atol=0)
            worker.close()
            worker = None
            graph.reset()
    finally:
        if worker is not None:
            worker.close()
        graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("rows", [1, 8])
def test_native_compiled_glm_mlps(world_size, rows, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() == 4
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    mp.spawn(_rank, args=(world_size, (tmp_path / "rendezvous").as_uri(), str(snapshot), rows),
             nprocs=world_size, join=True)
