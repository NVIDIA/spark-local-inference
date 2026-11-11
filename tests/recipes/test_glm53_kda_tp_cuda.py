# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head-sharded GLM KDA graphs with b12x reductions and native rank workers."""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four explicitly assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1")


def _rank(rank, world, rendezvous, snapshot, mode):
    from b12x.comm.pcie import AllReduce

    from ignition.components.collectives import PreparedAllReduce
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.recurrent import DeltaMetadata, DeltaState
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.kda import KDAScratch, PreparedGlmKDA
    from ignition.recipes.glm53.model import create_model_tree
    from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    model = create_model_tree(Path(snapshot))
    manifest = inspect_indexed_checkpoint(snapshot)
    with GlmWeightLoader(manifest, device, shard=TensorParallelShard(rank, world),
                         prefixes=(f"{TEXT_PREFIX}.layers.0.self_attn.",)) as loader:
        weights = loader.load_attention(0)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=2))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    runtime = AllReduce.from_exchange_group(exchange_group=group, device=device, single_channel=True)
    worker = None
    graph = torch.cuda.CUDAGraph()
    try:
        capacity = 65 if mode == "prefill" else 2
        count = capacity if mode == "prefill" else 1
        heads = 64 // world
        state = DeltaState.allocate(6, device, heads=heads, channels=3 * heads * 128)
        metadata = DeltaMetadata.allocate(2, device)
        scratch = KDAScratch.allocate(capacity, heads, device)
        source = torch.zeros((capacity, 4096), dtype=torch.bfloat16, device=device)
        output = torch.empty_like(source)
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
                                        workload_schema="glm-kda-tp-v1", numerical_policy="default",
                                        coordinator=coordinator) as session:
            session.register_scratch(arena)
            operator = PreparedGlmKDA(model.model.language_model.layers[0].self_attn,
                weights, source, scratch, state, metadata, mode=mode, session=session,
                name=f"kda.{mode}", workspace=arena, checkpoint_export=mode == "prefill")
            reduction = PreparedAllReduce(scratch.output, output, runtime=runtime, session=session,
                                           coordinator=coordinator, name="kda.reduce")
            operator.functional(session=session)
            reduction.functional(session=session)
            session.tune()
            arena.release_preparation()

            def forward():
                output.copy_(reduction(operator(source, metadata)))

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                compiled()
            torch.cuda.synchronize(device)
            session.freeze()
            bundle = PreparedGraphBundle(device=rank, graphs={mode: graph},
                buffers={"source": source, "output": output, "state": state.recurrent, "history": state.convolution},
                resources=(compiled, operator, reduction, runtime, session, arena, stream, metadata))
            worker = bundle.start_worker()
            generator = torch.Generator().manual_seed(53)
            source[:count].copy_((torch.randn((count, 4096), generator=generator) * .1).to(torch.bfloat16))
            metadata.query_start_loc.copy_(torch.tensor([0, count, count], dtype=torch.int32))
            metadata.convolution_slots.copy_(torch.tensor([1, -1], dtype=torch.int64))
            metadata.state_indices[0, 0] = 1
            metadata.initial_state_indices[0] = 1
            metadata.final_state_indices[0] = 1
            metadata.checkpoint_state_indices[0] = 2 if count >= 16 else 0
            metadata.checkpoint_offsets[0] = 16 if count >= 16 else 0
            metadata.num_seqs.fill_(1)
            metadata.num_tokens.fill_(count)
            for repetitions in (1, 8):
                initial = (state.recurrent.clone(), state.convolution.clone())
                for _ in range(repetitions):
                    forward()
                expected = (output.clone(), state.recurrent.clone(), state.convolution.clone())
                state.recurrent.copy_(initial[0])
                state.convolution.copy_(initial[1])
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                dist.barrier(group=group)
                ticket = worker.submit(0, repetitions=repetitions)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                for actual, reference in zip((output, state.recurrent, state.convolution), expected, strict=True):
                    assert bool(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                assert not state.recurrent[0].count_nonzero()
                gathered = [None] * world
                dist.all_gather_object(gathered, output.cpu(), group=group)
                for other in gathered:
                    torch.testing.assert_close(other, expected[0].cpu(), rtol=0, atol=0)
            worker.close()
            worker = None
            graph.reset()
    finally:
        if worker is not None:
            worker.close()
        graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_actual_kda_native_tp_graph(world, mode, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() == 4
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri(), str(snapshot), mode),
             nprocs=world, join=True)
