# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sharded vocabulary projection and column gathering through native graph replay."""

import os
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires explicitly assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1")


@torch.no_grad()
def _rank(rank, world, rendezvous, vocab, indexed=False, output_dtype=torch.bfloat16):
    from ignition.components.vocabulary import PreparedVocabularyProjection
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_gather_runtime,
    )
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=2))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    hidden, capacity = 256, 4
    width = (vocab + world * 64 - 1) // (world * 64) * 64
    torch.manual_seed(42)
    full_weight = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=device) * .0625
    weight = torch.zeros(width, hidden, dtype=torch.bfloat16, device=device)
    valid_columns = min(width, vocab - rank * width)
    weight[:valid_columns].copy_(full_weight[rank * width:rank * width + valid_columns])
    seed = torch.randn(capacity, hidden, dtype=torch.bfloat16, device=device) * .125
    source = seed.clone()
    local = torch.empty(capacity, width, dtype=output_dtype, device=device)
    sample_capacity = 2 * capacity if indexed else capacity
    output = torch.empty(sample_capacity, vocab, dtype=output_dtype, device=device)
    indices = torch.zeros(sample_capacity, dtype=torch.int64, device=device) if indexed else None
    runtime = create_all_gather_runtime(group, device, max_rows=capacity,
        columns=width * world * (output.element_size() // 2))
    graphs, heads, compiled = {}, {}, {}
    control = {"frame_index": torch.zeros(1, dtype=torch.int32, device=device),
               "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
               "completion_records": torch.zeros((2, 4, 32), dtype=torch.int64, device=device)}
    worker = None
    try:
        with RecordedPreparationSession(device=device, checkpoint="vocabulary-test",
                workload_schema="column-gather-v1", numerical_policy="default", coordinator=coordinator,
                compile_workers=2) as session:
            for rows in (4, 1):
                result = output if indexed else output[:rows]
                heads[rows] = PreparedVocabularyProjection(source[:rows], weight, local[:rows], result,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"head{rows}", row_indices=indices)
            session.tune()
            stream = torch.cuda.Stream(device=device)
            pool = torch.cuda.graph_pool_handle()
            for rows, head in heads.items():
                compiled[rows] = compile_forward(lambda head=head, rows=rows: head(source[:rows]))
                compiled[rows]()
                torch.cuda.synchronize(device)
                stream.wait_stream(torch.cuda.current_stream(device))
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph, stream=stream, pool=pool):
                    compiled[rows]()
                graphs[f"head{rows}"] = graph
            session.freeze()
            bundle = PreparedExecution(device=rank, graphs=graphs,
                signatures={f"head{rows}": GraphSignature("decode", rows, 1) for rows in heads},
                buffers=control,
                resources=(heads, compiled, runtime, session, stream, pool, weight))
            worker = bundle.start_executor()
            for index, (rows, repetitions) in enumerate(((1, 13), (4, 8), (1, 31), (4, 17))):
                source.copy_(seed + index * .03125)
                expected = (source[:rows].float() @ full_weight.float().T).to(output_dtype)
                if indexed:
                    indices.copy_((torch.arange(sample_capacity, device=device) * 3 + index) % rows)
                    expected = expected[indices]
                output.fill_(float("nan"))
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                dist.barrier(group=group)
                for _ in range(repetitions):
                    ticket = worker.submit(bundle.graph_indices[f"head{rows}"], [])
                    assert ticket is not None
                    deadline = time.monotonic() + 120
                    completion = worker.poll()
                    while completion is None and time.monotonic() < deadline:
                        time.sleep(.001)
                        completion = worker.poll()
                    assert completion is not None and completion.fence == ticket
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                result = output if indexed else output[:rows]
                torch.testing.assert_close(result, expected,
                    rtol=.016 if output_dtype == torch.bfloat16 else 2e-5,
                    atol=.002 if output_dtype == torch.bfloat16 else 2e-5)
                selected_local = local[:rows] if indices is None else local[:rows][indices]
                torch.testing.assert_close(result[:, rank * width:rank * width + valid_columns],
                                           selected_local[:, :valid_columns], rtol=0, atol=0)
                if not indexed and rows < capacity:
                    assert output[rows:].isnan().all()
                dist.barrier(group=group)
            worker.close()
            worker = None
            for graph in graphs.values():
                graph.reset()
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("vocab", [154880, 128896])
def test_sharded_vocabulary_native_replay(world, vocab, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} assigned CUDA devices")
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri(), vocab), nprocs=world, join=True)


def test_sharded_vocabulary_expands_rows_after_gather_under_native_replay(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four assigned CUDA devices")
    mp.spawn(_rank, args=(4, (tmp_path / "rendezvous").as_uri(), 129280, True), nprocs=4, join=True)


def test_fp32_vocabulary_expands_rows_after_gather_under_native_replay(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two assigned CUDA devices")
    mp.spawn(_rank, args=(2, (tmp_path / "rendezvous").as_uri(), 129280, True, torch.float32),
             nprocs=2, join=True)


@pytest.mark.parametrize("world", [2, 4])
@torch.no_grad()
def test_quantized_draft_shards_preserve_logits_under_compiled_replay(world):
    from b12x.preparation import PreparationSession
    from ignition.components.vocabulary import PreparedVocabularyProjection, nvfp4_draft_head
    from ignition.preparation.compilation import compile_forward

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda", 0)
    capacity, vocab, hidden = 4, 768, 256
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                           0., -.5, -1., -1.5, -2., -3., -4., -6.], device=device)
    codes = (torch.arange(vocab, device=device)[:, None]
             + 3 * torch.arange(hidden, device=device)[None, :]) % 16
    # Every scale block contains +/-6, so NVFP4 represents these weights exactly.
    weight = levels[codes].bfloat16()
    original = weight.clone()
    source = torch.zeros(capacity, hidden, dtype=torch.bfloat16, device=device)
    output = torch.empty(capacity, vocab, dtype=torch.bfloat16, device=device)
    shards = [nvfp4_draft_head(part.contiguous()) for part in weight.chunk(world)]
    graphs, compiled, heads = {}, {}, {}
    stream = torch.cuda.Stream(device=device)
    try:
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            for rows in (1, capacity):
                heads[rows] = []
                width = vocab // world
                for rank, shard in enumerate(shards):
                    local = torch.empty(rows, width, dtype=source.dtype, device=device)
                    heads[rows].append(PreparedVocabularyProjection(source[:rows], shard, local,
                        output[:rows, rank * width:(rank + 1) * width], runtime=None,
                        session=session, coordinator=None, name=f"draft{rows}.shard{rank}"))

                def forward(rows=rows):
                    for head in heads[rows]:
                        head(source[:rows])

                compiled[rows] = compile_forward(forward)
                compiled[rows]()
                torch.cuda.synchronize(device)
                stream.wait_stream(torch.cuda.current_stream(device))
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph, stream=stream):
                    compiled[rows]()
                graphs[rows] = graph
            session.freeze()
            for turn, rows in enumerate((1, capacity, 1, capacity)):
                source.copy_(torch.sin(torch.arange(source.numel(), device=device)
                    .reshape_as(source).float() + turn).bfloat16())
                expected = (source[:rows].float() @ weight.float().T).bfloat16()
                output.fill_(float("nan"))
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                graphs[rows].replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
                torch.testing.assert_close(output[:rows], expected, rtol=.016, atol=.002)
                assert output[rows:].isnan().all()
            assert torch.equal(weight, original)
    finally:
        stream.synchronize()
        for graph in graphs.values():
            graph.reset()
