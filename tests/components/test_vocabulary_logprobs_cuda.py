# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation logprob shard reduction preserves raw scores and completion lifetimes."""

import os

import pytest
import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.vocabulary_logprobs import PreparedVocabularyParallelLogprobs
from tests.components.test_generation_logprobs_cuda import _assert_entry

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="CUDA opt-in")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [1, 3, 8])
@pytest.mark.parametrize("requested_ids", [False, True])
@pytest.mark.parametrize("functional", [False, True])
def test_raw_scores_frames_ties_and_mutated_replay(dtype, width, requested_ids, functional):
    vocabulary = 4097
    outputs = CompletionOutputs.allocate((2, 4, 32), device="cuda")
    logits = torch.randn((2, width, vocabulary), dtype=dtype, device="cuda")
    logits[0, :, :40] = 20
    slots = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
    epochs = torch.tensor([9, 10], dtype=torch.int64, device="cuda")
    live = torch.tensor([2], dtype=torch.int32, device="cuda")
    frame = torch.tensor([0], dtype=torch.int32, device="cuda")
    outputs.requested[2] = 1
    record = outputs.records[0, 2]
    record[0], record[8] = 9, width
    selected = [vocabulary - 1 - index for index in range(width)]
    record[16:16 + width] = torch.tensor(selected, device="cuda")
    ids = list(reversed(range(128))) if requested_ids else None
    if ids:
        outputs.token_id_counts[2] = len(ids)
        outputs.token_ids[2, :len(ids)] = torch.tensor(ids, device="cuda", dtype=torch.int32)
    operator = PreparedVocabularyParallelLogprobs(logits, outputs, slots, epochs, live, frame,
        vocabulary=vocabulary, offset=0, runtime=None, session=None, coordinator=None, name="logprobs")
    operator.prepare()
    stats, keys = operator.stats, operator.keys
    if functional:
        operator.functional(session=None)
        def call():
            return operator.run(logits)
    else:
        call = operator
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        logits[0, :, 40:] += .5
        stats.fill_(float("nan"))
        keys.fill_(-1)
        outputs.logprobs.fill_(-777)
        graph.replay()
        result = outputs.logprobs.cpu()
        assert torch.all(result[1] == -777)
        assert torch.all(result[0, :2] == -777)
        assert torch.all(result[0, 3] == -777)
        assert torch.all(result[0, 2, width:] == -777)
        for column, token in enumerate(selected):
            _assert_entry(result[0, 2, column], logits[0, column], token, ids=ids)
    outputs.requested.zero_()
    outputs.logprobs.fill_(-777)
    graph.replay()
    assert torch.all(outputs.logprobs == -777)
    outputs.requested[2] = 1
    epochs[0] = 8
    graph.replay()
    assert torch.all(outputs.logprobs == -777)
    epochs[0] = 9
    frame.fill_(1)
    graph.replay()
    assert torch.all(outputs.logprobs == -777)


def _parallel_logprob_rank(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(minutes=3))
    device = torch.device("cuda", rank)
    torch.manual_seed(61523)
    full = torch.randn(2, 8, 4097, dtype=torch.float32)
    full[0, :, 2090:2130] = 20  # Cutoff ties cross the vocabulary partition.
    local = torch.full((2, 8, 2112), 1e9, device=device)
    offset = rank * local.shape[-1]
    size = min(local.shape[-1], full.shape[-1] - offset)
    local[:, :, :size].copy_(full[:, :, offset:offset + size])
    outputs = CompletionOutputs.allocate((2, 4, 32), device=device)
    slots = torch.tensor([2, 0], dtype=torch.int32, device=device)
    epochs = torch.tensor([9, 10], dtype=torch.int64, device=device)
    live = torch.tensor([2], dtype=torch.int32, device=device)
    frame = torch.tensor([0], dtype=torch.int32, device=device)
    outputs.requested[slots.long()] = 1
    selected = [[4096, 0, 2112, 2111, 3333, 44, 1000, 3000], [0, 2112, 4096, 90, 3500, 100, 120, 130]]
    queried = [(index * 43) % full.shape[-1] for index in range(128)]
    outputs.token_id_counts[0] = len(queried)
    outputs.token_ids[0, :len(queried)] = torch.tensor(queried, dtype=torch.int32, device=device)
    for row, slot in enumerate([2, 0]):
        outputs.records[0, slot, 0] = 9 + row
        outputs.records[0, slot, 8] = 8
        outputs.records[0, slot, 16:24] = torch.tensor(selected[row], device=device)
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    runtime = create_all_gather_runtime(group, device, max_rows=16, columns=1024)
    graph = call = None
    try:
        with RecordedPreparationSession(device=device, checkpoint="raw-logprob-shards",
                workload_schema="generation-shards-v1", numerical_policy="default",
                coordinator=coordinator, autotune_effort="off", compile_workers=4) as session:
            operator = PreparedVocabularyParallelLogprobs(local, outputs, slots, epochs, live, frame,
                vocabulary=4097, offset=offset, runtime=runtime, session=session,
                coordinator=coordinator, name="logprobs")
            session.tune()
            operator.prepare()
            call = compile_forward(operator)
            call()
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph, stream=stream):
                call()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            for iteration in range(4):
                full[:, :, 40:80] += .5
                local[:, :, :size].copy_(full[:, :, offset:offset + size])
                operator.stats.fill_(float("nan"))
                operator.keys.fill_(-1)
                outputs.logprobs.fill_(-777)
                live.fill_(2 if iteration != 2 else 1)
                torch.cuda.synchronize()
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                result = outputs.logprobs.cpu()
                for row, slot in enumerate([2, 0]):
                    if iteration == 2 and row == 1:
                        assert torch.all(result[0, slot] == -777)
                        continue
                    for column, token in enumerate(selected[row]):
                        _assert_entry(result[0, slot, column], full[row, column], token,
                                      ids=queried if row else None)
                assert torch.all(result[1] == -777)
            dist.barrier()
    finally:
        torch.cuda.synchronize()
        del graph, call
        runtime.close()
        dist.destroy_process_group()


def test_two_rank_compiled_logprobs_preserve_cross_shard_ties_and_requested_ids(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two explicitly assigned CUDA devices")
    import torch.multiprocessing as mp
    mp.spawn(_parallel_logprob_rank, args=("file://" + str(tmp_path / "logprob-ranks"),),
             nprocs=2, join=True)
