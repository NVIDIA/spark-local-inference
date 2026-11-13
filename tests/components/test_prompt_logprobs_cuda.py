# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prompt reductions use bounded logits and preserve frame and block ownership."""

import gc
import os

import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='CUDA opt-in')


@pytest.mark.parametrize('vocab', [31, 1025, 4096, 248320])
@pytest.mark.parametrize('compiled', [False, True])
def test_prompt_reduction(vocab, compiled):
    from ignition.components.prompt_logprobs import PromptOutputs, PreparedPromptReduction
    rows, start = 5, 16
    torch.manual_seed(42)
    logits = torch.randn((rows, vocab), device='cuda', dtype=torch.bfloat16)
    logits[1].zero_()
    logits[2, 1:].fill_(-float('inf'))
    targets = [vocab - 1, 7, 0, -1, 2]
    outputs = PromptOutputs(32, 'cuda')
    outputs.targets[start:start + rows] = torch.tensor(targets, device='cuda')
    outputs.values.fill_(-777)
    frame = torch.ones(1, device='cuda', dtype=torch.int32)
    op = PreparedPromptReduction(logits, outputs.targets, outputs.values, frame,
        start=start, vocabulary=vocab, offset=0, runtime=None, session=None,
        coordinator=None, name='test')
    from ignition.preparation.compilation import compile_forward
    op = op.functional(session=None)
    call = compile_forward(lambda: op(logits)) if compiled else lambda: op(logits)
    call()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    gc.collect()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats()['allocation.all.allocated']
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
    assert torch.cuda.max_memory_allocated() == baseline
    for _ in range(3):
        outputs.values.fill_(-777)
        graph.replay()
        result = outputs.values.cpu()
        assert torch.all(result[0] == -777)
        assert torch.all(result[1, :start] == -777)
        assert torch.all(result[1, start + rows:] == -777)
        for row, target in enumerate(targets):
            packed = result[1, start + row]
            if target < 0:
                assert packed[0] == -1 and packed[2] == 0
                continue
            ref = logits[row].cpu().double().log_softmax(0)
            values = packed[1::2].view(torch.float32)
            assert abs(values[0].item() - ref[target].item()) < 3e-5
            ids = packed[3::2].tolist()
            # BF16 inputs have exact ties; the reported FP32 subtraction may create more.
            reported = logits[row].cpu().float() - torch.logsumexp(logits[row].cpu().float(), 0)
            expected = sorted(range(vocab), key=lambda i: (-float(reported[i]), i))[:20]
            assert ids[:min(20, vocab)] == expected
            assert packed[2].item() == 1 + int((reported > reported[target]).sum())
            torch.testing.assert_close(packed[4::2].view(torch.float32)[:len(expected)].double(),
                                       ref[expected], atol=3e-5, rtol=0)


@pytest.mark.parametrize('world', [2, 4])
def test_prompt_partial_combination_excludes_padding_and_counts_cross_shard_ties(world):
    from ignition.components.prompt_logprobs import PromptOutputs, PreparedPromptReduction
    vocab, rows = 1025, 3
    width = (vocab + world * 64 - 1) // (world * 64) * 64
    torch.manual_seed(7)
    raw = torch.randn(rows, vocab, device='cuda', dtype=torch.float32)
    raw[0, :64].fill_(3.)
    raw[1].zero_()
    targets = torch.tensor([5, vocab - 1, width], device='cuda', dtype=torch.int32)
    outputs = [PromptOutputs(rows, 'cuda') for _ in range(world)]
    frame = torch.zeros(1, device='cuda', dtype=torch.int32)
    parts, shards = [], []
    for rank in range(world):
        shard = torch.full((rows, width), 1000., device='cuda')
        size = min(width, vocab - rank * width)
        shard[:, :size].copy_(raw[:, rank * width:rank * width + size])
        outputs[rank].targets.copy_(targets)
        op = PreparedPromptReduction(shard, outputs[rank].targets, outputs[rank].values, frame,
            start=0, vocabulary=vocab, offset=rank * width, runtime=None, session=None,
            coordinator=None, name=f'shard{rank}')
        parts.append(op)
        shards.append(shard)
    # Simulate the bit-preserving collective between the two reduction phases.
    from ignition.components.prompt_logprobs import _global_stats, _finish, PROMPT_FIELDS
    from ignition.components.completion import MAX_TOP_LOGPROBS
    from ignition.preparation.triton import prepare_triton
    gathered = torch.zeros((rows, world * 64), device='cuda', dtype=torch.int32)
    norms = [op.operations[2].args[1] for op in parts]
    for op in parts:
        op.operations[0]()
        op.operations[1]()
    gathered.copy_(torch.cat([op.partial for op in parts], -1))
    for op, norm in zip(parts, norms):
        prepare_triton(_global_stats, (gathered, norm, world, world), (rows, 1, 1), num_warps=4)()
        op.operations[3]()
        op.operations[4]()
    gathered.copy_(torch.cat([op.partial for op in parts], -1))
    prepare_triton(_finish, (targets, gathered, norms[0], outputs[0].values, frame, vocab, 0, rows,
        world, MAX_TOP_LOGPROBS, 32 * world, world, PROMPT_FIELDS), (rows, 1, 1), num_warps=4)()
    actual = outputs[0].values[0].cpu()
    ref = raw.cpu().double().log_softmax(-1)
    for row, target in enumerate(targets.cpu().tolist()):
        assert abs(actual[row, 1:2].view(torch.float32).item() - ref[row, target]) < 3e-5
        assert actual[row, 2] == 1 + (ref[row] > ref[row, target]).sum()
        expected = sorted(range(vocab), key=lambda i: (-float(ref[row, i]), i))[:20]
        assert actual[row, 3::2].tolist() == expected


@pytest.mark.parametrize('logits_dtype', [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_blocked_projection_capture_crosses_row_blocks_with_fixed_peak_memory(logits_dtype):
    from ignition.components.prompt_logprobs import PromptScoring, BLOCK_ROWS
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    rows, vocab, hidden = BLOCK_ROWS * 2 + 1, 4096, 256
    source = torch.randn(rows, hidden, device='cuda', dtype=torch.bfloat16) * .125
    weight = torch.randn(vocab, hidden, device='cuda', dtype=torch.bfloat16) * .125
    frame = torch.ones(1, dtype=torch.int32, device='cuda')
    scoring = PromptScoring(rows, 'cuda', frame=frame)
    outputs = scoring.outputs
    outputs.targets.copy_(torch.arange(rows, device='cuda', dtype=torch.int32) % vocab)
    with RecordedPreparationSession(device=torch.device('cuda'), checkpoint='prompt-projection-test',
            workload_schema='prompt-blocks-v1', numerical_policy='default') as session:
        op = scoring.prepare(source, weight, vocabulary=vocab,
            session=session, coordinator=None, name='score', logits_dtype=logits_dtype)
        session.tune()
        declaration_source = source
        source = source.clone()
        declaration_source.fill_(float('nan'))
        call = compile_forward(lambda: op(source))
        call()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph, stream=stream):
            call()
        torch.cuda.current_stream().wait_stream(stream)
        source.add_(.03125)
        before = torch.cuda.memory_stats()['allocation.all.allocated']
        peak = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
        assert torch.cuda.max_memory_allocated() == peak
        print(dict(rows=rows, vocabulary=vocab, replay_peak_bytes=peak,
                   replay_allocations=0))
        # Every declared projection output has the fixed block bound, including the tail.
        assert max(count for _, count, _, _ in op.parts) == BLOCK_ROWS
        assert op.parts[-1][1] == 1
        assert all(projection.output is None for _, _, projection, _ in op.parts)
        assert all(projection.output_layout[2] == logits_dtype for _, _, projection, _ in op.parts)
        logits = (source.cpu().float() @ weight.cpu().float().T).to(logits_dtype).double()
        ref = logits.log_softmax(-1)
        result = outputs.values[1].cpu()
        for row in [0, BLOCK_ROWS - 1, BLOCK_ROWS, rows - 1]:
            actual = result[row, 1:2].view(torch.float32).item()
            assert abs(actual - ref[row, row].item()) < (.03 if logits_dtype == torch.bfloat16 else 5e-5)


def _prompt_tp_rank(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist
    from ignition.components.prompt_logprobs import PromptOutputs, PreparedPromptReduction
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(minutes=3))
    device = torch.device('cuda', rank)
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    runtime = create_all_gather_runtime(group, device, max_rows=4, columns=256)
    torch.manual_seed(42)
    full = torch.randn(4, 1025, dtype=torch.bfloat16)
    full[0, :64] = 5
    full[1].zero_()
    width = 576
    local = torch.full((4, width), 1000., dtype=torch.bfloat16, device=device)
    size = min(width, 1025 - rank * width)
    local[:, :size].copy_(full[:, rank * width:rank * width + size])
    outputs = PromptOutputs(4, device)
    outputs.targets.copy_(torch.tensor([5, 1024, 700, -1], device=device))
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    graph = None
    call = None
    try:
        with RecordedPreparationSession(device=device, checkpoint='prompt-tp-test',
                workload_schema='prompt-partials-v1', numerical_policy='default', coordinator=coordinator) as session:
            op = PreparedPromptReduction(local, outputs.targets, outputs.values, frame, start=0,
                vocabulary=1025, offset=rank * width, runtime=runtime, session=session,
                coordinator=coordinator, name='scores')
            hashes = [None, None]
            dist.all_gather_object(hashes, op.operations[0].compiled.hash)
            assert hashes[0] == hashes[1]
            op = op.functional(session=session)
            session.tune()
            call = compile_forward(lambda: op(local))
            call()
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph, stream=stream):
                call()
            for _ in range(5):
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                actual = outputs.values[0].cpu()
                ref = full.double().log_softmax(-1)
                for row, target in enumerate([5, 1024, 700]):
                    assert abs(actual[row, 1:2].view(torch.float32).item() - ref[row, target]) < 3e-5
                    assert actual[row, 2] == 1 + (ref[row] > ref[row, target]).sum()
                assert actual[3, 0] == -1 and actual[3, 2] == 0
            dist.barrier()
    finally:
        torch.cuda.synchronize()
        del graph, call
        runtime.close()
        dist.destroy_process_group()


def test_two_rank_partial_exchange_capture_replay(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip('requires two explicitly assigned CUDA devices')
    import torch.multiprocessing as mp
    mp.spawn(_prompt_tp_rank, args=('file://' + str(tmp_path / 'prompt-tp'),), nprocs=2, join=True)


def test_prompt_reported_value_ties_and_negative_infinity_targets():
    from ignition.components.prompt_logprobs import PromptOutputs, PreparedPromptReduction
    for logits, target in [
        (torch.arange(4096, dtype=torch.float32, device='cuda')[None] * 1e-12, 4095),
        (torch.tensor([[0., -float('inf'), -float('inf')]], device='cuda'), 2),
    ]:
        outputs = PromptOutputs(1, 'cuda')
        outputs.targets.fill_(target)
        frame = torch.zeros(1, dtype=torch.int32, device='cuda')
        op = PreparedPromptReduction(logits, outputs.targets, outputs.values, frame, start=0,
            vocabulary=logits.shape[1], offset=0, runtime=None, session=None, coordinator=None,
            name='ties').functional(session=None)
        op(logits)
        actual = outputs.values[0, 0].cpu()
        assert actual[2] == (2 if target == 2 else 1)
        assert actual[3:3 + 2 * min(20, logits.shape[1]):2].tolist() == list(range(min(20, logits.shape[1])))
        if target == 2:
            assert actual[1:2].view(torch.float32).isneginf().all()
