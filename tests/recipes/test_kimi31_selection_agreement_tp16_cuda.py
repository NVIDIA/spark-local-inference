"""TP16 agreement and complete candidate-merge timings with a broadcast control."""

from datetime import timedelta
import json
import os
import statistics

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_TP_CUDA_TESTS') != '1',
                              reason='requires sixteen explicitly assigned CUDA devices')


class MergePipeline:
    def __init__(self, rows, rank, runtime, session, coordinator, buffers, *, ordered):
        from ignition.components.candidate_rows import PreparedCandidateRows
        from ignition.components.collectives import PreparedAllGather, PreparedBroadcast
        from ignition.components.ordered_row_topk import PreparedOrderedRowTopK
        from ignition.components.row_topk import PreparedRowTopK
        from ignition.recipes.kimi3.dcp import ContextShards
        from ignition.recipes.kimi31.indexer import SharedBuffers

        device = runtime.device
        self.shards = ContextShards.on(rank, 16, device)
        storage = SharedBuffers(buffers, device)
        self.local_ids = storage.buffer('local_ids', (rows, 2048), torch.int32)
        self.local_scores = storage.buffer('local_scores', (rows, 2048), torch.float32)
        self.active = storage.buffer('active', (rows,), torch.bool)
        self.packed = storage.buffer('packed', (rows, 2048, 2), torch.float32)
        self.gathered = storage.buffer('gathered', (rows, 32768, 2), torch.float32)
        self.values = storage.buffer('values', (rows, 32768), torch.float32)
        self.ids = storage.buffer('ids', (rows, 32768), torch.int32)
        self.lengths = storage.buffer('lengths', (rows,), torch.int32)
        self.output_values = storage.buffer('output_values', (rows, 2048), torch.float32)
        self.output_ids = storage.buffer('output_ids', (rows, 2048), torch.int32)
        name = f'{"ordered" if ordered else "broadcast"}.{rows}'
        self.gather = PreparedAllGather(self.packed.view(torch.bfloat16).view(rows, -1),
            self.gathered.view(torch.bfloat16).view(rows, -1), runtime=runtime,
            session=session, coordinator=coordinator, name=name + '.candidates')
        self.compact = PreparedCandidateRows(self.gathered, self.active, self.values, self.ids, self.lengths,
            topk=2048, buffers=SharedBuffers(buffers, device, prefix='compact.'))
        args = (self.values, self.lengths, self.ids, self.output_values, self.output_ids)
        if ordered:
            self.fold = PreparedOrderedRowTopK(*args, session=session, name=name + '.fold',
                buffers=SharedBuffers(buffers, device, prefix='ordered.'))
            self.broadcast = None
        else:
            self.fold = PreparedRowTopK(*args, session=session, name=name + '.fold')
            self.broadcast = PreparedBroadcast((self.output_ids,), runtime=runtime, session=session,
                coordinator=coordinator, name=name + '.consensus',
                buffers=SharedBuffers(buffers, device, prefix='broadcast.'))

    def __call__(self):
        from ignition.recipes.kimi31.dcp import pack_candidates

        pack_candidates(self.local_ids, self.local_scores, self.packed, self.shards)
        self.gather(self.packed.view(torch.bfloat16).view(self.packed.shape[0], -1))
        self.compact()
        self.fold()
        if self.broadcast is not None:
            self.broadcast()
        return self.output_ids


def source_rows(rows, seed, count=None):
    from tests.components.test_ordered_row_topk import adversarial_rows

    scores, lengths, ids, labels = adversarial_rows(seed=seed)
    chosen = torch.arange(rows).add(seed).remainder(scores.shape[0])
    scores, lengths, ids = scores[chosen].clone(), lengths[chosen].clone(), ids[chosen].clone()
    ids[torch.arange(32768)[None] >= lengths[:, None]] = -1
    if count is not None:
        generator = torch.Generator().manual_seed(seed)
        scores = torch.stack([torch.randperm(32768, generator=generator) for _ in range(rows)]).float() / 32768
        ids = torch.arange(2048, dtype=torch.int32).repeat(16)[None].expand(rows, -1).clone()
        live = torch.arange(32768).remainder(2048) * 16 + torch.arange(32768).div(2048, rounding_mode='floor')
        ids[:, live >= count] = -1
        lengths.fill_(count)
    scores[ids < 0] = -torch.inf
    return scores, ids, lengths > 0


def expected_packed(scores, ids):
    result = torch.empty((*scores.shape, 2), dtype=torch.float32)
    result[:, :, 0].copy_(scores)
    rank = torch.arange(32768, dtype=torch.int32).div(2048, rounding_mode='floor')
    result.view(torch.int32)[:, :, 1].copy_(torch.where(ids >= 0, ids * 16 + rank, -1))
    return result


def worker(rank, rendezvous, destination):
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch.distributed as dist
    from ignition.preparation.collectives import CollectivePreparation, create_head_exchange
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tuning import RecordedPreparationSession
    from tests.components.test_ordered_row_topk import ordered_reference

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=16, timeout=timedelta(minutes=15))
    graphs, runtime = [], None
    try:
        with PackageStorage(rank=rank, world=16, device=rank) as storage:
            runtime = create_head_exchange(dist.group.WORLD, device, max_rows=128, heads=16, head_dim=8192,
                                           ipc=storage.peer_storage)
            coordinator = CollectivePreparation(dist.group.WORLD)
            with RecordedPreparationSession(device=device, checkpoint='identical-gathered-candidates',
                    workload_schema='tp16-selection-agreement', numerical_policy='ordered-uint32-input-ties',
                    coordinator=coordinator, compile_workers=16) as session:
                pools, cases, calls = {False: {}, True: {}}, {}, {}
                for rows in (128, 16, 1):
                    for ordered in (False, True):
                        cases[rows, ordered] = MergePipeline(rows, rank, runtime, session, coordinator,
                                                             pools[ordered], ordered=ordered)
                tuning = session.tune(show_progress=False)
                captures = {}
                for key, operation in cases.items():
                    scores, ids, active = source_rows(key[0], 0)
                    operation.local_scores.copy_(scores[:, rank * 2048:(rank + 1) * 2048])
                    operation.local_ids.copy_(ids[:, rank * 2048:(rank + 1) * 2048])
                    operation.active.copy_(active)
                    call = compile_forward(operation)
                    calls[key] = call
                    call()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    dist.barrier()
                    with session.capture(), torch.cuda.graph(graph):
                        call()
                    graphs.append(graph)
                    captures[key] = graph
                for seed in (0, 1, 2):
                    for key in ((1, True), (128, False), (16, True), (1, False), (128, True), (16, False)):
                        operation = cases[key]
                        scores, ids, active = source_rows(key[0], seed)
                        operation.local_scores.copy_(scores[:, rank * 2048:(rank + 1) * 2048])
                        operation.local_ids.copy_(ids[:, rank * 2048:(rank + 1) * 2048])
                        operation.active.copy_(active)
                        operation.output_ids.fill_(-91)
                        torch.cuda.synchronize()
                        dist.barrier()
                        before = torch.cuda.memory_stats()['allocation.all.allocated']
                        captures[key].replay()
                        torch.cuda.synchronize()
                        assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
                        observed = operation.output_ids.cpu()
                        assert torch.equal(operation.gathered.view(torch.int32).cpu(), expected_packed(scores, ids).view(torch.int32))
                        peers = [None] * 16
                        dist.all_gather_object(peers, observed)
                        assert all(torch.equal(value, peers[0]) for value in peers)
                        if key[1]:
                            expected_values, expected_ids, _ = ordered_reference(operation.values.cpu(), operation.lengths.cpu(), operation.ids.cpu(), 2048)
                            assert torch.equal(observed, expected_ids)
                            assert torch.equal(operation.output_values.view(torch.int32).cpu(), expected_values)
                sequence = ((1, True), (128, False), (16, True), (1, False), (128, True), (16, False))
                snapshots = [torch.empty_like(cases[key].output_ids) for key in sequence]
                score_snapshots = [torch.empty_like(cases[key].output_values) for key in sequence]
                patterns, expected = [], []
                for index, key in enumerate(sequence):
                    scores, ids, active = source_rows(key[0], index, count=32768)
                    packed = expected_packed(scores, ids)
                    expected.append(ordered_reference(packed[:, :, 0].contiguous(),
                        torch.full((key[0],), 32768, dtype=torch.int32),
                        packed.view(torch.int32)[:, :, 1].contiguous(), 2048))
                    patterns.append((scores[:, rank * 2048:(rank + 1) * 2048].to(device),
                                     ids[:, rank * 2048:(rank + 1) * 2048].to(device), active.to(device)))
                torch.cuda.synchronize()
                dist.barrier()
                if rank == 15:
                    torch.cuda._sleep(2000000)
                for index, key in enumerate(sequence):
                    operation = cases[key]
                    scores, ids, active = patterns[index]
                    operation.local_scores.copy_(scores)
                    operation.local_ids.copy_(ids)
                    operation.active.copy_(active)
                    operation.output_ids.fill_(-91)
                    captures[key].replay()
                    snapshots[index].copy_(operation.output_ids)
                    score_snapshots[index].copy_(operation.output_values)
                torch.cuda.synchronize()
                for index, key in enumerate(sequence):
                    observed = snapshots[index].cpu()
                    peers = [None] * 16
                    dist.all_gather_object(peers, observed)
                    assert all(torch.equal(value, peers[0]) for value in peers), key
                    if key[1]:
                        assert torch.equal(observed, expected[index][1])
                        assert torch.equal(score_snapshots[index].view(torch.int32).cpu(), expected[index][0])
                    else:
                        assert torch.equal(observed.sort().values, expected[index][1].sort().values)
                timings = []
                for rows, count in ((1, 1024), (1, 32768), (128, 32768)):
                    saved, events, samples = {}, {}, {False: [], True: []}
                    scores, ids, active = source_rows(rows, 77, count=count)
                    for ordered in (False, True):
                        operation = cases[rows, ordered]
                        operation.local_scores.copy_(scores[:, rank * 2048:(rank + 1) * 2048])
                        operation.local_ids.copy_(ids[:, rank * 2048:(rank + 1) * 2048])
                        operation.active.copy_(active)
                        pairs = [(torch.cuda.Event(enable_timing=True, external=True), torch.cuda.Event(enable_timing=True, external=True)) for _ in range(4)]
                        graph = torch.cuda.CUDAGraph()
                        dist.barrier()
                        with session.capture(), torch.cuda.graph(graph):
                            for begin, end in pairs:
                                begin.record()
                                calls[rows, ordered]()
                                end.record()
                        graphs.append(graph)
                        saved[ordered], events[ordered] = graph, pairs
                    for iteration in range(6):
                        for ordered in ((False, True) if iteration % 2 == 0 else (True, False)):
                            dist.barrier()
                            saved[ordered].replay()
                            torch.cuda.synchronize()
                            peers = [None] * 16
                            dist.all_gather_object(peers, [begin.elapsed_time(end) for begin, end in events[ordered]])
                            samples[ordered].extend(max(values) for values in zip(*peers, strict=True))
                    timings.append(dict(rows=rows, valid_candidates=count, samples_ms=samples,
                        median_ms={str(key): statistics.median(value) for key, value in samples.items()}))
                if rank == 0:
                    from pathlib import Path
                    Path(destination).write_text(json.dumps(dict(tuning=tuning,
                        effective_b12x={key: value for key, value in os.environ.items() if key.startswith('B12X_')},
                        paths={'false': 'gather+compact+topk+broadcast', 'true': 'gather+compact+ordered_topk'},
                        timings=timings), indent=2) + '\n')
                for graph in graphs:
                    graph.reset()
                graphs.clear()
            runtime.close()
            runtime = None
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    finally:
        for graph in graphs:
            graph.reset()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def test_tp16_ordered_selection_and_broadcast_control(tmp_path):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == 16
    mp.spawn(worker, args=((tmp_path / 'gloo').as_uri(), str(tmp_path / 'selection_pipeline.json')), nprocs=16, join=True)
