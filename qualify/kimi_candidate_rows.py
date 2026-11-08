"""Measure TP16 candidate unpacking, top-k selection and rank-zero consensus.

This is a synthetic component diagnostic, not model throughput. Every timed
iteration includes the complete control or candidate pipeline. A separate GPU
collective precedes the timed region to reduce host-arrival skew; input resets
and that alignment collective are excluded equally from both measurements.
"""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import statistics
import tempfile

import torch


WORLD, WIDTH, TOPK = 16, 32768, 2048


class FoldBroadcast:
    def __init__(self, rows, *, compact, device, session, runtime, coordinator):
        from ignition.components.candidate_rows import PreparedCandidateRows
        from ignition.components.collectives import PreparedBroadcast
        from ignition.components.row_topk import PreparedRowTopK
        from ignition.recipes.kimi31.indexer import SharedBuffers

        self.compact = compact
        self.packed = torch.full((rows, WIDTH, 2), -torch.inf, device=device)
        self.packed.view(torch.int32)[:, :, 1].fill_(-1)
        self.active = torch.zeros(rows, dtype=torch.bool, device=device)
        self.values = torch.empty((rows, WIDTH), dtype=torch.float32, device=device)
        self.ids = torch.empty((rows, WIDTH), dtype=torch.int32, device=device)
        self.lengths = torch.zeros(rows, dtype=torch.int32, device=device)
        self.output_values = torch.empty((rows, TOPK), dtype=torch.float32, device=device)
        self.output_ids = torch.full((rows, TOPK), -1, dtype=torch.int32, device=device)
        buffers = SharedBuffers({}, device)
        self.unpack = PreparedCandidateRows(self.packed, self.active, self.values, self.ids,
            self.lengths, topk=TOPK, buffers=buffers) if compact else None
        name = f"{'compact' if compact else 'padded'}{rows}"
        self.fold = PreparedRowTopK(self.values, self.lengths, self.ids, self.output_values,
            self.output_ids, session=session, name=name + ".fold")
        self.broadcast = PreparedBroadcast((self.output_ids,), runtime=runtime, session=session,
            coordinator=coordinator, name=name + ".consensus", buffers=buffers)

    def __call__(self):
        if self.compact:
            self.unpack()
        else:
            self.values.copy_(self.packed[:, :, 0])
            self.ids.copy_(self.packed.view(torch.int32)[:, :, 1])
            self.lengths.copy_(torch.where(self.active, WIDTH, 0))
        self.fold()
        self.broadcast()
        return self.output_ids


def inputs(rows, count, seed):
    generator = torch.Generator().manual_seed(seed)
    packed = torch.full((rows, WIDTH, 2), -torch.inf)
    packed.view(torch.int32)[:, :, 1].fill_(-1)
    active = torch.ones(rows, dtype=torch.bool)
    tokens = torch.arange(count, dtype=torch.int32)
    positions = (tokens % WORLD).long() * TOPK + tokens.long() // WORLD
    for row in range(rows):
        packed[row, positions, 0] = torch.randperm(count, generator=generator).float() / max(count, 1)
        packed.view(torch.int32)[row, positions, 1] = tokens
    return packed, active


def rank_main(rank, rendezvous, args):
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch.distributed as dist
    from ignition.components.collectives import PreparedAllReduce
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime, create_head_exchange
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=15))
    graphs, runtimes = [], []
    samples = []
    try:
        with PackageStorage(rank=rank, world=WORLD, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_head_exchange(dist.group.WORLD, device, max_rows=max(args.rows),
                heads=WORLD, head_dim=TOPK * 4, ipc=storage.peer_storage)
            alignment_runtime = create_all_reduce_runtime(dist.group.WORLD, device,
                collective_tuning='static', max_tokens=1, hidden_size=8, ipc=storage.peer_storage)
            runtimes = [runtime, alignment_runtime]
            with RecordedPreparationSession(device=device, checkpoint='candidate-row-pipeline',
                    workload_schema='padded-versus-underfull-with-consensus', numerical_policy='fp32-pairs',
                    autotune_effort='default', coordinator=coordinator) as session:
                session.register_scratch(ScratchArena(device))
                zero = torch.zeros((1, 8), dtype=torch.bfloat16, device=device)
                ready = torch.zeros_like(zero)
                align = PreparedAllReduce(zero, ready, runtime=alignment_runtime, session=session,
                    coordinator=coordinator, name='timing-alignment')
                variants = {rows: [FoldBroadcast(rows, compact=compact, device=device, session=session,
                                                runtime=runtime, coordinator=coordinator)
                                   for compact in (False, True)] for rows in args.rows}
                session.tune(show_progress=False)
                bound = {rows: [compile_forward(operation) for operation in operations]
                         for rows, operations in variants.items()}
                captures = {}
                for rows, operations in bound.items():
                    captures[rows] = []
                    for operation in operations:
                        align(zero)
                        operation()
                        torch.cuda.synchronize()
                        dist.barrier()
                        graph = torch.cuda.CUDAGraph()
                        start = torch.cuda.Event(enable_timing=True, external=True)
                        end = torch.cuda.Event(enable_timing=True, external=True)
                        with session.capture(), torch.cuda.graph(graph):
                            align(zero)
                            start.record()
                            for _ in range(args.iterations):
                                operation()
                            end.record()
                        graphs.append(graph)
                        captures[rows].append((graph, start, end))
                session.freeze()
                for rows in args.rows:
                    for count in args.counts:
                        for seed in args.seeds:
                            packed, active = inputs(rows, count, seed)
                            for operation in variants[rows]:
                                operation.packed.copy_(packed)
                                operation.active.copy_(active)
                            torch.cuda.synchronize()
                            for graph, _, end in captures[rows]:
                                dist.barrier()
                                graph.replay()
                                end.synchronize()
                            first, second = variants[rows]
                            assert torch.equal(first.output_ids.sort(-1).values, second.output_ids.sort(-1).values)
                            for pair in range(args.pairs):
                                order = (0, 1) if pair % 2 == 0 else (1, 0)
                                times = {}
                                for choice in order:
                                    graph, start, end = captures[rows][choice]
                                    dist.barrier()
                                    graph.replay()
                                    end.synchronize()
                                    times[choice] = start.elapsed_time(end) * 1000 / args.iterations
                                gathered = [None] * WORLD
                                dist.all_gather_object(gathered, times)
                                if rank == 0:
                                    sample = dict(rows=rows, valid_candidates=count, seed=seed, pair=pair,
                                        control_us=max(item[0] for item in gathered),
                                        candidate_us=max(item[1] for item in gathered),
                                        rank_control_us=[item[0] for item in gathered],
                                        rank_candidate_us=[item[1] for item in gathered])
                                    samples.append(sample)
                                    print(json.dumps(sample), flush=True)
                for graph in graphs:
                    graph.reset()
                graphs.clear()
            for runtime in reversed(runtimes):
                runtime.close()
            runtimes.clear()
        if rank == 0:
            summaries = []
            for rows in args.rows:
                for count in args.counts:
                    group = [item for item in samples if item['rows'] == rows and item['valid_candidates'] == count]
                    control = statistics.median(item['control_us'] for item in group)
                    candidate = statistics.median(item['candidate_us'] for item in group)
                    summaries.append(dict(rows=rows, valid_candidates=count, control_us=control,
                        candidate_us=candidate, latency_change_percent=(candidate/control - 1) * 100))
            args.output.write_text(json.dumps(dict(world=WORLD, width=WIDTH, topk=TOPK,
                iterations_per_sample=args.iterations, samples=samples, summaries=summaries,
                timed='Compiled unpack or count/compact, row top-k, and identical rank-zero broadcast; maximum rank duration.',
                excluded='Input reset, one captured alignment collective before the timing events, and host coordination.',
                scope='Synthetic component diagnostic; not model or native service throughput.'), indent=2) + '\n')
            print(json.dumps(summaries, indent=2), flush=True)
    finally:
        for graph in graphs:
            graph.reset()
        for runtime in reversed(runtimes):
            runtime.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=[1, 4, 128])
    parser.add_argument('--counts', type=int, nargs='+', default=[17, 1021, 2048, 2049, 4221, 32768])
    parser.add_argument('--seeds', type=int, nargs='+', default=[122, 123])
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=8)
    args = parser.parse_args()
    if (min(args.rows) < 1 or max(args.rows) > 128 or min(args.counts) < 1 or max(args.counts) > WIDTH
            or args.pairs < 1 or args.iterations < 1):
        parser.error('positive rows up to 128, counts up to 32768, pairs and iterations are required')
    if args.output.exists():
        parser.error('choose an unused output filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == WORLD, 'assign exactly sixteen CUDA devices'
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory(prefix='candidate-rows-tp16-') as directory:
        mp.spawn(rank_main, args=((Path(directory) / 'rendezvous').as_uri(), args), nprocs=WORLD, join=True)


if __name__ == '__main__':
    main()
