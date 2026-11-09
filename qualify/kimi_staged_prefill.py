"""Synthetic TP16 diagnostics for staged Kimi latent-record transport.

Timings include record addressing, gathering and BF16 attention. The staged path
also includes packing, publication, peer reads and the consumption collective.
They exclude input preparation and do not measure complete model throughput.
"""

import argparse
from datetime import timedelta
import json
import math
from pathlib import Path
import statistics
import tempfile

import torch

WORLD, PAGE, WIDTH, TOPK, SLICE = 16, 128, 576, 2048, 128


class SelectedHistory:
    def __init__(self, *, cache, direct_peers, stage, output, signals, rows, maximum,
                 runtime, coordinator, session, arena, device, name="fixture"):
        from b12x.attention import dense_mla
        from ignition.components.dense_mla import PreparedDenseMLA
        from ignition.components.collectives import PreparedAllReduce
        from ignition.components.interleaved_history import PreparedInterleavedHistory
        from ignition.components.peer_record_gather import PreparedPeerRecordGather
        from ignition.recipes.kimi3.dcp import ContextShards

        rank = coordinator.ranks.index(torch.distributed.get_rank())
        self.shards = ContextShards.on(rank, WORLD, device)
        self.cache, self.maximum, self.rows = cache, maximum, rows
        self.end = torch.zeros(1, dtype=torch.int64, device=device)
        self.table = torch.arange(cache.shape[0] - 1, -1, -1, dtype=torch.int32, device=device)
        self.selected = torch.full((rows, TOPK), -1, dtype=torch.int32, device=device)
        self.positions = torch.full((rows,), -1, dtype=torch.int64, device=device)
        self.current = torch.zeros((rows, WIDTH), dtype=torch.bfloat16, device=device)
        self.transfer = PreparedInterleavedHistory(cache, self.table, self.end, self.shards.index,
            stage.tensor.view(-1, WIDTH), stage.addresses, output, signals, runtime=runtime,
            session=session, coordinator=coordinator, name=name + ".history")
        self.output = output
        self.local_address = torch.tensor([output.data_ptr()], dtype=torch.int64, device=device)
        self.query = torch.full((SLICE, 6, WIDTH), .015625, dtype=torch.bfloat16, device=device)
        self.starts = torch.arange(SLICE + 1, dtype=torch.int32, device=device)
        self.pages = torch.arange(SLICE * TOPK, dtype=torch.int32, device=device).view(SLICE, TOPK)
        self.parts = []
        self.reference_output = torch.empty((rows, 6, 512), dtype=torch.bfloat16, device=device)
        self.staged_output = torch.empty_like(self.reference_output)
        self.exact_records = torch.empty((rows // SLICE,), dtype=torch.bool, device=device)
        self.exact_counts = torch.empty_like(self.exact_records)
        self.done_input = torch.zeros((1, 8), dtype=torch.bfloat16, device=device)
        self.done_output = torch.zeros_like(self.done_input)
        self.done = PreparedAllReduce(self.done_input, self.done_output, runtime=runtime,
            session=session, coordinator=coordinator, name=name + ".reference-consumed")
        records = [torch.empty((SLICE, TOPK, WIDTH), dtype=torch.float8_e4m3fn, device=device) for _ in range(2)]
        attended = [torch.empty((SLICE, 6, 512), dtype=torch.bfloat16, device=device) for _ in range(2)]
        self.records = records
        for variant in range(2):
            owners = torch.empty((SLICE, TOPK), dtype=torch.int32, device=device)
            slots = torch.empty_like(owners)
            counts = torch.empty((SLICE,), dtype=torch.int32, device=device)
            gather = PreparedPeerRecordGather(direct_peers if variant == 0 else self.local_address,
                owners, slots, counts, records[variant],
                peer_records=cache.shape[0] * PAGE if variant == 0 else maximum,
                session=session, name=f"{name}.records{variant}",
                dependencies=(cache,) if variant == 0 else (output, self.transfer.completed))
            caps = dense_mla.Caps(device=device, mode="decode", q_dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn, num_q_heads=6, max_total_q=SLICE, max_batch=SLICE,
                page_size=1, max_cache_tokens=TOPK, max_page_table_width=TOPK,
                num_cache_pages=SLICE * TOPK, use_cuda_graph=True, partial_dtype=torch.float32)
            attention = PreparedDenseMLA(self.query, records[variant].view(-1, 1, WIDTH), attended[variant],
                self.pages, counts, self.starts, caps=caps, scale=192 ** -.5,
                session=session, name=f"{name}.attention{variant}", workspace=arena)
            self.parts.append((owners, slots, counts, gather, attention, attended[variant]))
        owners, slots, counts, *_ = self.parts[0]
        self.timed_staged_gather = PreparedPeerRecordGather(self.local_address, owners, slots, counts, records[0],
            peer_records=maximum, session=session, name=name + ".timed-staged-records",
            dependencies=(output, self.transfer.completed))

    def write_current(self):
        from ignition.recipes.kimi3.dcp import write_records

        table = self.table[None].expand(self.rows, -1)
        current_slots = self.shards.token_slots(self.positions, table)
        write_records(self.current, current_slots, self.cache.view(-1, WIDTH))

    def run_variant(self, staged):
        """Use one record buffer, attention plan and scratch for either timed path."""
        from ignition.recipes.kimi31.dcp import selection_slots

        table = self.table[None].expand(self.rows, -1)
        owners, slots, counts, direct_gather, attention, attended = self.parts[0]
        if staged:
            self.transfer()
        target = self.staged_output if staged else self.reference_output
        for begin in range(0, self.rows, SLICE):
            end = begin + SLICE
            selection_slots(self.selected[begin:end], self.positions[begin:end], table[begin:end],
                owners, slots, counts, self.shards, self.cache.shape[0], local=False, global_records=staged)
            if staged:
                self.timed_staged_gather()
            else:
                direct_gather()
            attention()
            target[begin:end].copy_(attended)
        return target

    def __call__(self):
        from ignition.recipes.kimi31.dcp import selection_slots

        table = self.table[None].expand(self.rows, -1)
        self.write_current()
        self.transfer()
        for begin in range(0, self.rows, SLICE):
            end = begin + SLICE
            for variant, (owners, slots, counts, gather, attention, attended) in enumerate(self.parts):
                selection_slots(self.selected[begin:end], self.positions[begin:end], table[begin:end],
                    owners, slots, counts, self.shards, self.cache.shape[0], local=False,
                    global_records=variant == 1)
                gather()
                attention()
                target = self.reference_output if variant == 0 else self.staged_output
                target[begin:end].copy_(attended)
            self.exact_records[begin // SLICE] = torch.eq(self.records[0].view(torch.uint8),
                                                        self.records[1].view(torch.uint8)).all()
            self.exact_counts[begin // SLICE] = torch.eq(self.parts[0][2], self.parts[1][2]).all()
        # The direct-reference readers must drain before the next queued input
        # mutates their source cache. This barrier is fixture-only work.
        self.done_input.copy_(self.exact_records.all().to(torch.bfloat16).expand_as(self.done_input))
        self.done(self.done_input)
        return self.staged_output


class Variant:
    def __init__(self, operation, staged):
        self.operation, self.staged = operation, staged

    def __call__(self):
        return self.operation.run_variant(self.staged)


def summarize_pairs(records):
    """Pair rank maxima by seed and iteration; retain every measured observation."""
    pairs = {}
    for record in records:
        key = record['seed'], record['pair']
        pair = pairs.setdefault(key, {})
        if record['path'] in pair:
            raise ValueError('duplicate timing observation')
        pair[record['path']] = max(record['rank_ms'])
    if not pairs or any(set(pair) != {'direct', 'staged'} for pair in pairs.values()):
        raise ValueError('timing requires complete direct/staged pairs')
    return dict(pairs=len(pairs), direct_mean_ms=statistics.mean(pair['direct'] for pair in pairs.values()),
        staged_mean_ms=statistics.mean(pair['staged'] for pair in pairs.values()),
        paired_difference_mean_ms=statistics.mean(pair['staged'] - pair['direct'] for pair in pairs.values()),
        aggregation='Arithmetic means of per-replay maximum-rank CUDA event intervals.')


def _benchmark_rank(rank, rendezvous, destination, pairs, seeds):
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch.distributed as dist
    from ignition.components.shared_pages import SharedPages
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device, maximum = torch.device('cuda', rank), 1048576
    dist.init_process_group('gloo', rank=rank, world_size=WORLD, init_method=rendezvous,
                            timeout=timedelta(minutes=30))
    graphs, runtime, shared_cache, stage = [], None, None, None
    results = []
    try:
        with PackageStorage(rank=rank, world=WORLD, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            shared_cache = SharedPages(dist.group.WORLD, device, maximum // WORLD * WIDTH, ipc=storage.peer_storage)
            stage = SharedPages(dist.group.WORLD, device, maximum // WORLD * WIDTH, ipc=storage.peer_storage)
            cache = shared_cache.tensor.view(torch.float8_e4m3fn).view(-1, PAGE, WIDTH)
            output = torch.empty((maximum, WIDTH), dtype=cache.dtype, device=device)
            signals = tuple(torch.zeros((1, 8), dtype=torch.bfloat16, device=device) for _ in range(4))
            runtime = create_all_reduce_runtime(dist.group.WORLD, device,
                collective_tuning='static', max_tokens=1, hidden_size=8, ipc=storage.peer_storage)
            with RecordedPreparationSession(device=device, checkpoint='kimi31-staged-prefill-synthetic',
                    workload_schema='paired-transfer-gather-bf16-attention', numerical_policy='exact-bf16',
                    autotune_effort='default', compile_workers=16, coordinator=coordinator) as session:
                arena = ScratchArena(device)
                session.register_scratch(arena)
                operations = {rows: SelectedHistory(cache=cache, direct_peers=shared_cache.addresses,
                    stage=stage, output=output, signals=signals, rows=rows, maximum=maximum,
                    runtime=runtime, coordinator=coordinator, session=session, arena=arena, device=device,
                    name=f'rows{rows}') for rows in (128, 4096)}
                session.tune(show_progress=False)
                for operation in operations.values():
                    operation()
                    torch.cuda.synchronize(device)
                prepared = {}
                for rows, operation in operations.items():
                    variants = {}
                    for name, staged in (('direct', False), ('staged', True)):
                        call = compile_forward(Variant(operation, staged))
                        call()
                        torch.cuda.synchronize(device)
                        graph = torch.cuda.CUDAGraph()
                        start = torch.cuda.Event(enable_timing=True, external=True)
                        stop = torch.cuda.Event(enable_timing=True, external=True)
                        with session.capture(), torch.cuda.graph(graph):
                            start.record()
                            call()
                            stop.record()
                        graphs.append(graph)
                        variants[name] = graph, start, stop, call
                    prepared[rows] = variants
                session.freeze()
                for rows, operation in operations.items():
                    samples = []
                    for visible in (4096, 32768, maximum):
                        samples.clear()
                        for seed in seeds:
                            generator = torch.Generator(device=device).manual_seed(seed)
                            # Independent rank data forbids reconstructing a peer's
                            # current chunk from this rank's projected values.
                            raw = torch.randint(0, 127, cache.shape, dtype=torch.uint8, device=device,
                                                generator=generator)
                            raw.bitwise_xor_(rank % 16)
                            raw.remainder_(127)
                            row_ids = torch.arange(rows, dtype=torch.int64, device=device)
                            columns = torch.randperm(TOPK, device=device, generator=generator)
                            lengths = visible - rows + row_ids + 1
                            steps = []
                            for length in range(visible - rows + 1, visible + 1):
                                step = max(1, length // TOPK)
                                while math.gcd(step, length) != 1:
                                    step += 1
                                steps.append(step)
                            strides = torch.tensor(steps, device=device, dtype=torch.int64)
                            ids = ((row_ids[:, None] * 31 + columns[None] * strides[:, None]) % lengths[:, None]).int()
                            ids = torch.where(columns[None] < lengths[:, None], ids, -1)
                            query = (torch.randn(operation.query.shape, device=device, generator=generator) * .125).bfloat16()

                            def reset():
                                cache.view(torch.uint8).copy_(raw)
                                operation.end.fill_(visible)
                                operation.positions.copy_(visible - rows + row_ids)
                                operation.selected.copy_(ids)
                                operation.current.fill_((rank + 1) / 64)
                                operation.query.copy_(query)
                                operation.write_current()
                                stage.tensor.fill_(255)
                                output.view(torch.uint8).fill_(255)
                                for record in operation.records:
                                    record.view(torch.uint8).fill_(255)
                                torch.cuda.synchronize(device)
                                dist.barrier()

                            # Check every slice's record order with the independent
                            # two-buffer reference before using the timed captures.
                            reset()
                            operation()
                            torch.cuda.synchronize(device)
                            assert operation.exact_records.all() and operation.exact_counts.all(), (rank, rows, visible)
                            torch.testing.assert_close(operation.staged_output, operation.reference_output, rtol=0, atol=0)
                            dist.barrier()
                            for name in ('direct', 'staged'):
                                reset()
                                for _ in range(2):
                                    prepared[rows][name][0].replay()
                                torch.cuda.synchronize(device)
                                dist.barrier()
                            for pair in range(pairs):
                                order = ('direct', 'staged') if pair % 2 == 0 else ('staged', 'direct')
                                for name in order:
                                    reset()
                                    graph, start, stop, _ = prepared[rows][name]
                                    before = torch.cuda.memory_stats(device)['allocation.all.allocated']
                                    graph.replay()
                                    torch.cuda.synchronize(device)
                                    assert torch.cuda.memory_stats(device)['allocation.all.allocated'] == before
                                    elapsed = start.elapsed_time(stop)
                                    per_rank = [None] * WORLD
                                    dist.all_gather_object(per_rank, elapsed)
                                    # The control has no in-path collective after its
                                    # reads. Every rank drains before input reset.
                                    samples.append(dict(seed=seed, pair=pair, path=name, order=list(order), rank_ms=per_rank))
                                torch.testing.assert_close(operation.staged_output, operation.reference_output, rtol=0, atol=0)
                        record = dict(rows=rows, live_history=visible, prepared_history=maximum,
                            seeds=list(seeds), warmups=2, repeats_per_seed=pairs, samples=list(samples),
                            summary=summarize_pairs(samples), correctness='exact records/order and BF16 output passed')
                        results.append(record)
                        if rank == 0:
                            report = dict(boundary='Synthetic component diagnostic; not complete-model throughput.',
                                world=WORLD, query_dtype='bfloat16', record_dtype='float8_e4m3fn',
                                included='record addressing, selected gather, BF16 attention, output copy; staged also packs, publishes, reads peers, and joins',
                                excluded='input reset, compilation, warmup, timing exchanges and host barriers',
                                tuning='default', cache_policy='identical input/poison reset before each path; no separate L2 flush',
                                cases=results)
                            Path(destination).write_text(json.dumps(report, indent=2) + '\n')
                            print(json.dumps({key: record[key] for key in ('rows', 'live_history', 'summary')}), flush=True)
                for graph in graphs:
                    graph.reset()
                graphs.clear()
            runtime.close()
            runtime = None
            stage.close()
            shared_cache.close()
            stage = shared_cache = None
    finally:
        for graph in graphs:
            graph.reset()
        if runtime is not None:
            runtime.close()
        if stage is not None:
            stage.close()
        if shared_cache is not None:
            shared_cache.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7331, 2909])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must not already exist')
    if args.pairs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('pairs must be positive and seeds distinct')
    if torch.cuda.device_count() != WORLD:
        parser.error('assign exactly sixteen idle CUDA devices')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory(prefix='kimi-staged-prefill-') as directory:
        mp.spawn(_benchmark_rank, args=((Path(directory) / 'gloo').as_uri(), str(args.output),
                                       args.pairs, args.seeds), nprocs=WORLD, join=True)


if __name__ == '__main__':
    main()
