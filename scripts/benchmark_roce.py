# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure b12x RoCEnante graph replay between GPU ranks on separate hosts.

Defaults match Qwen TP2 transport capacities and BF16 hidden reductions.
The primary CUDA-event interval follows one untimed collective inside the graph
to reduce launch skew. Samples include staging, proxy progress, and peer waits;
they are amortized communication latency, not pure wire latency. This component
diagnostic does not execute a Flare runtime core.
"""

import argparse
from contextlib import ExitStack
from datetime import timedelta
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world', type=int, default=2)
    parser.add_argument('--rendezvous', required=True, help='rank-zero HOST:PORT')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hca-names', nargs='+')
    parser.add_argument('--gid-index', type=int, default=3)
    parser.add_argument('--threads', type=int, default=512)
    parser.add_argument('--blocks', type=int, default=8)
    parser.add_argument('--max-size', type=int, default=2097152)
    parser.add_argument('--gather-bytes', type=int, default=31784960)
    parser.add_argument('--pipe-bytes', type=int, default=20971520)
    parser.add_argument('--sizes', type=int, nargs='+', default=[16, 5120, 15360, 20480, 40960, 81920])
    parser.add_argument('--gather-rows', type=int, nargs='*', default=[])
    parser.add_argument('--gather-cols', type=int, default=124160)
    parser.add_argument('--ops', type=int, default=20)
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1781)
    parser.add_argument('--output', type=Path, help='rank-local JSONL file; must not exist')
    args = parser.parse_args()
    if not 2 <= args.world <= 16 or not 0 <= args.rank < args.world:
        parser.error('world must be in [2,16] and rank must be in [0,world)')
    if min(args.ops, args.reps, args.warmup, args.gather_cols) < 1 or args.gid_index < 0:
        parser.error('counts must be positive and GID index must be nonnegative')
    if not 32 <= args.threads <= 1024 or args.threads % 32:
        parser.error('threads must be a multiple of 32 in [32,1024]')
    if args.blocks < 1 or args.blocks & (args.blocks - 1):
        parser.error('blocks must be a positive power of two')
    capacities = (args.max_size, args.gather_bytes, args.pipe_bytes)
    if any(size <= 0 or size % 16 for size in capacities):
        parser.error('byte capacities must be positive multiples of 16')
    if any(size <= 0 or size % 16 or size > max(args.max_size, args.pipe_bytes) for size in args.sizes):
        parser.error('reduction sizes must be positive 16-byte packs within transport capacity')
    if args.gather_cols % 8 or any(rows < 1 or rows * args.gather_cols * 2 > args.gather_bytes
                                 for rows in args.gather_rows):
        parser.error('gathers require 8-column packs and positive rows within gather capacity')
    if args.output is not None and args.output.exists():
        parser.error(f'output already exists: {args.output}')

    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.cli.preparation import Host

    configuration = Configuration.load(args.config)
    available = {gpu.index: gpu for gpu in Host.discover().gpus}
    if args.gpu not in available:
        parser.error(f'GPU {args.gpu} is not visible')
    apply(configuration, (available[args.gpu],))

    import torch
    import torch.distributed as dist
    from b12x.comm import roce
    from b12x.comm.roce import _preparation
    from b12x.preparation import PreparationSession, PreparedCall

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    dist.init_process_group('gloo', init_method=f'tcp://{args.rendezvous}',
        rank=args.rank, world_size=args.world, timeout=timedelta(seconds=180))
    with ExitStack() as resources:
        resources.callback(dist.destroy_process_group)
        output_file = None if args.output is None else resources.enter_context(args.output.open('x'))

        def emit(record):
            line = json.dumps(dict(rank=args.rank, **record))
            print(line, flush=True)
            if output_file is not None:
                output_file.write(line + '\n')
                output_file.flush()

        settings = dict(world=args.world, max_size=args.max_size, gather_bytes=args.gather_bytes,
            pipe_bytes=args.pipe_bytes, varlen_bytes=0, sizes=args.sizes,
            threads=args.threads, blocks=args.blocks,
            gather_rows=args.gather_rows, gather_cols=args.gather_cols,
            ops=args.ops, reps=args.reps, warmup=args.warmup, seed=args.seed)
        peer_settings = [None] * args.world
        dist.all_gather_object(peer_settings, settings)
        if any(peer != settings for peer in peer_settings):
            raise ValueError('ranks must use identical workload and transport capacities')
        runtime = roce.AllReduce(exchange_group=dist.group.WORLD, device=device,
            max_size=args.max_size, max_gather_bytes=args.gather_bytes, pipe_bytes=args.pipe_bytes,
            max_varlen_bytes=0, hca_names=args.hca_names, gid_index=args.gid_index,
            threads=args.threads, blocks=args.blocks)
        resources.callback(runtime.close)
        query = roce.query_from_runtime(runtime, surface='AllReduce.all_reduce',
            call={'dtypes': ('bfloat16',)}, topology='roce_rdma',
            peer_hosts=tuple(f'rank-{rank}' for rank in range(args.world)))
        plan = roce.plan(query, runtime=runtime)
        seed = torch.zeros(8, dtype=torch.bfloat16, device=device)

        def prepare(state):
            reduce = _preparation.prepared_call(state, inp=seed)
            gather = _preparation.prepared_gather_call(state, inp=seed)
            return PreparedCall(run=lambda: (reduce.run(), gather.run()))

        session = PreparationSession(device=device, autotune=False, compile_workers=2)
        resources.callback(session.close)
        session.prepare((plan.request(name='roce', prepare_call=prepare),))
        stream = torch.cuda.Stream(device=device)
        emit(dict(kind='configuration', **settings, dtype='bfloat16',
            timing='external CUDA-event nodes inside graph after one untimed collective; includes peer waits',
            primary_timing='segment_cuda_us_per_collective', untimed_prefix_ops=1,
            graph_ops=args.ops + 1,
            replay_timing='cuda_us_per_collective and host_us_per_collective cover the full graph divided by graph_ops',
            clocks_locked=False, runtime=runtime.stats(), region_bytes=runtime._region.numel(),
            gid_index=runtime.gid_index))

        cases = [('all_reduce', (size // 2,)) for size in args.sizes]
        cases += [('all_gather', (rows, args.gather_cols)) for rows in args.gather_rows]
        for case_index, (operation, shape) in enumerate(cases):
            source = torch.empty(shape, dtype=torch.bfloat16, device=device)
            output_shape = (*shape[:-1], shape[-1] * args.world) if operation == 'all_gather' else shape
            output = torch.empty(output_shape, dtype=torch.bfloat16, device=device)

            def set_input(iteration):
                peers = [torch.randn(shape, generator=torch.Generator().manual_seed(
                    args.seed + case_index * 100000 + iteration * args.world + rank),
                    dtype=torch.bfloat16) for rank in range(args.world)]
                source.copy_(peers[args.rank])
                output.fill_(float('nan'))
                if operation == 'all_gather':
                    return torch.cat(peers, dim=-1)
                expected = peers[0].float()
                for peer in peers[1:]:
                    expected.add_(peer.float())
                return expected.to(torch.bfloat16)

            def call():
                if operation == 'all_reduce':
                    runtime.all_reduce(source, out=output, plan=plan)
                else:
                    runtime.all_gather(source, dim=-1, out=output, plan=plan)

            def check(expected, stable=True):
                actual = output.cpu()
                peers = [torch.empty_like(actual) for _ in range(args.world)]
                dist.all_gather(peers, actual)
                errors = [None] * args.world
                dist.all_gather_object(errors, dict(reference=not torch.equal(actual, expected),
                    rank_agreement=any(not torch.equal(actual, peer) for peer in peers),
                    allocated_during_replay=not stable))
                if any(any(error.values()) for error in errors):
                    raise AssertionError(f'{operation} {shape}: {errors}')

            expected = set_input(0)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(args.warmup):
                    call()
            stream.synchronize()
            check(expected)
            dist.barrier()
            segment_begin, segment_end = (torch.cuda.Event(enable_timing=True, external=True)
                                          for _ in range(2))
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            for event in (segment_begin, segment_end, begin, end):
                event.record()
            end.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), runtime.capture(stream), torch.cuda.graph(graph, stream=stream):
                    call()
                    segment_begin.record(stream)
                    for _ in range(args.ops):
                        call()
                    segment_end.record(stream)
                graph.replay()
                torch.cuda.synchronize(device)
                check(expected)
                segment_samples, cuda_samples, host_samples = [], [], []
                for iteration in range(1, args.reps + 1):
                    expected = set_input(iteration)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    allocated = torch.cuda.memory_stats(device)['allocation.all.allocated']
                    host_begin = time.perf_counter()
                    begin.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    host_samples.append((time.perf_counter() - host_begin) * 1e6 / (args.ops + 1))
                    cuda_samples.append(begin.elapsed_time(end) * 1000 / (args.ops + 1))
                    segment_samples.append(segment_begin.elapsed_time(segment_end) * 1000 / args.ops)
                    stable = torch.cuda.memory_stats(device)['allocation.all.allocated'] == allocated
                    runtime.check_health()
                    check(expected, stable)
                emit(dict(kind='measurement', operation=operation, shape=shape,
                    bytes=source.numel() * source.element_size(),
                    segment_cuda_us_per_collective=segment_samples,
                    median_segment_cuda_us=statistics.median(segment_samples),
                    cuda_us_per_collective=cuda_samples, host_us_per_collective=host_samples,
                    median_cuda_us=statistics.median(cuda_samples),
                    runtime=runtime.stats()))
            finally:
                torch.cuda.synchronize(device)
                graph.reset()
            dist.barrier()
        dist.barrier()


if __name__ == '__main__':
    main()
