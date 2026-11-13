# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time a checkpoint-backed Qwen MoE shard with cold-cache CUDA graph replay.

Loads one layer, including all routed experts and the shared expert, and uses
seeded Gaussian activations. Timings include routing, shared/routed overlap,
and their output sum. They exclude TP communication and
the surrounding decoder. Native serving measurements establish end-to-end gains.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tp-size', type=int, default=2)
    parser.add_argument('--tp-rank', type=int, default=0)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--scale-storage', choices=('expanded', 'compressed'), default='expanded',
        help='retain expanded scales or lossless CSF scales with shared expansion scratch')
    parser.add_argument('--rows', type=int, nargs='+', default=[1, 3, 8, 16])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1781, 1782, 1783])
    parser.add_argument('--compile-workers', type=int, default=4)
    parser.add_argument('--lock-gpu-clocks', action='store_true', help='lock clocks only during autotuning')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--output', type=Path, help='write JSONL to a file that does not already exist')
    args = parser.parse_args()
    if args.tp_size < 1 or not 0 <= args.tp_rank < args.tp_size:
        parser.error('TP size must be positive and TP rank must be in range')
    if 640 % args.tp_size or (640 // args.tp_size) % 16:
        parser.error('TP intermediate partitions must contain complete 16-channel scale groups')
    if not 0 <= args.layer < 48 or any(rows < 1 for rows in args.rows):
        parser.error('layer must be in [0, 48) and row counts must be positive')
    if len(set(args.rows)) != len(args.rows):
        parser.error('row counts must be distinct')
    if min(args.warmup, args.samples, args.compile_workers) < 1:
        parser.error('warmup, samples, and compile workers must be positive')
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
    from flashinfer.testing.utils import bench_gpu_time_with_cuda_event
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation import inspect_qwen38_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.qwen38 import REVISION, TEXT_PREFIX
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.moe import PreparedQwenMoe, QwenMoeScratch, QwenMoeWeights
    from ignition.recipes.qwen38.tuning import preparation_session
    from ignition.recipes.qwen38.weights import QwenWeightLoader, allocate_expert_scale_scratch

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        parser.error('this benchmark requires SM120 or SM121')
    manifest = inspect_qwen38_checkpoint(args.snapshot)
    shard = TensorParallelShard(args.tp_rank, args.tp_size)
    prefix = f'{TEXT_PREFIX}.layers.{args.layer}.mlp.'
    output_file = None if args.output is None else args.output.open('x')

    def emit(record):
        line = json.dumps(record)
        print(line, flush=True)
        if output_file is not None:
            output_file.write(line + '\n')
            output_file.flush()

    try:
        with torch.no_grad(), QwenWeightLoader(
                manifest, device, prefixes=(prefix,), shard=shard) as loader:
            scale_scratch = (allocate_expert_scale_scratch(device, shard=shard)
                             if args.scale_storage == 'compressed' else None)
            weights = QwenMoeWeights.load(loader, args.layer, scale_scratch=scale_scratch)
        assert weights.experts.has_csf_blocks == (args.scale_storage == 'compressed')
        decoder = weights.experts._impl.nvfp4_csf
        compressed_scale_bytes = (0 if decoder is None else sum(
            plane.scales.storage.numel() * plane.scales.storage.element_size()
            for plane in (decoder.first, decoder.second)))
        with torch.no_grad(), preparation_session(
                device=device, checkpoint=REVISION, compile_workers=args.compile_workers,
                lock_gpu_clocks=args.lock_gpu_clocks, autotune_effort='default',
                namespace={'config': manifest.metadata_sha256['config.json'],
                           'headers': manifest.headers_sha256, 'world_size': args.tp_size},
                headroom_bytes=1 << 30) as session, ExitStack() as resources:
            workspace = ScratchArena(device)
            session.register_scratch(workspace)
            cases = []
            for rows in args.rows:
                source = torch.zeros((rows, 2560), dtype=torch.bfloat16, device=device)
                scratch = QwenMoeScratch.allocate(rows, device, intermediate=640 // args.tp_size)
                moe = PreparedQwenMoe(source, weights, session=session, name=f'moe.rows{rows}',
                    scratch=scratch, workspace=workspace, projection_workspace=workspace)
                cases.append((source, moe))
            report = session.tune()
            emit(dict(kind='configuration', snapshot=str(manifest.snapshot), layer=args.layer,
                tp_size=args.tp_size, tp_rank=args.tp_rank, gpu=available[args.gpu].uuid,
                scale_storage=args.scale_storage, compressed_scale_bytes=compressed_scale_bytes,
                scale_scratch_bytes=sum(buffer.numel() * buffer.element_size()
                                        for buffer in scale_scratch or ()),
                rows=args.rows, seeds=args.seeds, activation_distribution='Gaussian(mean=0,std=0.1)',
                autotune_effort='default', compile_workers=args.compile_workers,
                tuning_clock_lock=args.lock_gpu_clocks, timing_clock_lock=False,
                timing='CUDA events around compiled graph replay; L2 flushed before each sample',
                warmup=args.warmup, samples=args.samples, selections=report['selections']))

            stream = torch.cuda.Stream(device=device)
            graphs, compiled = [], []

            def release_graphs():
                torch.cuda.synchronize()
                for graph in graphs:
                    graph.reset()

            resources.callback(release_graphs)
            for source, moe in cases:
                forward = compile_forward(lambda source=source, moe=moe: moe(source))
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    forward()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                graphs.append(graph)
                with session.capture(), torch.cuda.graph(graph, stream=stream):
                    forward()
                compiled.append(forward)
            session.freeze()

            for (source, moe), graph in zip(cases, graphs, strict=True):
                for seed in args.seeds:
                    generator = torch.Generator().manual_seed(seed)
                    values = (torch.randn(source.shape, generator=generator) * .1).to(torch.bfloat16)
                    source.copy_(values)
                    moe.output.fill_(float('nan'))
                    torch.cuda.synchronize()
                    allocated = torch.cuda.memory_stats(device)['allocation.all.allocated']
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize()
                    if torch.cuda.memory_stats(device)['allocation.all.allocated'] != allocated:
                        raise RuntimeError('captured MoE replay allocated GPU storage')
                    if not torch.isfinite(moe.output).all().item():
                        raise RuntimeError(f'non-finite output for rows={source.shape[0]}, seed={seed}')
                    with torch.cuda.nvtx.range(f'qwen_moe.rows{source.shape[0]}.seed{seed}'):
                        times = bench_gpu_time_with_cuda_event(graph.replay,
                            dry_run_iters=args.warmup, repeat_iters=args.samples, cold_l2_cache=True)
                    if not torch.isfinite(moe.output).all().item():
                        raise RuntimeError(f'non-finite timed output for rows={source.shape[0]}, seed={seed}')
                    emit(dict(kind='measurement', rows=source.shape[0], seed=seed,
                        implementation=moe.experts.binding.implementation,
                        unique_routed_experts=moe.router.ids.unique().numel(),
                        milliseconds=times, median_ms=statistics.median(times)))
    finally:
        if output_file is not None:
            output_file.close()


if __name__ == '__main__':
    main()
