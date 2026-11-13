# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture whole-graph GPU decode time with identical prompt token workloads.

Generate a corpus with ``prepare`` and replay it with ``flare``.
Run under Nsight with ``--trace=cuda-sw,nvtx``,
``--cuda-graph-trace=graph:host-only``, ``--capture-range=cudaProfilerApi``,
``--capture-range-end=stop`` and ``--export=sqlite``. Place ``flare-trace.sqlite``
beside the ``flare`` output directory, then run
``python -m qualify.gpu_replay_analysis DIRECTORY``.

For kernel timings, pass ``--ncu`` and collect with Nsight
Compute's ``--nvtx --nvtx-include kernel.b1/ --graph-profiling node
--metrics gpu__time_duration.sum --cache-control none --clock-control none``.
The range selects one warmed decode at position 575, including sampling.
Use the whole-graph Nsight Systems measurement for end-to-end GPU time;
Nsight Compute serializes the individual kernel measurements.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    from transformers import AutoTokenizer

    from qualify.performance import make_prompt

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    groups = []
    for repeat in range(4):
        for batch in args.batches:
            cases = []
            for lane in range(batch):
                request, tokens, _ = make_prompt(tokenizer, f'gpu-b{batch}-r{repeat}-lane{lane}', 512, 256)
                request.update(temperature=1, top_k=20, top_p=.95)
                case = {'request': request, 'prompt_tokens': tokens, 'output_tokens': 256}
                cases.append(case)
            groups.append({'batch': batch, 'repeat': repeat, 'measured': repeat > 0, 'cases': cases})
    save(args.output, {'prompt_tokens': 512, 'output_tokens': 256, 'groups': groups})


def flare(args, corpus):
    import torch

    from ignition.recipes.qwen38.bootstrap import PreparationConfig, prepare_model
    from qualify.performance_trace import prepared_graph_map
    from qualify.replay_floor import ReplayCase

    result = {'engine': 'flare', 'compiled_decoder': args.compile, 'groups': []}
    with prepare_model(args.snapshot, PreparationConfig(prefill_chunk_size=4096),
                       compile_decoder=args.compile,
                       progress=lambda message: print(message, flush=True)) as model:
        save(args.output / 'tuning.json', model.memory_report['tuning'])
        if not args.ncu:
            save(args.output / 'graphs.json', prepared_graph_map(model.manifest()))
        runner = ReplayCase(model)
        profiling = False
        try:
            for group in corpus['groups']:
                batch, repeat = group['batch'], group['repeat']
                initial = runner.prepare(group['cases'])
                if group['measured'] and not profiling:
                    torch.cuda.profiler.start()
                    profiling = True
                graph = model.graphs[f'decode{batch}']
                before = torch.cuda.memory_stats()
                with torch.cuda.stream(runner.stream):
                    with torch.cuda.nvtx.range(f'flare.decode.b{batch}.r{repeat}'):
                        for step in range(255):
                            selected = args.ncu and repeat == 1 and step == 63
                            scope = torch.cuda.nvtx.range(f'kernel.b{batch}') if selected else nullcontext()
                            with scope:
                                graph.replay()
                runner.stream.synchronize()
                after = torch.cuda.memory_stats()
                _, records = runner.snapshot(group['cases'])
                keys = ('allocation.all.allocated', 'allocation.all.freed',
                        'segment.all.allocated', 'segment.all.freed')
                stable = all(before[key] == after[key] for key in keys)
                if not stable:
                    raise RuntimeError('decode replay allocated or freed CUDA storage')
                result['groups'].append({'batch': batch, 'repeat': repeat,
                    'measured': group['measured'], 'initial_tokens': initial,
                    'records': records, 'allocator_stable': stable})
                save(args.output / 'results.json', result)
                print(f'flare batch={batch} repeat={repeat} completed 256 tokens per request', flush=True)
        finally:
            if profiling:
                torch.cuda.profiler.stop()
            runner.stream.synchronize()
        if args.check_compiled:
            from qualify.compiled_replay import check_compiled_replay

            try:
                result['compiled_validation'] = check_compiled_replay(model, corpus['groups'], args.output)
            except Exception as error:
                result['compiled_validation'] = {'status': 'failed', 'error': str(error)}
                save(args.output / 'results.json', result)
                raise
            save(args.output / 'results.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', choices=('prepare', 'flare'))
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--batches', type=int, nargs='+', choices=(1, 2, 4), default=(1, 2, 4))
    parser.add_argument('--ncu', action='store_true',
        help='annotate one warmed decode at context position 575 per batch for Nsight Compute')
    parser.add_argument('--compile', action='store_true',
        help='compile Flare prefill, decode, and resume forwards before CUDA graph capture')
    parser.add_argument('--check-compiled', action='store_true',
        help='compare compiled and uncompiled replay bytewise; requires deterministic kernels')
    args = parser.parse_args()
    if args.compile and args.engine != 'flare':
        parser.error('--compile applies only to Flare')
    if args.check_compiled and not args.compile:
        parser.error('--check-compiled requires --compile')
    if len(set(args.batches)) != len(args.batches):
        parser.error('--batches must be unique')
    if args.output.exists():
        parser.error('output must be fresh')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.engine == 'prepare':
        return prepare(args)
    if not args.corpus:
        parser.error('require --corpus')
    args.output.mkdir()
    corpus = json.loads(args.corpus.read_text())
    corpus['groups'] = [group for group in corpus['groups'] if group['batch'] in args.batches]
    expected = {(batch, repeat) for batch in args.batches for repeat in range(4)}
    if {(group['batch'], group['repeat']) for group in corpus['groups']} != expected or len(corpus['groups']) != len(expected):
        parser.error('corpus must contain one warmup and three measured groups per selected batch')
    flare(args, corpus)


if __name__ == '__main__':
    main()
