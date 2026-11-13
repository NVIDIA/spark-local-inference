# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure checkpoint loading and native preparation of compact CSF scales.

Times one complete Qwen expert layer with the production b12x direct loader.
Header discovery is excluded. Timed loads include direct GPU reads, TP gathering,
validation, and compressed native-layout preparation.
"""

import argparse
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from b12x._lib.quant._csf_prepare import prepare_csf_payload, validate_source
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_qwen38_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--trials', type=int, default=7)
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--tp-rank', type=int, default=0)
    args = parser.parse_args()
    if args.trials <= 0 or args.tp_size <= 0 or not 0 <= args.tp_rank < args.tp_size:
        parser.error('trials and TP size must be positive, and TP rank must be in range')
    local = 640 // args.tp_size
    if 640 % args.tp_size or local % 32:
        parser.error('TP partitions must contain complete 32-channel groups')
    first, last = args.tp_rank * local, (args.tp_rank + 1) * local
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    manifest = inspect_qwen38_checkpoint(args.snapshot)
    prefix = f'model.language_model.layers.{args.layer}.mlp.experts.'
    with DirectWeightLoader(manifest, 'cuda:0', prefixes=(prefix,)) as loader:
        def load():
            sources = (
                loader._queue_csf_sources([f'{prefix}{expert}.{proj}.weight_scale'
                    for expert in range(512) for proj in ('up_proj', 'gate_proj')],
                    rows=640, columns=160, join=2, row_range=(first, last)),
                loader._queue_csf_sources([f'{prefix}{expert}.down_proj.weight_scale' for expert in range(512)],
                    rows=2560, columns=40, column_range=(first // 16, last // 16)),
            )
            loader._session.flush()
            return sources

        def convert(sources):
            for source in sources:
                validate_source(source)
            return tuple(prepare_csf_payload(source) for source in sources)

        planes = load()
        expected = convert(planes)
        torch.cuda.synchronize()
        timings = {'checkpoint+conversion': [], 'resident-conversion': []}
        for _ in range(args.trials):
            for name in timings:
                torch.cuda.synchronize()
                started = perf_counter()
                result = convert(load() if name == 'checkpoint+conversion' else planes)
                torch.cuda.synchronize()
                timings[name].append((perf_counter() - started) * 1000)
                for actual, reference in zip(result, expected, strict=True):
                    assert torch.equal(actual.storage, reference.storage)
                del result
        print(f'Layer {args.layer}: 512 experts, hidden=2560, intermediate=640, '
              f'TP={args.tp_size} rank={args.tp_rank}, reader={loader._session.read_mode}')
        print(f'Checkpoint CSF: {sum(t.nbytes for p in planes for t in p.components) / 2**20:.3f} MiB')
        print(f'Resident CSF: {sum(p.storage.nbytes for p in expected) / 2**20:.3f} MiB')
        for name, values in timings.items():
            print(f'{name} wall ms: {[round(v, 3) for v in values]}, median={statistics.median(values):.3f}')


if __name__ == '__main__':
    main()
