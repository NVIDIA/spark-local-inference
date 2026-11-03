# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Measure the gated reduction separately from model replay and host transfers.

Run from the repository root:
    python -m scripts.benchmark_logprobs --device cuda:1 --vocabulary 248320 --repeats 3
"""

import argparse
import json

import torch
from triton.testing import do_bench_cudagraph

from ignition.components.completion import (
    CompletionOutputs, EMITTED_COUNT_FIELD, VECTOR_TOKENS_FIELD,
)
from ignition.components.logprobs import PreparedLogprobs


def benchmark(rows, width, vocab, repeats, device):
    outputs = CompletionOutputs.allocate((2, rows, VECTOR_TOKENS_FIELD + width), device=device)
    outputs.records[0, :, 0] = 1
    outputs.records[0, :, 5] = 1
    outputs.records[0, :, EMITTED_COUNT_FIELD] = width
    outputs.records[0, :, VECTOR_TOKENS_FIELD:VECTOR_TOKENS_FIELD+width] = 0
    logits = torch.randn((rows, width, vocab), dtype=torch.bfloat16, device=device)
    slots = torch.arange(rows, dtype=torch.int32, device=device)
    epochs = torch.ones(rows, dtype=torch.int64, device=device)
    live = torch.tensor([rows], dtype=torch.int32, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    reduction = PreparedLogprobs(logits, outputs, slots, epochs, live, frame)
    reduction.prepare()
    marker = torch.zeros(1, dtype=torch.int32, device=device)
    def base():
        marker.add_(1)
    def reported():
        base()
        reduction()
    for trial in range(repeats):
        # Alternate order so changing clocks affect both comparisons.
        for state in (('base', 'off', 'on') if trial % 2 == 0 else ('on', 'off', 'base')):
            outputs.requested.fill_(int(state == 'on'))
            elapsed = do_bench_cudagraph(base if state == 'base' else reported, rep=30)
            print(json.dumps(dict(rows=rows, width=width, vocabulary=vocab, state=state,
                device=str(device), gpu=torch.cuda.get_device_name(device), dtype=str(logits.dtype),
                trial=trial, microseconds=elapsed * 1000, scratch_bytes=outputs.scratch.nbytes)), flush=True)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=torch.device, required=True, help='CUDA device, e.g. cuda:1')
    parser.add_argument('--vocabulary', type=positive_int, default=248320)
    parser.add_argument('--repeats', type=positive_int, default=3)
    args = parser.parse_args()
    if args.device.type != "cuda" or args.device.index is None:
        parser.error("--device must select a CUDA index, e.g. cuda:1")
    torch.cuda.set_device(args.device)
    for width in (1, 6):
        for rows in (1, 8, 16):
            benchmark(rows, width, args.vocabulary, args.repeats, args.device)
