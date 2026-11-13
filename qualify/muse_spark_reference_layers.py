# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record canonical Muse Spark decoder outputs for exact teacher-forced tokens."""

import argparse
import json
from pathlib import Path
import time

import torch

from ignition.reference.muse_spark import text_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--attention-implementation', choices=('sdpa', 'eager'), default='sdpa',
                        help='sdpa requires the FlashAttention backend; eager retains BF16 score/probability rounding')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    tokens = json.loads(args.tokens.read_text())['token_ids']
    layers = []
    started = time.monotonic()

    def record(index, hidden):
        if index != len(layers):
            raise AssertionError('canonical decoder callbacks are out of order')
        layers.append(hidden.squeeze(0).detach().cpu().clone())
        print(f'layer={index} tokens={hidden.shape[1]} '
              f'elapsed_seconds={time.monotonic() - started:.3f}', flush=True)

    hidden, logits = text_logits(args.snapshot, tokens, device=args.device, progress=record,
                                 attention_implementation=args.attention_implementation)
    torch.save(dict(token_ids=torch.tensor(tokens), layers=torch.stack(layers),
                    hidden=hidden, logits=logits, attention_implementation=args.attention_implementation,
                    attention_backend='FLASH_ATTENTION' if args.attention_implementation == 'sdpa' else 'eager'), args.output)
    print(f'output={args.output} layers={len(layers)} '
          f'peak_tensor_bytes={torch.cuda.max_memory_allocated(args.device)}', flush=True)


if __name__ == '__main__':
    main()
