# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the production model, optionally measure replay, and bind native HTTP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time

from .provenance import write_json



def main():
    started = time.perf_counter_ns()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bind', default='127.0.0.1:38761')
    parser.add_argument('--prefill-chunk-size', type=int, choices=(512, 4096), required=True)
    parser.add_argument('--replay-corpus', type=Path)
    parser.add_argument('--trace-graph-ids', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from ignition.recipes.qwen38.bootstrap import PreparationConfig, prepare_model

    def progress(message):
        elapsed = (time.perf_counter_ns() - started) / 1e9
        print(f"{elapsed:.1f}s {message}", flush=True)

    config = PreparationConfig(prefill_chunk_size=args.prefill_chunk_size)
    with prepare_model(args.snapshot, config, progress=progress) as model:
        write_json(args.output / 'memory.json', model.memory_report)
        if args.replay_corpus:
            from .performance import validate_corpus
            from .replay_floor import measure
            corpus = json.loads(args.replay_corpus.read_text())
            validate_corpus(corpus)
            measure(model, corpus, args.output / 'replay')
        if args.trace_graph_ids:
            from .performance_trace import prepared_graph_map
            write_json(args.output / 'trace-graphs.json', prepared_graph_map(model.manifest()))
        server = model.start_server(bind=args.bind)
        print(f'Listening on {server.address}', flush=True)

        def stop(signum, frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            while server.wait(1000):
                pass
            raise RuntimeError('native HTTP server stopped unexpectedly')
        except KeyboardInterrupt:
            return 0
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
