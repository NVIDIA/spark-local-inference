# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure closed-loop native HTTP serving with timestamped SSE token bursts."""
from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import math
import statistics
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit


def connection(url, timeout=600):
    parsed = urlsplit(url)
    cls = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
    return cls(parsed.hostname, parsed.port, timeout=timeout)


def request(url, body, stream=False, path="/v1/chat/completions"):
    client = connection(url)
    start = time.perf_counter()
    try:
        client.request('POST', path, json.dumps(body),
                       {'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
        response = client.getresponse()
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {response.read().decode()}')
        if not stream:
            return json.load(response)
        groups, usage, finishes, done = [], None, [], False
        for line in response:
            if not line.startswith(b'data: '):
                continue
            elapsed = time.perf_counter() - start
            data = line[6:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            if 'error' in event:
                raise RuntimeError(event['error'])
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                if choice.get('index') != 0:
                    raise RuntimeError(f'unexpected streaming choice index: {choice.get("index")}')
                tokens = choice.get('token_ids') or []
                if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
                    raise RuntimeError('invalid generated token IDs')
                if tokens:
                    groups.append({'seconds': elapsed, 'token_ids': tokens})
                if choice.get('finish_reason') is not None:
                    finishes.append(choice['finish_reason'])
        count = sum(len(group['token_ids']) for group in groups)
        if (not done or not groups or not usage or usage['completion_tokens'] != count
                or len(finishes) != 1 or not isinstance(finishes[0], str)
                or not 0 < count <= body['max_tokens']
                or usage.get('total_tokens') != usage.get('prompt_tokens', 0) + count):
            raise RuntimeError(f'incomplete token stream: done={done}, groups={len(groups)}, count={count}, finishes={finishes}, usage={usage}')
        return {'started': start, 'seconds': time.perf_counter() - start, 'groups': groups,
                'usage': usage, 'finish_reason': finishes[0]}
    finally:
        client.close()


def prompt(url, model, length, lane, *, context=None):
    tasks = [
        "Review this queue implementation for ownership and cancellation bugs, then provide a corrected implementation and explain each change.",
        "Design a bounded asynchronous request scheduler. Explain fairness, backpressure, timeout handling, and failure recovery with concrete examples.",
        "Explain how to diagnose a service with high tail latency and low average CPU utilization. Include measurements, likely causes, and experiments.",
        "Implement a prefix cache for immutable token sequences. Discuss branching, eviction, memory accounting, and concurrent readers.",
    ]
    default_context = """A local inference service receives interactive requests while background jobs submit longer prompts. Each request owns input buffers until the GPU completes its last read. Cancellation may occur before admission, during execution, or while output is waiting for a client. The runtime must keep cache state valid across these transitions. Request identifiers are unique, and a completion event can contain several accepted tokens. Latency measurements must retain actual event boundaries. The scheduler estimates execution costs from observed graph durations and chooses bounded work that fits prepared capacities.

class RequestQueue:
    def __init__(self):
        self.pending = []
        self.active = {}
    def submit(self, request):
        self.pending.append(request)
    def cancel(self, request_id):
        self.active.pop(request_id, None)
    def complete(self, request_id, result):
        request = self.active.pop(request_id)
        request.future.set_result(result)

The implementation must handle a client disconnect without reusing storage that a submitted operation still references. A stalled output consumer should not prevent other requests from progressing. Describe how the API reports errors, how shutdown drains work, and how a test can force each race deterministically. Use explicit resource ownership and concrete invariants. Avoid adding configuration switches without a demonstrated need.
"""
    prefix = f'Workload lane {lane}. ' + tasks[lane % len(tasks)] + '\nContext:\n'
    if context is None:
        text = prefix + default_context * max(1, length // 300 + 1)
    else:
        text = prefix + context * max(1, length * 8 // len(context) + 1)
    low, high = len(prefix), len(text)
    cut, padding = (low + high) // 2, 0
    for _ in range(32):
        body = {'model': model, 'messages': [{'role': 'user', 'content': text[:cut] + ' detail' * padding}],
                'max_tokens': 1, 'seed': 0, 'return_token_ids': True}
        result = request(url, {'model': model, 'messages': body['messages']},
                         path='/v1/messages/count_tokens')
        actual = result['input_tokens']
        if actual == length:
            return body
        if actual > length:
            high = cut - 1
        else:
            low = cut + 1
        if low > high:
            if actual < length and length - actual <= 4:
                padding += length - actual
            else:
                cut -= 1
                padding = 0
        else:
            cut = (low + high) // 2
    raise ValueError(f'cannot produce {length}-token prompt for lane {lane}: {actual}')


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)] if values else None


def metrics(url):
    client = connection(url)
    try:
        client.request('GET', '/metrics')
        response = client.getresponse()
        if response.status != 200:
            raise RuntimeError(f'metrics HTTP {response.status}')
        return response.read().decode()
    finally:
        client.close()


def trial(url, bodies, duration):
    before_metrics = metrics(url)
    barrier = threading.Barrier(len(bodies) + 1)
    deadline = [0.0]
    def worker(lane, body):
        results = []
        barrier.wait()
        while time.perf_counter() < deadline[0]:
            row = request(url, body, stream=True)
            row['lane'] = lane
            results.append(row)
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(bodies)) as pool:
        jobs = [pool.submit(worker, lane, body) for lane, body in enumerate(bodies)]
        start = time.perf_counter()
        deadline[0] = start + duration
        barrier.wait()
        rows = [row for job in jobs for row in job.result()]
    elapsed = time.perf_counter() - start
    groups = [group for row in rows for group in row['groups']]
    gaps = [b['seconds'] - a['seconds'] for row in rows for a, b in zip(row['groups'], row['groups'][1:])]
    ttfts = [row['groups'][0]['seconds'] for row in rows]
    tokens = sum(row['usage']['completion_tokens'] for row in rows)
    return {'concurrency': len(bodies), 'issue_duration_seconds': duration, 'elapsed_seconds': elapsed,
            'requests': len(rows), 'generated_tokens': tokens, 'tokens_per_second': tokens / elapsed,
            'prompt_token_counts': sorted({row['usage']['prompt_tokens'] for row in rows}),
            'cached_prompt_token_counts': sorted({(row['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', -1) for row in rows}),
            'cached_prompt_tokens': sum((row['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0) for row in rows),
            'ttft_median_seconds': statistics.median(ttfts), 'ttft_p99_seconds': percentile(ttfts, .99),
            'burst_gap_median_seconds': statistics.median(gaps) if gaps else None,
            'burst_gap_p99_seconds': percentile(gaps, .99),
            'coalesced_groups': sum(len(g['token_ids']) > 1 for g in groups),
            'early_eos_requests': sum(row['finish_reason'] != 'length' for row in rows),
            'metrics_before': before_metrics, 'metrics_after': metrics(url), 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='DeepSeek-V4.1-Flash')
    parser.add_argument('--concurrency', default='1,2,4,8,16,32')
    parser.add_argument('--prompt-tokens', type=int, default=512)
    parser.add_argument('--output-tokens', type=int, default=256)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--warmup-seconds', type=float, default=30)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--expected-cached-prompt-tokens', type=int)
    parser.add_argument('--pair-index', type=int)
    parser.add_argument('--variant', choices=('baseline', 'candidate'))
    parser.add_argument('--pair-order', choices=('baseline,candidate', 'candidate,baseline'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-requests', type=Path, help='Read identical requests from a previous measurement JSONL file.')
    args = parser.parse_args()
    concurrencies = [int(c) for c in args.concurrency.split(',')]
    if args.reuse_requests:
        with args.reuse_requests.open() as previous:
            bodies = json.loads(previous.readline())['requests']
        if len(bodies) < max(concurrencies):
            raise ValueError('request file has insufficient concurrency lanes')
    else:
        bodies = [prompt(args.base_url, args.model, args.prompt_tokens, lane) for lane in range(max(concurrencies))]
    for body in bodies:
        body.update(max_tokens=args.output_tokens, stream=True, stream_options={'include_usage': True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps({'configuration': {**vars(args), 'output': str(args.output), 'reuse_requests': str(args.reuse_requests) if args.reuse_requests else None}, 'requests': bodies}) + '\n')
        output.flush()
        for concurrency in concurrencies:
            if args.warmup_seconds:
                trial(args.base_url, bodies[:concurrency], args.warmup_seconds)
            for repetition in range(args.trials):
                result = trial(args.base_url, bodies[:concurrency], args.seconds)
                result['prompt_geometry_valid'] = result['prompt_token_counts'] == [args.prompt_tokens]
                result['cache_condition_valid'] = (args.expected_cached_prompt_tokens is None
                    or result['cached_prompt_token_counts'] == [args.expected_cached_prompt_tokens])
                result['trial'] = repetition
                output.write(json.dumps(result) + '\n')
                output.flush()
                if not result['prompt_geometry_valid']:
                    raise ValueError(f"prompt geometry differs: {result['prompt_token_counts']}")
                if not result['cache_condition_valid']:
                    raise ValueError(f"cache condition differs: {result['cached_prompt_token_counts']}")
                print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'metrics_before', 'metrics_after')}), flush=True)


if __name__ == '__main__':
    main()
