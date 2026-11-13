# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure streamed decode throughput while every request remains active."""

import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx


PROMPT = """Implement a complete Python 3.12 asyncio job scheduler library.
It must support bounded concurrency, job priorities with FIFO ties, delayed jobs,
per-job timeout, cancellation, retries with exponential backoff, graceful shutdown,
and an async iterator of completion results. Use only the standard library.
Explain the API and concurrency invariants briefly, then provide the complete
implementation with type hints, followed by a substantial unittest suite using
IsolatedAsyncioTestCase. Cover race conditions, failures, cancellation, priority
ordering, retries, and shutdown. Include an executable usage example. Do not omit
method bodies or use pseudocode. Make the code readable and self-contained."""


async def request(client, args, lane, repeat, budget, on_tokens=None):
    template = dict(reasoning_effort="low")
    template["enable_thinking" if args.engine == "flare" else "thinking"] = True
    body = dict(
        model=args.model, messages=[dict(role="user", content=args.prompt)],
        max_tokens=budget, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        seed=1234 + repeat * 16 + lane,
        chat_template_kwargs=template,
        stream=True, stream_options=dict(include_usage=True), return_token_ids=True,
    )
    clock_start_ns = [time.time_ns(), time.perf_counter_ns(), time.time_ns()]
    started = clock_start_ns[1] / 1e9
    events, ids, text, reasoning = [], [], [], []
    usage, finish, prompt_ids = None, None, None
    done = False
    async with client.stream("POST", args.url + "/v1/chat/completions", json=body) as response:
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {(await response.aread()).decode()}")
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                done = True
                continue
            obj = json.loads(data)
            if "error" in obj:
                raise RuntimeError(obj["error"])
            usage = obj.get("usage") or usage
            prompt_ids = obj.get("prompt_token_ids") or prompt_ids
            for choice in obj.get("choices", []):
                tokens = choice.get("token_ids") or []
                if tokens:
                    ids.extend(tokens)
                    events.append([time.perf_counter(), len(ids)])
                    if on_tokens is not None:
                        on_tokens(lane, len(ids))
                delta = choice.get("delta") or {}
                text.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
                finish = choice.get("finish_reason") or finish
    if not done or not usage or not finish or len(events) < 2:
        raise RuntimeError(f"Incomplete stream: done={done}, usage={usage}, finish={finish}, events={len(events)}")
    if usage["completion_tokens"] != len(ids):
        raise RuntimeError(f"Token usage mismatch: usage={usage}, ids={len(ids)}")
    clock_end_ns = [time.time_ns(), time.perf_counter_ns(), time.time_ns()]
    return dict(lane=lane, start=started, end=clock_end_ns[1] / 1e9, usage=usage,
                clock_start_ns=clock_start_ns, clock_end_ns=clock_end_ns,
                finish=finish, events=events, token_ids=ids, prompt_token_ids=prompt_ids,
                content="".join(text), reasoning="".join(reasoning), request=body)


def summarize(rows):
    start = max(row["events"][0][0] for row in rows)
    end = min(row["events"][-1][0] for row in rows)
    if end <= start:
        raise RuntimeError("No common decode interval")
    counts = []
    for row in rows:
        before = max((n for t, n in row["events"] if t <= start), default=0)
        after = max((n for t, n in row["events"] if t <= end), default=0)
        counts.append(after - before)
    rates = [(r["events"][-1][1] - r["events"][0][1]) /
             (r["events"][-1][0] - r["events"][0][0]) for r in rows]
    return dict(common_decode_seconds=end-start, common_decode_tokens=sum(counts),
                aggregate_tok_s=sum(counts)/(end-start),
                per_request_tok_s=sum(counts)/(end-start)/len(rows),
                full_request_decode_tok_s=rates,
                output_tokens=[r["usage"]["completion_tokens"] for r in rows],
                prompt_tokens=[r["usage"]["prompt_tokens"] for r in rows],
                finishes=[r["finish"] for r in rows])


async def main(args):
    args.prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else PROMPT
    args.output.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800), trust_env=False) as client:
        for url in [args.url, *args.peer_url]:
            for attempt in range(720):
                try:
                    response = await client.get(url + "/health", timeout=2)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(5)
            else:
                raise RuntimeError(f"{url} did not become healthy within one hour")
        for c in args.concurrency:
            for repeat in range(-1, args.repeats):
                warmup = repeat == -1
                metrics = (await client.get(args.url + "/metrics")).text
                rows = await asyncio.gather(*(request(client, args, i, repeat,
                    128 if warmup else args.max_tokens) for i in range(c)))
                try:
                    measurement = summarize(rows)
                except RuntimeError:
                    if not warmup:
                        raise
                    measurement = dict(
                        output_tokens=[r['usage']['completion_tokens'] for r in rows],
                        prompt_tokens=[r['usage']['prompt_tokens'] for r in rows],
                        finishes=[r['finish'] for r in rows])
                    print('Warmup requests completed without a shared decode interval.', flush=True)
                result = dict(engine=args.engine, spec=args.spec, concurrency=c,
                              repeat=repeat, warmup=warmup, **measurement)
                stem = f"{args.engine}-spec-{args.spec}-c{c}-r{repeat}"
                (args.output / f"{stem}.json").write_text(json.dumps(dict(summary=result, requests=rows)))
                (args.output / f"{stem}-metrics-before.txt").write_text(metrics)
                (args.output / f"{stem}-metrics-after.txt").write_text((await client.get(args.url + "/metrics")).text)
                with (args.output / "results.jsonl").open("a") as f:
                    f.write(json.dumps(result) + "\n")
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--peer-url", action="append", default=[])
    p.add_argument("--model", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--spec", choices=("on", "off"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--prompt-file", type=Path, help="UTF-8 user prompt; defaults to the programming task")
    p.add_argument("--max-tokens", type=int, default=4096, help="measured request output budget")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    args = p.parse_args()
    if args.max_tokens < 1:
        p.error('--max-tokens must be positive')
    asyncio.run(main(args))
