# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch-one GLM HTTP decode measurements and CUDA graph tracing.

Create one corpus and reuse it for both engines. The ``flare`` command prepares
the production rank group and runs the client against Helios. The ``http``
command targets an HTTP server exposing CUDA profiler controls. Each runs
one warmup, three unprofiled requests, and three requests inside the profiler
capture range. Nsight Systems should use graph tracing and cudaProfilerApi.
"""

import argparse
import ctypes
import http.client
import json
import os
import statistics
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def corpus(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    cases = []
    for index in range(7):
        prefix = (f"Performance request {index}. Write a detailed numbered list of at least 400 programming facts. "
                  "Continue the list without a conclusion. Background notes:")
        message = {"role": "user", "content": prefix}
        count = 400
        for _ in range(10):
            message["content"] = prefix + " apple" * count
            tokens = tokenizer.apply_chat_template([message], tokenize=True, return_dict=False,
                add_generation_prompt=True, reasoning_effort="low")
            if len(tokens) == 512:
                break
            count += 512 - len(tokens)
        else:
            raise ValueError("cannot construct the 512-token prompt")
        request = {"model": args.snapshot.name, "messages": [message], "reasoning_effort": "low",
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 42, "max_tokens": 256,
            "stream": True, "return_token_ids": True, "stream_options": {"include_usage": True}}
        cases.append({"index": index, "phase": "warmup" if index == 0 else "unprofiled" if index < 4 else "profiled",
                      "request": request, "prompt_tokens": tokens})
    save(args.output, {"snapshot": str(args.snapshot.resolve()), "cases": cases})


def connect(base):
    url = urlsplit(base)
    if url.scheme != "http" or url.path not in ("", "/"):
        raise ValueError("require an HTTP origin")
    return closing(http.client.HTTPConnection(url.hostname, url.port, timeout=180))


def profile_http(base, start):
    with connect(base) as client:
        client.request("POST", "/start_profile" if start else "/stop_profile")
        response = client.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"profiler HTTP {response.status}: {body.decode()}")


def request(base, case, *, native, model, output):
    body = dict(case["request"])
    body["model"] = model
    if native:
        body["return_token_timestamps"] = True
    events, tokens, delivered, core, prompts, finishes = [], [], [], [], [], []
    usage = None
    done = False
    with connect(base) as client:
        client.connect()
        start = time.perf_counter_ns()
        client.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = client.getresponse()
        if response.status != 200:
            raise RuntimeError(f"generation HTTP {response.status}: {response.read().decode()}")
        while line := response.readline():
            if not line.startswith(b"data:"):
                continue
            value = line[5:].strip()
            elapsed = time.perf_counter_ns() - start
            if value == b"[DONE]":
                done = True
                break
            event = json.loads(value)
            events.append({"elapsed_ns": elapsed, "data": event})
            if "error" in event:
                save(output, events)
                raise RuntimeError(event["error"])
            if event.get("prompt_token_ids") is not None:
                prompts.append(event["prompt_token_ids"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                ids = choice.get("token_ids") or []
                tokens.extend(ids)
                delivered.extend([elapsed] * len(ids))
                core.extend(choice.get("token_timestamps_ns") or [])
                if choice.get("finish_reason"):
                    finishes.append(choice["finish_reason"])
    save(output, events)
    if not done or prompts != [case["prompt_tokens"]] or len(tokens) != 256 or finishes != ["length"]:
        raise RuntimeError(f"workload differs: done={done}, prompt_match={prompts == [case['prompt_tokens']]}, "
                           f"outputs={len(tokens)}, finish={finishes}; see {output}")
    if usage is None or usage["prompt_tokens"] != 512 or usage["completion_tokens"] != 256:
        raise RuntimeError(f"usage differs: {usage}")
    if native and len(core) != 256:
        raise RuntimeError("native completion timestamps are incomplete")
    def timing(times):
        intervals = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
        return {"mean_ms": statistics.mean(intervals), "median_ms": statistics.median(intervals),
                "p95_ms": sorted(intervals)[int(len(intervals) * .95)],
                "tokens_per_second": 255e9 / (times[-1] - times[0])}
    result = {"index": case["index"], "phase": case["phase"], "ttft_ms": delivered[0] / 1e6,
              "delivery": timing(delivered), "usage": usage}
    if core:
        result["core"] = timing(core)
    print(json.dumps(result), flush=True)
    return result


def measure(args, start, stop, *, native):
    data = json.loads(args.corpus.read_text())
    if Path(data["snapshot"]) != args.snapshot.resolve():
        raise ValueError("corpus checkpoint differs")
    with connect(args.base) as client:
        client.request("GET", "/v1/models")
        response = client.getresponse()
        if response.status != 200:
            raise RuntimeError(f"model discovery HTTP {response.status}: {response.read().decode()}")
        model, = json.load(response)["data"]
    results, profiling = [], False
    try:
        for case in data["cases"]:
            if case["phase"] == "profiled" and not profiling:
                start()
                profiling = True
            results.append(request(args.base, case, native=native, model=model["id"],
                                   output=args.output / f"request-{case['index']}.json"))
            save(args.output / "results.json", results)
    finally:
        if profiling:
            stop()


def flare(args):
    import torch

    from ignition.preparation.rank_group import prepare_group
    from ignition.recipes.glm53.bootstrap import PreparationConfig, prepare_rank
    from qualify.performance_trace import loaded_cupti_path

    config = PreparationConfig(devices=tuple(range(args.tp)), context=16384,
        prefill_chunk_size=4096, pages=256, snapshot_budget_bytes=1 << 30)
    with prepare_group(args.snapshot, config, prepare_rank, progress=lambda s: print(s, flush=True)) as group:
        save(args.output / "memory.json", group.memory_report)
        library = ctypes.CDLL(str(loaded_cupti_path(Path("/proc/self/maps").read_text())),
                              mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        query = library.cuptiGetGraphExecId
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        query.restype = ctypes.c_int
        graphs = []
        for graph in group.manifest()["graphs"]:
            identifier = ctypes.c_uint32()
            status = query(ctypes.c_void_p(graph["handle"]), ctypes.byref(identifier))
            if status:
                raise RuntimeError(f"cuptiGetGraphExecId returned {status}")
            graphs.append({"name": graph["name"], "id": identifier.value})
        save(args.output / "graphs.json", {"pid": os.getpid(), "graphs": graphs})
        server = group.start_server(bind=urlsplit(args.base).netloc)
        print(f"Listening on {server.address}", flush=True)
        measure(args, torch.cuda.profiler.start, torch.cuda.profiler.stop, native=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("corpus", "flare", "http"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--tp", type=int, choices=(2, 4), default=2)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be fresh")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "corpus":
        return corpus(args)
    if not args.corpus:
        parser.error("require --corpus")
    args.output.mkdir()
    if args.mode == "flare":
        flare(args)
    else:
        measure(args, lambda: profile_http(args.base, True), lambda: profile_http(args.base, False), native=False)


if __name__ == "__main__":
    main()
