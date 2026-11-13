# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect CUDA-node and CPU timelines during the streamed decode workload."""

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import shlex
import time

import httpx

from benchmark_ds41_decode import PROMPT, request, summarize


def cpu_snapshot(roots):
    started = time.time_ns()
    pending, seen, tasks = list(roots), set(), []
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        process = Path('/proc') / str(pid)
        for task in sorted((process / 'task').iterdir()):
            pending.extend(map(int, (task / 'children').read_text().split()))
            before = time.time_ns()
            runtime, delay, slices = map(int, (task / 'schedstat').read_text().split())
            after = time.time_ns()
            switches = {k: int(v.strip()) for line in (task / 'status').read_text().splitlines()
                        if line.startswith(('voluntary_ctxt_switches:', 'nonvoluntary_ctxt_switches:'))
                        for k, v in [line.split(':', 1)]}
            tasks.append(dict(pid=pid, tid=int(task.name), name=(task / 'comm').read_text().strip(),
                              read_before_ns=before, read_after_ns=after, runtime_ns=runtime,
                              runqueue_delay_ns=delay, timeslices=slices, **switches))
    return dict(start_ns=started, end_ns=time.time_ns(), tasks=tasks,
                schedstats_enabled=Path('/proc/sys/kernel/sched_schedstats').read_text().strip() == '1')


async def command(*argv):
    process = await asyncio.create_subprocess_exec(
        *map(str, argv), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    output, _ = await process.communicate()
    print(output.decode(), flush=True)
    if process.returncode:
        raise RuntimeError(f"{argv[0]} exited {process.returncode}")


async def capture_nsys(args, stem, host=None):
    async def run(*argv):
        if host:
            await command("ssh", host, shlex.join(map(str, argv)))
        else:
            await command(*argv)

    if host:
        await run("mkdir", "-p", "--", args.output)
    prefix = ("sudo", "-n", "nsys") if host else ("nsys",)
    await run(*prefix, "start", "--session", args.nsys_session,
              f"--sample={args.nsys_cpu_sampling}", "--backtrace=dwarf",
              f"--cpuctxsw={args.nsys_cpu_sampling}", "--export=sqlite",
              "--output", args.output / stem)
    try:
        await asyncio.sleep(args.duration)
    finally:
        await run(*prefix, "stop", "--session", args.nsys_session)


async def main(args):
    args.prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else PROMPT
    args.output.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800), trust_env=False) as client:
        for url in [args.url, *args.peer_url]:
            for _ in range(720):
                try:
                    if (await client.get(url + "/health", timeout=2)).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(5)
            else:
                raise RuntimeError(f"{url} did not become healthy")
        for c in args.concurrency:
            await asyncio.gather(*(request(client, args, lane, -1, 128) for lane in range(c)))
            counts = [0] * c
            ready = asyncio.Event()

            def progress(lane, tokens):
                counts[lane] = tokens
                if min(counts) >= args.arm_tokens:
                    ready.set()

            stem = f"{args.engine}-spec-{args.spec}-c{c}"

            async def capture():
                await ready.wait()
                if args.profiler == "cpu":
                    before = cpu_snapshot(args.cpu_pid)
                    await asyncio.sleep(args.duration)
                    after = cpu_snapshot(args.cpu_pid)
                    (args.output / f"{stem}-cpu.json").write_text(json.dumps(dict(before=before, after=after)))
                elif args.profiler == "nsys":
                    captures = await asyncio.gather(
                        *(capture_nsys(args, stem, host) for host in args.nsys_host or [None]),
                        return_exceptions=True)
                    for result in captures:
                        if isinstance(result, BaseException):
                            raise result
                elif args.profiler == "cuda":
                    print("Starting CUDA profiler range.", flush=True)
                    (await client.post(args.url + "/start_profile")).raise_for_status()
                    await asyncio.sleep(args.duration)
                    (await client.post(args.url + "/stop_profile")).raise_for_status()
                    print("Stopped CUDA profiler range.", flush=True)
                else:
                    cpu_session = stem + "-cpu"
                    await command("sudo", "-n", "nsys", "start", "--session-new", cpu_session,
                                  "--sample=system-wide", "--backtrace=dwarf",
                                  "--cpuctxsw=system-wide", "--export=sqlite",
                                  "--output", args.output / cpu_session)
                    spies = [asyncio.create_task(command(
                        "sudo", "-n", shutil.which("py-spy") or "py-spy", "record",
                        "--pid", pid, "--duration", "3", "--rate", "100",
                        "--native", "--format", "speedscope",
                        "--output", args.output / f"{stem}-pid{pid}.speedscope.json"))
                        for pid in args.py_spy_pid]
                    try:
                        (await client.post(args.url + "/start_profile")).raise_for_status()
                        await asyncio.sleep(args.duration)
                        (await client.post(args.url + "/stop_profile")).raise_for_status()
                        await asyncio.gather(*spies)
                    finally:
                        await command("sudo", "-n", "nsys", "stop", "--session", cpu_session)

            before = (await client.get(args.url + "/metrics")).text
            rows, _ = await asyncio.gather(
                asyncio.gather(*(request(client, args, lane, 0, args.max_tokens, progress) for lane in range(c))),
                capture())
            result = dict(engine=args.engine, spec=args.spec, concurrency=c, **summarize(rows))
            (args.output / f"{stem}.json").write_text(json.dumps(dict(summary=result, requests=rows)))
            (args.output / f"{stem}-metrics-before.txt").write_text(before)
            (args.output / f"{stem}-metrics-after.txt").write_text((await client.get(args.url + "/metrics")).text)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--peer-url", action="append", default=[])
    p.add_argument("--model", default="DeepSeek-V4.1-Flash")
    p.add_argument("--engine", choices=("flare", "vllm"), required=True)
    p.add_argument("--spec", choices=("on", "off"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profiler", choices=("nsys", "torch", "cuda", "cpu"), default="nsys")
    p.add_argument("--cpu-pid", type=int, action="append", default=[])
    p.add_argument("--nsys-session")
    p.add_argument("--nsys-cpu-sampling", choices=("none", "process-tree"), default="process-tree",
                   help="CPU sampling and context-switch collection for nsys sessions; "
                        "none disables both (default: process-tree).")
    p.add_argument("--nsys-host", action="append", default=[],
                   help="SSH host running the named nsys session; repeat for multiple ranks. "
                        "Remote captures use sudo and save traces under --output on each host.")
    p.add_argument("--py-spy-pid", type=int, action="append", default=[])
    p.add_argument("--duration", type=float, default=1)
    p.add_argument("--arm-tokens", type=int, default=256)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--prompt-file", type=Path, help="UTF-8 user prompt; defaults to the programming task")
    p.add_argument("--max-tokens", type=int, default=4096, help="profiled request output budget")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    args = p.parse_args()
    if args.profiler == 'cpu' and not args.cpu_pid:
        p.error('--profiler cpu requires at least one --cpu-pid')
    if args.arm_tokens < 1 or args.max_tokens <= args.arm_tokens:
        p.error('--max-tokens must exceed the positive --arm-tokens threshold')
    asyncio.run(main(args))
