# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify fatal model-worker HTTP propagation using malformed graph completions.

The graph catalog implements the transport shapes but no model arithmetic. Its
initialization graph writes a nonzero reserved completion field. The installed
executor must reject that record, terminate its model worker, fail the waiting
response, and expose unhealthy status. Production code is not patched.

Without --run this prints the CPU-only case plan and performs no CUDA or HTTP
operations. An explicitly assigned CUDA_VISIBLE_DEVICES UUID is required to run.
"""

from __future__ import annotations

import argparse
import gc
import http.client
import json
import os
from pathlib import Path
import sys
import time
import weakref

from qualify_startup_validation import ABI_VERSION, SNAPSHOT, fixture

ERROR = "invalid or stale device completion record"


def plan():
    return {"scope": "Real native ModelRuntime and HTTP failure propagation over a deliberately malformed tiny CUDA graph catalog; no checkpoint inference.",
        "abi_version": ABI_VERSION,
        "cases": ["waiting-json", "waiting-sse"],
        "trigger": "Initialization graph writes 1 into completion_records[:, :, 7] (reserved ABI field).",
        "path": ["GraphExecutor::poll/decode_records", "run_loop returns Err",
            "ModelRuntime marks unhealthy and fails active/pending/queued requests",
            "executor drains model/copy streams", "HTTP health returns 503"],
        "checks": ["health 200 before submission", "one real initialization graph submitted",
            "waiting JSON returns 500 with exact fatal error",
            "waiting SSE returns 200 with one terminal error, no [DONE] or successful finish",
            "health 503 after fatal worker exit", "subsequent chat returns 503",
            "Server.healthy and wait are false", "close retains error and confirms drain",
            "owner retained until close then released", "CPU scheduler resources fully returned"],
        "http_timeout_seconds": 30,
        "gpu_policy": "One explicitly assigned UUID; no model weights, no existing endpoint contacted.",
        "limitations": ["Does not force a Rust panic or a CUDA execution fault.",
            "Does not validate checkpoint outputs, multi-request error fanout, or failed CUDA drain."]}


def catalog(snapshot):
    import torch
    from ignition.preparation.execution import GraphSignature, PreparedExecution

    metadata, config = fixture(snapshot)
    buffers = {item["name"]: torch.zeros(item["shape"], dtype=getattr(torch, item["dtype"]), device="cuda")
        for item in metadata["buffers"]}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs, signatures = {}, {}
    for item in metadata["graphs"]:
        inputs = tuple(binding["buffer_name"] for binding in item["input_bindings"]
                       if binding["buffer_name"] not in ("frame_index", "submission_sequence"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            buffers["completion_records"][:, :, 7].fill_(1)
        graphs[item["name"]] = graph
        signatures[item["name"]] = GraphSignature(item["kind"], item["batch_capacity"],
            item["token_capacity"], inputs)
    torch.cuda.synchronize()
    owner = PreparedExecution(device=0, graphs=graphs, buffers=buffers,
        signatures=signatures, resources=(stream,))
    return owner, config


class Checks:
    def __init__(self):
        self.count = 0

    def check(self, case, name, passed, **details):
        print(json.dumps(dict(case=case, name=name, passed=bool(passed), **details)), flush=True)
        assert passed, f"{case}: {name}: {details}"
        self.count += 1

    def http(self, address, label, path, body=None):
        connection = http.client.HTTPConnection(address, timeout=30)
        started = time.monotonic()
        try:
            connection.request("GET" if body is None else "POST", path,
                body=None if body is None else json.dumps(body),
                headers={} if body is None else {"Content-Type": "application/json"})
            response = connection.getresponse()
            return {"status": response.status, "headers": dict(response.getheaders()),
                "body": response.read().decode("utf-8"), "elapsed_seconds": time.monotonic() - started}
        finally:
            connection.close()


def qualify_case(evidence, snapshot, stream):
    import flare_heliopause

    case = "waiting-sse" if stream else "waiting-json"
    owner, config = catalog(snapshot)
    manifest = owner.manifest()
    retained = weakref.ref(owner)
    server = flare_heliopause.Server(owner, manifest, json.dumps(config))
    del owner
    gc.collect()
    close_error = None
    try:
        evidence.check(case, "owner-retained-after-handoff", retained() is not None)
        before = evidence.http(server.address, case + "-health-before", "/health")
        evidence.check(case, "ready-before-submission", before["status"] == 200 and server.healthy)
        body = {"model": config["model_id"], "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 2, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}
        response = evidence.http(server.address, case, "/v1/chat/completions", body)
        if stream:
            values = [line[5:].strip() for line in response["body"].splitlines() if line.startswith("data:")]
            events = [json.loads(value) for value in values if value != "[DONE]"]
            errors = [event for event in events if "error" in event]
            successful_finish = any(choice.get("finish_reason") is not None
                for event in events for choice in event.get("choices", []))
            evidence.check(case, "waiting-sse-fails-once", response["status"] == 200 and len(errors) == 1
                and errors[0]["error"]["message"] == ERROR and events[-1] == errors[0]
                and "[DONE]" not in values and not successful_finish)
        else:
            evidence.check(case, "waiting-json-fails", response["status"] == 500
                and json.loads(response["body"]).get("error", {}).get("message") == ERROR)
        evidence.check(case, "worker-exit-visible", not server.wait(timeout_ms=30000) and not server.healthy)
        after = evidence.http(server.address, case + "-health-after", "/health")
        evidence.check(case, "health-unavailable", after["status"] == 503
            and json.loads(after["body"]).get("error", {}).get("message") == "prepared model runtime is unavailable")
        rejected = evidence.http(server.address, case + "-after-failure", "/v1/chat/completions", body)
        evidence.check(case, "subsequent-request-rejected", rejected["status"] == 503)
        metrics = evidence.http(server.address, case + "-metrics", "/metrics")
        counters = {line.split()[0]: int(line.split()[1]) for line in metrics["body"].splitlines()
            if line and not line.startswith("#")}
        evidence.check(case, "one-fatal-initialization", counters.get('flare_graph_launches_total{kind="init"}') == 1
            and counters.get("flare_requests_admitted_total") == 1
            and counters.get("flare_requests_failed_total") == 1, counters=counters)
        evidence.check(case, "resources-returned-after-drain", counters.get("flare_free_kv_pages") == config["pages"]
            and counters.get("flare_free_snapshots") == config["snapshots"]
            and counters.get("flare_free_request_slots") == 4)
        evidence.check(case, "owner-held-before-close", retained() is not None)
    finally:
        try:
            server.close()
        except RuntimeError as error:
            close_error = str(error)
        evidence.check(case, "close-confirms-drain-and-error", server.drained and close_error == ERROR,
            drained=server.drained, error=close_error)
        gc.collect()
        evidence.check(case, "owner-released-after-drain", retained() is None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--device-uuid")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(plan(), indent=2))
        return 0
    if not args.device_uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != args.device_uuid:
        parser.error("--run requires --device-uuid matching CUDA_VISIBLE_DEVICES")
    import torch
    import flare_heliopause

    torch.cuda.set_device(0)
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if actual_uuid.removeprefix("GPU-") != args.device_uuid.removeprefix("GPU-"):
        raise RuntimeError(f"CUDA UUID {actual_uuid} does not match assigned {args.device_uuid}")
    checks = Checks()
    for stream in (False, True):
        qualify_case(checks, args.snapshot, stream)
    assert checks.count == 22
    print(json.dumps({"passed": True, "checks": checks.count, "device_uuid": actual_uuid}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
