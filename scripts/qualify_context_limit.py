# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the exact Qwen context boundary through the native HTTP server."""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import http.client
import importlib.metadata
import json
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qualify.server import Run, digest, metric_delta, resource_totals, write_json


MODEL = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD"
REVISION = "dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2"
NATIVE_SHA256 = "735bf767999b26affeacc5ae574b6e52c6a5e1dc18f42cc208ee8b9f0b2a18ab"
CONTEXT = 262_144
PROMPT_TOKENS = 262_136
OUTPUT_TOKENS = 8
PAGE_TOKENS = 64
PREFILL_CAPACITY = 512
LITERAL = "x " * 128 + "END"


def token_bytes(tokenizer_json):
    """Decode the checkpoint's ByteLevel vocabulary without Unicode replacement."""
    if tokenizer_json["decoder"]["type"] != "ByteLevel":
        raise ValueError("qualification requires the pinned ByteLevel tokenizer")
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = list(values)
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + extra)
            extra += 1
    inverse = {chr(character): value for character, value in zip(characters, values, strict=True)}
    result = {index: bytes(inverse[character] for character in word)
              for word, index in tokenizer_json["model"]["vocab"].items()}
    for token in tokenizer_json["added_tokens"]:
        if token["special"]:
            result.pop(token["id"], None)
        else:
            result[token["id"]] = token["content"].encode("utf-8")
    return result


def minimum_literal_tokens(literal, vocabulary):
    data = literal.encode("ascii")
    candidates = sorted({word for word in vocabulary.values() if word and word in data})
    unreachable = len(data) + 1
    best = [0] + [unreachable] * len(data)
    for offset in range(len(data)):
        if best[offset] == unreachable:
            continue
        for word in candidates:
            if data.startswith(word, offset):
                end = offset + len(word)
                best[end] = min(best[end], best[offset] + 1)
    if best[-1] == unreachable:
        raise ValueError("literal cannot be expressed by the ordinary checkpoint vocabulary")
    return {"minimum_tokens": best[-1], "candidate_token_byte_strings": len(candidates),
            "literal_utf8_bytes": len(data), "method": "Exact shortest segmentation over non-special checkpoint token bytes."}


def load_tokenizer(snapshot):
    import tokenizers
    import transformers

    fixture = json.loads((ROOT / "tests/frontend/qwen38.json").read_text())
    assets = {name: digest(snapshot / name) for name in fixture["assets"]}
    if assets != fixture["assets"]:
        raise ValueError("snapshot tokenizer/template assets differ from the pinned fixture")
    origin = json.loads(importlib.metadata.distribution("transformers").read_text("direct_url.json"))
    if (origin.get("vcs_info", {}).get("commit_id") != fixture["transformers_commit"]
            or tokenizers.__version__ != fixture["tokenizers_version"]):
        raise ValueError("qualification requires pinned Transformers and tokenizers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    return tokenizer, {"assets_sha256": assets, "transformers_version": transformers.__version__,
        "transformers_origin": origin, "tokenizers_version": tokenizers.__version__}


def render(tokenizer, body, *, tokenize):
    return tokenizer.apply_chat_template(body["messages"], tokenize=tokenize,
        return_dict=False, enable_thinking=False, add_generation_prompt=True)


def prepare(args):
    tokenizer, provenance = load_tokenizer(args.snapshot)
    plan_path = args.artifacts / "case-plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["model"] != MODEL or plan["revision"] != REVISION or plan["provenance"] != provenance:
            raise ValueError("prepared fixture identity differs from the pinned checkpoint")
        for name, expected in plan["artifacts_sha256"].items():
            if digest(args.artifacts / name) != expected:
                raise ValueError(f"prepared fixture changed: {name}")
        return tokenizer, plan, json.loads((args.artifacts / "valid-request.json").read_text())
    if args.artifacts.exists() and any(args.artifacts.iterdir()):
        raise ValueError("fixture directory is nonempty without a case plan")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    prefix = f"Context address limit fixture {nonce}."
    body = {"model": MODEL, "messages": [{"role": "user", "content": prefix}],
        "temperature": 0, "seed": 42, "max_tokens": OUTPUT_TOKENS,
        "enable_thinking": False, "add_generation_prompt": True, "stream": False,
        "grammar": "start: " + json.dumps(LITERAL)}
    count = PROMPT_TOKENS - len(render(tokenizer, body, tokenize=True))
    for _ in range(4):
        body["messages"][0]["content"] = prefix + " x" * count
        ids = render(tokenizer, body, tokenize=True)
        if len(ids) == PROMPT_TOKENS:
            break
        count += PROMPT_TOKENS - len(ids)
    else:
        raise ValueError("failed to construct the exact requested prompt length")
    text = render(tokenizer, body, tokenize=False)
    if tokenizer.encode(text, add_special_tokens=False) != ids:
        raise ValueError("rendered prompt bytes and template token IDs disagree")
    grammar = minimum_literal_tokens(LITERAL, token_bytes(json.loads((args.snapshot / "tokenizer.json").read_text())))
    if grammar["minimum_tokens"] <= OUTPUT_TOKENS:
        raise ValueError("grammar can finish before the required generation budget")
    encoded = array("I", ids)
    if encoded.itemsize != 4:
        raise ValueError("uint32 prompt serialization requires four-byte unsigned ints")
    if sys.byteorder != "little":
        encoded.byteswap()
    (args.artifacts / "prompt-token-ids.u32le").write_bytes(encoded.tobytes())
    write_json(args.artifacts / "prompt-token-ids.json", ids)
    (args.artifacts / "rendered-prompt.txt").write_text(text)
    write_json(args.artifacts / "valid-request.json", body)
    invalid = {**body, "max_tokens": OUTPUT_TOKENS + 1}
    write_json(args.artifacts / "invalid-request.json", invalid)
    names = ("prompt-token-ids.u32le", "prompt-token-ids.json", "rendered-prompt.txt", "valid-request.json", "invalid-request.json")
    plan = {"schema": "flare-context-limit-fixture-v1", "model": MODEL, "revision": REVISION,
        "context_tokens": CONTEXT, "prompt_tokens": len(ids), "max_new_tokens": OUTPUT_TOKENS,
        "invalid_prompt_plus_output": len(ids) + invalid["max_tokens"], "nonce": nonce,
        "expected_consumed_tokens": PROMPT_TOKENS + OUTPUT_TOKENS - 1,
        "expected_prefill_launches": (PROMPT_TOKENS + PREFILL_CAPACITY - 1) // PREFILL_CAPACITY,
        "expected_decode_launches": OUTPUT_TOKENS - 1, "reserved_pages": CONTEXT // PAGE_TOKENS,
        "terminal_checkpoint_tokens": PROMPT_TOKENS + OUTPUT_TOKENS - 1,
        "grammar": {"literal": LITERAL, **grammar}, "provenance": provenance,
        "token_ids_semantics": "Pinned canonical chat-template rendering; HTTP usage independently checks the prompt length.",
        "artifacts_sha256": {name: digest(args.artifacts / name) for name in names}}
    write_json(plan_path, plan)
    return tokenizer, plan, body


class ContextPost:
    def __init__(self, run, body, deadline):
        self.run, self.body, self.deadline = run, body, deadline
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.connection = None
        self.transport_socket = None
        self.record = None
        self.expired = False

    def abort(self):
        with self.lock:
            self.expired = True
            if self.connection is not None:
                transport = self.transport_socket or self.connection.sock
                if transport is not None:
                    try:
                        transport.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                self.connection.close()

    def execute(self):
        run = self.run
        with run.lock:
            run.counter += 1
            stem = f"{run.counter:03d}-context-boundary"
        request_path = run.args.artifacts / f"{stem}.request.json"
        write_json(request_path, self.body)
        started = time.monotonic()
        record = {"name": "context-boundary", "method": "POST", "path": "/v1/chat/completions",
            "stream": False, "started_monotonic": started, "request_artifact": str(request_path)}
        raw_path = run.args.artifacts / f"{stem}.response.json"
        connection = None
        try:
            connection = http.client.HTTPConnection(run.url.hostname, run.url.port,
                timeout=max(0.001, self.deadline - time.monotonic()))
            with self.lock:
                if self.expired:
                    raise TimeoutError("context request deadline expired before dispatch")
                self.connection = connection
            encoded = json.dumps(self.body, ensure_ascii=False).encode()
            connection.request("POST", run.url.path.rstrip("/") + record["path"], body=encoded,
                headers={"Content-Type": "application/json", "Accept": "application/json"})
            with self.lock:
                self.transport_socket = connection.sock
                if self.expired:
                    raise TimeoutError("context request deadline expired during dispatch")
            response = connection.getresponse()
            record.update(status=response.status, headers=dict(response.getheaders()),
                headers_seconds=time.monotonic() - started, response_artifact=str(raw_path))
            raw = response.read(run.args.max_response_bytes + 1)
            raw_path.write_bytes(raw)
            if len(raw) > run.args.max_response_bytes:
                raise ValueError("context JSON response exceeded byte bound")
            record["response_sha256"] = digest(raw_path)
            record["body"] = json.loads(raw)
        except Exception as error:
            record["client_error"] = f"{type(error).__name__}: {error}"
        finally:
            if connection is not None:
                connection.close()
            record.update(elapsed_seconds=time.monotonic() - started,
                finished_monotonic=time.monotonic(), hard_deadline_expired=self.expired
                or ("client_error" in record and time.monotonic() >= self.deadline))
            with run.lock:
                run.results["requests"].append(record)
            self.record = record
            self.done.set()


def idle(run, label, timeout=60):
    deadline = time.monotonic() + timeout
    index = 0
    while True:
        metrics = run.metrics(f"{label}-{index}", wait_idle=False)
        if all(metrics[key] == 0 for key in ("flare_active_requests", "flare_retiring_request_slots",
                "flare_pending_requests", "flare_http_inflight_requests")):
            return metrics
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{label}: model did not reach idle within {timeout}s")
        index += 1
        time.sleep(0.25)


def execute(args, tokenizer, plan, body):
    memory = json.loads(args.memory_report.read_text())
    if memory["cached_token_capacity"] != memory["pages"] * PAGE_TOKENS or memory["pages"] < plan["reserved_pages"]:
        raise ValueError("memory report cannot represent the full requested context")
    args.require_metrics, args.cases = True, None
    args.timeout = 5.0
    run = Run(args)
    run.results["progress"] = []
    active = None
    try:
        models = run.request("models", path="/v1/models")
        loaded = [item for item in models.get("body", {}).get("data", []) if item.get("id") == MODEL]
        run.check("loaded_model_identity", models.get("status") == 200 and len(loaded) == 1,
            models=models.get("body"))
        if not run.results["checks"][-1]["passed"]:
            raise ValueError("server model identity differs from the exact case")
        before = idle(run, "before")
        totals = resource_totals(before)
        run.check("memory_report_matches_native_pool", totals == {"pages": memory["pages"],
            "snapshots": memory["snapshot_capacity"], "slots": 4}, observed=totals)
        invalid = run.request("context-total262145", {**body, "max_tokens": OUTPUT_TOKENS + 1})
        run.check("reject_total262145", invalid.get("status") == 400
            and "exceeds this server's context of" in invalid.get("body", {}).get("error", {}).get("message", ""),
            response=invalid)
        rejected = idle(run, "after-rejection")
        reject_delta = metric_delta(before, rejected)
        run.check("rejection_no_model_admission", reject_delta["flare_requests_admitted_total"] == 0
            and all(reject_delta[f'flare_graph_launches_total{{kind="{kind}"}}'] == 0
                for kind in ("init", "restore", "resume", "prefill", "decode", "vision", "snapshot")), delta=reject_delta)
        if not all(check["passed"] for check in run.results["checks"]):
            raise ValueError("context prerequisites failed; exact input remains preserved")
        before = rejected
        started = time.monotonic()
        deadline = started + args.deadline_seconds
        active = ContextPost(run, body, deadline)
        thread = threading.Thread(target=active.execute, name="context-limit-http", daemon=True)
        thread.start()
        while not active.done.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                active.abort()
                if not active.done.wait(10):
                    raise TimeoutError("HTTP worker did not drain after the context deadline")
                break
            metrics = run.metrics(f"context-progress-{len(run.results['progress']):04d}", wait_idle=False)
            delta = metric_delta(before, metrics)
            submitted = delta['flare_graph_launches_total{kind="prefill"}']
            sample = {"elapsed_seconds": time.monotonic() - started, "metrics": metrics,
                "prefill_graphs_submitted": submitted,
                "completed_prefill_token_lower_bound": max(0, submitted - 1) * PREFILL_CAPACITY}
            with run.lock:
                run.results["progress"].append(sample)
            print(json.dumps({"event": "context_progress", "elapsed_seconds": round(sample["elapsed_seconds"], 2),
                "prefill_graphs": submitted, "expected_prefill_graphs": plan["expected_prefill_launches"],
                "active_requests": metrics["flare_active_requests"]}), flush=True)
            active.done.wait(min(args.poll_seconds, max(0, deadline - time.monotonic())))
        thread.join(timeout=1)
        result = active.record
        if result is None:
            raise RuntimeError("context HTTP worker has no final response record")
        after = idle(run, "after-context")
        delta = metric_delta(before, after)
        response = result.get("body", {})
        usage = response.get("usage", {})
        choices = response.get("choices", [])
        run.check("context_complete_exact_usage", result.get("status") == 200 and "client_error" not in result
            and "error" not in response and usage == {"prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": OUTPUT_TOKENS, "total_tokens": CONTEXT}, usage=usage, response=result)
        content = choices[0].get("message", {}).get("content", "") if len(choices) == 1 else ""
        run.check("context_finishes_at_length", len(choices) == 1 and choices[0].get("finish_reason") == "length"
            and bool(content) and LITERAL.startswith(content) and content != LITERAL,
            content=content, choices=choices, grammar_minimum_tokens=plan["grammar"]["minimum_tokens"])
        run.check("context_executes_all_prefill_and_decode", delta['flare_graph_launches_total{kind="prefill"}'] == 512
            and delta['flare_graph_launches_total{kind="decode"}'] == 7
            and delta['flare_graph_launches_total{kind="init"}'] == 1
            and delta['flare_graph_launches_total{kind="snapshot"}'] == 1
            and all(delta[f'flare_graph_launches_total{{kind="{kind}"}}'] == 0 for kind in ("restore", "resume", "vision"))
            and delta["flare_cache_hits_total"] == delta["flare_cache_reused_tokens_total"] == 0,
            delta=delta, derived_consumed_tokens=262143, last_consumed_zero_based_position=262142,
            final_sample_pending=True, pending_sample_zero_based_position=262143)
        run.check("context_no_request_errors", delta["flare_requests_admitted_total"] == 1
            and delta["flare_requests_completed_total"] == 1
            and delta["flare_requests_failed_total"] == delta["flare_requests_cancelled_total"] == 0, delta=delta)
        samples = run.results["progress"]
        run.check("context_page_reservation_observed", any(sample["metrics"]["flare_active_requests"] == 1
            and resource_totals(sample["metrics"])["pages"] == totals["pages"] - plan["reserved_pages"]
            and sample["metrics"]["flare_free_request_slots"] == 3 for sample in samples),
            pages_per_request=plan["reserved_pages"], samples=len(samples))
        run.check("context_prefill_progress_observed", len({sample["prefill_graphs_submitted"] for sample in samples}) > 1,
            observed_prefill_counts=[sample["prefill_graphs_submitted"] for sample in samples])
        run.reclaimed("context", before, after)
        short = {**body, "messages": [{"role": "user", "content": "Post-context admission " + plan["nonce"]}],
            "max_tokens": OUTPUT_TOKENS}
        short_tokens = len(render(tokenizer, short, tokenize=True))
        saved_timeout = args.timeout
        args.timeout = 120
        try:
            subsequent = run.request("subsequent-short", short)
        finally:
            args.timeout = saved_timeout
        run.complete(subsequent, short_tokens)
        run.check("subsequent_short_exact_budget", subsequent.get("body", {}).get("usage", {}).get("completion_tokens") == OUTPUT_TOKENS)
        final = idle(run, "final")
        run.reclaimed("subsequent-short", after, final)
        run.results["context_counter_delta"] = delta
    except Exception as error:
        if active is not None and not active.done.is_set():
            active.abort()
            active.done.wait(10)
        run.results["case_errors"].append({"case": "context-limit", "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()})
    finally:
        run.results["finished_at"] = datetime.now(timezone.utc).isoformat()
        run.results["passed"] = not run.results["case_errors"] and all(check["passed"] for check in run.results["checks"])
    return 0 if run.results["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:38761")
    parser.add_argument("--snapshot", type=Path, default=Path.home() / ".cache/huggingface/hub" /
        ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--memory-report", type=Path)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--deadline-seconds", type=float, default=3600)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--max-response-bytes", type=int, default=8 << 20)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.deadline_seconds <= 0 or args.poll_seconds <= 0 or args.max_response_bytes <= 0:
        parser.error("deadline, poll interval and response byte bound must be positive")
    if args.base_url != "http://127.0.0.1:38761":
        parser.error("context qualification is assigned to http://127.0.0.1:38761")
    tokenizer, plan, body = prepare(args)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "http_requests": 0, "plan": str(args.artifacts / "case-plan.json"),
            "prompt_tokens": plan["prompt_tokens"], "grammar_minimum_tokens": plan["grammar"]["minimum_tokens"],
            "expected_consumed_tokens": plan["expected_consumed_tokens"]}), flush=True)
        return 0
    return execute(args, tokenizer, plan, body)


if __name__ == "__main__":
    raise SystemExit(main())
