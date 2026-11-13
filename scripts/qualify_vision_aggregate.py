# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify four full-capacity images through a running native Qwen HTTP server."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import sys
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qualify.server import Run, comparable_message, digest, metric_delta, write_json


SEEDS = (13, 29, 47, 71)
SIDE = 2048
PATCH = 16
BODY_LIMIT = 32 << 20


@contextmanager
def deadline(seconds):
    def expired(signum, frame):
        raise TimeoutError(f"absolute HTTP deadline exceeded ({seconds:g} seconds)")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class BoundedRun(Run):
    def request(self, *args, **kwargs):
        with deadline(self.args.timeout):
            return super().request(*args, **kwargs)


def coordinate_pattern(seed):
    import numpy as np

    y, x = np.indices((SIDE // PATCH, SIDE // PATCH), dtype=np.uint16)
    blocks = np.stack(((3 * x + 5 * y + 17 * seed) % 256,
                       (11 * x + 7 * y + 29 * seed) % 256,
                       (13 * x + 19 * y + 31 * seed) % 256), axis=-1).astype(np.uint8)
    return blocks.repeat(PATCH, axis=0).repeat(PATCH, axis=1)


def process_identity(args):
    process = Path(f"/proc/{args.server_pid}")
    argv = [value.decode() for value in (process / "cmdline").read_bytes().split(b"\0") if value]
    bind = argv[argv.index("--bind") + 1]
    if args.base_url != "http://" + bind:
        raise ValueError("server PID bind address differs from the qualification endpoint")
    memory_path = Path(argv[argv.index("--memory-report") + 1]).resolve()
    if memory_path != args.memory_report.resolve():
        raise ValueError("server PID memory-report argument differs from the selected report")
    mappings = [line.split(maxsplit=5) for line in (process / "maps").read_text().splitlines()
                if "flare_heliopause" in line and ".so" in line]
    paths = {row[5] for row in mappings}
    if len(paths) != 1:
        raise ValueError("server must map exactly one identifiable native extension")
    extension = Path(paths.pop())
    if digest(extension) != args.expected_extension_sha256:
        raise ValueError("server native extension fingerprint differs from the selected build")
    if any(int(row[4]) != extension.stat().st_ino for row in mappings):
        raise ValueError("native extension file identity differs from the mapped server library")
    environment = dict(value.decode().split("=", 1) for value in (process / "environ").read_bytes().split(b"\0") if value)
    return {"pid": args.server_pid, "argv": argv, "native_extension": str(extension),
            "native_extension_sha256": digest(extension), "native_extension_inode": extension.stat().st_ino,
            "process_start_ticks": (process / "stat").read_text().split()[21],
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "cuda_home": environment.get("CUDA_HOME")}


def prepare_fixture(args):
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel
    from transformers.vision_utils import (
        get_vision_cu_seqlens, get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    def forbid_cuda(*unused_args, **unused_kwargs):
        raise RuntimeError("qualification client must not initialize CUDA")

    torch.cuda._lazy_init = forbid_cuda
    torch.set_num_threads(1)
    golden_path = ROOT / "tests/frontend/qwen38_images.json"
    golden = json.loads(golden_path.read_text())
    origin = json.loads(importlib.metadata.distribution("transformers").read_text("direct_url.json"))
    if origin.get("vcs_info", {}).get("commit_id") != golden["transformers_commit"]:
        raise ValueError("Transformers source differs from the canonical fixture pin")
    versions = {name: importlib.metadata.version(name) for name in golden["versions"]}
    if versions != golden["versions"] or torch.version.git_version != golden["torch_commit"]:
        raise ValueError("canonical processor dependencies differ from the pinned fixture")
    snapshot = args.snapshot.resolve()
    assets = {name: digest(snapshot / name) for name in golden["assets"]}
    if assets != golden["assets"]:
        raise ValueError("checkpoint frontend assets differ from the canonical fixture")
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        (args.artifacts / name).write_bytes((snapshot / name).read_bytes())
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    positional_model = SimpleNamespace(config=config)
    positional_model.get_vision_position_ids = MethodType(Qwen4ExpModel.get_vision_position_ids, positional_model)
    images, inputs, content = [], [], [{"type": "text", "text": "Describe these four coordinate-pattern images briefly."}]
    for index, seed in enumerate(SEEDS):
        rgb = coordinate_pattern(seed)
        path = args.artifacts / f"coordinate-{index}-seed-{seed}.png"
        Image.fromarray(rgb).save(path, format="PNG")
        with Image.open(path) as image:
            image.load()
            decoded = np.asarray(image.convert("RGB"))
            if image.size != (SIDE, SIDE) or not np.array_equal(rgb, decoded):
                raise ValueError("PNG encoding changed the generated RGB input")
            images.append(image.copy())
        png = path.read_bytes()
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}})
        inputs.append({"path": str(path.resolve()), "seed": seed, "width": SIDE, "height": SIDE,
                       "png_bytes": len(png), "png_sha256": hashlib.sha256(png).hexdigest(),
                       "rgb_bytes": rgb.nbytes, "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
    if len({entry["rgb_sha256"] for entry in inputs}) != 4:
        raise ValueError("aggregate fixture requires four different images")
    messages = [{"role": "user", "content": content}]
    request = {"model": golden["model_id"], "messages": messages, "temperature": 0,
               "seed": 1234, "max_tokens": 32, "enable_thinking": False, "add_generation_prompt": True}
    rendered = processor.tokenizer.apply_chat_template(messages, tokenize=False,
        enable_thinking=False, add_generation_prompt=True)
    result = processor(text=rendered, images=images, return_tensors="pt", return_mm_token_type_ids=True)
    grids = result["image_grid_thw"]
    if grids.tolist() != [[1, 128, 128]] * 4 or tuple(result["pixel_values"].shape) != (65536, 1536):
        raise ValueError("canonical processor did not preserve four full-capacity images")
    token_ids = result["input_ids"][0].tolist()
    if token_ids.count(config.image_token_id) != 16384:
        raise ValueError("canonical expanded prompt does not contain 16384 image features")
    cu_seqlens = get_vision_cu_seqlens(grids).tolist()
    if cu_seqlens != [0, 16384, 32768, 49152, 65536]:
        raise ValueError("canonical patch offsets differ from four complete images")
    positions = get_vision_position_ids(grids, 2)
    indices, weights = get_vision_interpolation_indices_and_weights(grids, 48,
        mode="bilinear", align_corners=True, spatial_merge_size=2)
    mrope, rope_delta = Qwen4ExpModel.get_rope_index(positional_model, result["input_ids"],
        result["mm_token_type_ids"], image_grid_thw=grids)

    def tensor_receipt(tensor):
        if tensor.device.type != "cpu":
            raise ValueError("canonical processor unexpectedly produced a device tensor")
        array = tensor.contiguous().numpy()
        return {"shape": list(array.shape), "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}

    fixture = {"name": "four_coordinate_images_16384_patches_each", "model_id": golden["model_id"],
        "revision": golden["revision"], "purpose": "aggregate capacity and image indexing; no caption-quality oracle",
        "inputs": inputs, "generator": {"script_sha256": digest(Path(__file__)),
            "block_side": PATCH, "seeds": list(SEEDS), "resized": False},
        "assets_sha256": assets, "versions": versions, "transformers_origin": origin,
        "torch_commit": torch.version.git_version, "canonical_fixture_sha256": digest(golden_path),
        "processor_class": f"{type(processor.image_processor).__module__}.{type(processor.image_processor).__name__}",
        "processor_backend": processor.image_processor.backend, "rendered": rendered,
        "token_ids": token_ids, "mm_token_type_ids": result["mm_token_type_ids"][0].tolist(),
        "grid_thw": grids.tolist(), "cu_seqlens": cu_seqlens,
        "expected_feature_offsets": [value // 4 for value in cu_seqlens[:-1]],
        "aggregate_patches": 65536, "aggregate_features": 16384,
        "prompt_tokens": len(token_ids), "mrope": mrope[:, 0, :].tolist(),
        "rope_delta": rope_delta[0, 0].item(),
        "tensors": {name: tensor_receipt(value) for name, value in {
            "pixel_values": result["pixel_values"], "position_ids": positions,
            "interpolation_indices": indices, "interpolation_weights": weights,
            "mrope": mrope[:, 0, :],
        }.items()},
        "json_request_bytes": len(json.dumps({**request, "stream": False}, ensure_ascii=False).encode()),
        "sse_request_bytes": len(json.dumps({**request, "stream": True}, ensure_ascii=False).encode()),
        "cuda_initialized": torch.cuda.is_initialized()}
    if max(fixture["json_request_bytes"], fixture["sse_request_bytes"]) > BODY_LIMIT or fixture["cuda_initialized"]:
        raise ValueError("aggregate fixture exceeds the HTTP body limit or initialized CUDA")
    write_json(args.artifacts / "canonical-fixture.json", fixture)
    write_json(args.artifacts / "request.json", request)
    return request, fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--server-pid", required=True, type=int)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--memory-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.artifacts.exists() and any(args.artifacts.iterdir()):
        parser.error("receipt and artifact paths must not contain an earlier run")
    if not 0 < args.timeout <= 300:
        parser.error("HTTP timeout must be positive and no larger than 300 seconds")
    args.max_response_bytes = 64 << 20
    args.require_metrics = True
    args.cases = None
    args.artifacts.mkdir(parents=True, exist_ok=True)
    identity = process_identity(args)
    request, fixture = prepare_fixture(args)
    run = BoundedRun(args)

    def health(label):
        record = run.request(label, path="/health")
        run.check(label, record.get("status") == 200 and record.get("body", {}).get("status") == "ok", response=record)

    def aggregate():
        health("aggregate-health-before")
        models = run.request("aggregate-models", path="/v1/models")
        run.check("aggregate.exact_model", models.get("status") == 200 and fixture["model_id"] in
            [entry.get("id") for entry in models.get("body", {}).get("data", [])])
        before = run.metrics("aggregate-before")
        run.results["idle_before"] = before
        memory = json.loads(args.memory_report.read_text())
        run.check("aggregate.idle_capacity", before["flare_free_request_slots"] == 4
            and before["flare_free_kv_pages"] + before["flare_cached_kv_pages"] == memory["pages"]
            and before["flare_free_snapshots"] + before["flare_cached_checkpoints"] == memory["snapshot_capacity"])
        previous = before
        choices = []
        for index, stream in enumerate((False, False, True)):
            name = ("aggregate-cold-json", "aggregate-repeat-json", "aggregate-repeat-sse")[index]
            record = run.request(name, request, stream=stream)
            if not stream:
                choice = run.complete(record, fixture["prompt_tokens"])
                usage = record["body"].get("usage", {})
                count = usage.get("completion_tokens")
                run.check(name + ".usage", isinstance(count, int) and 0 < count <= request["max_tokens"]
                    and usage.get("total_tokens") == fixture["prompt_tokens"] + count,
                    expected_prompt_tokens=fixture["prompt_tokens"], usage=usage)
                choices.append(choice)
            else:
                combined = record.get("sse", {})
                run.check(name + ".clean_terminal", record.get("status") == 200 and record.get("done_count") == 1
                    and not record.get("client_error") and not record.get("incomplete_event") and not combined.get("errors")
                    and combined.get("finish_reasons") in (["stop"], ["length"], ["tool_calls"]), response=record)
                choices.append({"message": combined.get("message", {}),
                    "finish_reason": (combined.get("finish_reasons") or [None])[0]})
            after = run.metrics(name + "-after")
            delta = metric_delta(previous, after)
            run.results.setdefault("model_metric_deltas", []).append({"case": name, "delta": delta})
            run.check(name + ".four_full_vision_graphs", delta['flare_graph_launches_total{kind="vision"}'] == 4,
                delta=delta, expected_patches_each=16384, expected_feature_offsets=fixture["expected_feature_offsets"])
            run.check(name + ".completed", delta["flare_requests_admitted_total"] == delta["flare_requests_completed_total"] == 1
                and delta["flare_requests_cancelled_total"] == delta["flare_requests_failed_total"] == 0, delta=delta)
            run.check(name + ".cache_path", delta["flare_cache_hits_total"] == int(index > 0)
                and delta["flare_cache_misses_total"] == int(index == 0)
                and delta['flare_graph_launches_total{kind="restore"}'] == int(index > 0), delta=delta)
            run.reclaimed(name, before, after)
            previous = after
        for index in (1, 2):
            run.check(f"aggregate.cold_repeat_{index}_greedy_equivalence",
                comparable_message(choices[0]["message"]) == comparable_message(choices[index]["message"])
                and choices[0]["finish_reason"] == choices[index]["finish_reason"], expected=choices[0], observed=choices[index])
        health("aggregate-health-after")

    run.case("aggregate_vision", aggregate)
    def final_idle():
        after = run.metrics("aggregate-final-idle")
        run.results["idle_after"] = after
        if "idle_before" in run.results:
            run.reclaimed("aggregate-final", run.results["idle_before"], after)
        run.check("aggregate.server_identity_unchanged", process_identity(args) == identity)

    run.case("aggregate_final_idle", final_idle)
    run.results["finished_at"] = datetime.now(timezone.utc).isoformat()
    run.results["summary"] = {"requests": len(run.results["requests"]), "checks": len(run.results["checks"]),
        "failed_checks": sum(not check["passed"] for check in run.results["checks"]), "case_errors": len(run.results["case_errors"])}
    run.results["passed"] = not run.results["summary"]["failed_checks"] and not run.results["case_errors"]
    print(json.dumps({**run.results["summary"], "passed": run.results["passed"]}), flush=True)
    return 0 if run.results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
