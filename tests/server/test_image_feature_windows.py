# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native chunked prefill exceeding a Qwen core's image feature arena.

Uses the GPU assignment and TP1 Qwen core configured for
test_prefix_cache_offload.py. Each image contributes 4,096 feature tokens;
five images exceed the core's 16,384-token arena without exceeding context.
"""

import base64
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from PIL import Image, ImageDraw, ImageFont

from tests.server.test_conditioned_input_cache import (
    DECODED, PREPARED, VISION, ask, content, measured, settled,
)
from tests.server.test_prefix_cache_offload import pytestmark, server


def large_image(label, color):
    pixels = Image.new("RGB", (2048, 2048), color)
    draw = ImageDraw.Draw(pixels)
    draw.rectangle((80, 750, 1968, 1250), fill="white")
    draw.text((200, 925), label, fill="black", font=ImageFont.load_default(size=160))
    encoded = io.BytesIO()
    pixels.save(encoded, "WEBP", lossless=True)
    return {"type": "image_url", "image_url": {
        "url": "data:image/webp;base64," + base64.b64encode(encoded.getvalue()).decode()
    }}


def conversation(count=5, tag="Read all images."):
    labels = [f"WINDOW {111 * (index + 1)}" for index in range(count)]
    colors = ("red", "blue", "green", "orange", "purple", "cyan", "yellow", "pink", "gray")
    parts = [large_image(label, colors[index % len(colors)])
             for index, label in enumerate(labels)]
    parts.append({"type": "text", "text": "Read the printed text in every image, "
                  "in image order. Reply with only the text, one line per image."})
    return [{"role": "system", "content": tag},
            {"role": "user", "content": parts}], labels


@pytest.mark.parametrize(("drafts", "tier"), [
    (0, "disabled"), (5, "disabled"), (5, "memory"), (5, "disk"), (5, "combined"),
])
def test_image_history_exceeds_arena_across_cache_tiers(tmp_path, drafts, tier):
    directory = tmp_path / "records"
    flags = ["--offline", "--embedding-table-storage", "ram"]
    flags += ["--prefix-cache-memory", "8G" if tier in ("memory", "combined") else "0"]
    flags += ["--prefix-cache-disk", str(directory), "--prefix-cache-disk-limit",
              "16G" if tier in ("disk", "combined") else "0"]
    messages, labels = conversation()
    with server(tmp_path, f"windows-{drafts}-{tier}", drafts=drafts, flags=flags) as service:
        model = json.loads(service.get("/v1/models"))["data"][0]
        assert model["flare"]["image_feature_window_tokens"] == 16384
        assert model["flare"]["max_image_tokens_per_prompt"] > 5 * 4096
        before = settled(service)
        first, work = measured(service, messages, f"windows-{drafts}-{tier}-cold")
        assert content(first).splitlines() == labels
        details = first["usage"]["prompt_tokens_details"]
        assert details["multimodal_tokens"]["image"] == 5 * 4096
        assert details["cached_tokens"] == 0
        assert work[VISION] == work[PREPARED] == work[DECODED] == 5
        after = settled(service)
        assert after['flare_graph_launches_total{kind="restore"}'] - before['flare_graph_launches_total{kind="restore"}'] == 1

        repeated, work = measured(service, messages, f"windows-{drafts}-{tier}-repeat")
        assert repeated["choices"][0]["token_ids"] == first["choices"][0]["token_ids"]
        assert work[VISION] == work[PREPARED] == 0
        assert work[DECODED] == (5 if tier == "disabled" else 0)
        assert repeated["usage"]["prompt_tokens_details"]["cached_tokens"] == repeated["usage"]["prompt_tokens"]

        appended = messages + [
            {"role": "assistant", "content": content(first)},
            {"role": "user", "content": [
                large_image("WINDOW 666", "cyan"),
                {"type": "text", "text": "Read this additional image. "
                 "Reply with only its printed text."},
            ]},
        ]
        answer, work = measured(service, appended, f"windows-{drafts}-{tier}-append")
        assert content(answer) == "WINDOW 666"
        assert answer["usage"]["prompt_tokens_details"]["multimodal_tokens"]["image"] == 6 * 4096
        assert work[VISION] == work[PREPARED] == 1
        assert work[DECODED] == (6 if tier == "disabled" else 1)

        changed = [{"role": "system", "content": "Read the supplied image labels."}, *messages[1:]]
        answer, work = measured(service, changed, f"windows-{drafts}-{tier}-changed-prefix")
        assert content(answer).splitlines() == labels
        assert answer["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert work[VISION] == 5
        assert work[PREPARED] == work[DECODED] == (5 if tier == "disabled" else 0)
        if tier == "disabled":
            assert not directory.exists()
            assert all(value == 0 for name, value in settled(service).items()
                       if name.startswith("flare_host_cache_"))

    if tier not in ("disk", "combined"):
        return
    with server(tmp_path, f"windows-{drafts}-{tier}-restart", drafts=drafts, flags=flags) as service:
        restored, work = measured(service, messages, f"windows-{drafts}-{tier}-restored")
        assert restored["choices"][0]["token_ids"] == first["choices"][0]["token_ids"]
        assert work[VISION] == work[PREPARED] == work[DECODED] == 0
        assert settled(service)["flare_host_cache_fetch_failed_total"] == 0
        assert settled(service)["flare_host_cache_fetched_total"] > 0
    if tier == "disk":
        checkpoints = list(directory.rglob("*.ckpt"))
        assert checkpoints
        for checkpoint in checkpoints:
            with checkpoint.open("r+b") as file:
                file.seek(-1, 2)
                byte = file.read(1)
                file.seek(-1, 2)
                file.write(bytes([byte[0] ^ 255]))
        with server(tmp_path, "windows-corrupt-checkpoint", drafts=drafts, flags=flags) as service:
            rebuilt, work = measured(service, messages, "windows-failed-host-restore")
            assert content(rebuilt).splitlines() == labels
            assert rebuilt["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
            assert work[VISION] == 5
            assert work[PREPARED] == work[DECODED] == 0
            assert settled(service)["flare_host_cache_fetch_failed_total"] > 0


def test_three_feature_windows_without_host_cache(tmp_path):
    messages, labels = conversation(9)
    with server(tmp_path, "three-windows", flags=[
        "--offline", "--embedding-table-storage", "ram",
        "--prefix-cache-memory", "0", "--prefix-cache-disk-limit", "0",
    ]) as service:
        before = settled(service)
        answer, work = measured(service, messages, "three-windows-cold")
        assert content(answer).splitlines() == labels
        details = answer["usage"]["prompt_tokens_details"]
        assert details["multimodal_tokens"]["image"] == 9 * 4096
        assert details["cached_tokens"] == 0
        assert work[VISION] == work[PREPARED] == work[DECODED] == 9
        after = settled(service)
        assert after['flare_graph_launches_total{kind="restore"}'] - before['flare_graph_launches_total{kind="restore"}'] == 2


def test_small_host_cache_recomputes_evicted_inputs_across_windows(tmp_path):
    messages, labels = conversation()
    with server(tmp_path, "bounded-input-retention", flags=[
        "--offline", "--embedding-table-storage", "ram",
        "--prefix-cache-memory", "2G", "--prefix-cache-disk-limit", "0",
    ]) as service:
        first, work = measured(service, messages, "bounded-input-retention-cold")
        assert content(first).splitlines() == labels
        assert work[VISION] == work[PREPARED] == 5
        assert 0 < settled(service)["flare_input_cache_memory_bytes"] <= 2**28
        repeated, work = measured(service, messages, "bounded-input-retention-repeat")
        assert repeated["choices"][0]["token_ids"] == first["choices"][0]["token_ids"]
        assert work[VISION] == work[PREPARED] == 0
        changed = [{"role": "system", "content": "Read the supplied image labels."}, *messages[1:]]
        answer, work = measured(service, changed, "bounded-input-retention-recompute")
        assert content(answer).splitlines() == labels
        assert work[VISION] == 5
        assert 3 <= work[PREPARED] <= 5
        assert 0 < settled(service)["flare_input_cache_memory_bytes"] <= 2**28


def test_cancel_between_windows_and_reuse_feature_slots(tmp_path):
    messages, _ = conversation(9, "Image request to cancel.")
    with server(tmp_path, "cancel-windows", flags=[
        "--offline", "--embedding-table-storage", "ram", "--max-concurrency", "4",
        "--prefix-cache-memory", "0", "--prefix-cache-disk-limit", "0",
    ]) as service:
        before = settled(service)
        model = json.loads(service.get("/v1/models"))["data"][0]["id"]
        body = {"model": model, "messages": messages, "stream": True, "max_tokens": 256, "seed": 1234,
                "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(service.url + "/v1/chat/completions",
                                         json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300):
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                metrics = service.metrics()
                if metrics['flare_graph_launches_total{kind="restore"}'] > before['flare_graph_launches_total{kind="restore"}']:
                    assert metrics[VISION] - before[VISION] < 9
                    break
                time.sleep(.01)
            else:
                pytest.fail("request did not reach its first feature-window restore")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            metrics = service.metrics()
            if (metrics["flare_requests_cancelled_total"] > before["flare_requests_cancelled_total"]
                    and metrics["flare_active_requests"] == metrics["flare_retiring_request_slots"] == 0):
                break
            time.sleep(.01)
        else:
            pytest.fail("cancelled window request retained execution resources")

        def complete(index):
            prompt, labels = conversation(5, f"Concurrent image request {index}.")
            answer, seconds = ask(service, prompt)
            print(f"after-window-cancel-{index}", {"seconds": round(seconds, 4),
                                                  "answer": content(answer)})
            assert content(answer).splitlines() == labels
            assert answer["usage"]["prompt_tokens_details"]["cached_tokens"] == 0

        vision_before = settled(service)[VISION]
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(complete, range(6)))
        assert settled(service)[VISION] - vision_before == 30
        assert service.get("/health")


def test_image_windows_progress_during_continuous_decode(tmp_path):
    messages, labels = conversation()
    with server(tmp_path, "windows-during-decode", drafts=0, flags=[
        "--offline", "--embedding-table-storage", "ram", "--max-concurrency", "4",
        "--prefix-cache-memory", "0", "--prefix-cache-disk-limit", "0",
    ]) as service:
        model = json.loads(service.get("/v1/models"))["data"][0]["id"]
        started, stop = Event(), Event()
        deliveries = []

        def decode():
            body = {"model": model, "stream": True, "max_tokens": 4096,
                    "ignore_eos": True, "seed": 1234,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content":
                                  "Count upward from one, one number per line."}]}
            request = urllib.request.Request(service.url + "/v1/chat/completions",
                                             json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                for line in response:
                    if stop.is_set():
                        break
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                        deliveries.append(1)
                        started.set()

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(decode)
            try:
                assert started.wait(30), "decoder produced no output"
                before = len(deliveries)
                answer, work = measured(service, messages, "windows-with-active-decoder")
                assert content(answer).splitlines() == labels
                assert work[VISION] == work[PREPARED] == 5
                assert len(deliveries) > before
                assert not future.done(), "image request waited for the decoder to finish"
                print("decoder deliveries during image request", len(deliveries) - before)
            finally:
                stop.set()
                future.result(timeout=60)
