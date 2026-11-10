# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native conditioned-input reuse through GPU, memory, and disk prefix caches.

Uses the GPU assignment and compatible core configured for
test_prefix_cache_offload.py. Build flare-cli/target/release/flare before running.
"""

import base64
import io
import json
import time
import urllib.request

import pytest
from PIL import Image, ImageDraw, ImageFont

from tests.server.test_prefix_cache_offload import pytestmark, server


VISION = 'flare_graph_launches_total{kind="vision"}'
PREPARED = "flare_input_prepared_images_total"
DECODED = "flare_input_decoded_images_total"


def image(text, color):
    pixels = Image.new("RGB", (640, 320), color)
    draw = ImageDraw.Draw(pixels)
    draw.rectangle((20, 80, 620, 240), fill="white")
    draw.text((35, 125), text, fill="black", font=ImageFont.load_default(size=60))
    encoded = io.BytesIO()
    pixels.save(encoded, "WEBP", lossless=True)
    return {"type": "image_url", "image_url": {
        "url": "data:image/webp;base64," + base64.b64encode(encoded.getvalue()).decode()
    }}


def messages(tag="Image cache test."):
    background = (
        tag + " Use the user images to answer. Reference background: "
        + " ".join(f"Item {i} records code {i * 17 % 997}." for i in range(600))
    )
    return [
        {"role": "system", "content": background},
        {"role": "user", "content": [
            {"type": "text", "text": "Read the printed text in these two images. "
             "Reply with only their text, one line per image."},
            image("CACHE 314", "red"), image("CACHE 628", "blue"),
        ]},
    ]


def ask(service, conversation):
    model = json.loads(service.get("/v1/models"))["data"][0]["id"]
    body = {"model": model, "messages": conversation, "seed": 1234,
            "max_tokens": 256, "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(
        service.url + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    assert result["choices"][0]["finish_reason"] != "length", result
    return result, time.monotonic() - start


def content(result):
    return result["choices"][0]["message"]["content"].strip()


def settled(service):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        metrics = service.settled()
        if metrics["flare_input_cache_queued"] == metrics["flare_input_cache_pending_writes"] == 0:
            return metrics
        time.sleep(.01)
    pytest.fail("prepared-input jobs did not drain")


def measured(service, conversation, label):
    before = settled(service)
    result, seconds = ask(service, conversation)
    after = settled(service)
    delta = {name: after.get(name, 0) - before.get(name, 0)
             for name in (VISION, PREPARED, DECODED)}
    print(label, {"seconds": round(seconds, 4), "usage": result["usage"],
                  "work": delta, "answer": content(result)})
    return result, delta


@pytest.mark.parametrize("tier", ["disabled", "memory", "disk", "combined"])
def test_prefix_and_preprocessing_reuse_across_host_tiers(tmp_path, tier):
    directory = tmp_path / "records"
    flags = ["--offline", "--embedding-table-storage", "ram"]
    if tier in ("memory", "combined"):
        flags += ["--prefix-cache-memory", "2G"]
    if tier in ("disk", "combined"):
        flags += ["--prefix-cache-disk", str(directory), "--prefix-cache-disk-limit", "16G"]
    if tier == "disabled":
        flags += ["--prefix-cache-memory", "0",
                  "--prefix-cache-disk", str(directory), "--prefix-cache-disk-limit", "0"]
    conversation = messages()
    with server(tmp_path, tier, flags=flags) as service:
        first, work = measured(service, conversation, tier + "-cold")
        assert content(first) == "CACHE 314\nCACHE 628"
        assert work[VISION] == work[PREPARED] == work[DECODED] == 2

        repeated, work = measured(service, conversation, tier + "-repeat")
        assert repeated["choices"][0]["token_ids"] == first["choices"][0]["token_ids"]
        assert work[VISION] == work[PREPARED] == 0
        assert work[DECODED] == (2 if tier == "disabled" else 0)
        assert repeated["usage"]["prompt_tokens_details"]["cached_tokens"] == repeated["usage"]["prompt_tokens"]

        appended = conversation + [
            {"role": "assistant", "content": content(first)},
            {"role": "user", "content": [
                {"type": "text", "text": "Here is one more image. Read its printed text. "
                 "Reply with only that text."},
                image("CACHE 942", "green"),
            ]},
        ]
        answer, work = measured(service, appended, tier + "-append")
        assert content(answer) == "CACHE 942"
        assert work[VISION] == work[PREPARED] == 1
        assert work[DECODED] == (3 if tier == "disabled" else 1)

        changed, work = measured(service, messages("Separate reference."), tier + "-changed-prefix")
        assert content(changed) == content(first)
        assert work[VISION] == 2
        assert work[PREPARED] == work[DECODED] == (2 if tier == "disabled" else 0)
        assert changed["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        if tier == "disabled":
            assert not directory.exists()
        if tier in ("memory", "combined"):
            metrics = settled(service)
            assert 0 < metrics["flare_input_cache_memory_bytes"] <= 2**28
        if tier in ("disk", "combined"):
            assert list(directory.rglob("*.input"))
            metrics = settled(service)
            assert metrics["flare_host_cache_disk_bytes"] + metrics["flare_input_cache_disk_bytes"] <= 16 * 2**30

    if tier not in ("disk", "combined"):
        return
    with server(tmp_path, tier + "-restart", flags=flags) as service:
        restored, work = measured(service, conversation, tier + "-restored")
        assert restored["choices"][0]["token_ids"] == first["choices"][0]["token_ids"]
        assert work[VISION] == work[PREPARED] == work[DECODED] == 0
        metrics = settled(service)
        assert metrics["flare_host_cache_hits_total"] > 0
        assert metrics["flare_host_cache_fetched_total"] > 0
        assert metrics["flare_host_cache_fetch_failed_total"] == 0
        assert restored["usage"]["prompt_tokens_details"]["cached_tokens"] == restored["usage"]["prompt_tokens"]
        changed, work = measured(service, messages("Another reference."), tier + "-disk-payload")
        assert content(changed) == content(first)
        assert work[VISION] == 2
        assert work[PREPARED] == work[DECODED] == 0
        assert settled(service)["flare_input_cache_disk_hits_total"] >= 4


def test_eight_images_and_corrupt_preprocessing_preserve_prefix_state(tmp_path):
    directory = tmp_path / "records"
    flags = ["--offline", "--embedding-table-storage", "ram",
             "--prefix-cache-disk", str(directory), "--prefix-cache-disk-limit", "16G"]
    conversation = messages()
    with server(tmp_path, "input-corruption-source", flags=flags) as service:
        first, work = measured(service, conversation, "corruption-cold")
        assert content(first) == "CACHE 314\nCACHE 628"
        assert work[VISION] == 2
        labels = ["FIRST 128", "SECOND 256", "THIRD 512", "FOURTH 768",
                  "FIFTH 1024", "SIXTH 2048", "SEVENTH 4096", "EIGHTH 8192"]
        colors = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "pink"]
        parts = [image(label, color) for label, color in zip(labels, colors)]
        parts.append({"type": "text", "text": "Read the printed text in every image, "
                      "in image order. Reply with only the text, one line per image."})
        eight, work = measured(service, [{"role": "user", "content": parts}], "eight-images")
        assert content(eight).splitlines() == labels
        assert work[VISION] == work[PREPARED] == work[DECODED] == 8

    payloads = [path for path in directory.rglob("*.input") if path.stat().st_size > 4096]
    assert payloads
    for path in payloads:
        with path.open("r+b") as file:
            file.seek(-1, 2)
            value = file.read(1)
            file.seek(-1, 2)
            file.write(bytes([value[0] ^ 255]))
    with server(tmp_path, "input-corruption-restart", flags=flags) as service:
        restored, work = measured(service, conversation, "corruption-prefix")
        assert content(restored) == content(first)
        assert work[VISION] == work[PREPARED] == work[DECODED] == 0
        changed, work = measured(service, messages("Rebuilt input reference."), "corruption-rebuilt")
        assert content(changed) == content(first)
        assert work[VISION] == work[PREPARED] == work[DECODED] == 2
        assert settled(service)["flare_input_cache_failures_total"] >= 2
