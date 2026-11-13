# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent requests through a Qwen service loaded with mixed target graphs.

Use FLARE_RUN_CUDA_TESTS=1 and FLARE_SCHEDULER_TEST_URL for an idle service
with concurrency four and 4096-token chunks. Exercise ordinary, fixed-depth,
and adaptive settings by starting the service with the corresponding flags.
"""

import base64
import io
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.server.test_scheduler import (
    Stream, check_answer, decode_request, measure, metrics, prompt_request, service, stream, pytestmark,
)


@pytest.mark.parametrize("decode_count", [1, 3])
def test_mixed_graphs_execute_during_prompt_ingestion(service, decode_count):
    decodes = [decode_request(f"decode-{i}") for i in range(decode_count)]
    prompt = prompt_request("prefill-long", words=32768)
    delta = measure(service, f"mixed_prefill_{decode_count}_decodes", [*decodes, prompt], stagger_prefill=True)
    assert delta['flare_graph_launches_total{kind="mixed"}'] > 0, delta
    for decode in decodes:
        assert any(prompt.started < at < prompt.events[0][0] for at, _ in decode.events)


def test_image_prompt_during_decode(service):
    from PIL import Image

    data = io.BytesIO()
    Image.new("RGB", (224, 224), "red").save(data, format="PNG")
    image = {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()}}
    prompt = Stream("image", [
        {"type": "text", "text": "Ignore this filler: " + " apple" * 4096}, image,
        {"type": "text", "text": "What color is the image? Reply with the color only."}], 32)
    decode = decode_request("decode-image-peer")
    before = metrics(service)
    assert before["flare_active_requests"] == 0
    with ThreadPoolExecutor(max_workers=2) as workers:
        ongoing = workers.submit(stream, service, decode)
        assert decode.first.wait(timeout=30)
        stream(service, prompt)
        check_answer(ongoing.result(timeout=180))
    assert "red" in prompt.text.lower(), prompt.text
    after = metrics(service)
    assert after['flare_graph_launches_total{kind="vision"}'] > before['flare_graph_launches_total{kind="vision"}']
    assert after['flare_graph_launches_total{kind="mixed"}'] > before['flare_graph_launches_total{kind="mixed"}']
    assert after["flare_requests_failed_total"] == before["flare_requests_failed_total"]
