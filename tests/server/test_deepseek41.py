# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4.1 C32 native replay with shared text, vision, and speculative scratch.

Set FLARE_RUN_CUDA_TESTS=1 and FLARE_DEEPSEEK41_TEST_URL to a TP4 service
with concurrency 32, 4096-token prefill, vision, and five adaptive drafts.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import threading
import urllib.request

import pytest

from .test_deepseek4 import image_part

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("FLARE_DEEPSEEK41_TEST_URL"),
    reason="requires a native DeepSeek V4.1 C32 service",
)


@pytest.fixture(scope="module")
def service():
    url = os.environ["FLARE_DEEPSEEK41_TEST_URL"].rstrip("/")
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]
    assert "deepseek" in model["id"].lower() and "4.1" in model["id"]
    assert model["flare"]["max_concurrency"] == 32
    assert model["flare"]["prefill_chunk_size"] == 4096
    assert "image" in model["flare"]["input_modalities"]

    def ask(content, **options):
        body = dict(model=model["id"], messages=[dict(role="user", content=content)],
                    temperature=1, top_k=1, top_p=1, seed=1234, max_tokens=512,
                    enable_thinking=False, return_token_ids=True)
        body.update(options)
        request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        assert result["choices"][0]["finish_reason"] != "length", result
        return result

    def metrics():
        with urllib.request.urlopen(url + "/metrics", timeout=10) as response:
            return {parts[0]: float(parts[1]) for line in response.read().decode().splitlines()
                    if not line.startswith("#") and len(parts := line.split()) == 2}

    return ask, metrics


def _answer(result):
    return result["choices"][0]["message"]["content"].strip()


def test_full_concurrency_preserves_independent_tokens_and_cache_state(service):
    ask, metrics = service
    before = metrics()
    barrier = threading.Barrier(32)

    def request(index):
        barrier.wait()
        result = ask(f"Write the label FLARE{index}128, then count from 1 through 48 inclusive, "
                     "separated by commas. Output only the label and numbers, with no omissions.")
        answer = _answer(result)
        assert f"FLARE{index}128" in answer, answer
        numbers = answer.replace(f"FLARE{index}128", "")
        assert [int(value) for value in re.findall(r"\d+", numbers)] == list(range(1, 49)), answer

    with ThreadPoolExecutor(max_workers=32) as workers:
        list(workers.map(request, range(32)))
    after = metrics()
    assert after["flare_requests_completed_total"] - before["flare_requests_completed_total"] == 32
    assert after["flare_requests_failed_total"] == before["flare_requests_failed_total"]
    assert sum(after[key] - before[key] for key in after
               if key.startswith('flare_speculative_rounds_total{batch="32",')) > 0


def test_chunked_prefill_and_resume_preserve_exact_repeated_output(service):
    ask, metrics = service
    text = " ".join(f"Item {i} records code {i * 17 % 997}." for i in range(1200))
    prompt = text + "\nWhat code does item 211 record? Reply with only the integer."
    before = metrics()
    first, second = ask(prompt), ask(prompt)
    assert first["usage"]["prompt_tokens"] > 4096
    assert _answer(first) == _answer(second) == "596"
    assert first["choices"][0]["token_ids"] == second["choices"][0]["token_ids"]
    assert metrics()["flare_cache_reused_tokens_total"] > before["flare_cache_reused_tokens_total"]


def test_vision_and_text_reuse_scratch_across_prefill_boundaries(service):
    ask, _ = service
    question = {"type": "text", "text": "Read the label in the image. Reply with only its text."}
    for color, label in (("red", "SPARK 128"), ("blue", "FLASH 256")):
        result = ask([{"type": "text", "text": "filler " * 4050}, image_part(color, label), question])
        assert result["usage"]["prompt_tokens"] > 4096
        assert label in _answer(result)
        assert _answer(ask("What is 17 times 23? Reply with only the integer.")) == "391"
