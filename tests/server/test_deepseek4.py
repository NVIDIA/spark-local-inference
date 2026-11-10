# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live DeepSeek V4 Flash Vision requests through the native HTTP service.

Set FLARE_RUN_CUDA_TESTS=1 and FLARE_DEEPSEEK4_TEST_URL to a Flare service
loaded with a V4 Flash Vision core, concurrency four, and a 4096-token
prefill chunk. Run with DSpark enabled and with --mtp-draft-tokens 0.
"""

import base64
import io
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("FLARE_DEEPSEEK4_TEST_URL"),
    reason="requires a native DeepSeek V4 Flash Vision service",
)


@pytest.fixture(scope="module")
def service():
    url = os.environ["FLARE_DEEPSEEK4_TEST_URL"].rstrip("/")
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    assert "deepseek" in model.lower() and "4.1" not in model, model

    def ask(content, **options):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 1,
            "seed": 1234,
            "max_tokens": 128,
            "enable_thinking": False,
        }
        body.update(options)
        request = urllib.request.Request(
            url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        assert result["choices"][0]["finish_reason"] != "length", result
        return result

    return ask


def answer(result):
    return result["choices"][0]["message"]["content"].strip()


def image_part(color, label):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, 384), color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=64)
    draw.rectangle((32, 128, 608, 240), fill="white")
    draw.text((48, 144), label, fill="black", font=font)
    data = io.BytesIO()
    image.save(data, format="PNG")
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()},
    }


def test_text_and_parallel_requests(service):
    assert "391" in answer(service("What is 17 times 23? Reply with only the integer."))
    prompts = [f"Repeat exactly this code and nothing else: FLARE{i}128" for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(service, prompts))
    for i, result in enumerate(results):
        assert f"FLARE{i}128" in answer(result)


def test_reasoning_is_separate_from_visible_answer(service):
    result = service(
        "What is 17 times 23? Put only the integer in the final answer.",
        enable_thinking=True,
        reasoning_effort="low",
        max_tokens=512,
    )
    message = result["choices"][0]["message"]
    assert message["reasoning_content"].strip()
    assert answer(result) == "391"


def test_long_context_and_reused_prefix(service):
    text = " ".join(f"Item {i} records code {i * 17 % 997}." for i in range(1200))
    prompt = text + "\nWhat code does item 211 record? Reply with only the integer."
    first, second = service(prompt), service(prompt)
    assert first["usage"]["prompt_tokens"] > 4096
    assert answer(first) == answer(second) == "596"



def test_decode_across_multiple_compression_boundaries(service):
    result = service(
        "Count from 1 through 96 inclusive, separated by commas. Output only the numbers, with no omissions.",
        max_tokens=512,
    )
    assert result["usage"]["completion_tokens"] > 256
    assert [int(value) for value in re.findall(r"\d+", answer(result))] == list(range(1, 97))

def test_vision_and_image_span_across_prefill_boundary(service):
    image = image_part("red", "SPARK 128")
    question = {"type": "text", "text": "Read the label in the image. Reply with only its text."}
    assert "SPARK 128" in answer(service([image, question]))
    content = [{"type": "text", "text": "filler " * 4050}, image, question]
    result = service(content)
    assert result["usage"]["prompt_tokens"] > 4096
    assert "SPARK 128" in answer(result)


def test_multiple_images_keep_feature_offsets_and_order(service):
    content = [
        image_part("red", "FIRST 128"),
        image_part("blue", "SECOND 256"),
        {"type": "text", "text": "Read both image labels in their original order. Only output the labels."},
    ]
    result = answer(service(content))
    assert "FIRST 128" in result and "SECOND 256" in result
    assert result.index("FIRST 128") < result.index("SECOND 256")


def test_dsml_tool_call(service):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Get the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    result = service("Use lookup_weather to get the weather in Paris.", tools=tools)
    calls = result["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "lookup_weather"
    assert json.loads(calls[0]["function"]["arguments"])["city"].lower() == "paris"
