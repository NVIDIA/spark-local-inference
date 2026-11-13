# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Bolt text, tool, image, and recurrent-prefix requests.

Set FLARE_RUN_CUDA_TESTS=1 and FLARE_NEMOTRON_TEST_URL to a native service
with concurrency eight and a 4096-token prefill chunk. Run with ordinary
decode, fixed seven-token LiLiCorr, and adaptive verification.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import re
import threading
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("FLARE_NEMOTRON_TEST_URL"),
    reason="requires a native Nemotron Bolt service",
)


@pytest.fixture(scope="module")
def service():
    url = os.environ["FLARE_NEMOTRON_TEST_URL"].rstrip("/")
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    assert "nemotron" in model.lower(), model

    def ask(content, *, barrier=None, **options):
        body = dict(model=model, messages=[dict(role="user", content=content)], temperature=1,
                    top_k=1, top_p=1, seed=42, max_tokens=128, return_token_ids=True,
                    chat_template_kwargs=dict(enable_thinking=False))
        body.update(options)
        request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        if barrier is not None:
            barrier.wait(timeout=10)
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
        assert result["choices"][0]["finish_reason"] != "length", result
        assert len(result["choices"][0]["token_ids"]) == result["usage"]["completion_tokens"]
        return result

    return ask


def answer(result):
    return result["choices"][0]["message"]["content"].strip()


def image(color, size=(512, 512)):
    from PIL import Image, ImageDraw

    picture = Image.new("RGB", size, "white")
    width, height = size
    ImageDraw.Draw(picture).rectangle((width // 6, height // 4, 5 * width // 6, 3 * height // 4), fill=color)
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return dict(type="image_url", image_url=dict(url="data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode()))


@pytest.mark.parametrize("batch", [1, 3, 4, 5, 7, 8])
def test_concurrent_partial_batches(service, batch):
    barrier = threading.Barrier(batch)

    def lane(index):
        return service(f"Compute {104 + index} + 207. Reply with only the result.", barrier=barrier)

    with ThreadPoolExecutor(max_workers=batch) as workers:
        results = list(workers.map(lane, range(batch)))
    assert [answer(result) for result in results] == [str(311 + index) for index in range(batch)]


def test_default_reasoning(service):
    result = service("Compute 7 multiplied by 8. Give the final answer as a number.",
                     max_tokens=512, chat_template_kwargs={})
    assert result["choices"][0]["message"]["reasoning_content"].strip()
    assert re.search(r"\b56\b", answer(result))


def test_wrapped_tool_arguments_and_result(service):
    tools = [dict(type="function", function=dict(name="lookup_weather", description="Get the weather for a city.",
        parameters=dict(type="object", properties=dict(city=dict(type="string")),
                        required=["city"], additionalProperties=False)))]
    history = [dict(role="system", content="Follow the user instructions precisely."),
               dict(role="user", content="Use lookup_weather to get the weather in Paris.")]
    result = service("", messages=history, tools=tools, tool_choice="required", max_tokens=256)
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    message = result["choices"][0]["message"]
    call, = message["tool_calls"]
    assert call["function"]["name"] == "lookup_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    result = service("", messages=history + [message, dict(role="tool", tool_call_id=call["id"],
        content='{"temperature_c": 21, "condition": "sunny"}')], tools=tools)
    assert re.search(r"\b21\b", answer(result))


def test_recurrent_prefix_branch_and_exact_restore(service):
    prefix = "A cache preserves data so repeated computations can reuse earlier results. " * 650
    history = [dict(role="system", content="Read the supplied context and follow the final user instruction."),
               dict(role="user", content=prefix), dict(role="assistant", content="I have read the context.")]
    for marker in ("472901", "395106", "472901"):
        result = service("", messages=history + [dict(role="user", content=f"Reply with exactly this number: {marker}.")])
        assert answer(result) == marker
    assert result["usage"]["prompt_tokens"] > 4096
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == result["usage"]["prompt_tokens"]


def test_image_cache_identity(service):
    question = dict(type="text", text="What color is the rectangle? Reply with only the color name.")
    results = []
    for color in ("red", "blue", "red"):
        result = service([image(color), question])
        assert answer(result).lower().rstrip(".") == color
        results.append(result)
    first, changed, repeated = results
    assert first["prompt_token_ids"] == changed["prompt_token_ids"] == repeated["prompt_token_ids"]
    assert repeated["usage"]["prompt_tokens_details"]["cached_tokens"] == repeated["usage"]["prompt_tokens"]
    assert changed["usage"]["prompt_tokens_details"]["cached_tokens"] < changed["usage"]["prompt_tokens"]


@pytest.mark.parametrize("colors,size,features", [
    (("red", "blue"), (2048, 1536), 6144),
    (("red", "blue", "green", "yellow"), (1656, 2048), 13312),
])
def test_image_features_across_prefill_chunks(service, colors, size, features):
    content = [image(color, size) for color in colors]
    content.append(dict(type="text", text="Name the rectangle colors in image order. Reply with only the colors."))
    result = service(content)
    assert result["usage"]["prompt_tokens"] > 4096
    assert result["usage"]["prompt_tokens_details"]["multimodal_tokens"]["image"] == features
    text = answer(result).lower()
    assert all(re.search(r"\b" + color + r"\b", text) for color in colors), text
    positions = [text.index(color) for color in colors]
    assert positions == sorted(positions), text
