# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Kimi K3 text, XTML, image, and continuation requests.

Set FLARE_RUN_CUDA_TESTS=1 and FLARE_KIMI3_TEST_URL to a native service with
concurrency four and a 4096-token prefill chunk.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import re
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("FLARE_KIMI3_TEST_URL"),
    reason="requires a native Kimi K3 service",
)


@pytest.fixture(scope="module")
def service():
    url = os.environ["FLARE_KIMI3_TEST_URL"].rstrip("/")
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    assert "kimi-k3" in model.lower(), model

    def ask(content, **options):
        body = dict(model=model, messages=[dict(role="user", content=content)], temperature=1,
                    seed=1234, max_tokens=1024, enable_thinking=False, reasoning_effort="low")
        body.update(options)
        request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                if body.get("stream"):
                    events = []
                    for line in response:
                        if line.startswith(b"data: "):
                            payload = line.removeprefix(b"data: ").strip()
                            if payload == b"[DONE]":
                                return events
                            events.append(json.loads(payload))
                    pytest.fail("stream ended without [DONE]")
                result = json.load(response)
        except urllib.error.HTTPError as error:
            pytest.fail(f"HTTP {error.code}: {error.read().decode()}")
        assert result["choices"][0]["finish_reason"] != "length", result
        return result

    return ask


def answer(result):
    return result["choices"][0]["message"]["content"].strip()


def image(color, label, size=(640, 384), format="PNG"):
    from PIL import Image, ImageDraw, ImageFont

    picture = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(picture)
    draw.rectangle((32, 128, 608, 240), fill="white")
    draw.text((48, 144), label, fill="black", font=ImageFont.load_default(size=64))
    buffer = io.BytesIO()
    picture.save(buffer, format=format)
    return dict(type="image_url", image_url=dict(url=f"data:image/{format.lower()};base64,"
                + base64.b64encode(buffer.getvalue()).decode()))


def test_text_and_concurrent_requests(service):
    assert answer(service("What is 17 times 23? Reply with only the integer.")) == "391"
    prompts = [f"Repeat exactly this code and nothing else: FLARE{i}128" for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(service, prompts))
    for index, result in enumerate(results):
        assert f"FLARE{index}128" in answer(result)


@pytest.mark.parametrize("options", [
    {"enable_thinking": True},
    {"enable_thinking": False},
    {"chat_template_kwargs": {"thinking": False}},
])
def test_reasoning_is_separate(service, options):
    result = service("What is 17 times 23? Put only the integer in the final answer.",
                     **options)
    assert result["choices"][0]["message"]["reasoning_content"].strip()
    assert answer(result) == "391"


def test_streamed_reasoning_and_answer(service):
    events = service("What is 17 times 23? Put only the integer in the final answer.",
        enable_thinking=False,
        stream=True, stream_options=dict(include_usage=True))
    choices = [choice for event in events for choice in event["choices"]]
    reasoning = "".join(choice.get("delta", {}).get("reasoning_content") or "" for choice in choices)
    content = "".join(choice.get("delta", {}).get("content") or "" for choice in choices)
    assert reasoning.strip()
    assert content.strip() == "391"
    assert choices[-1]["finish_reason"] == "stop"
    assert any(event.get("usage", {}).get("completion_tokens", 0) > 0
               for event in events if event.get("usage"))


def test_xtml_tool_call_and_result(service):
    tools = [dict(type="function", function=dict(name="lookup_weather", description="Get a city's weather.",
        parameters=dict(type="object", properties=dict(city=dict(type="string")), required=["city"])))]
    prompt = "Use lookup_weather to get the weather in Paris."
    result = service(prompt, tools=tools, tool_choice="required")
    message = result["choices"][0]["message"]
    assert message["reasoning_content"].strip()
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    call, = message["tool_calls"]
    assert call["function"]["name"] == "lookup_weather"
    assert json.loads(call["function"]["arguments"])["city"] == "Paris"
    result = service("", tools=tools, messages=[dict(role="user", content=prompt), message,
        dict(role="tool", tool_call_id=call["id"], content='{"temperature_c": 21, "condition": "sunny"}')])
    assert "21" in answer(result)


def test_structured_response(service):
    schema = dict(type="object", properties=dict(result=dict(type="integer")), required=["result"],
                  additionalProperties=False)
    result = service("Return the product of 17 and 23 as result.",
        response_format=dict(type="json_schema", json_schema=dict(name="product", strict=True, schema=schema)))
    assert json.loads(answer(result)) == {"result": 391}


def test_images_and_chunked_prefill(service):
    content = [image("red", "FIRST 128"), image("blue", "SECOND 256"),
        dict(type="text", text="Read both image labels in their original order. Only output the labels.")]
    for prefix in ([], [dict(type="text", text="filler " * 4100)]):
        response = service(prefix + content)
        if prefix:
            assert response["usage"]["prompt_tokens"] > 4096
        result = answer(response)
        assert "FIRST 128" in result and "SECOND 256" in result
        assert result.index("FIRST 128") < result.index("SECOND 256")


@pytest.mark.parametrize("format", ["PNG", "WEBP"])
def test_five_images_share_the_visual_token_budget(service, format):
    labels = ["FIRST 128", "SECOND 256", "THIRD 512", "FOURTH 768", "FIFTH 1024"]
    content = [image(color, label, format=format) for color, label in
               zip(("red", "blue", "green", "orange", "purple"), labels)]
    result = answer(service(content + [dict(type="text",
        text="Read every image label in its original order. Only output the labels.")]))
    assert all(label in result for label in labels), result
    assert [result.index(label) for label in labels] == sorted(result.index(label) for label in labels)


def test_image_budget_rejection_preserves_service(service):
    url = os.environ["FLARE_KIMI3_TEST_URL"].rstrip("/")
    content = [image("red", "CAPACITY", (3584, 3584))] * 2
    body = dict(messages=[dict(role="user", content=content)], max_tokens=16)
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as rejected:
        urllib.request.urlopen(request, timeout=60)
    assert rejected.value.code == 400
    detail = rejected.value.read().decode()
    assert "131072 visual patches" in detail and "65536" in detail
    assert answer(service("What is 17 times 23? Reply with only the integer.")) == "391"


def test_long_context_and_reused_prefix(service):
    prefix = " ".join(f"Item {i} records code {i * 17 % 997}." for i in range(1200))
    prompt = prefix + "\nWhat code does item 211 record? Reply with only the integer."
    first, second = service(prompt), service(prompt)
    assert first["usage"]["prompt_tokens"] > 4096
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    assert answer(first) == answer(second) == "596"


def test_largest_image_graph(service):
    result = service([image("red", "CAPACITY", (3584, 3584)), dict(type="text",
        text="What is the dominant background color? Reply with only the color name.")])
    assert result["usage"]["prompt_tokens_details"]["multimodal_tokens"]["image"] == 16384
    assert answer(result).lower().rstrip(".") == "red"


def test_decode_across_latent_page_boundaries(service):
    result = service("Count from 1 through 160 inclusive, separated by commas. Output only the numbers, "
                     "with no omissions.", max_tokens=2048)
    assert result["usage"]["completion_tokens"] > 256
    assert [int(value) for value in re.findall(r"\d+", answer(result))] == list(range(1, 161))
