# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance for an Inkling NVFP4 package exported with prompt logprobs and served by Flare.

Set FLARE_RUN_CUDA_TESTS=1, FLARE_INKLING_TEST_URL and FLARE_INKLING_SNAPSHOT.
The package uses the default 262144-token context and concurrency four.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or not os.environ.get("FLARE_INKLING_TEST_URL"), reason="requires an exported Inkling package served by flare run")


LONG_INPUTS = (4097, 8193, 16385, 32769, 65537, 262080)
CASES = ([("text", i) for i in range(32)] + [("tool", i) for i in range(12)]
         + [("long", i) for i in range(6)])


@pytest.fixture(scope="module")
def checkpoint_frontend():
    from transformers import PreTrainedTokenizerFast

    root = Path(os.environ["FLARE_INKLING_SNAPSHOT"])
    return PreTrainedTokenizerFast.from_pretrained(root), (root / "chat_template.jinja").read_text()


@pytest.fixture(scope="module")
def service(checkpoint_frontend):
    tokenizer, template = checkpoint_frontend
    url = os.environ["FLARE_INKLING_TEST_URL"].rstrip("/")
    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url + path, data, {"Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("FLARE_API_TEST_KEY", "local")})
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.load(response)
    card, = request("/v1/models")["data"]
    assert "inkling" in card["id"].lower()
    assert card["max_model_len"] == 262144
    assert card["flare"]["max_concurrency"] == 4
    assert card["flare"]["input_modalities"] == ["text"]
    assert card["flare"]["sampling_defaults"]["temperature"] == 1.
    def ask(messages, **options):
        body = dict(model=card["id"], messages=messages, seed=58123, max_tokens=128,
            return_token_ids=True, chat_template_kwargs=dict(reasoning_effort="none"))
        body.update(options)
        result = request("/v1/chat/completions", body)
        expected_messages = messages
        if options.get("tool_choice") == "required":
            expected_messages = [dict(role="system", content="You must call at least one of the provided tools in your next response. Choose the appropriate tool and arguments."), *messages]
        expected = tokenizer.apply_chat_template(expected_messages, chat_template=template, tokenize=True,
            return_dict=False, add_generation_prompt=True, tools=options.get("tools"), **body["chat_template_kwargs"])
        assert result["prompt_token_ids"] == expected
        assert result["usage"]["completion_tokens"] == len(result["choices"][0]["token_ids"])
        return result
    return ask


def text(result):
    return (result["choices"][0]["message"].get("content") or "").strip()


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "max"])
def test_reasoning_headers(service, effort):
    result = service([dict(role="user", content="A store has 7 boxes of 13 bolts, sells 19 bolts and receives two more boxes. How many bolts remain? Put only the integer in the final answer.")],
        max_tokens=2048, chat_template_kwargs=dict(reasoning_effort=effort))
    assert text(result) == "98"
    assert result["choices"][0]["message"]["reasoning_content"]


def test_prefix_branches_preserve_convolution_histories(service):
    prefix = "Preserve this context before answering the final question. " * 1600
    history = [dict(role="user", content=prefix), dict(role="assistant", content="I have read the context.")]
    def ask(index):
        return service([*history, dict(role="user", content=f"Reply with only INKLING-BRANCH-{index}.")])
    first, second = ask(0), ask(0)
    assert first["choices"][0]["token_ids"] == second["choices"][0]["token_ids"]
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] > 4096
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(ask, range(4)))
    assert [text(result) for result in results] == [f"INKLING-BRANCH-{i}" for i in range(4)]


def test_prompt_scores(service):
    result = service([dict(role="user", content="A cache preserves context. café 世界.")], max_tokens=1, prompt_logprobs=2)
    assert len(result["prompt_logprobs"]) == len(result["prompt_token_ids"])
    assert result["prompt_logprobs"][0] is None
    for token, scores in zip(result["prompt_token_ids"][1:], result["prompt_logprobs"][1:]):
        assert str(token) in scores
        assert all(math.isfinite(value["logprob"]) and value["logprob"] <= 0 for value in scores.values())


@pytest.mark.parametrize("content", [
    [dict(type="image_url", image_url=dict(url="data:image/png;base64,AA=="))],
    [dict(type="input_audio", input_audio=dict(data="AA==", format="wav"))],
], ids=["image", "audio"])
def test_unsupported_media_is_rejected(service, content):
    with pytest.raises(urllib.error.HTTPError) as rejected:
        service([dict(role="user", content=content)])
    assert rejected.value.code == 400


def long_prompt(frontend, target):
    tokenizer, template = frontend
    head = "The secret code is INKLING-84127. Remember it while reading the filler.\n"
    tail = "\nWhat is the secret code? Reply with only the code."

    def length(prompt):
        return len(tokenizer.apply_chat_template([dict(role="user", content=prompt)],
            chat_template=template, tokenize=True, return_dict=False,
            add_generation_prompt=True, reasoning_effort="none"))

    # Each additional filler contributes one token in the pinned checkpoint tokenizer.
    count = target - length(head + " filler" + tail) + 1
    prompt = head + " filler" * count + tail
    assert length(prompt) == target
    return prompt


@pytest.mark.parametrize("kind,index", CASES, ids=[f"{kind}-{i}" for kind, i in CASES])
def test_fifty_generation_cases(service, checkpoint_frontend, kind, index):
    options = dict(max_tokens=64)
    if kind == "text":
        if index < 20:
            a, b = 31 + index, 47 - index
            prompt = f"What is {a} times {b}? Reply with only the integer."
            expected = str(a * b)
        else:
            expected = ["INKLING-64283", "café noir", "世界和平", "naïve", "á", "Δοκιμή",
                        "مرحبا", "日本語", "Straße", "한국어", "mañana", "résumé"][index - 20]
            prompt = "Repeat exactly this text, without quotation marks or explanation: " + expected
    elif kind == "tool":
        expected = dict(value=index + 11, label=f"café 世界 {index}")
        function = dict(name="record_value", description="Record the supplied value exactly.",
            parameters=dict(type="object", properties=dict(value=dict(type="integer"), label=dict(type="string")),
                            required=["value", "label"], additionalProperties=False))
        options["tools"] = [dict(type="function", function=function)]
        options["tool_choice"] = ("required" if index % 2 else
                                 dict(type="function", function=dict(name="record_value")))
        prompt = "Call record_value once with exactly these arguments:\n" + json.dumps(expected)
    else:
        prompt = long_prompt(checkpoint_frontend, LONG_INPUTS[index])
        expected = "INKLING-84127"

    result = service([dict(role="user", content=prompt)], **options)
    choice = result["choices"][0]
    assert 0 < result["usage"]["completion_tokens"] <= 64
    assert result["usage"]["prompt_tokens"] == len(result["prompt_token_ids"])
    if kind == "tool":
        call, = choice["message"]["tool_calls"]
        assert call["function"]["name"] == "record_value"
        assert json.loads(call["function"]["arguments"]) == expected
        assert choice["finish_reason"] == "tool_calls"
    else:
        assert text(result) == expected
        assert choice["finish_reason"] == "stop"
    if kind == "long":
        assert result["usage"]["prompt_tokens"] == LONG_INPUTS[index]
