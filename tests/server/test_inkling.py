# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance for a TP8 Inkling package exported with prompt logprobs and served by Flare.

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


@pytest.fixture(scope="module")
def service():
    from transformers import PreTrainedTokenizerFast

    root = Path(os.environ["FLARE_INKLING_SNAPSHOT"])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(root)
    template = (root / "chat_template.jinja").read_text()
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
    return ask, request, card["id"]


def text(result):
    return (result["choices"][0]["message"].get("content") or "").strip()


@pytest.mark.parametrize("index", range(20))
def test_text_and_unicode(service, index):
    ask, _, _ = service
    if index < 12:
        a, b = index + 13, 29 - index
        prompt, expected = f"What is {a} times {b}? Reply with only the integer.", str(a*b)
    else:
        expected = ["INKLING-317", "café", "世界", "naïve", "á", "Δοκιμή", "مرحبا", "日本語"][index-12]
        prompt = "Repeat exactly this text, without quotation marks or explanation: " + expected
    result = ask([dict(role="user", content=prompt)])
    assert text(result) == expected
    assert result["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "max"])
def test_reasoning_headers(service, effort):
    result = service[0]([dict(role="user", content="A store has 7 boxes of 13 bolts, sells 19 bolts and receives two more boxes. How many bolts remain? Put only the integer in the final answer.")],
        max_tokens=2048, chat_template_kwargs=dict(reasoning_effort=effort))
    assert text(result) == "98"
    assert result["choices"][0]["message"]["reasoning_content"]


@pytest.mark.parametrize("required", [True, False])
def test_json_tool_envelope(service, required):
    value = dict(query="café 世界", count=3)
    function = dict(name="record", parameters=dict(type="object", properties=dict(query=dict(type="string"),
        count=dict(type="integer")), required=["query", "count"]))
    choice = "required" if required else dict(type="function", function=dict(name="record"))
    result = service[0]([dict(role="user", content="Call record with exactly " + json.dumps(value))],
        tools=[dict(type="function", function=function)], tool_choice=choice)
    call, = result["choices"][0]["message"]["tool_calls"]
    assert call["function"]["name"] == "record"
    assert json.loads(call["function"]["arguments"]) == value
    assert result["choices"][0]["finish_reason"] == "tool_calls"


def test_prefix_branches_preserve_convolution_histories(service):
    prefix = "Preserve this context before answering the final question. " * 1600
    history = [dict(role="user", content=prefix), dict(role="assistant", content="I have read the context.")]
    def ask(index):
        return service[0]([*history, dict(role="user", content=f"Reply with only INKLING-BRANCH-{index}.")])
    first, second = ask(0), ask(0)
    assert first["choices"][0]["token_ids"] == second["choices"][0]["token_ids"]
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] > 4096
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(ask, range(4)))
    assert [text(result) for result in results] == [f"INKLING-BRANCH-{i}" for i in range(4)]


def test_prompt_scores_and_text_only_rejection(service):
    result = service[0]([dict(role="user", content="A cache preserves context. café 世界.")], max_tokens=1, prompt_logprobs=2)
    assert len(result["prompt_logprobs"]) == len(result["prompt_token_ids"])
    assert result["prompt_logprobs"][0] is None
    for token, scores in zip(result["prompt_token_ids"][1:], result["prompt_logprobs"][1:]):
        assert str(token) in scores
        assert all(math.isfinite(value["logprob"]) and value["logprob"] <= 0 for value in scores.values())
    for content in ([dict(type="image_url", image_url=dict(url="data:image/png;base64,AA=="))],
                    [dict(type="input_audio", input_audio=dict(data="AA==", format="wav"))]):
        with pytest.raises(urllib.error.HTTPError) as rejected:
            service[1]("/v1/chat/completions", dict(model=service[2], messages=[dict(role="user", content=content)]))
        assert rejected.value.code == 400
