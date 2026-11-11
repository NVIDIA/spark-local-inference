# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared physical KV/snapshot backing through native Qwen and GLM services.

Uses FLARE_OFFLOAD_TEST_CORE, all assigned CUDA_VISIBLE_DEVICES, and optional
FLARE_TEST_PUBLISHER_KEYS. The core must hold at least four concurrent requests
with 130,000 output tokens reserved each, including MTP3 graphs.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import urllib.request

import pytest
from google.protobuf.json_format import MessageToDict

from tests.server.test_prefix_cache_offload import pytestmark, server
from ignition.preparation.core_file import read_index


@pytest.mark.parametrize("resident_prompts", [False, True], ids=["reservations", "resident-prompts"])
def test_concurrent_requests_reclaim_checkpoint_backing(tmp_path, resident_prompts):
    core, _ = read_index(os.environ["FLARE_OFFLOAD_TEST_CORE"])
    config = MessageToDict(core.contract, preserving_proto_field_name=True)
    concurrency = min(8, core.ranks[0].execution.max_active)
    assert concurrency >= 4
    flags = ("--max-concurrency", str(concurrency), "--mtp-adaptive-verification")
    with server(tmp_path, "shared-cache", drafts=3, flags=flags) as service:
        model = json.loads(service.get("/v1/models"))["data"][0]["id"]
        page_tokens = int(config["model_contract"]["cache"]["page_tokens"])
        capacity = int(service.metrics()["flare_free_kv_pages"] * page_tokens)

        def ask(tag, copies=1, maximum=1024):
            text = f"Reference {tag}. " + "The sum of seventeen and twenty-five is forty-two. " * copies
            messages = [{"role": "user", "content": text + "What is 17 + 25? Reply with only the number."}]
            body = {"model": model, "messages": messages, "temperature": 1,
                    "seed": 1234, "max_tokens": maximum, "enable_thinking": False,
                    "chat_template_kwargs": {"reasoning_effort": "low"}}
            if resident_prompts:
                # Exercise adaptive widths with deterministic token selection during long prefill and restore.
                body["top_k"] = 1
            request = urllib.request.Request(service.url + "/v1/chat/completions",
                json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
            assert result["choices"][0]["message"]["content"].strip() == "42", result
            assert result["choices"][0]["finish_reason"] == "stop", result
            return result

        copies, maximum = 600, 130000
        if resident_prompts:
            short = ask(100, 10)["usage"]["prompt_tokens"]
            long = ask(100, 20)["usage"]["prompt_tokens"]
            tokens_per_copy = (long - short) // 10
            assert tokens_per_copy > 0
            target = min(int(capacity * .85) // concurrency, int(config["max_sequence_tokens"]) - 2048)
            copies = (target - short) // tokens_per_copy + 10
            maximum = 1024
        for tag in range(24):
            ask(tag)
        before = service.metrics()
        assert before["flare_cached_checkpoints"] > 0
        assert before["flare_snapshot_backing_pages"] > 0
        peak = 0
        resident_prefill = 0
        least_snapshot_pages = before["flare_snapshot_backing_pages"]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(ask, 100 + index, copies, maximum) for index in range(concurrency)]
            while not all(f.done() for f in futures):
                counters = service.metrics()
                peak = max(peak, counters["flare_active_requests"])
                if counters["flare_active_requests"] == concurrency:
                    resident_prefill = max(resident_prefill,
                        counters["flare_prefill_tokens_total"] - before["flare_prefill_tokens_total"])
                least_snapshot_pages = min(least_snapshot_pages, counters["flare_snapshot_backing_pages"])
                time.sleep(.01)
            results = [future.result() for future in futures]
        after = service.metrics()
        assert peak == concurrency
        assert least_snapshot_pages < before["flare_snapshot_backing_pages"]
        assert after["flare_snapshot_evictions_total"] > before["flare_snapshot_evictions_total"]
        assert after["flare_active_requests"] == 0
        if resident_prompts:
            assert sum(result["usage"]["prompt_tokens"] for result in results) > capacity * .8
            assert resident_prefill > capacity * .75
        ask(103, copies)
        ask(103, copies)
        assert service.metrics()["flare_cache_reused_tokens_total"] > after["flare_cache_reused_tokens_total"]
        print({"peak_active": peak, "resident_prefill_tokens": resident_prefill,
               "checkpoint_pages_before": before["flare_snapshot_backing_pages"],
               "checkpoint_pages_under_pressure": least_snapshot_pages,
               "evicted": after["flare_snapshot_evictions_total"] - before["flare_snapshot_evictions_total"]})
