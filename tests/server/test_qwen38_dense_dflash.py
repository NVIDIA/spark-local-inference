# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 acceptance and complete prefix restore through an exported Qwen core.

Set FLARE_RUN_CUDA_TESTS=1, FLARE_QWEN_DFLASH_TEST_URL, and
FLARE_QWEN_DFLASH_SNAPSHOT. Serve a Qwen3.8-27B core through flare run
with fixed verification (--no-mtp-adaptive-verification). Set
FLARE_QWEN_DFLASH_VERIFY_WIDTH to the prepared verification width, including
the pending token: 8 by default, or 16 for fifteen drafts.
Metric checks require an otherwise idle service.
"""

import os
import time
from pathlib import Path

import pytest

from tests.server.test_muse_spark import NativeMuse
from tests.server.test_muse_spark_dflash import rounds, test_dflash_full_prompt_hit_and_changed_suffix

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("FLARE_QWEN_DFLASH_TEST_URL"),
    reason="requires an exported Qwen3.8-27B DFlash2 core running through flare run",
)


class NativeQwen(NativeMuse):
    def __init__(self):
        from transformers import PreTrainedTokenizerFast
        self.url = os.environ["FLARE_QWEN_DFLASH_TEST_URL"].rstrip("/")
        self.key = os.environ.get("FLARE_API_TEST_KEY", "local")
        self.snapshot = Path(os.environ["FLARE_QWEN_DFLASH_SNAPSHOT"])
        self.verify_width = int(os.environ.get("FLARE_QWEN_DFLASH_VERIFY_WIDTH", "8"))
        assert self.verify_width > 1
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.snapshot)
        self.template = (self.snapshot / "chat_template.jinja").read_text()
        cards = self.request("/v1/models")["data"]
        assert len(cards) == 1
        self.card = cards[0]
        self.model = self.card["id"]
        assert self.model == "Qwen3.8-27B"
        assert self.card["flare"]["input_modalities"] == ["text"]

    def body(self, prompt, **options):
        body = dict(model=self.model, messages=[dict(role="user", content=prompt)],
            temperature=.7, top_p=.8, top_k=20, seed=41873, max_tokens=512,
            return_token_ids=True, chat_template_kwargs=dict(enable_thinking=False, add_generation_prompt=True))
        body.update(options)
        return body


@pytest.fixture(scope="module")
def service():
    return NativeQwen()


def test_dflash_short_tails_and_fixed_rounds_commit_accepted_prefixes(service):
    prompt = (f"Counting request {time.time_ns()}. Count from 1 through 500, "
              "one integer per line, with no omissions.")
    before = service.metrics()
    for budget in (2, 64, 2):
        result = service.request("/v1/chat/completions", service.body(prompt, max_tokens=budget))
        choice = result["choices"][0]
        assert choice["finish_reason"] == "length", result
        assert result["usage"]["completion_tokens"] == len(choice["token_ids"]) == budget
        after = service.metrics()
        assert rounds(after, service.verify_width) > rounds(before, service.verify_width), (budget, before, after)
        accepted = after["flare_accepted_draft_tokens_total"] - before["flare_accepted_draft_tokens_total"]
        offered = after["flare_verification_draft_tokens_total"] - before["flare_verification_draft_tokens_total"]
        draft_rounds = after["flare_draft_request_rounds_total"] - before["flare_draft_request_rounds_total"]
        if budget == 64:
            assert 0 < accepted <= offered and draft_rounds > 0, (accepted, offered, draft_rounds)
        else:
            # The first output is sampled by prefill or resume; one output remains.
            assert accepted == offered == 0, (accepted, offered)
        print(dict(output_budget=budget, verification_width=service.verify_width,
            verification_rounds=rounds(after, service.verify_width) - rounds(before, service.verify_width),
            accepted_drafts=accepted, offered_drafts=offered, draft_rounds=draft_rounds))
        assert after["flare_requests_failed_total"] == before["flare_requests_failed_total"]
        before = after
