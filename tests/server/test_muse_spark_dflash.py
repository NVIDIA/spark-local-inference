# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DFlash width transitions and complete prefix restores through native HTTP.

Use the exported-core configuration documented in test_muse_spark.py, with
seven drafts and fixed verification (--no-mtp-adaptive-verification). Run these
metric-delta checks without unrelated requests to the same server.
"""

import time

from tests.server.test_muse_spark import pytestmark, service


def rounds(metrics, width):
    return metrics[f'flare_speculative_rounds_total{{batch="1",width="{width}"}}']


def test_dflash_width_one_eight_one_and_accepted_commits(service):
    nonce = time.time_ns()
    prompt = f'Counting request {nonce}. Count from 1 through 500, one integer per line, with no omissions.'
    before = service.metrics()
    for budget, width in ((2, 1), (64, 8), (2, 1)):
        result = service.request('/v1/chat/completions', service.body(prompt, max_tokens=budget))
        choice = result['choices'][0]
        assert choice['finish_reason'] == 'length', result
        assert result['usage']['completion_tokens'] == len(choice['token_ids']) == budget
        after = service.metrics()
        assert rounds(after, width) > rounds(before, width), (width, before, after)
        if width == 8:
            accepted = after['flare_accepted_draft_tokens_total'] - before['flare_accepted_draft_tokens_total']
            offered = after['flare_verification_draft_tokens_total'] - before['flare_verification_draft_tokens_total']
            draft_rounds = after['flare_draft_request_rounds_total'] - before['flare_draft_request_rounds_total']
            assert 0 < accepted <= offered and draft_rounds > 0, (accepted, offered, draft_rounds)
            print(dict(accepted_drafts=accepted, offered_drafts=offered, draft_rounds=draft_rounds,
                       accepted_per_round=accepted / draft_rounds, accepted_fraction=accepted / offered))
        assert after['flare_requests_failed_total'] == before['flare_requests_failed_total']
        before = after


def test_dflash_full_prompt_hit_and_changed_suffix(service):
    reference = f'DFlash reference {time.time_ns()}.\n' + ' '.join(
        f'Item {index} records code {index * 17 % 997}.' for index in range(500))
    history = [dict(role='user', content=reference), dict(role='assistant', content='I have read the reference.')]

    def body(question):
        return service.body('', messages=[*history, dict(role='user', content=question)], max_tokens=512)

    question = 'What code does item 211 record? Reply with only the integer.'
    request = body(question)
    if len(service.expected_prompt(request)) % 128 == 0:
        request = body('Reference question. ' + question)
    tokens = service.expected_prompt(request)
    assert len(tokens) > 4096 and len(tokens) % 128 != 0
    cold = service.request('/v1/chat/completions', request)
    before = service.metrics()
    repeated = service.request('/v1/chat/completions', request)
    after = service.metrics()
    assert cold['prompt_token_ids'] == repeated['prompt_token_ids'] == tokens
    assert cold['choices'][0]['token_ids'] == repeated['choices'][0]['token_ids']
    assert repeated['usage']['prompt_tokens_details']['cached_tokens'] == len(tokens)
    assert after['flare_prefill_tokens_total'] == before['flare_prefill_tokens_total']
    for kind in ('restore', 'resume'):
        key = f'flare_graph_launches_total{{kind="{kind}"}}'
        assert after[key] > before[key], (kind, before, after)
    changed_request = body('What code does item 212 record? Reply with only the integer.')
    changed = service.request('/v1/chat/completions', changed_request)
    changed_tokens = service.expected_prompt(changed_request)
    assert changed['prompt_token_ids'] == changed_tokens
    cached = changed['usage']['prompt_tokens_details']['cached_tokens']
    common = next((index for index, pair in enumerate(zip(tokens, changed_tokens)) if pair[0] != pair[1]),
                  min(len(tokens), len(changed_tokens)))
    assert 4096 < cached <= common, (cached, common)
    final = service.metrics()
    assert final['flare_requests_failed_total'] == before['flare_requests_failed_total']
    assert final['flare_active_requests'] == final['flare_http_inflight_requests'] == 0
