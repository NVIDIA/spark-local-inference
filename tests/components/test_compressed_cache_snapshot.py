# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation declares immutable snapshot operands and bounded priming storage."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import compressed_mla


@pytest.mark.parametrize("requests,width", [(1, 8), (3, 7)])
def test_snapshot_writer_declares_operands_and_primes_valid_private_pages(monkeypatch, requests, width):
    source = torch.zeros(requests * width, 512, dtype=torch.bfloat16)
    cache = torch.empty(requests, (128 + width) * 528, dtype=torch.uint8)
    history = torch.empty(9, 128 * 528, dtype=torch.uint8)
    rows = torch.arange(requests, dtype=torch.int64)
    slots = torch.empty(requests * width, dtype=torch.int64)
    requests_out, calls = [], []

    def plan(query, *, device):
        return SimpleNamespace(query=query, handle=1,
            request=lambda **kwargs: SimpleNamespace(**kwargs))

    monkeypatch.setattr(compressed_mla.mla, "plan_cache_writer", plan)
    monkeypatch.setattr(compressed_mla, "declare", lambda session, values: requests_out.extend(values))
    monkeypatch.setattr(compressed_mla, "_write", lambda *args: calls.append(args))
    writer = compressed_mla.PreparedCompressedCacheWriter(source, cache, slots,
        page_size=128 + width, kind="swa", session=object(), name="window",
        history=history, history_rows=rows, snapshot_tokens=128)
    assert writer.plan.query.snapshot_tokens == 128
    assert writer.plan.query.max_requests == requests
    observed = []
    call = requests_out[0].prepare_call(SimpleNamespace(run=lambda *args, **kwargs: observed.append((args, kwargs))))
    call.invoke()
    args, kwargs = observed[0]
    assert args[1].shape == cache.shape
    assert kwargs["history"].shape == (requests + 1, 128 * 528)
    assert torch.equal(kwargs["history_rows"], torch.arange(1, requests + 1))
    assert torch.all(args[2] % (128 + width) >= 128)
    writer(source)
    assert calls[0][4] is history and calls[0][5] is rows


@pytest.mark.parametrize("history,rows,tokens", [(None, None, 128), (True, None, 128), (True, True, 0)])
def test_snapshot_writer_rejects_incomplete_operands(history, rows, tokens):
    source = torch.empty(8, 512, dtype=torch.bfloat16)
    cache = torch.empty(1, 136 * 528, dtype=torch.uint8)
    selected = torch.zeros(1, dtype=torch.int64)
    with pytest.raises(ValueError):
        compressed_mla.PreparedCompressedCacheWriter(source, cache, torch.empty(8, dtype=torch.int64),
            page_size=136, kind="swa", session=object(), name="window",
            history=cache if history else None, history_rows=selected if rows else None,
            snapshot_tokens=tokens)
