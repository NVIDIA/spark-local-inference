# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The replay qualifier observes borrowed-state restoration before resetting it."""

from types import SimpleNamespace

import pytest
import torch

from qualify import graph_execution
from ignition.recipes.qwen38.state import EOS_TOKEN_ID


@pytest.mark.parametrize("corruption", [None, "state", "null_history", "history"])
def test_pristine_state_counts_all_bytes_and_checks_eos_history(corruption):
    backing = torch.zeros((3, 32), dtype=torch.uint8)
    history = backing[:, 8:24].view(torch.int64)
    history[1:].fill_(EOS_TOKEN_ID)
    if corruption == "state":
        backing[2, 31] = 1
    elif corruption == "null_history":
        history[0, 0] = EOS_TOKEN_ID
    elif corruption == "history":
        history[1, 0] = EOS_TOKEN_ID + 1
    before = backing.clone()
    checks = []
    evidence = SimpleNamespace(check=lambda name, passed, **details: checks.append((name, passed, details)))
    graph_execution.verify_prepared_hybrid_state(SimpleNamespace(backing=backing, ple_history=history), evidence)
    assert len(checks) == 1
    name, passed, details = checks[0]
    assert name == "prepared_hybrid_state_pristine"
    assert passed is (corruption is None)
    assert details["expected_nonzero_bytes"] == 12
    assert torch.equal(backing, before)


@pytest.mark.parametrize("corruption", [False, True])
def test_pristine_state_inspects_late_padding_with_bounded_reductions(monkeypatch, corruption):
    verify = graph_execution.verify_prepared_hybrid_state
    monkeypatch.setattr(graph_execution, "_PRISTINE_COUNT_CHUNK_BYTES", 64)
    backing = torch.zeros((3, 80), dtype=torch.uint8)
    history = backing[:, 64:80].view(torch.int64)
    history[1:].fill_(EOS_TOKEN_ID)
    if corruption:
        backing[2, 63] = 11
    before = backing.clone()
    reduce = torch.count_nonzero
    reductions = []

    def bounded_count(tensor, *args, **kwargs):
        assert tensor.nbytes <= 64
        if tensor.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr():
            reductions.append((tensor.storage_offset(), tensor.numel()))
        return reduce(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "count_nonzero", bounded_count)
    checks = []
    evidence = SimpleNamespace(check=lambda name, passed, **details: checks.append((name, passed, details)))
    verify(SimpleNamespace(backing=backing, ple_history=history), evidence)
    assert reductions == [(0, 64), (64, 64), (128, 64), (192, 48)]
    assert checks[0][1] is not corruption
    assert checks[0][2]["observed_nonzero_bytes"] == 12 + int(corruption)
    assert checks[0][2]["history_valid"] is True
    assert checks[0][2]["count_chunk_bytes"] == 64
    assert torch.equal(backing, before)
