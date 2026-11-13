# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Arithmetic metadata reaches prepared MLA without changing its cache ABI."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import compressed_mla


def test_auto_passes_selector_to_the_constraint_api(monkeypatch):
    calls = []

    def recording_caps(*, device, num_q_heads, max_q_rows, max_width, swa_width, indexed_width,
                    max_page_table_width, swa_page_size, indexed_page_size, cache_format, mode,
                    decode_row_capacity, compute_mode):
        calls.append(locals())
        return object()

    monkeypatch.setattr(compressed_mla.mla, "Caps", recording_caps)
    monkeypatch.setattr(compressed_mla.mla, "plan", lambda *args, **kwargs: SimpleNamespace(request=lambda **kwargs: object()))
    monkeypatch.setattr(compressed_mla, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(compressed_mla, "register_finalizer", lambda *args, **kwargs: None)
    q = torch.empty((3, 16, 512), dtype=torch.bfloat16)
    cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    indices = torch.empty((3, 128), dtype=torch.int32)
    lengths = torch.empty(3, dtype=torch.int32)
    operation = compressed_mla.PreparedCompressedMLA(q, cache, indices, lengths, None, torch.empty_like(q),
        session=object(), name="attention", workspace=object())
    assert calls[0]["cache_format"] == "deepseek_v41"
    assert operation.window is cache and cache.dtype == torch.uint8
    assert calls[0]["compute_mode"] == "auto"


@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_explicit_bf16_is_operator_metadata_and_has_no_complete_override(monkeypatch, mode):
    monkeypatch.setattr(compressed_mla, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(compressed_mla, "register_finalizer", lambda *args, **kwargs: None)
    q = torch.empty((3, 16, 512), dtype=torch.bfloat16)
    cache = torch.empty((2, 128 * 528), dtype=torch.uint8)
    indices = torch.empty((3, 128), dtype=torch.int32)
    lengths = torch.empty(3, dtype=torch.int32)
    operation = compressed_mla.PreparedCompressedMLA(q, cache, indices, lengths, None, torch.empty_like(q),
        session=object(), name="attention", workspace=object(), compute_mode="bf16", mode=mode)
    assert operation.plan.query.compute_mode == "bf16"
    assert operation.plan.override is None
    assert operation.window is cache and operation.plan.query.kv_dtype == "uint8"


def test_preparation_config_rejects_missing_api_before_weight_loading(monkeypatch):
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    monkeypatch.setattr(compressed_mla.mla, "Caps", lambda: None)
    assert PreparationConfig().mla_compute_mode == "auto"
    with pytest.raises(ValueError, match="preparation API"):
        PreparationConfig(mla_compute_mode="bf16")


def test_target_prefill_and_draft_factories_keep_the_selected_arithmetic(monkeypatch):
    from ignition.recipes.deepseek41 import bootstrap

    calls = []
    monkeypatch.setattr(bootstrap, "PreparedDecoder", lambda **kwargs: calls.append(("target", kwargs)))
    monkeypatch.setattr(bootstrap, "PreparedDraft", lambda **kwargs: calls.append(("draft", kwargs)))
    target, draft = bootstrap._forward_factories("bf16", window_staging=object(),
        suffix_scratch=object(), vision=object())
    for mode in ("prefill", "decode", "verify"):
        target(mode=mode)
    draft()
    assert [kind for kind, _ in calls] == ["target", "target", "target", "draft"]
    assert all(arguments["mla_compute_mode"] == "bf16" for _, arguments in calls)
