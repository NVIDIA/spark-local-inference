# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled forwards persist under the configured kernel cache."""

import types

import torch

from ignition.preparation import aot


def test_forward_cache_uses_the_configured_root(monkeypatch, tmp_path):
    properties = types.SimpleNamespace(name="RTX PRO 6000 Blackwell", major=12, minor=0, multi_processor_count=188)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    configured = aot.ForwardCache(namespace={"model": "root"}, device="cpu", root=tmp_path)
    assert configured.root.parent == tmp_path / "forwards"
    default = aot.ForwardCache(namespace={"model": "root"}, device="cpu")
    assert default.root.parent == aot.DEFAULT_CACHE_ROOT / "forwards"
    assert configured.root.name == default.root.name
