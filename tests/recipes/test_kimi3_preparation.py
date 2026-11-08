# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 and K3.1 preparation preserve explicit vision capacity."""

import importlib

import pytest


@pytest.fixture(params=("kimi3",))
def config_type(request, monkeypatch):
    import torch

    def no_cuda(*args, **kwargs):
        pytest.fail("preparation configuration tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return importlib.import_module(f"ignition.recipes.{request.param}.bootstrap").PreparationConfig


@pytest.mark.parametrize("capacity", [None, 0])
def test_text_only_has_no_image_slots(config_type, capacity):
    config = config_type(vision=False, image_concurrency=capacity)
    assert config.image_concurrency == 0 and not config.vision


@pytest.mark.parametrize("capacity", [1, -1, False, 0.0])
def test_text_only_rejects_image_capacity(config_type, capacity):
    with pytest.raises(ValueError, match="image concurrency"):
        config_type(vision=False, image_concurrency=capacity)


@pytest.mark.parametrize("active,capacity,expected", [(1, None, 1), (4, None, 4), (4, 1, 1)])
def test_vision_keeps_its_default_and_explicit_capacity(config_type, active, capacity, expected):
    config = config_type(max_active=active, image_concurrency=capacity)
    assert config.vision and config.image_concurrency == expected


@pytest.mark.parametrize("capacity", [0, 5])
def test_vision_rejects_capacity_outside_active_requests(config_type, capacity):
    with pytest.raises(ValueError, match="image concurrency"):
        config_type(max_active=4, image_concurrency=capacity)
