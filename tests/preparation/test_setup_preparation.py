# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation setup updates dependencies with the native build policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("setup_preparation", ROOT / "scripts/setup_preparation.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def test_setup_updates_preparation_dependencies(monkeypatch):
    """Command-construction test; it does not run uv or prove resolution."""
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert setup.main() == 0
    assert len(calls) == 1
    command = calls[0][0][0]
    assert command == [
        "uv", "sync", "--upgrade-package", "flashinfer-python",
        "--extra", "prepare", "--extra", "test",
        "--reinstall-package", "flashinfer-python", "--reinstall-package", "flare-heliopause",
    ]
    assert "--locked" not in command
    assert "--frozen" not in command
    assert calls[0][1]["env"]["BUILD_NVEP"] == "0"
    assert calls[0][1]["cwd"] == ROOT
    assert calls[0][1]["check"] is True


@pytest.mark.parametrize("error", [OSError("uv unavailable"), subprocess.CalledProcessError(1, "uv")])
def test_failed_sync_reports_error(monkeypatch, capsys, error):
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(setup.subprocess, "run", fail)
    assert setup.main() == 1
    assert calls == 1
    assert str(error) in capsys.readouterr().err
