# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU clock control commands and cleanup."""

from types import SimpleNamespace

import pytest

from ignition.errors import PreparationError
from ignition.preparation import clocks


@pytest.fixture
def commands(monkeypatch):
    from cuda.bindings import nvml

    monkeypatch.setattr(nvml, "init_v2", lambda: None)
    monkeypatch.setattr(nvml, "shutdown", lambda: None)
    monkeypatch.setattr(nvml, "device_get_handle_by_uuid", lambda uuid: uuid)
    monkeypatch.setattr(nvml, "device_get_uuid", lambda handle: "GPU-selected")
    monkeypatch.setattr(nvml, "device_get_max_clock_info", lambda handle, kind: 2500)
    monkeypatch.setattr(clocks.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(clocks.os, "geteuid", lambda: 1000)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="clock changed", stderr="")

    monkeypatch.setattr(clocks.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("fail", [False, True])
def test_maximum_lock_and_reset(commands, fail, capsys):
    def use():
        with clocks.locked_gpu_clocks("selected"):
            assert len(commands) == 1
            if fail:
                raise RuntimeError("race failed")

    if fail:
        with pytest.raises(RuntimeError, match="race failed"):
            use()
    else:
        use()
    assert commands == [
        ["sudo", "-n", "/usr/bin/nvidia-smi", "--id", "GPU-selected", "--lock-gpu-clocks=2500,2500"],
        ["sudo", "-n", "/usr/bin/nvidia-smi", "--id", "GPU-selected", "--reset-gpu-clocks"],
    ]
    assert capsys.readouterr() == ("", "")


def test_lock_permission_failure_does_not_run_tuning(commands, monkeypatch):
    monkeypatch.setattr(clocks.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="", stderr="permission denied"))
    with pytest.raises(PreparationError, match="permission denied"):
        with clocks.locked_gpu_clocks("GPU-selected"):
            pytest.fail("tuning must not proceed unlocked")


def test_reset_failure_preserves_tuning_failure(commands, monkeypatch):
    run = clocks.subprocess.run

    def reset_fails(command, **kwargs):
        if command[-1] == "--reset-gpu-clocks":
            return SimpleNamespace(returncode=1, stdout="", stderr="device lost")
        return run(command, **kwargs)

    monkeypatch.setattr(clocks.subprocess, "run", reset_fails)
    with pytest.raises(ExceptionGroup) as failure:
        with clocks.locked_gpu_clocks("GPU-selected"):
            raise ValueError("bad candidate")
    assert isinstance(failure.value.exceptions[0], ValueError)
    assert isinstance(failure.value.exceptions[1], PreparationError)
