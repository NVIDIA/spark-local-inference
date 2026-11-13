# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup producers and compiled consumers survive export into native TP4 ranks."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four assigned GPUs and FLARE_RUN_TP_CUDA_TESTS=1",
)


@pytest.fixture(scope="module")
def startup_core(tmp_path_factory):
    import torch

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign four CUDA devices explicitly"
    assert torch.cuda.device_count() == 4
    root = Path(__file__).resolve().parents[2]
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "helios/target"))
    subprocess.run([
        "cargo", "build", "-j", "4", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-core,test-tools",
        "--bin", "test_fixed_runtime_core",
    ], cwd=root, check=True, timeout=600)
    directory = tmp_path_factory.mktemp("startup-core")
    exported = subprocess.run([
        sys.executable, str(root / "helios/tests/collective_core_fixture.py"),
        str(directory), "--startup-tuning",
    ], cwd=root, text=True, capture_output=True, timeout=600)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    assert "all four preparation ranks exited" in exported.stdout
    return root, target / "debug/test_fixed_runtime_core", directory


@pytest.mark.parametrize("failure", (False, True), ids=("persistent-shared-outputs", "rank-status-failure"))
def test_exported_startup_producers_precede_native_readiness(startup_core, failure):
    root, executable, directory = startup_core
    command = [str(executable), str(directory), "--tp4-collective"]
    if failure:
        command.append("--startup-failure")
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = ("native startup tuning rejected failing rank before request readiness"
                if failure else "native TP4 collective core replay passed after Python exit")
    assert expected in result.stdout
