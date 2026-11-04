# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom prepared operators survive Python exit and native package replay."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA GPU")
def test_inkling_operator_package_native_replay(tmp_path):
    root = Path(__file__).resolve().parents[2]
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "helios/target"))
    subprocess.run(["cargo", "build", "-j", "4", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", "test_fixed_runtime_package"],
        cwd=root, check=True, timeout=600)
    exported = subprocess.run([sys.executable, str(root / "helios/tests/inkling_operator_package_fixture.py"), str(tmp_path)],
        cwd=root, text=True, capture_output=True, timeout=600)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    assert "exported prepared relative attention and residual convolution" in exported.stdout
    replay = subprocess.run([str(target / "debug/test_fixed_runtime_package"), str(tmp_path)],
        cwd=root, text=True, capture_output=True, timeout=180)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "native fixed-address replay passed" in replay.stdout
