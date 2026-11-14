# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU preparation core and command-line smoke checks."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tomllib


ROOT = pathlib.Path(__file__).resolve().parents[2]

FLASHINFER_REQUIREMENT = re.compile(
    r"flashinfer-python\[b12x\] @ git\+(?P<repository>https://\S+)@main"
)


def test_pyproject_declares_preparation_entrypoint_and_cores():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    includes = set(pyproject["tool"]["setuptools"]["cores"]["find"]["include"])

    assert pyproject["project"]["name"] == "ignition"
    assert "ignition*" in includes
    assert not any(name.startswith(("flare", "serve")) for name in includes)
    assert pyproject["project"]["dependencies"] == []
    assert pyproject["project"]["scripts"] == {"ignition": "ignition.__main__:main"}


def test_flashinfer_declaration_tracks_main_statically():
    """Static declaration check. `.github/workflows/flashinfer-main.yml` proves
    resolution, installation, and import; this test only rejects a hand-edited
    revision."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = [
        requirement
        for requirement in pyproject["project"]["optional-dependencies"]["prepare"]
        if requirement.startswith("flashinfer-python")
    ]
    assert len(requirements) == 1, requirements
    match = FLASHINFER_REQUIREMENT.fullmatch(requirements[0])
    assert match, requirements
    repository = match["repository"]
    assert "flashinfer-python" in pyproject["tool"]["uv"]["upgrade-package"]

    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    flashinfer = [package for package in lock["package"] if package["name"] == "flashinfer-python"]
    assert len(flashinfer) == 1
    locked_source = re.compile(re.escape(f"{repository}?rev=main") + r"#[0-9a-f]{40}")
    assert locked_source.fullmatch(flashinfer[0]["source"]["git"]), flashinfer[0]["source"]

    ignition = [package for package in lock["package"] if package["name"] == "ignition"]
    assert len(ignition) == 1
    requires_dist = [
        requirement
        for requirement in ignition[0]["metadata"]["requires-dist"]
        if requirement["name"] == "flashinfer-python"
    ]
    assert requires_dist == [
        {
            "name": "flashinfer-python",
            "extras": ["b12x"],
            "marker": "extra == 'prepare'",
            "git": f"{repository}?rev=main",
        }
    ]


def test_core_import_has_no_runtime_or_cuda_side_effects():
    result = subprocess.run(
        [sys.executable, "-c", "import sys, ignition; "
         "assert ignition.__all__ == ['__version__']; "
         "assert 'torch' not in sys.modules; "
         "assert 'flare_heliopause' not in sys.modules; "
         "assert not any(n == 'serve' or n.startswith('serve.') for n in sys.modules)"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (ROOT / "serve").exists()


def test_server_cli_help_does_not_prepare_model():
    result = subprocess.run(
        [sys.executable, "-m", "ignition", "--help"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    for command in ("export", "debug-serve", "bench"):
        assert f"\n    {command} " in result.stdout
    for command in ("serve", "run", "prepare", "plan", "models", "pull", "status", "config", "skill"):
        assert f"\n    {command} " not in result.stdout
