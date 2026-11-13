# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coding-client acceptance against an actual Flare endpoint and temporary projects.

Set FLARE_API_TEST_URL and FLARE_RUN_CLIENT_TESTS=1. Client settings are isolated
from personal configuration; tools execute only the requested temporary-project task.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("FLARE_API_TEST_URL") or os.environ.get("FLARE_RUN_CLIENT_TESTS") != "1",
    reason="requires a native service and explicit coding-client test opt-in",
)
ROOT = Path(__file__).resolve().parents[2]
TASK = "Inspect calculator.py and test_calculator.py. Fix the addition bug in calculator.py, run python3 -m unittest, and report the result. Use tools to inspect, edit, and test the actual files. Do not modify the tests."
TEST_SOURCE = "import unittest\nfrom calculator import add\n\nclass Addition(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n        self.assertEqual(add(-1, 1), 0)\n"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b + 1\n")
    (tmp_path / "test_calculator.py").write_text(TEST_SOURCE)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def verify(project):
    assert (project / "test_calculator.py").read_text() == TEST_SOURCE
    result = subprocess.run(["python3", "-m", "unittest"], cwd=project, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "return a + b + 1" not in (project / "calculator.py").read_text()


def test_codex(project):
    assert shutil.which("codex"), "Codex must be installed"
    catalog = ROOT / "docs/clients/codex-models.json"
    base = os.environ["FLARE_API_TEST_URL"].rstrip("/")
    config = {
        "model": "flare", "model_provider": "flare", "model_catalog_json": str(catalog),
        "model_providers.flare.name": "Flare", "model_providers.flare.base_url": base + "/v1",
        "model_providers.flare.wire_api": "responses", "model_providers.flare.supports_websockets": False,
        "model_providers.flare.requires_openai_auth": False, "web_search": "disabled",
        "model_providers.flare.experimental_bearer_token": os.environ.get("FLARE_API_TEST_KEY", "local"),
    }
    command = ["codex", "exec", "--ignore-user-config", "--ephemeral", "--sandbox", "workspace-write", "--json"]
    for key, value in config.items():
        command += ["-c", f"{key}={json.dumps(value)}"]
    result = subprocess.run(command + [TASK], cwd=project, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"type":"turn.completed"' in result.stdout, result.stdout + result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert any("unittest" in e.get("item", {}).get("command", "") for e in events), result.stdout
    verify(project)


@pytest.mark.parametrize("bare", [False, True], ids=["normal", "bare"])
def test_claude_code(project, bare):
    assert shutil.which("claude"), "Claude Code must be installed"
    settings = project / "client-settings.json"
    settings.write_text(json.dumps({"env": {
        "ANTHROPIC_BASE_URL": os.environ["FLARE_API_TEST_URL"].rstrip("/"),
        "ANTHROPIC_API_KEY": os.environ.get("FLARE_API_TEST_KEY", "local"),
        "ANTHROPIC_MODEL": "fable",
    }}))
    command = [
        "claude", *(["--bare"] if bare else []),
        "--setting-sources", "", "--settings", str(settings),
        "--model", "fable", "--effort", "xhigh",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--no-session-persistence", "--tools", "Read,Edit,Write,Bash",
        "--allowedTools", "Read,Edit,Write,Bash", "--output-format", "json", "-p", TASK,
    ]
    result = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    response = json.loads(result.stdout)
    if isinstance(response, list):
        calls = [block for message in response for block in message.get("message", {}).get("content", []) if block.get("type") == "tool_use"]
        assert any("unittest" in call.get("input", {}).get("command", "") for call in calls), response
        response = response[-1]
    assert not response.get("is_error"), response
    verify(project)
