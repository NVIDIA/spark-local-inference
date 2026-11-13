# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification CLIs share package implementations without importing CUDA."""

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qualify import server

ROOT = Path(__file__).resolve().parents[2]


def invoke(arguments, cwd, *, package_path=False):
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(CUDA_VISIBLE_DEVICES="", COLUMNS="240")
    if package_path:
        env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("module", ["server", "graph_execution"])
def test_script_and_module_clis_have_identical_help_outside_checkout(tmp_path, module):
    script = invoke([str(ROOT / f"scripts/qualify_{module}.py"), "--help"], tmp_path)
    package = invoke(["-m", f"qualify.{module}", "--help"], tmp_path, package_path=True)
    assert script.returncode == package.returncode == 0, (script.stderr, package.stderr)
    assert (
        script.stdout.replace(f"qualify_{module}.py", "QUALIFIER").split()
        == package.stdout.replace(f"{module}.py", "QUALIFIER").split()
    )


@pytest.mark.parametrize(
    "script", ["prefill_capacity", "context_limit", "vision_aggregate"]
)
def test_http_dependents_import_package_outside_checkout(tmp_path, script):
    result = invoke([str(ROOT / f"scripts/qualify_{script}.py"), "--help"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "--base-url" in result.stdout
    assert "--artifacts" in result.stdout


def test_package_import_has_no_runtime_initialization_or_path_mutation(tmp_path):
    result = invoke(
        [
            "-c",
            """
import sys
before = list(sys.path)
import qualify.server
import qualify.graph_execution
assert sys.path == before
assert not any(name in sys.modules for name in ('torch', 'flashinfer', 'b12x', 'flare_heliopause'))
assert qualify.server.Run.__module__ == 'qualify.server'
assert qualify.graph_execution.Checks.__module__ == 'qualify.graph_execution'
""",
        ],
        tmp_path,
        package_path=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("module", "arguments", "error"),
    [
        (
            "server",
            [
                "--base-url",
                "http://127.0.0.1:1",
                "--gdn-intermediate-dtype",
                "bfloat16",
                "--timeout",
                "0",
            ],
            "timeout, response byte bound and image repetitions must be positive",
        ),
        (
            "graph_execution",
            ["--prefill-chunk-size", "513"],
            "qualification requires a 64-aligned prefill chunk size of at least 512",
        ),
    ],
)
def test_cli_validation_precedes_runtime_and_output_creation(
    tmp_path, module, arguments, error
):
    output, artifacts = tmp_path / "unused.json", tmp_path / "artifacts"
    tail = [*arguments, "--artifacts", str(artifacts)]
    for command, package_path in [
        ([str(ROOT / f"scripts/qualify_{module}.py")], False),
        (["-m", f"qualify.{module}"], True),
    ]:
        result = invoke([*command, *tail], tmp_path, package_path=package_path)
        assert result.returncode == 2
        assert error in result.stderr
        assert "Traceback" not in result.stderr
        assert not output.exists()
        assert not artifacts.exists()


@pytest.mark.parametrize("stream", [False, True])
def test_http_request_bytes_and_results_preserve_utf8_and_sse(
    tmp_path, monkeypatch, stream
):
    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hé", "reasoning_content": "why"},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {"index": 0, "delta": {"content": "llo"}, "finish_reason": "length"}
            ]
        },
    ]
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "café"}],
        "temperature": 0,
        "seed": 1234,
        "max_tokens": 32,
    }
    completion = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "héllo"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    }
    raw = (
        (
            b"".join(
                b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n"
                for event in events
            )
            + b"data: [DONE]\n\n"
        )
        if stream
        else json.dumps(completion).encode()
    )
    content_type = "text/event-stream; charset=utf-8" if stream else "application/json"

    class Response(io.BytesIO):
        status = 200

        def getheaders(self):
            return [("Content-Type", content_type)]

        def getheader(self, name, default=None):
            return dict(self.getheaders()).get(name, default)

    connections = []

    class Connection:
        def __init__(self, host, port, *, timeout):
            self.endpoint = (host, port, timeout)
            self.closed = False
            connections.append(self)

        def request(self, method, path, *, body, headers):
            self.sent = (method, path, body, headers)

        def getresponse(self):
            return Response(raw)

        def close(self):
            self.closed = True

    monkeypatch.setattr(server.http.client, "HTTPConnection", Connection)
    args = SimpleNamespace(
        base_url="http://127.0.0.1:1/api/",
        timeout=17,
        max_response_bytes=4096,
        require_metrics=True,
        cases=None,
        artifacts=tmp_path,
        output=tmp_path / "unused.json",
    )
    run = server.Run(args)
    record = run.request("completion", body, stream=stream)
    expected_body = {**body, "stream": stream}
    (connection,) = connections
    assert connection.endpoint == ("127.0.0.1", 1, 17)
    assert connection.sent == (
        "POST",
        "/api/v1/chat/completions",
        json.dumps(expected_body, ensure_ascii=False).encode(),
        {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        },
    )
    assert connection.closed
    assert "stream" not in body
    assert (
        Path(record["request_artifact"]).read_bytes()
        == (json.dumps(expected_body, ensure_ascii=False, indent=2) + "\n").encode()
    )
    assert Path(record["response_artifact"]).read_bytes() == raw
    assert record["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "client_error" not in record
    if stream:
        assert record["events"] == events
        assert record["sse"] == {
            "message": {
                "role": "assistant",
                "content": "héllo",
                "reasoning_content": "why",
            },
            "finish_reasons": ["length"],
            "errors": [],
        }
        assert record["done_count"] == 1
        assert record["incomplete_event"] is False
    else:
        assert record["body"] == completion
        assert run.complete(record, prompt_tokens=7) == completion["choices"][0]
        assert all(check["passed"] for check in run.results["checks"])
    assert not args.output.exists()
    assert run.results["requests"] == [record]
