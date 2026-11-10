# SPDX-License-Identifier: Apache-2.0

"""Synthetic Kimi benchmark planning and result handling without CUDA work."""

import importlib.util
import json
from pathlib import Path

import pytest

from ignition.cli import bench, environment
from ignition.cli.preparation import Host
from ignition.errors import UsageError
from ignition.planning import Gpu


@pytest.fixture
def synthetic_cli(monkeypatch, tmp_path):
    source = Path(__file__).parents[2] / "tools" / "kimi31_synthetic_bench.py"
    spec = importlib.util.spec_from_file_location("kimi31_synthetic_bench", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = tmp_path / "target"
    draft = tmp_path / "draft"
    for directory, architecture in ((target, "KimiK31ForConditionalGeneration"), (draft, "DFlash2DraftModel")):
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    config = tmp_path / "config.toml"
    config.write_text("")
    output = tmp_path / "results" / "rounds.json"
    arguments = ["--target", str(target), "--draft", str(draft), "--config", str(config),
                 "--output", str(output), "--seed", "123", "--seeds", "0",
                 "--warmup", "2", "--repetitions", "5"]

    gpus = tuple(Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 << 30, 96 << 30, (12, 0))
                 for index in range(16))
    monkeypatch.setattr(Host, "discover", lambda: Host(gpus, None))

    def run(*extra):
        return module.run(module.arguments().parse_args([*arguments, *extra]))

    return run, output, draft


@pytest.fixture
def stub_preflight(monkeypatch):
    import torch
    from flashinfer import sampling

    monkeypatch.setattr(environment, "apply", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(sampling, "get_sampling_module", lambda: None)


def test_reduced_dflash2_plan_without_preparation(synthetic_cli, stub_preflight, monkeypatch):
    import huggingface_hub

    run, output, draft = synthetic_cli
    captured = {}

    def forbidden_download(*args, **kwargs):
        pytest.fail("synthetic benchmark attempted a checkpoint download")

    def measure(model, plan, configuration, settings, progress, **kwargs):
        captured.update(plan=plan, settings=settings, options=kwargs)
        return {"status": "complete", "device": {"name": "planning fixture", "capability": [12, 0]},
                "cases": [], "aggregate_ms": 0.0}

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden_download)
    monkeypatch.setattr(bench, "_measure", measure)
    assert run() == 0
    plan = captured["plan"]
    assert plan.benchmark_selection.speculative_round_shapes == ((1, 8, 8),)
    assert plan.speculation.method == "dflash2"
    assert plan.speculation.draft_tokens == 7
    assert plan.speculation.graph_budget == 16
    assert plan.language_model_only
    prepare = captured["options"]["prepare_options"]
    assert prepare["synthetic_seed"] == 123
    assert Path(prepare["draft_checkpoint"]) == draft
    assert captured["options"]["diagnostic_metadata"]["synthetic_weights"]["payload_bytes_read"] == 0
    assert json.loads(output.read_text())["status"] == "complete"


@pytest.mark.parametrize(("options", "error", "message"), [
    (("--seed", "-1"), ValueError, "seed must be a uint63"),
    (("--contexts", "129"), ValueError, "positive multiples of 128"),
    (("--contexts", "1048576"), UsageError, "exceed the prepared context"),
    (("--compile-workers", "1"), UsageError, "at least one process per local serving rank"),
])
def test_invalid_benchmark_inputs_do_not_create_output(synthetic_cli, monkeypatch, options, error, message):
    run, output, _ = synthetic_cli

    def unexpected_preflight(*args):
        pytest.fail("invalid benchmark inputs reached CUDA preflight")

    monkeypatch.setattr(environment, "apply", unexpected_preflight)
    with pytest.raises(error, match=message):
        run(*options)
    assert not output.exists()
    assert not output.parent.exists()


@pytest.mark.parametrize("recorded_samples", (False, True))
def test_preparation_failure_writes_failure_or_preserves_partial_result(
        synthetic_cli, stub_preflight, monkeypatch, recorded_samples):
    run, output, _ = synthetic_cli
    partial = {"status": "incomplete", "cases": [{"samples": [{"times_ms": [1.0]}]}]}

    def fail(*args, **kwargs):
        assert kwargs["output"] == output
        assert output.is_file()
        if recorded_samples:
            bench.save_result(partial, output)
        raise RuntimeError("benchmark device failed")

    monkeypatch.setattr(bench, "_measure", fail)
    with pytest.raises(RuntimeError, match="benchmark device failed"):
        run()
    result = json.loads(output.read_text())
    if recorded_samples:
        assert result == partial
    else:
        assert result["status"] == "failed"
        assert result["error_type"] == "RuntimeError"
        assert result["error"] == "benchmark device failed"
        assert result["synthetic_weights"]["seed"] == 123
