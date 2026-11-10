# SPDX-License-Identifier: Apache-2.0

"""Synthetic Kimi benchmark planning stops before payload I/O and preparation."""

import importlib.util
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires a sixteen-GPU planning host")


def test_reduced_dflash2_plan_without_preparation(monkeypatch, tmp_path):
    target = os.environ.get("FLARE_KIMI31_CHECKPOINT")
    draft = os.environ.get("FLARE_KIMI31_DFLASH_CHECKPOINT")
    if not target or not draft:
        pytest.skip("set FLARE_KIMI31_CHECKPOINT and FLARE_KIMI31_DFLASH_CHECKPOINT")
    source = Path(__file__).parents[2] / "tools" / "kimi31_synthetic_bench.py"
    spec = importlib.util.spec_from_file_location("kimi31_synthetic_bench", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from ignition.cli import bench, environment

    monkeypatch.setattr(environment, "apply", lambda *args: None)
    captured = {}

    def measure(model, plan, configuration, settings, progress, **kwargs):
        captured.update(plan=plan, settings=settings, options=kwargs)
        return {"status": "planned"}

    monkeypatch.setattr(bench, "_measure", measure)
    monkeypatch.setattr(bench, "write_result", lambda result, output: None)
    config = tmp_path / "config.toml"
    config.write_text("")
    output = tmp_path / "result.json"
    options = module.arguments().parse_args([
        "--target", target, "--draft", draft, "--config", str(config),
        "--output", str(output), "--seed", "123", "--seeds", "0",
        "--warmup", "2", "--repetitions", "5",
    ])
    assert module.run(options) == 0
    plan = captured["plan"]
    assert plan.benchmark_selection.speculative_round_shapes == ((1, 8, 8),)
    assert plan.speculation.method == "dflash2"
    assert plan.speculation.draft_tokens == 7
    assert plan.speculation.graph_budget == 16
    assert plan.language_model_only
    prepare = captured["options"]["prepare_options"]
    assert prepare["synthetic_seed"] == 123
    assert Path(prepare["draft_checkpoint"]) == Path(draft).resolve()
    assert captured["options"]["diagnostic_metadata"]["synthetic_weights"]["payload_bytes_read"] == 0
