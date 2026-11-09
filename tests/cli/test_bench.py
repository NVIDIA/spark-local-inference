# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark CLI workload selection and GPU latency presentation."""

import json

import pytest

from ignition.cli import _parser, main
from ignition.cli.bench import Settings, table, write_result
from ignition.errors import UsageError


def settings(*flags):
    return Settings.from_args(_parser().parse_args(["bench", *flags]))


def test_default_matrix_and_measurement_counts():
    value = settings()
    assert value.batches == (1, 2, 4)
    assert value.contexts == (512, 2048, 8192, 16384)
    assert value.width == 8
    assert value.warmup == 5
    assert value.repetitions == 20
    assert value.seeds == (0, 1, 2)
    assert [(batch, value.width, batch * value.width) for batch in value.batches] == [
        (1, 8, 8), (2, 8, 16), (4, 8, 32)]


def test_explicit_workload_and_preparation_options(tmp_path):
    args = _parser().parse_args(["bench", "qwen3.8-27b", "--batches", "2,4", "--contexts", "2048,8192",
        "--verify-width", "8", "--mtp-draft-tokens", "7", "--seeds", "14,29", "--warmup", "2",
        "--repetitions", "7", "--output", str(tmp_path / "rounds.json"), "--gpus", "3",
        "--compile-workers", "12", "--autotune-effort", "off", "--speculative-token-capacity", "128"])
    value = Settings.from_args(args)
    assert (value.batches, value.contexts, value.width) == ((2, 4), (2048, 8192), 8)
    assert (value.warmup, value.repetitions, value.seeds) == (2, 7, (14, 29))
    assert (args.gpus, args.compile_workers, args.autotune_effort, args.speculative_token_capacity) == (
        "3", 12, "off", 128)
    assert args.output == tmp_path / "rounds.json"


def test_explicit_draft_count_and_unordered_selections():
    value = settings("--mtp-draft-tokens", "7", "--batches", "4,1", "--contexts", "2048,512")
    assert value.width == 8
    assert value.batches == (1, 4)
    assert value.contexts == (512, 2048)


def test_kimi_eight_request_round_with_autotuning():
    args = _parser().parse_args(["bench", "kimi-k3.1-pro", "--gpus", "0-15",
        "--batches", "8", "--max-concurrency", "8", "--verify-width", "8",
        "--speculative-token-capacity", "64", "--online-mxfp8", "--autotune-effort", "default"])
    value = Settings.from_args(args)
    assert (value.batches, value.width) == ((8,), 8)
    assert args.autotune_effort == "default" and not args.no_compile
    assert args.max_concurrency == 8 and args.speculative_token_capacity == 64


def test_verification_matrix_keeps_the_full_draft_block():
    value = settings("--verify-prefixes", "8,1,6,4", "--mtp-draft-tokens", "7")
    assert value.verification_prefixes == (1, 4, 6, 8)
    assert value.width == 8
    assert value.batches == (1,)
    assert settings("--verify-prefixes", "4,6").width == 8


@pytest.mark.parametrize("flags", [
    ("--verify-prefixes", "0,8"), ("--verify-prefixes", "4,9"),
    ("--verify-prefixes", "1,8", "--batches", "2"),
    ("--verify-prefixes", "1,8", "--prefill"),
    ("--verify-prefixes", "1,8", "--speculation-study"),
    ("--verify-prefixes", "1,8", "--mtp-draft-tokens", "3"),
])
def test_invalid_verification_matrix(flags):
    with pytest.raises(UsageError):
        settings(*flags)


def test_kimi_rejects_private_dflash_studies_before_preparation(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text("")
    assert main(["bench", "kimi-k3.1-pro", "--speculation-study", "--config", str(config)]) == 2
    assert "--speculation-study requires Qwen3.8-27B DFlash2" in capsys.readouterr().err


def test_prefill_defaults_and_target_only_selection():
    value = settings("--prefill")
    assert value.prefill and value.batches == (1,) and value.contexts == (0,)
    args = _parser().parse_args(["bench", "--prefill", "--prefill-chunk-size", "2048",
                                "--contexts", "4096,0", "--no-mtp"])
    assert Settings.from_args(args).contexts == (0, 4096)
    assert args.prefill_chunk_size == 2048 and args.no_mtp


@pytest.mark.parametrize("flags,message", [
    (("--batches", "2"), "requires --batches 1"),
    (("--verify-width", "8"), "does not apply"),
    (("--contexts", "-128"), "multiples of 128"),
    (("--contexts", "129"), "multiples of 128"),
])
def test_prefill_rejects_unsupported_geometry(flags, message):
    with pytest.raises(UsageError, match=message):
        settings("--prefill", *flags)


def test_prefill_report_names_chunk_and_prefix():
    from ignition.bench.runner import PrefillCase, measurement_result

    result = measurement_result((PrefillCase(2048, 4096),), [[dict(seed=0, times_ms=[2., 4.])]],
                                warmup=1, repetitions=2, seeds=(0,), status="complete")
    result["device"] = dict(name="NVIDIA GB10", capability=[12, 1])
    rendered = table(result)
    assert "GPU graph prefill latency" in rendered and "Chunk" in rendered
    assert "2048" in rendered and "4096" in rendered and "3.0000" in rendered
    assert result["metric"] == "cuda_graph_prefill_latency_ms"


@pytest.mark.parametrize("flags,message", [
    (("--verify-width", "4", "--mtp-draft-tokens", "7"), "must equal"),
    (("--verify-width", "9"), "requires --verify-width 8"),
    (("--verify-width", "4"), "requires --verify-width 8"),
    (("--mtp-draft-tokens", "3"), "requires --verify-width 8"),
    (("--batches", "0"), "from 1,2,4"),
    (("--batches", "16"), "from 1,2,4,8"),
    (("--contexts", "0"), "positive multiples"),
    (("--contexts", "513"), "positive multiples"),
    (("--warmup", "-1"), "positive"),
    (("--warmup", "0"), "positive"),
    (("--repetitions", "0"), "positive"),
    (("--seeds", "-1"), "0 through"),
    (("--no-mtp",), "requires draft state"),
    (("--mixed-prefill",), "speculative rounds"),
    (("--prompt-logprobs",), "speculative rounds"),
    (("--official-source", "https://example.com/model"), "package publishing"),
])
def test_invalid_measurement_settings(flags, message):
    with pytest.raises(UsageError, match=message):
        settings(*flags)


@pytest.mark.parametrize("flag,value", [("--batches", "1,1"), ("--contexts", "512,"), ("--seeds", "x")])
def test_malformed_lists_are_usage_errors(flag, value):
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(["bench", flag, value])
    assert error.value.code == 2


def test_bench_help_describes_measured_boundary(capsys):
    with pytest.raises(SystemExit) as error:
        main(["bench", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "Kimi K3.1 Pro DSpark" in output
    assert "excludes preparation" in output
    assert "not serving throughput" in output
    assert "--repetitions" in output


def test_unsupported_recipe_fails_before_preparation(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text("")
    assert main(["bench", "qwen3.8-flash", "--config", str(config)]) == 2
    assert "supports Qwen3.8-27B DFlash2 and Kimi K3.1 Pro DSpark" in capsys.readouterr().err


def test_report_keeps_case_latency_and_equal_weight_score_separate(tmp_path, capsys):
    result = {"device": {"name": "NVIDIA RTX PRO 6000 Blackwell", "capability": [12, 0]},
        "aggregate_ms": 3.5,
        "cases": [{"batch": 1, "width": 8, "packed_tokens": 8, "context_tokens": 512,
                   "median_ms": 2.0, "p25_ms": 1.8, "p75_ms": 2.2,
                   "samples": [{"seed": 0, "times_ms": [1.8, 2.0, 2.2]}]},
                  {"batch": 4, "width": 8, "packed_tokens": 32, "context_tokens": 8192,
                   "median_ms": 5.0, "p25_ms": 4.8, "p75_ms": 5.2}]}
    text = table(json.loads(json.dumps(result)))
    assert "SM120" in text
    assert "2.0000" in text and "5.0000" in text
    assert "1.8000–2.2000" in text
    assert "Equal-weight mean of case medians: 3.5000 ms" in text
    assert "lower is better" in text
    assert "GPU class scores are separate" in text
    destination = tmp_path / "rounds.json"
    write_result(result, destination)
    assert capsys.readouterr().out.strip() == text
    assert json.loads(destination.read_text()) == result


def test_json_write_failure_reports_destination(tmp_path):
    result = {"device": {"name": "NVIDIA GB10", "capability": [12, 1]}, "aggregate_ms": 1.0, "cases": []}
    with pytest.raises(UsageError, match="cannot write benchmark JSON"):
        write_result(result, tmp_path / "missing" / "rounds.json")


def test_partial_result_is_saved_without_printing_success_table(tmp_path, capsys):
    from ignition.cli.bench import save_result

    result = {"status": "incomplete", "ignore_eos": True, "failure": "invalid completion",
              "device": {"name": "NVIDIA GB10", "capability": [12, 1]},
              "configuration": {"compiled_forwards": True},
              "cases": [{"complete": False, "samples": [{"seed": 0, "times_ms": [1.2]}]}]}
    output = tmp_path / "rounds.json"
    save_result(result, output)
    assert json.loads(output.read_text()) == result
    assert "aggregate_ms" not in json.loads(output.read_text())
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("width", [8, 13, 16])
def test_speculation_study_selects_private_mode(width):
    value = settings("--speculation-study", "--verify-width", str(width),
                     "--mtp-draft-tokens", str(width - 1), "--contexts", "8192")
    assert value.speculation_study and value.batches == (1,) and value.width == width
    if width == 13:
        with pytest.raises(UsageError):
            settings("--verify-width", str(width))


@pytest.mark.parametrize("flags", [
    ("--mtp-draft-tokens", "15"),
    ("--verify-width", "16"),
    ("--mtp-draft-tokens", "15", "--verify-width", "16"),
])
def test_fifteen_drafts_select_native_rounds(flags):
    value = settings("--batches", "1", "--contexts", "4096", *flags)
    assert (value.batches, value.contexts, value.width) == ((1,), (4096,), 16)
    assert not value.speculation_study and not value.study_widths


def test_fifteen_drafts_reject_mismatched_verification_width():
    with pytest.raises(UsageError, match="must equal"):
        settings("--mtp-draft-tokens", "15", "--verify-width", "8")


@pytest.mark.parametrize("flags", [("--batches", "2"), ("--prefill",), ("--verify-width", "9")])
def test_speculation_study_rejects_other_regimes(flags):
    with pytest.raises(UsageError):
        settings("--speculation-study", *flags)


def test_study_width_catalog_uses_one_maximum_capacity():
    value = settings("--speculation-study", "--study-widths", "16,8,13")
    assert value.study_widths == (8, 13, 16)
    assert value.width == 16
    assert value.contexts == (8192,)
    assert value.batches == (1,)


@pytest.mark.parametrize("flags", [
    ("--study-widths", "8,13"),
    ("--speculation-study", "--study-widths", "8,9"),
    ("--speculation-study", "--study-widths", "8,16", "--verify-width", "16"),
    ("--speculation-study", "--study-widths", "8,16", "--mtp-draft-tokens", "15"),
])
def test_study_width_catalog_rejects_ambiguous_or_unsupported_selection(flags):
    with pytest.raises(UsageError):
        settings(*flags)
