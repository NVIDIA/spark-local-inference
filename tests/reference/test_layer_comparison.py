# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU layer measurements preserve paired contexts and expose every mismatch."""

import json
import math

import pytest
import torch

from qualify.layer_comparison import (
    compare_layers,
    layer_trace_from_artifact,
    load_layer_trace,
)

VOCAB = 16


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("layer comparison must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def artifact(*, steps=1, dtype=torch.float32, legacy=False):
    teachers = list(range(4, 4 + steps - 1))
    value = {
        "model_id": "checkpoint", "revision": "revision",
        "logits": [torch.arange(VOCAB).float() for _ in range(steps)],
        "layer_outputs": [torch.ones(48, 2 if step == 0 else 1, 10240, dtype=dtype)
                          for step in range(steps)],
        "stage_outputs": [{"embedding": torch.ones(2 if step == 0 else 1, 32, dtype=dtype)}
                          for step in range(steps)],
    }
    if legacy:
        value.update(prompt=[2, 3], tokens=teachers + [15])
    else:
        value.update(schema_version=1, prompt_tokens=[2, 3], teacher_tokens=teachers)
    return value


def trace(value):
    return layer_trace_from_artifact(value, vocab_size=VOCAB)


def compare(actual, reference):
    return compare_layers(flare=trace(actual), reference=trace(reference))


def test_lossless_bank_conversion_is_equal_and_preserves_recorded_dtype_metadata():
    actual, reference = artifact(steps=4, dtype=torch.bfloat16, legacy=True), artifact(steps=4)
    reference["stage_storage_dtype"] = "float32"
    reference["stage_source_dtypes"] = "BF16/FP16/FP32 copied exactly into FP32"
    report = compare(actual, reference)
    assert [step["context_length"] for step in report["steps"]] == [2, 3, 4, 5]
    assert [step["input_token_ids"] for step in report["steps"]] == [[2, 3], [4], [5], [6]]
    assert report["sources"]["reference"]["recorded_dtype_metadata"] == {
        "stage_storage_dtype": "float32",
        "stage_source_dtypes": "BF16/FP16/FP32 copied exactly into FP32",
    }
    for step in report["steps"]:
        assert step["first_fp32_value_mismatch_layer"] is None
        assert step["first_last_row_fp32_value_mismatch_layer"] is None
        assert len(step["layers"]) == 48
        for layer in step["layers"]:
            assert layer["relative_l2"] == 0
            assert layer["fp32_value_bits_equal"]
            assert layer["storage_bitwise_equal"] is None
            assert layer["flare_storage_dtype"] == "torch.bfloat16"
            assert layer["reference_storage_dtype"] == "torch.float32"
    assert "passed" not in report
    assert report["numerical_acceptance"] == "not evaluated; no thresholds supplied"
    json.dumps(report, allow_nan=False)


def test_first_layer_and_last_row_mismatches_report_independently_without_stopping():
    actual, reference = artifact(), artifact()
    actual["layer_outputs"][0][7, 0, 9] = 3
    actual["layer_outputs"][0][19, 1, 17] = 2
    actual["layer_outputs"][0][47, 0, 5] = 0
    saved = actual["layer_outputs"][0].clone()
    step = compare(actual, reference)["steps"][0]
    assert step["first_fp32_value_mismatch_layer"] == 7
    assert step["first_last_row_fp32_value_mismatch_layer"] == 19
    assert [row["layer"] for row in step["layers"]] == list(range(48))
    layer = step["layers"][7]
    assert layer["max_absolute_error"] == 2
    assert layer["relative_l2"] == pytest.approx(2 / math.sqrt(2 * 10240))
    assert layer["rows"][0]["relative_l2"] == pytest.approx(2 / math.sqrt(10240))
    assert not layer["rows"][0]["fp32_value_bits_equal"]
    assert layer["last_row"]["fp32_value_bits_equal"]
    assert not step["layers"][47]["fp32_value_bits_equal"]
    assert torch.equal(actual["layer_outputs"][0], saved)


def test_reference_denominator_is_reference_for_layers_and_stages():
    actual, reference = artifact(), artifact()
    actual["layer_outputs"][0].mul_(6)
    reference["layer_outputs"][0].mul_(2)
    actual["stage_outputs"][0]["embedding"].mul_(6)
    reference["stage_outputs"][0]["embedding"].mul_(2)
    step = compare(actual, reference)["steps"][0]
    assert step["layers"][23]["relative_l2"] == pytest.approx(2)
    assert step["stages"]["embedding"]["relative_l2"] == pytest.approx(2)
    reverse = compare(reference, actual)["steps"][0]
    assert reverse["layers"][23]["relative_l2"] == pytest.approx(2 / 3)


def test_signed_zero_is_detected_even_when_all_error_metrics_are_zero():
    actual, reference = artifact(), artifact()
    actual["layer_outputs"][0].zero_()
    reference["layer_outputs"][0].zero_()
    actual["layer_outputs"][0][3, 1, 0] = -0.0
    step = compare(actual, reference)["steps"][0]
    assert step["first_fp32_value_mismatch_layer"] == 3
    row = step["layers"][3]
    assert row["relative_l2"] is None
    assert row["max_absolute_error"] == 0
    assert not row["fp32_value_bits_equal"]
    assert not row["storage_bitwise_equal"]


def test_zero_reference_norm_is_explicit_without_nonfinite_json():
    actual, reference = artifact(), artifact()
    reference["layer_outputs"][0][47].zero_()
    layer = compare(actual, reference)["steps"][0]["layers"][47]
    assert layer["relative_l2"] is None
    assert layer["difference_l2_norm"] > 0
    assert layer["max_absolute_error"] == 1
    json.dumps(layer, allow_nan=False)


def test_missing_stage_keys_are_explicit_and_not_renamed_or_silently_matched():
    actual, reference = artifact(), artifact()
    actual["stage_outputs"][0]["gdn_projected_qkv"] = torch.ones(2, 10240)
    reference["stage_outputs"][0]["gdn_projected_qkvz"] = torch.ones(2, 16384)
    actual["stage_outputs"][0]["moe_ids"] = torch.ones(2, 10, dtype=torch.int32)
    step = compare(actual, reference)["steps"][0]
    assert step["stage_coverage"] == {
        "compared": ["embedding"],
        "missing_in_flare": ["gdn_projected_qkvz"],
        "missing_in_reference": ["gdn_projected_qkv", "moe_ids"],
        "identical_key_sets": False, "has_comparable_stages": True,
    }
    assert set(step["stages"]) == {"embedding"}
    assert report_dtype(actual, reference) == "torch.int32"


def report_dtype(actual, reference):
    return compare(actual, reference)["sources"]["flare"]["stage_bank_dtypes"][0]["moe_ids"]


def test_empty_stage_coverage_is_not_a_success_predicate():
    actual, reference = artifact(), artifact()
    actual["stage_outputs"][0].clear()
    reference["stage_outputs"][0].clear()
    coverage = compare(actual, reference)["steps"][0]["stage_coverage"]
    assert coverage["identical_key_sets"]
    assert not coverage["has_comparable_stages"]


def test_shared_stage_shape_mismatch_fails_with_step_and_key():
    actual, reference = artifact(), artifact()
    actual["stage_outputs"][0]["embedding"] = torch.ones(2, 31)
    with pytest.raises(ValueError, match="step 0, stage embedding: tensor shapes differ"):
        compare(actual, reference)


def test_integer_stage_values_retain_exact_identity():
    actual, reference = artifact(), artifact()
    actual["stage_outputs"][0]["routes"] = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    reference["stage_outputs"][0]["routes"] = torch.tensor([[1, 2], [3, 5]], dtype=torch.int64)
    stage = compare(actual, reference)["steps"][0]["stages"]["routes"]
    assert stage["max_absolute_error"] == 1
    assert stage["rows"][0]["fp32_value_bits_equal"]
    assert not stage["last_row"]["fp32_value_bits_equal"]
    assert stage["storage_bitwise_equal"] is None


@pytest.mark.parametrize("change,message", [
    ({"prompt_tokens": [2, 9]}, "prompt token IDs"),
    ({"teacher_tokens": [9]}, "consumed teacher token IDs"),
    ({"model_id": "different"}, "model_id differs"),
    ({"revision": "different"}, "revision differs"),
])
def test_unpaired_contexts_or_identity_rejected(change, message):
    actual, reference = artifact(steps=2), artifact(steps=2)
    actual.update(change)
    with pytest.raises(ValueError, match=message):
        compare(actual, reference)


@pytest.mark.parametrize("field,replacement,message", [
    ("layer_outputs", [], "exactly 1 logit steps"),
    ("stage_outputs", [], "exactly 1 logit steps"),
    ("layer_outputs", [torch.ones(47, 2, 10240)], "shape "),
    ("layer_outputs", [torch.ones(48, 1, 10240)], "shape "),
    ("layer_outputs", [torch.ones(48, 2, 10239)], "shape "),
    ("layer_outputs", [torch.ones(48, 2, 10240, dtype=torch.int32)], "unsupported tensor dtype"),
    ("layer_outputs", [torch.ones(48, 2, 10240, device="meta")], "CPU"),
    ("stage_outputs", [{"embedding": torch.ones(1, 32)}], "shape "),
    ("stage_outputs", [{"embedding": torch.ones(2, 0)}], "empty"),
    ("stage_outputs", [{"embedding": torch.ones(2, 32, dtype=torch.complex64)}], "unsupported tensor dtype"),
    ("stage_outputs", [{"unpaired": torch.full((2, 32), float("nan"))}], "finite"),
    ("stage_outputs", [{"routes": torch.full((2, 1), 2**24 + 1, dtype=torch.int64)}], "exactly representable"),
])
def test_invalid_layer_or_stage_contract_rejected(field, replacement, message):
    value = artifact()
    value[field] = replacement
    with pytest.raises(ValueError, match=message):
        trace(value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_late_layer_is_rejected(bad):
    value = artifact()
    value["layer_outputs"][0][47, 1, 10000] = bad
    with pytest.raises(ValueError, match=r"layer_outputs\[0\]\[47\].*finite"):
        trace(value)


def test_full_vocabulary_validation_runs_before_layer_comparison():
    value = artifact()
    value["logits"][0] = torch.ones(VOCAB - 1)
    with pytest.raises(ValueError, match="full vocabulary"):
        trace(value)


def test_fp32_metric_overflow_fails_instead_of_emitting_invalid_json():
    actual, reference = artifact(), artifact()
    reference["layer_outputs"][0][0].fill_(1e38)
    with pytest.raises(ValueError, match="FP32 metric computation"):
        compare(actual, reference)


def test_file_loading_preserves_teacher_context(tmp_path):
    path = tmp_path / "decoder.pt"
    torch.save(artifact(steps=4, legacy=True), path)
    loaded = load_layer_trace(path, vocab_size=VOCAB)
    assert loaded.logits.teacher_tokens == (4, 5, 6)
    assert loaded.logits.source["artifact_format"] == "flare_decoder"
    assert loaded.logits.source["path"] == str(path)
    assert all(value.device.type == "cpu" for value in loaded.layers)


def test_cli_preserves_all_layers_and_refuses_existing_output(tmp_path, monkeypatch):
    from qualify.layer_comparison import main
    actual, reference = artifact(legacy=True), artifact()
    for value in (actual, reference):
        value["logits"] = [torch.zeros(248320)]
    a, r, output = (tmp_path / name for name in ("actual.pt", "reference.pt", "comparison.json"))
    torch.save(actual, a)
    torch.save(reference, r)
    monkeypatch.setattr("sys.argv", ["layer_comparison", "--flare", str(a), "--reference", str(r),
                                    "--output", str(output)])
    main()
    data = json.loads(output.read_text())
    assert len(data["steps"][0]["layers"]) == 48
    assert data["sources"]["flare"]["path"] == str(a)
    original = output.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        main()
    assert output.read_bytes() == original
