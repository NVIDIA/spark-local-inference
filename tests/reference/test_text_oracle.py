# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU measurements reject unpaired contexts before comparing serving logits."""

import json
import math
from dataclasses import replace

import pytest
import torch

from qualify.text_oracle import compare_traces, load_trace, trace_from_artifact

VOCAB = 16


def artifact(*, logits=None, teachers=None):
    return {
        "schema_version": 1, "model_id": "checkpoint", "revision": "revision",
        "prompt_tokens": [2, 3], "teacher_tokens": [4] if teachers is None else teachers,
        "logits": [torch.arange(VOCAB).float() / 8, torch.arange(VOCAB).float() / 4]
        if logits is None else logits,
    }


def trace(value=None):
    return trace_from_artifact(artifact() if value is None else value, vocab_size=VOCAB)


def compare(actual, reference):
    return compare_traces(flare=actual, reference=reference, canonical=reference)


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("text comparison must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def test_all_three_pairs_use_named_reference_norm_and_preserve_storage():
    reference = trace()
    actual = trace(artifact(logits=[row * 2 for row in reference.logits]))
    serving = trace(artifact(logits=[row * 3 for row in reference.logits]))
    originals = [row.clone() for row in actual.logits]
    report = compare_traces(flare=actual, reference=serving, canonical=reference)
    assert set(report["pairs"]) == {
        "flare_vs_reference", "flare_vs_canonical", "reference_vs_canonical",
    }
    for pair, expected in (("flare_vs_reference", 1 / 3),
                           ("flare_vs_canonical", 1), ("reference_vs_canonical", 2)):
        for row in report["pairs"][pair]["steps"]:
            assert row["relative_l2"] == pytest.approx(expected)
            assert row["top10_overlap_count"] == 10
            assert row["top10_overlap_fraction"] == 1
            assert row["argmax_equal"]
    assert report["context_lengths"] == [2, 3]
    assert all(torch.equal(a, b) for a, b in zip(actual.logits, originals, strict=True))
    assert "passed" not in report
    assert report["numerical_acceptance"] == "not evaluated; no thresholds supplied"
    json.dumps(report, allow_nan=False)


def test_kl_direction_matches_independent_probability_formula():
    actual = [0.4 * i for i in range(VOCAB)]
    reference = [0.2 * (i % 5) for i in range(VOCAB)]
    a, r = (trace(artifact(logits=[torch.tensor(x)], teachers=[])) for x in (actual, reference))
    row = compare(a, r)["pairs"]["flare_vs_canonical"]["steps"][0]
    def probabilities(values):
        exp = [math.exp(v) for v in values]
        return [v / sum(exp) for v in exp]
    p, q = probabilities(reference), probabilities(actual)
    expected = sum(x * math.log(x / y) for x, y in zip(p, q, strict=True))
    reverse = sum(y * math.log(y / x) for x, y in zip(p, q, strict=True))
    assert abs(expected - reverse) > 0.005
    assert row["kl_reference_to_actual"] == pytest.approx(
        expected, abs=VOCAB * torch.finfo(torch.float32).eps)


def test_extreme_logits_use_stable_log_softmax_and_keep_large_kl():
    reference = torch.full((VOCAB,), -1000.0)
    actual = reference.clone()
    reference[0], actual[1] = 1000.0, 1000.0
    row = compare(trace(artifact(logits=[actual], teachers=[])),
                  trace(artifact(logits=[reference], teachers=[])))["pairs"]["flare_vs_canonical"]["steps"][0]
    assert row["kl_reference_to_actual"] == 2000
    assert row["reference"]["argmax"] == 0
    assert row["actual"]["argmax"] == 1
    assert row["reference"]["top1_margin"] == 2000


def test_same_argmax_does_not_hide_logit_change_or_additive_offset():
    reference = trace()
    actual = trace(artifact(logits=[row + 20 for row in reference.logits]))
    rows = compare(actual, reference)["pairs"]["flare_vs_canonical"]["steps"]
    for row in rows:
        assert row["argmax_equal"]
        assert row["relative_l2"] > 1
        assert row["kl_reference_to_actual"] == pytest.approx(0, abs=1e-6)


def test_top10_overlap_count_and_fraction():
    reference = trace(artifact(logits=[torch.arange(VOCAB).float()], teachers=[]))
    actual = trace(artifact(logits=[torch.arange(VOCAB).flip(0).float()], teachers=[]))
    row = compare(actual, reference)["pairs"]["flare_vs_canonical"]["steps"][0]
    assert row["top10_overlap_count"] == 4
    assert row["top10_overlap_fraction"] == 0.4


def test_legal_zero_logits_report_undefined_relative_l2_without_infinity():
    zero = trace(artifact(logits=[torch.zeros(VOCAB)], teachers=[]))
    actual = trace(artifact(logits=[torch.ones(VOCAB)], teachers=[]))
    row = compare(actual, zero)["pairs"]["flare_vs_canonical"]["steps"][0]
    assert row["relative_l2"] is None
    assert row["reference_l2_norm"] == 0
    assert row["kl_reference_to_actual"] == 0
    assert row["reference"]["argmax"] == 0
    assert row["reference"]["top1_margin"] == 0


def test_storage_dtypes_may_differ_and_are_recorded():
    canonical = trace()
    native = trace(artifact(logits=[x.bfloat16() for x in canonical.logits]))
    reference = trace(artifact(logits=torch.stack(canonical.logits)))
    report = compare_traces(flare=native, reference=reference, canonical=canonical)
    assert report["sources"]["flare"]["logit_dtypes"] == ["torch.bfloat16"] * 2
    assert report["sources"]["reference"]["logit_dtypes"] == ["torch.float32"] * 2
    assert report["pairs"]["flare_vs_reference"]["steps"][0]["relative_l2"] == 0


def test_decoder_formats_load_consumed_contexts(tmp_path):
    value = artifact()
    canonical = {"input_ids": torch.tensor([[2, 3]], dtype=torch.int64),
                 "teacher_tokens": [4], "model_id": "checkpoint", "revision": "revision",
                 "logits": value["logits"]}
    native = {"prompt": [2, 3], "tokens": [4, 15],
              "logits": [row.bfloat16() for row in value["logits"]]}
    traces = []
    for index, data in enumerate((canonical, native)):
        path = tmp_path / f"{index}.pt"
        torch.save(data, path)
        loaded = load_trace(path, vocab_size=VOCAB)
        assert loaded.source["path"] == str(path)
        assert loaded.teacher_tokens == (4,)
        assert all(row.device.type == "cpu" for row in loaded.logits)
        traces.append(loaded)
    result = compare(traces[1], traces[0])
    assert result["sources"]["flare"]["model_id"] is None
    assert result["sources"]["canonical"]["model_id"] == "checkpoint"


def test_self_fed_divergence_rejected_even_when_final_argmax_matches():
    canonical = trace()
    native = trace_from_artifact({"prompt": [2, 3], "tokens": [5, 15],
                                  "logits": canonical.logits}, vocab_size=VOCAB)
    with pytest.raises(ValueError, match="consumed teacher token IDs"):
        compare(native, canonical)


@pytest.mark.parametrize("change,message", [
    ({"prompt_tokens": [2, 5]}, "prompt token IDs"),
    ({"teacher_tokens": [6]}, "consumed teacher token IDs"),
    ({"model_id": "different-model"}, "model_id differs"),
    ({"revision": "different-revision"}, "revision differs"),
])
def test_mismatched_inputs_or_identity_rejected(change, message):
    value = artifact()
    value.update(change)
    with pytest.raises(ValueError, match=message):
        compare(trace(value), trace())


def test_mismatched_steps_and_vocabulary_rejected():
    with pytest.raises(ValueError, match="step count"):
        compare(trace(artifact(logits=[torch.ones(VOCAB)], teachers=[])), trace())
    larger = trace_from_artifact(artifact(logits=[torch.ones(17)] * 2), vocab_size=17)
    with pytest.raises(ValueError, match="vocabulary size"):
        compare(larger, trace())


@pytest.mark.parametrize("rows,message", [
    ([], "at least one step"),
    ([torch.ones(VOCAB - 1)], "full vocabulary"),
    ([torch.ones(1, VOCAB)], "full vocabulary"),
    ([torch.ones(VOCAB, dtype=torch.int64)], "floating"),
    ([torch.ones(VOCAB, dtype=torch.complex64)], "floating"),
    ([torch.ones(VOCAB, device="meta")], "CPU"),
    ([torch.full((VOCAB,), float("nan"))], "finite"),
    ([torch.full((VOCAB,), float("inf"))], "finite"),
    ([torch.full((VOCAB,), 1e300, dtype=torch.float64)], "FP32"),
])
def test_invalid_logits_rejected(rows, message):
    with pytest.raises(ValueError, match=message):
        trace(artifact(logits=rows, teachers=[]))


@pytest.mark.parametrize("field,value", [
    ("prompt_tokens", []), ("prompt_tokens", [True]),
    ("prompt_tokens", [-1]), ("prompt_tokens", [VOCAB]),
    ("teacher_tokens", []), ("teacher_tokens", [4, 5]),
    ("teacher_tokens", [4.0]), ("teacher_tokens", [VOCAB]),
    ("schema_version", 2), ("schema_version", True),
    ("model_id", None), ("revision", ""),
])
def test_invalid_tokens_lengths_and_schema_rejected(field, value):
    data = artifact()
    data[field] = value
    with pytest.raises(ValueError):
        trace(data)


def test_ambiguous_artifact_and_legacy_token_count_rejected():
    data = artifact()
    data["prompt"] = data["prompt_tokens"]
    with pytest.raises(ValueError, match="exactly one"):
        trace(data)
    with pytest.raises(ValueError, match="tokens count"):
        trace_from_artifact({"prompt": [2, 3], "tokens": [4], "logits": data["logits"]}, vocab_size=VOCAB)


def test_comparison_revalidates_trace_and_rejects_nonfinite_metric():
    valid = trace()
    mutated = replace(valid, logits=(torch.full((VOCAB,), float("nan")), valid.logits[1]))
    with pytest.raises(ValueError, match="finite"):
        compare(mutated, valid)
    large = trace(artifact(logits=[torch.full((VOCAB,), 1e38)] * 2))
    with pytest.raises(ValueError, match="nonfinite value"):
        compare(large, valid)
