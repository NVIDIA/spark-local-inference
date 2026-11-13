# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU acceptance checks for the complete-model prefix diagnostic receipt."""

import json

import pytest

from scripts.diagnose_prefix_replay import equality_summary, write_equality_receipt


@pytest.mark.parametrize("endpoint", [1, 2, 3, 10, 15, 63, 64, 65, 66, 67, 127])
def test_explicit_checkpoint_endpoint_is_not_rounded(endpoint):
    from scripts.diagnose_prefix_replay import checkpoint_endpoint

    assert checkpoint_endpoint(459, endpoint) == endpoint
    assert checkpoint_endpoint(459) == 448


@pytest.mark.parametrize("endpoint", [-1, 0, 459, 460, True, 10.5])
def test_checkpoint_endpoint_must_be_inside_prompt(endpoint):
    from scripts.diagnose_prefix_replay import checkpoint_endpoint

    with pytest.raises(ValueError, match="positive token count"):
        checkpoint_endpoint(459, endpoint)


def test_nested_attention_and_same_argmax_failures_are_reported(tmp_path):
    comparisons = {
        "cold_vs_tail": {
            "gdn.0.recurrent": {"equal": True, "relative_l2": 0.0},
            "logits": {"equal": False, "argmax_left": 9419, "argmax_right": 9419},
            "qsa": {
                "logical_rows": [[0, 448], [0, 449]],
                "tensors": {
                    "query": {"equal": True, "different_elements": 0},
                    "attended": {"equal": False, "different_elements": 7201},
                },
            },
            "metadata": {"enabled": False, "same_argmax": True, "count": 0},
        },
    }
    receipt = {"comparisons": comparisons, "tensor_sha256": "retained tensor digest"}
    output = tmp_path / "receipt.json"

    assert write_equality_receipt(receipt, output, require_bitwise=True) == 1
    saved = json.loads(output.read_text())
    assert saved["comparisons"] == comparisons
    assert saved["tensor_sha256"] == "retained tensor digest"
    assert saved["require_bitwise"] is True
    assert saved["equality_summary"] == {
        "comparison_count": 1,
        "equal_predicates": 4,
        "passed_predicates": 2,
        "failed_predicates": 2,
        "failed_paths": [
            "comparisons.cold_vs_tail.logits.equal",
            "comparisons.cold_vs_tail.qsa.tensors.attended.equal",
        ],
        "passed": False,
    }


def test_default_diagnostic_preserves_success_exit_with_failed_comparison(tmp_path):
    receipt = {"comparisons": {"cold_vs_tail": {"logits": {"equal": False}}}}
    output = tmp_path / "receipt.json"
    assert write_equality_receipt(receipt, output) == 0
    saved = json.loads(output.read_text())
    assert saved["require_bitwise"] is False
    assert saved["equality_summary"]["passed"] is False


@pytest.mark.parametrize("comparisons", [{}, {"metadata": {"same_argmax": True, "rows": []}}])
def test_no_equality_predicates_cannot_pass_gate(tmp_path, comparisons):
    output = tmp_path / "receipt.json"
    assert write_equality_receipt({"comparisons": comparisons}, output, require_bitwise=True) == 1
    summary = json.loads(output.read_text())["equality_summary"]
    assert summary["equal_predicates"] == 0
    assert summary["failed_paths"] == []
    assert summary["passed"] is False


def test_all_nested_equalities_pass_gate(tmp_path):
    comparisons = {"cold_vs_tail": {
        "logits": {"equal": True},
        "qsa": {"tensors": {"attended": {"equal": True}}},
    }}
    assert equality_summary(comparisons)["equal_predicates"] == 2
    assert write_equality_receipt({"comparisons": comparisons}, tmp_path / "receipt.json",
                                  require_bitwise=True) == 0


def test_exact_prefix_checkpoint_is_computed_separately_from_monolithic_run():
    from scripts.diagnose_prefix_replay import prefix_replay_sequence

    events = []
    live = {'consumed': 0, 'epoch': 0}
    def admit(epoch, *, restore=False):
        events.append(('admit', epoch, restore))
        live.update(consumed=448 if restore else 0, epoch=epoch)
    def prefill(begin, end, capacity):
        assert begin == live['consumed']
        events.append(('prefill', live['epoch'], begin, end, capacity))
        live['consumed'] = end
    def snapshot(consumed):
        assert consumed == live['consumed'] == 448
        events.append(('snapshot', live['epoch'], consumed))
    def observe(name, row=1, *, logits=True):
        events.append(('observe', name, row, logits))
    prefix_replay_sequence(admit, prefill, snapshot, observe,
                           tokens=459, boundary=448, full_capacity=512, tail_capacity=16)
    assert ('prefill', 1, 0, 459, 512) in events
    assert ('prefill', 5, 0, 459, 512) in events
    assert events.index(('prefill', 2, 0, 448, 512)) < events.index(('snapshot', 2, 448))
    assert ('prefill', 3, 448, 459, 16) in events
    assert ('prefill', 4, 448, 459, 512) in events
    assert ('prefill', 6, 0, 448, 512) in events
    assert ('prefill', 6, 448, 459, 16) in events
    assert not any('candidate' in str(event) for event in events)


def test_exact_transfer_staging_clears_inactive_rows_and_preserves_tables():
    import torch
    from scripts.diagnose_prefix_replay import stage_state_transfer

    host = {'state_' + name: torch.full((4,), 55, dtype=torch.int64) for name in
            ('source_slots', 'destination_slots', 'qsa_slots', 'expected_consumed', 'operation', 'accepted_tokens')}
    host['state_num_ops'] = torch.zeros(1, dtype=torch.int32)
    host['page_tables'] = torch.arange(128, dtype=torch.int32).reshape(4, 32)
    pages = host['page_tables'].clone()
    stage_state_transfer(host, source=1, destination=5, consumed=67)
    assert host['state_expected_consumed'].tolist() == [67, -1, -1, -1]
    assert host['state_source_slots'].tolist() == [1, 0, 0, 0]
    assert host['state_qsa_slots'].tolist() == [-1] * 4
    stage_state_transfer(host, source=5, destination=1, consumed=67)
    assert host['state_qsa_slots'].tolist() == [1, -1, -1, -1]
    assert host['state_accepted_tokens'].tolist() == [1, 0, 0, 0]
    stage_state_transfer(host, source=1, destination=1, consumed=0, reset=True)
    assert host['state_operation'].tolist() == [1, 0, 0, 0]
    assert torch.equal(host['page_tables'], pages)
    for consumed, reset in [(-1, False), (0, False), (1, True), (True, True)]:
        with pytest.raises(ValueError):
            stage_state_transfer(host, source=1, destination=5, consumed=consumed, reset=reset)
