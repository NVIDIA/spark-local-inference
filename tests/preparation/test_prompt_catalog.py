# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare real exports made with the same shape catalog and weights."""

import os

import pytest

from ignition.preparation import runtime_core_pb2 as proto
from ignition.preparation.core_file import read_index


@pytest.mark.skipif(not os.environ.get("FLARE_PROMPT_TEST_OFF_CORE"),
    reason="requires matching flag-off and flag-on exports")
def test_flag_off_export_preserves_ordinary_catalog_and_buffers():
    off, _ = read_index(os.environ["FLARE_PROMPT_TEST_OFF_CORE"])
    on, _ = read_index(os.environ["FLARE_PROMPT_TEST_CORE"])
    assert not any(g.kind == proto.SCORED_PREFILL for g in off.contract.model_contract.graphs)
    assert any(g.kind == proto.SCORED_PREFILL for g in on.contract.model_contract.graphs)
    assert [g.SerializeToString() for g in off.contract.model_contract.graphs] == [
        g.SerializeToString() for g in on.contract.model_contract.graphs if g.kind != proto.SCORED_PREFILL]
    assert len(off.ranks) == len(on.ranks)
    for before, after in zip(off.ranks, on.ranks):
        def graphs(rank):
            result = []
            for graph in rank.execution.graphs:
                if graph.kind == "scored_prefill":
                    continue
                value = proto.ExecutionGraph()
                value.CopyFrom(graph)
                value.ClearField("estimated_duration_ns")
                result.append(value.SerializeToString())
            return result
        def buffers(rank):
            result = []
            for buffer in rank.execution.buffers:
                if buffer.name in ("prompt_target_tokens", "prompt_logprobs"):
                    continue
                value = proto.Buffer()
                value.CopyFrom(buffer)
                value.ClearField("reference")
                result.append(value.SerializeToString())
            return result
        assert not any(b.name.startswith("prompt_") for b in before.execution.buffers)
        assert graphs(before) == graphs(after)
        assert buffers(before) == buffers(after)
