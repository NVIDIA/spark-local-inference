"""Completion capacity defines the record vector and all native sidecar axes."""

import pytest
import torch

from ignition.components.capacity import completion_token_capacity
from ignition.components.completion import CompletionOutputs, LOGPROB_FIELDS, validate_completion_buffers
from ignition.preparation.execution import GraphSignature
from tests.preparation.test_execution import _execution, _tensor


@pytest.mark.parametrize("wire,expected", [(0, 8), (1, 1), (8, 8), (13, 13), (16, 16), (32, 32), ((1 << 32) - 1, (1 << 32) - 1)])
def test_wire_capacity_zero_defaults_to_eight_and_positive_values_are_literal(wire, expected):
    assert completion_token_capacity(wire) == expected


@pytest.mark.parametrize("value", [-1, 1 << 32, True, 8., None])
def test_capacity_rejects_values_outside_uint32(value):
    with pytest.raises(ValueError, match="uint32"):
        completion_token_capacity(value)


@pytest.mark.parametrize("tokens", [1, 8, 13, 16, 32])
def test_completion_buffers_derive_all_output_axes_from_records(tokens):
    outputs = CompletionOutputs.allocate((2, 4, 16 + tokens), device="meta")
    assert outputs.token_capacity == tokens
    assert outputs.records.shape == (2, 4, 16 + tokens)
    assert outputs.logprobs.shape == (2, 4, tokens, LOGPROB_FIELDS)
    assert validate_completion_buffers(outputs.buffers(), 4, tokens) == tokens
    if tokens == 1:
        assert outputs.speculative_survival is None
        assert "completion_speculative_survival" not in outputs.buffers()
    else:
        assert outputs.speculative_survival.shape == (2, 4, tokens - 1)
    private = outputs.private()
    assert private.token_capacity == tokens
    assert private.records.shape == outputs.records.shape
    assert private.logprobs.shape == outputs.logprobs.shape


@pytest.mark.parametrize("tokens", [8, 13, 16, 32])
@pytest.mark.parametrize("sidecar", ["completion_logprobs", "completion_speculative_survival", "declared"])
def test_sidecars_and_explicit_capacity_must_match_record_geometry(tokens, sidecar):
    outputs = CompletionOutputs.allocate((2, 4, 16 + tokens), device="meta")
    buffers = outputs.buffers()
    if sidecar == "declared":
        with pytest.raises(ValueError, match="disagrees"):
            validate_completion_buffers(buffers, 4, tokens + 1)
    else:
        shape = list(buffers[sidecar].shape)
        shape[2] += 1
        buffers[sidecar] = torch.empty(shape, dtype=buffers[sidecar].dtype, device="meta")
        with pytest.raises(ValueError, match=sidecar):
            validate_completion_buffers(buffers, 4, tokens)


@pytest.mark.parametrize("fields", [16, 15, 0, 16 + (1 << 32)])
def test_record_geometry_rejects_empty_tokens_and_capacity_overflow_before_allocation(fields):
    with pytest.raises(ValueError):
        CompletionOutputs.allocate((2, 4, fields), device="meta")


@pytest.mark.parametrize("tokens", [1, 8, 13, 16, 32])
def test_prepared_execution_exports_explicit_capacity_and_protobuf_roundtrips(tokens, monkeypatch):
    from ignition.preparation import runtime_package_pb2 as p

    buffers = dict(_execution().buffers)
    buffers["completion_records"] = _tensor("int64", (2, 4, 16 + tokens))
    model = _execution(graphs={"round": object()},
        signatures={"round": GraphSignature("speculative", 1, tokens)}, buffers=buffers)
    spans = [(4096, 4), (8192, 8), (12288, 8 * 8 * (16 + tokens)), (16384, 128), (20480, 1)]
    monkeypatch.setattr(model, "_validate_resources", lambda: (9, [11], spans))
    manifest = model.manifest()
    assert manifest["completion_token_capacity"] == tokens
    execution = p.Execution(abi_version=manifest["abi_version"], completion_token_capacity=manifest["completion_token_capacity"])
    assert p.Execution.FromString(execution.SerializeToString()).completion_token_capacity == tokens
    assert completion_token_capacity(p.Execution().completion_token_capacity) == 8
    with pytest.raises(ValueError, match="completion token capacity"):
        _execution(graphs={"round": object()},
            signatures={"round": GraphSignature("speculative", 1, tokens + 1)}, buffers=buffers)


def test_execution_checks_full_proposal_depth_against_output_capacity():
    buffers = dict(_execution().buffers)
    buffers["completion_records"] = _tensor("int64", (2, 4, 24))
    with pytest.raises(ValueError, match="completion token capacity"):
        _execution(graphs={"round": object()}, signatures={
            "round": GraphSignature("speculative", 1, 3, draft_tokens=12, confidence_width=12)}, buffers=buffers)
    buffers["completion_records"] = _tensor("int64", (2, 4, 29))
    _execution(graphs={"round": object()}, signatures={
        "round": GraphSignature("speculative", 1, 3, draft_tokens=12, confidence_width=12)}, buffers=buffers)


def test_capacity_one_omits_zero_extent_survival_storage():
    outputs = CompletionOutputs.allocate((2, 4, 17), device="meta")
    buffers = outputs.buffers()
    buffers["completion_speculative_survival"] = torch.empty((2, 4, 0), dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="must be omitted"):
        validate_completion_buffers(buffers, 4)
