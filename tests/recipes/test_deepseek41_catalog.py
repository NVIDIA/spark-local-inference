# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V4.1 registers packed rounds while retaining target-only DSpark continuations."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.execution import GraphSignature
from ignition.recipes.deepseek41.bootstrap import PreparationConfig, _prepare_speculation_catalog


@pytest.fixture
def register_catalog(monkeypatch):
    from ignition.components import packed_speculation, parallel_draft

    constructed = []

    class DraftRound:
        def __init__(self, owner, batch, width, name, **options):
            self.owner, self.batch, self.width, self.name = owner, batch, width, name
            self.options, self.prepared = options, False
            constructed.append(self)

        def prepare(self):
            self.prepared = True

    class Packed:
        def __init__(self, source, target, prefix_limits, confidence_output, **options):
            self.source, self.target = source, target
            self.prefix_limits = prefix_limits
            assert source.prepared and target.prepared
            assert source.batch == target.batch
            assert confidence_output.shape[0] == 2

    monkeypatch.setattr(parallel_draft, "PreparedParallelDraftRound", DraftRound)
    monkeypatch.setattr(packed_speculation, "PreparedPackedSpeculativeRound", Packed)

    def register(config):
        constructed.clear()
        records, buffers, host_allocations = [], {}, []

        def host_allocate(name, shape, dtype):
            host_allocations.append((name, shape, dtype))
            value = torch.zeros(shape, dtype=dtype)
            return SimpleNamespace(device_view=value, host_view=value)

        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0, draft_tokens=None):
            records.append((name, forward, GraphSignature(kind, rows, tokens, tuple(fields),
                speculation_enabled=enabled, packed_token_capacity=packed_tokens, draft_tokens=draft_tokens,
                confidence_width=draft_tokens if "speculative_prefix_limits" in fields else 0)))

        confidence = _prepare_speculation_catalog(config, SimpleNamespace() if config.dspark else None,
            add=add, buffers=buffers,
            forward_inputs=("row_slot_ids", "query_lengths", "request_state_rows"),
            host_allocate=host_allocate, device="cpu")
        if confidence is not None:
            assert confidence.device_view is buffers["proposal_confidence"]
            assert confidence.host_view.shape == (2, config.request_capacity, config.draft_tokens)
        return records, buffers, None, list(constructed), host_allocations

    return register


def test_default_catalog_contains_only_packed_rounds_and_width_one_continuations(register_catalog):
    config = PreparationConfig()
    assert (config.max_active, config.prefill_chunk_size, config.draft_tokens,
            config.speculative_token_capacity) == (32, 4096, 5, 128)
    records, buffers, storage, constructed, hosts = register_catalog(config)
    capacities = defaultdict(list)
    for name, forward, signature in records:
        assert signature.speculation_enabled
        if "speculative_prefix_limits" in signature.inputs:
            capacities[signature.batch_capacity].append(signature.packed_token_capacity)
            assert signature.token_capacity == 6 and signature.draft_tokens == 5
            assert signature.confidence_width == 5
            assert "query_lengths" not in signature.inputs
            assert "speculative_prefix_limits" in signature.inputs
            assert forward.prefix_limits is buffers["speculative_prefix_limits"]
        else:
            assert signature.kind == "speculative" and signature.token_capacity == 1
            assert "query_lengths" in signature.inputs
            assert signature.packed_token_capacity == signature.batch_capacity
    assert {(batch, tokens) for batch, values in capacities.items() for tokens in values} == set(config.packed_speculative_shapes)
    assert sum(map(len, capacities.values())) == 64
    assert sorted(capacities[1]) == [2, 3, 4, 5, 6]
    for batch, physical in capacities.items():
        assert min(physical) == batch + 1
        assert max(physical) == min(128, batch * 6)
    assert len(records) == 70
    assert len({name for name, _, _ in records}) == len(records)
    targets = [round for round in constructed if not round.options.get("generate_proposals", True)]
    assert {(round.batch, round.options["target_token_capacity"]) for round in targets} == set(config.packed_speculative_shapes)
    assert {s.batch_capacity for _, _, s in records if s.kind == "speculative"} == {1, 2, 4, 8, 16, 32}
    assert all(not name.startswith(("proposal", "verification")) for name, _, _ in records)
    proposals = [round for round in constructed if round.options.get("proposal_only")]
    assert len(proposals) == 6 and all(round.width == 6 and round.prepared for round in proposals)
    assert hosts == [("proposal_confidence", (2, 32, 5), torch.float32)]
    assert storage is None


def test_small_physical_budget_keeps_every_width_one_batch(register_catalog):
    records, _, _, constructed, _ = register_catalog(PreparationConfig(speculative_token_capacity=6))
    packed = [s for _, _, s in records if "speculative_prefix_limits" in s.inputs]
    assert {(s.batch_capacity, s.packed_token_capacity) for s in packed} == {
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 3), (2, 4), (2, 5), (2, 6), (4, 5), (4, 6),
    }
    assert {s.batch_capacity for _, _, s in records if s.kind == "speculative"} == {1, 2, 4, 8, 16, 32}
    assert {round.batch for round in constructed if round.options.get("proposal_only")} == {1, 2, 4}


@pytest.mark.parametrize("fixed", [None, 1])
def test_zero_drafts_retains_dspark_context_without_proposal_storage(register_catalog, fixed):
    config = PreparationConfig(speculative_widths=(1,), speculative_fixed_width=fixed)
    assert not config.packed_verification
    assert config.draft_prefix_selection == "cpu"
    records, buffers, storage, constructed, hosts = register_catalog(config)
    assert len(records) == 6 and len(constructed) == 6
    assert all(s.kind == "speculative" and s.token_capacity == 1 for _, _, s in records)
    assert buffers == {} and storage is None and hosts == []


def test_no_mtp_omits_the_dspark_catalog(register_catalog):
    config = PreparationConfig(dspark=False)
    assert config.speculative_shapes == () and config.draft_tokens == 0
    assert not config.packed_verification
    assert register_catalog(config) == ([], {}, None, [], [])


@pytest.mark.parametrize("options", [
    {"draft_prefix_selection": "cpu"}, {"packed_verification": False},
    {"speculative_fixed_width": 6}, {"speculation_reference_graphs": True},
])
def test_positive_dspark_rejects_other_catalogs(options):
    with pytest.raises(ValueError, match="requires (packed GPU|adaptive packed GPU)"):
        PreparationConfig(**options)


def test_invalid_prefix_selection_is_rejected():
    with pytest.raises(ValueError, match="draft prefix selection must be cpu or gpu"):
        PreparationConfig(draft_prefix_selection="automatic")


@pytest.mark.parametrize("options", [
    {"dspark": False}, {"speculative_widths": (1,)},
])
def test_gpu_prefix_selection_requires_positive_draft_depth(options):
    with pytest.raises(ValueError, match="requires positive DSpark draft depth"):
        PreparationConfig(**options, draft_prefix_selection="gpu")
