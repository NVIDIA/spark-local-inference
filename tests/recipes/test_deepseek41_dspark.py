# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DSpark capacity declarations and packed verification coordinates."""

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.deepseek41.bootstrap import PreparationConfig
from ignition.recipes.deepseek41.config import DSparkConfig
from ignition.recipes.deepseek41.recipe import RECIPE


def test_dspark_preparation_and_recipe_draft_defaults():
    config = PreparationConfig(max_active=4)
    assert config.speculative_widths == (1, 2, 3, 4, 5, 6)
    assert config.draft_tokens == 5 and config.decode_tokens == 24
    assert config.packed_verification
    assert config.draft_prefix_selection == "gpu"
    assert RECIPE.speculation.draft_tokens == 7 and RECIPE.max_draft_tokens == 7
    assert RECIPE.speculation.adaptive
    assert PreparationConfig(dspark=False).speculative_shapes == ()
    assert PreparationConfig(dspark=False).draft_tokens == 0
    with pytest.raises(ValueError):
        PreparationConfig(speculative_widths=(1, 9))


@pytest.mark.parametrize("draft,markov", [(None, None), ("nvfp4", None), (None, "nvfp4"), ("nvfp4", "nvfp4")])
def test_draft_precision_preserves_graph_capacities(draft, markov):
    from ignition.recipes.deepseek4.bootstrap import PreparationConfig as V4Config

    for configuration in (PreparationConfig, V4Config):
        control = configuration(max_active=16)
        candidate = configuration(max_active=16, draft_head_quantization=draft,
                                  markov_head_quantization=markov)
        if configuration is V4Config:
            assert candidate.speculative_round_shapes == control.speculative_round_shapes
        else:
            assert candidate.speculative_shapes == control.speculative_shapes
        assert candidate.prefill_capacities == control.prefill_capacities
        if draft or markov:
            with pytest.raises(ValueError, match="positive learned"):
                configuration(dspark=False, draft_head_quantization=draft, markov_head_quantization=markov)


def test_mxfp8_draft_head_preserves_deepseek41_capacities_and_markov_precision():
    from dataclasses import replace
    from ignition.recipes.deepseek4.bootstrap import PreparationConfig as V4Config

    control = PreparationConfig(max_active=16)
    candidate = replace(control, draft_head_quantization="mxfp8")
    assert candidate.speculative_shapes == control.speculative_shapes
    assert candidate.prefill_capacities == control.prefill_capacities
    assert candidate.markov_head_quantization is None
    with pytest.raises(ValueError, match="DeepSeek V4 draft head"):
        V4Config(draft_head_quantization="mxfp8")


@pytest.mark.parametrize("width", range(1, 9))
def test_explicit_draft_limit_sizes_the_parallel_block(width):
    config = PreparationConfig(max_active=4, speculative_widths=tuple(range(1, width + 1)))
    assert config.draft_tokens == width - 1
    assert config.decode_tokens == 4 * width
    assert config.packed_verification == (width > 1)


def test_native_dspark_geometry():
    text = dict(num_nextn_predict_layers=3, dspark_block_size=5, dspark_noise_token_id=128799,
        dspark_target_layer_ids=[37, 38, 39], dspark_markov_rank=256, dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3, num_hidden_layers=40, vocab_size=129280, compress_ratios=[0] * 43)
    assert DSparkConfig.from_text(text).target_layers == (37, 38, 39)
    assert DSparkConfig.from_text({**text, "dspark_block_size": 7}).block_size == 7
    with pytest.raises(ValueError):
        DSparkConfig.from_text({**text, "dspark_target_layer_ids": [39, 37, 38]})


def test_verification_packs_partial_and_inactive_requests_without_crossing_sequences():
    metadata = Metadata.allocate(24, 4, 8, "cpu", samples=24)
    tokens = torch.arange(24).view(4, 6) + 100
    pages = torch.arange(32, dtype=torch.int32).view(4, 8)
    metadata.set(torch.tensor([127, 80, 200, 4]), torch.tensor([3, 0, 1, 5]),
        torch.tensor([3, 4, 5, 6]), pages, tokens, torch.tensor([4]))
    assert metadata.live_counts.tolist() == [9, 4]
    assert metadata.query_start_loc.tolist() == [0, 3, 3, 4, 9]
    assert metadata.token_ids[:9].tolist() == [100, 101, 102, 112, 118, 119, 120, 121, 122]
    assert metadata.positions[:9].tolist() == [127, 128, 129, 200, 4, 5, 6, 7, 8]
    assert metadata.positions[9:].eq(-1).all()
    assert metadata.state_rows.tolist() == [3, 0, 5, 6]
    assert metadata.sample_indices.view(4, 6)[:, 0].tolist() == [0, 3, 3, 4]


def test_single_token_verification_packs_page_tables_across_inactive_rows():
    metadata = Metadata.allocate(4, 4, 8, "cpu")
    pages = torch.arange(32, dtype=torch.int32).view(4, 8)
    metadata.set(torch.tensor([127, 80, 200, 4]), torch.tensor([1, 0, 1, 0]),
        torch.tensor([3, 4, 5, 6]), pages, torch.tensor([[10], [11], [12], [13]]), torch.tensor([4]))
    assert metadata.token_ids.tolist() == [10, 12, 0, 0]
    assert metadata.positions.tolist() == [127, 200, -1, -1]
    assert torch.equal(metadata.token_page_table[:2], pages[[0, 2]])
    assert torch.equal(metadata.page_table, pages)


@pytest.mark.parametrize("world", [1, 4])
def test_decode_page_tables_follow_packed_tokens_across_inactive_rows(world):
    from types import SimpleNamespace
    from ignition.components.paged_batch import BatchInputs, PreparedBatch

    inputs = BatchInputs.allocate(4, 8, "cpu", capacity=4)
    inputs.row_slot_ids.copy_(torch.tensor([3, 1, 0, 2]))
    inputs.live_rows.fill_(4)
    inputs.token_ids.copy_(torch.tensor([10, 11, 12, 13]))
    inputs.request_state_rows[:, 0].copy_(torch.tensor([1, 2, 3, 4]))
    inputs.page_tables.copy_(torch.arange(32).view(4, 8))
    batch = PreparedBatch.__new__(PreparedBatch)
    batch.inputs = inputs
    batch.requests = SimpleNamespace(capacity=4, status=torch.ones(4, dtype=torch.int32),
        error_code=torch.zeros(4, dtype=torch.int32), generation=torch.zeros(4, dtype=torch.int64))
    batch.metadata = None
    batch.token_state = SimpleNamespace(gather=lambda: (torch.tensor([20, 21, 22, 23]),
        torch.tensor([127, 80, 200, 4]), None, torch.tensor([1, 0, 1, 0])))
    batch.shard = SimpleNamespace(world_size=world)
    batch.rows, batch.tokens, batch.mode = 4, 4, "decode"
    batch.present = torch.ones(4, dtype=torch.bool)
    batch.sample_count, batch.max_query_tokens = 4, 1
    metadata = batch.begin(torch.tensor([[0, 0, 0, 0] + [batch.shard.world_size] * 4]), present=batch.present)
    assert metadata.token_ids.tolist() == [20, 22, 0, 0]
    assert metadata.positions.tolist() == [127, 200, -1, -1]
    assert torch.equal(metadata.token_page_table[:2], inputs.page_tables[[3, 0]])



def test_vocabulary_parallel_sampling_requires_exact_retained_proposal_formats():
    from dataclasses import replace

    base = PreparationConfig(vocabulary_parallel_sampling=True, markov_head_quantization="fp8")
    assert base.draft_tokens > 0
    assert replace(base, markov_head_quantization=None).vocabulary_parallel_sampling
    with pytest.raises(ValueError, match="BF16 or row-scaled FP8"):
        replace(base, markov_head_quantization="nvfp4")
    with pytest.raises(ValueError, match="positive learned"):
        replace(base, dspark=False)


def test_draft_confidence_consumes_local_hidden_before_head_normalization():
    from types import SimpleNamespace

    from ignition.components.parallel_draft import DraftProposalState
    from ignition.components.split_draft import PreparedDraftConfidence
    from ignition.recipes.deepseek41.dspark import PreparedDraft

    embedding = torch.arange(32, dtype=torch.float32).view(8, 4).bfloat16() / 8
    norm = torch.tensor([1., 2., 3., 4.], dtype=torch.bfloat16)
    head = torch.arange(12, dtype=torch.float32).view(3, 4) / 16
    markov = torch.arange(16, dtype=torch.float32).view(8, 2) / 32
    confidence_weight = torch.arange(6, dtype=torch.float32).view(1, 6) / 16
    draft = object.__new__(PreparedDraft)
    draft.weights = SimpleNamespace(draft=SimpleNamespace(norm=norm),
        config=SimpleNamespace(text={"rms_norm_eps": 1e-5}))
    draft.embedding = lambda *, token_ids, num_rows: embedding[token_ids]
    draft.window_metadata = lambda metadata: None
    draft.layers = ()
    draft.head = lambda values: torch.nn.functional.linear(values.float(), head).bfloat16()
    confidence = object.__new__(PreparedDraftConfidence)
    confidence.batch, confidence.depth, confidence.offset = 2, 1, 1
    confidence.feature_dtype = torch.float32
    confidence.embedding = lambda *, token_ids: markov[token_ids]
    confidence.projection = lambda values: torch.nn.functional.linear(values, confidence_weight)

    def forward(metadata, candidates, active):
        logits, _, features = draft(metadata)
        proposal = DraftProposalState(torch.zeros(2, dtype=torch.int64), active,
            candidates, candidates[:, 1:], active[:, None], torch.zeros(2, dtype=torch.int64),
            metadata, features)
        return logits, features["head_hidden"], confidence(proposal)

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    previous = None
    for tokens in (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[4, 5], [6, 7]])):
        metadata = Metadata.pack(torch.zeros(2, dtype=torch.int64), torch.full((2,), 2, dtype=torch.int32),
            torch.tensor([1, 2]), torch.zeros((2, 1), dtype=torch.int32), tokens,
            torch.tensor([2], dtype=torch.int32), token_capacity=4, sample_count=4)
        active = torch.tensor([True, False])
        logits, features, actual = compiled(metadata, tokens, active)
        expected_features = embedding[tokens]
        torch.testing.assert_close(features, expected_features, rtol=0, atol=0)
        hidden = expected_features.view(4, 4).float()
        normalized = (hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-5) * norm.float()).bfloat16()
        torch.testing.assert_close(logits, torch.nn.functional.linear(normalized.float(), head).bfloat16(), rtol=0, atol=0)
        preceding = torch.where(active, tokens[:, 0], 0)
        expected = torch.nn.functional.linear(
            torch.cat((expected_features[:, 1].float(), markov[preceding]), dim=1), confidence_weight).sigmoid()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if previous is not None:
            torch.testing.assert_close(previous, embedding[torch.tensor([[1, 2], [3, 4]])], rtol=0, atol=0)
        previous = features
