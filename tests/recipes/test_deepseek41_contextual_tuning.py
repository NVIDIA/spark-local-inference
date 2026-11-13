# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Joint tuning restores catalog-shaped inputs without rewinding TP state."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.window_indices import PreparedWindowIndices
from ignition.preparation.execution import GraphSignature
from ignition.recipes.deepseek41.contextual_tuning import CatalogLayerFixture


def catalog_layer(batch, tokens, width, mode, *, carry=False):
    metadata = Metadata.allocate(tokens, batch, 80, "cpu", max_query_tokens=width)
    window = torch.zeros((batch + 2, 128 * 528), dtype=torch.uint8)
    pool = SimpleNamespace(memory_capacity=torch.tensor([batch + 2, 80]))
    indices = PreparedWindowIndices(metadata, mode=mode, tokens=tokens,
        page_size=128 if mode == "decode" else 128 + width,
        width=128 + width if mode == "draft" else 128)
    indices.ratios = {}
    runtime = SimpleNamespace(sequence=torch.tensor([17]))
    attention = SimpleNamespace(pool=pool, mode=mode, global_kv=None, ratio=0,
        window=window, window_metadata=indices, replay_start=None,
        window_cache=window if mode == "decode" else torch.zeros((batch, (128 + width) * 528), dtype=torch.uint8))
    scratch = SimpleNamespace(residual=torch.zeros((tokens, 4, 16), dtype=torch.bfloat16),
        pre_mix=torch.zeros((tokens, 4)), mlp_output=torch.zeros((tokens, 16), dtype=torch.bfloat16),
        ffn_hc=SimpleNamespace(residual=torch.zeros((tokens, 4, 16), dtype=torch.bfloat16),
            post=torch.zeros((tokens, 4)), comb=torch.zeros((tokens, 4, 4))),
        attention=SimpleNamespace(query=torch.zeros((tokens, 2, 8), dtype=torch.bfloat16),
            output=torch.zeros((tokens, 16), dtype=torch.bfloat16),
            window_values=torch.zeros((tokens, 512), dtype=torch.bfloat16),
            selected=torch.full((tokens, 512), -1, dtype=torch.int32),
            candidates=torch.full((tokens, 512), -1, dtype=torch.int32),
            candidate_counts=torch.zeros(tokens, dtype=torch.int32)),
        routed=torch.zeros((tokens, 16), dtype=torch.bfloat16),
        router=SimpleNamespace(ids=torch.zeros((tokens, 6), dtype=torch.int32)))
    layer = SimpleNamespace(metadata=metadata, attention=attention, scratch=scratch,
        carry=carry, post=object(), engram=None, boundary=None,
        attention_reduce=SimpleNamespace(runtime=runtime), mlp_reduce=SimpleNamespace(runtime=runtime),
        router=SimpleNamespace(shape=(tokens, 384)))
    weights = SimpleNamespace(embedding=torch.arange(64 * 16).view(64, 16).bfloat16() / 1024,
                             config=SimpleNamespace(text={"vocab_size": 64}))
    decoder = SimpleNamespace(layers=(layer,), weights=weights, vision=None)
    kind = "prefill" if mode == "prefill" else "decode" if mode == "decode" else "speculative"
    signature = GraphSignature(kind, batch, width,
        packed_token_capacity=tokens if kind == "speculative" else 0)
    return decoder, signature, runtime


def initialize(fixture):
    fixture.initialize()
    fixture.output.zero_()
    scratch = fixture.scratch
    fixture.observations = {"attention.output": scratch.attention.output,
        "attention.window_values": scratch.attention.window_values,
        "attention.selected": scratch.attention.selected, "router.ids": scratch.router.ids}


@pytest.mark.parametrize("batch,tokens,width,mode", [
    (1, 1, 1, "decode"), (32, 32, 1, "decode"),
    (1, 3, 3, "verify"), (4, 9, 8, "verify"), (32, 128, 8, "verify"),
    (2, 14, 7, "draft"), (1, 4096, 4096, "prefill"),
])
def test_context_uses_actual_packed_geometry_and_restores_inactive_rows(batch, tokens, width, mode):
    decoder, signature, runtime = catalog_layer(batch, tokens, width, mode)
    fixture = CatalogLayerFixture("shape", signature, decoder, 0)
    assert fixture.trial_signature.batch_capacity == batch
    assert fixture.trial_signature.token_capacity == width
    if mode in ("verify", "draft"):
        assert fixture.trial_signature.packed_token_capacity == tokens
    initialize(fixture)
    layer = decoder.layers[0]
    fixture.restore_tune(layer)
    metadata = layer.metadata
    assert metadata.live_counts.tolist() == [tokens, batch]
    assert metadata.lengths.sum().item() == tokens
    assert metadata.lengths.max().item() <= width
    assert metadata.state_rows.tolist() == list(range(1, batch + 1))
    assert metadata.query_start_loc[-1].item() == tokens
    fixture.restore_tune(layer, case_index=1)
    if batch > 1:
        assert metadata.live_counts[1].item() == batch - 1
        assert metadata.lengths[-1].item() == metadata.state_rows[-1].item() == 0
    else:
        assert metadata.lengths.item() == max(1, tokens - 1)
    live = metadata.live_counts[0].item()
    assert metadata.positions[live:].eq(-1).all()
    assert metadata.token_ids[live:].eq(0).all()
    if batch > 1:
        assert torch.equal(metadata.token_page_table[:live], metadata.page_table[metadata.requests[:live]])
    layer.metadata.starts.zero_()
    fixture.window.fill_(255)
    fixture.restore_holdout(layer)
    assert metadata.starts.eq(fixture.start).all()
    assert torch.equal(fixture.window, fixture.cases[2].window)
    assert runtime.sequence.item() == 17
    fixture.close()
    assert not fixture.cases and fixture.active_case is None


def test_carry_inputs_and_vision_token_boundary_are_restored():
    decoder, signature, runtime = catalog_layer(4, 16, 8, "verify", carry=True)
    decoder.vision = SimpleNamespace(token_base=16)
    fixture = CatalogLayerFixture("packed", signature, decoder, 0)
    initialize(fixture)
    layer = decoder.layers[0]
    fixture.restore_tune(layer)
    assert layer.metadata.token_ids.max().item() < 16
    assert torch.equal(layer.scratch.ffn_hc.residual, layer.scratch.residual)
    assert layer.scratch.mlp_output.eq(0).all()
    assert layer.scratch.ffn_hc.post.eq(1).all()
    assert torch.equal(layer.scratch.ffn_hc.comb[0], torch.eye(4))
    assert runtime.sequence.item() == 17


def test_verify_history_validation_detects_mutation_and_replay_restores_it():
    decoder, signature, runtime = catalog_layer(4, 9, 8, "verify")
    fixture = CatalogLayerFixture("packed", signature, decoder, 0)
    initialize(fixture)
    layer = decoder.layers[0]
    fixture.restore_tune(layer)
    layer.scratch.attention.output.zero_()
    fixture.output.zero_()
    fixture.validate_result(layer)
    fixture.window[2, 0] ^= 1
    with pytest.raises(AssertionError, match="cache history"):
        fixture.validate_result(layer)
    fixture.restore_tune(layer)
    layer.scratch.attention.output.zero_()
    fixture.output.zero_()
    fixture.validate_result(layer)
    assert runtime.sequence.item() == 17


def test_context_rejects_unbacked_rows_and_external_layer_inputs():
    decoder, signature, _ = catalog_layer(4, 9, 8, "verify")
    decoder.layers[0].attention.pool.memory_capacity[0] = 4
    with pytest.raises(ValueError, match="backing"):
        CatalogLayerFixture("packed", signature, decoder, 0)
    decoder.layers[0].attention.pool.memory_capacity[0] = 6
    decoder.layers[0].engram = object()
    with pytest.raises(ValueError, match="restorable"):
        CatalogLayerFixture("packed", signature, decoder, 0)


def test_compressed_validation_checks_readonly_history_and_live_selected_bounds():
    decoder, signature, _ = catalog_layer(4, 9, 8, "verify")
    fixture = CatalogLayerFixture("packed", signature, decoder, 0)
    initialize(fixture)
    layer = decoder.layers[0]
    fixture.restore_tune(layer, case_index=1)
    layer.scratch.attention.output.zero_()
    fixture.output.zero_()
    layer.attention.ratio = 2
    selected = torch.full((9, 512), -1, dtype=torch.int32)
    layer.scratch.attention.selected = selected
    fixture.observations["attention.selected"] = selected
    selected[:, 0] = fixture.start // 2 - 1
    fixture.validate_result(layer)
    selected[0, 0] = (layer.metadata.positions[0] + 1) // 2
    with pytest.raises(AssertionError, match="valid cache prefix"):
        fixture.validate_result(layer)
    selected[0, 0] = -2
    with pytest.raises(AssertionError, match="valid cache prefix"):
        fixture.validate_result(layer)
    selected[0, 0] = fixture.start // 2 - 1
    from dataclasses import replace
    history = torch.ones(8, dtype=torch.uint8)
    fixture.active_case = replace(fixture.active_case, readonly=((history, history.clone()),))
    history[0] = 0
    with pytest.raises(AssertionError, match="read-only compressed history"):
        fixture.validate_result(layer)


def test_preparation_inputs_survive_serving_example_release():
    decoder, signature, _ = catalog_layer(4, 9, 8, "verify")
    fixture = CatalogLayerFixture("packed", signature, decoder, 0)
    initialize(fixture)
    layer = decoder.layers[0]
    layer.metadata = layer.scratch = None
    fixture.restore_tune(layer)
    assert fixture.metadata.lengths.sum().item() == 9
    assert fixture.incoming[0].shape == (9, 4, 16)
    fixture.release_examples()
    assert fixture.metadata is None and fixture.scratch is None and fixture.output is None


def test_trial_passes_explicit_inputs_after_serving_example_release():
    decoder, signature, _ = catalog_layer(4, 9, 8, "verify")
    fixture = CatalogLayerFixture("packed", signature, decoder, 0)
    initialize(fixture)
    fixture.restore_tune(fixture.layer)
    observed = []

    class Layer(SimpleNamespace):
        def __call__(self, incoming, pre_mix, metadata, indices, selection):
            observed.append((incoming, pre_mix, metadata, indices, selection))
            return (incoming[0] + 1,), pre_mix, None, selection

    layer = Layer(**vars(fixture.layer))
    layer.metadata = layer.scratch = None
    fixture.layer.metadata = fixture.layer.scratch = None
    fixture.trial_forward(layer)()
    torch.testing.assert_close(fixture.output, fixture.incoming[0] + 1, rtol=0, atol=0)
    incoming, pre_mix, metadata, indices, selection = observed[0]
    assert incoming is fixture.incoming and pre_mix is fixture.pre_mix
    assert metadata is fixture.metadata and selection is fixture.selection
    assert indices.window_lengths.eq(128).all()
