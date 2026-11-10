# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Residual boundaries, accepted histories, and runtime metadata remain explicit."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ignition.recipes.step5.decoder import PreparedDecoder, PreparedDecoderLayer
from ignition.recipes.step5.model import DecoderLayer, RMSNorm


class Norm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.operation = RMSNorm(weight, eps)

    def forward(self, source, update=None):
        if update is not None:
            source.add_(update)
        return self.operation(source).bfloat16()


class ReducedNorm(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.norm = Norm(weight)

    def forward(self, update, residual):
        residual.add_(update.float())
        return self.norm(residual)


class Attention(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, source, *, metadata, proposal=None):
        return source @ self.weight, (source.clone(), source.clone())

    def release_examples(self):
        pass


class FeedForward(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, source, *, metadata=None):
        return (source @ self.weight).bfloat16()


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("draft", [False, True])
def test_step_residual_rounding_matches_canonical_layers(fused, draft):
    torch.manual_seed(17)
    tokens, hidden, vocabulary = 3, 8, 16
    embedding = torch.randn((tokens, hidden)).bfloat16()
    decoder = PreparedDecoder.__new__(PreparedDecoder)
    decoder.weights = SimpleNamespace(draft=(object(),) if draft else None)
    decoder.mode, decoder.vision, decoder.scoring = "verify", None, None
    decoder.embedding = SimpleNamespace(lookup=lambda **kwargs: embedding.clone(), gather=lambda local: local.clone())
    decoder.layers, reference = [], []
    weights = [SimpleNamespace(input_norm=(torch.randn(hidden) * .1).bfloat16(),
        post_attention_norm=(torch.randn(hidden) * .1).bfloat16()) for _ in range(2)]
    for index, gains in enumerate(weights):
        aw, fw = ((torch.randn(hidden, hidden) * .1).bfloat16() for _ in range(2))
        layer = PreparedDecoderLayer.__new__(PreparedDecoderLayer)
        nn.Module.__init__(layer)
        layer.attention, layer.feed_forward = Attention(aw), FeedForward(fw)
        layer.input_norm = None if fused and index else Norm(gains.input_norm)
        layer.fused_attention = fused
        layer.post_attention_norm = ReducedNorm(gains.post_attention_norm) if fused else Norm(gains.post_attention_norm)
        layer.output_norm = ReducedNorm(weights[index + 1].input_norm) if fused and index + 1 < 2 else None
        decoder.layers.append(layer)
        reference.append(DecoderLayer(gains, lambda source, weight=aw: source @ weight, FeedForward(fw), 1e-6))
    final_weight = (torch.randn(hidden) * .1).bfloat16()
    decoder.norm = Norm(final_weight)
    head = (torch.randn(hidden, vocabulary) * .1).bfloat16()
    decoder.head = lambda source: source @ head
    metadata = SimpleNamespace(token_ids=torch.arange(tokens), live_counts=torch.tensor([tokens, 1], dtype=torch.int32),
                               sample_indices=torch.tensor([0, 2]))
    expected = embedding.float()
    for layer in reference:
        expected = layer(expected)
    hidden_states, logits, auxiliary, records = decoder(metadata)
    expected_hidden = Norm(final_weight)(expected[metadata.sample_indices].bfloat16())
    torch.testing.assert_close(hidden_states, expected_hidden, rtol=0, atol=0)
    torch.testing.assert_close(logits, expected_hidden @ head, rtol=0, atol=0)
    assert len(records) == 2
    if draft:
        torch.testing.assert_close(auxiliary[0], expected.bfloat16(), rtol=0, atol=0)
        assert torch.equal(auxiliary[1], embedding)
    else:
        assert auxiliary is None
    decoder.release_outputs()
    repeated = decoder(metadata)
    torch.testing.assert_close(repeated[0], hidden_states, rtol=0, atol=0)
    assert repeated[0].data_ptr() != hidden_states.data_ptr()


def test_minimax_explicit_metadata_preserves_canonical_unfused_forward():
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLDecoderLayer
    from ignition.recipes.minimax_m3.decoder import CanonicalPreparedLayer, _Attention, _FeedForward, _Norm

    torch.manual_seed(19)
    source = torch.randn(3, 8).bfloat16()
    metadata = SimpleNamespace(live_counts=torch.tensor([3, 1], dtype=torch.int32))
    module = MiniMaxM3VLDecoderLayer.__new__(MiniMaxM3VLDecoderLayer)
    nn.Module.__init__(module)
    module.input_layernorm = _Norm(Norm(torch.randn(8).bfloat16()))
    module.post_attention_layernorm = _Norm(Norm(torch.randn(8).bfloat16()))
    aw, fw = (torch.randn(8, 8).bfloat16() for _ in range(2))
    module.self_attn = _Attention(lambda value, **kwargs: value @ aw, lambda value: value)
    module.mlp = _FeedForward(FeedForward(fw), lambda value: value)
    layer = CanonicalPreparedLayer.__new__(CanonicalPreparedLayer)
    layer.module = module
    expected = module(source.clone(), metadata=metadata, lengths=None, schedule_storage=None)
    actual = layer(source.clone(), metadata=metadata, lengths=None, schedule_storage=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_step_accepted_context_keeps_only_the_committed_prefix():
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.step5.mtp import PreparedAcceptedContext

    hidden, width, rows = 4, 4, 3
    regions = {
        "draft.pending_tokens": torch.full((rows, width), 91, dtype=torch.int64),
        "draft.pending_hidden": torch.full((rows, width, hidden), 92, dtype=torch.bfloat16),
        "draft.pending_embedding": torch.full((rows, width, hidden), 93, dtype=torch.bfloat16),
        "draft.pending_count": torch.full((rows, 1), 94, dtype=torch.int32),
    }
    metadata = Metadata.allocate(6, 2, 1, "cpu")
    metadata.token_ids.copy_(torch.arange(6))
    metadata.query_start_loc.copy_(torch.tensor([0, 3, 6], dtype=torch.int32))
    metadata.state_rows.copy_(torch.tensor([1, 2]))
    target = torch.arange(24).reshape(6, hidden).bfloat16()
    context = PreparedAcceptedContext(SimpleNamespace(draft=(1, 2, 3)), metadata,
        SimpleNamespace(reserved_region=regions.__getitem__)).functional()
    context((target, target + 32), torch.tensor([2, 0], dtype=torch.int32), metadata)
    assert regions["draft.pending_tokens"][1].tolist() == [0, 1, 0, 0]
    assert torch.equal(regions["draft.pending_hidden"][1, :2], target[:2])
    assert torch.equal(regions["draft.pending_embedding"][1, :2], (target + 32)[:2])
    assert regions["draft.pending_count"].flatten().tolist() == [94, 2, 94]
    assert (regions["draft.pending_hidden"][[0, 2]] == 92).all()
    assert context.metadata is context.columns is None


@pytest.mark.parametrize("batch", [1, 2])
def test_step_proposal_state_threads_pending_rows_and_preserves_inactive_history(batch):
    from ignition.recipes.step5.mtp import PreparedProposals

    rows, pending_width, hidden = 3, 4, 2
    regions = {
        "draft.pending_count": torch.tensor([[0], [2], [1]], dtype=torch.int32),
        "draft.pending_tokens": torch.arange(rows * pending_width).view(rows, pending_width),
        "draft.pending_hidden": torch.arange(rows * pending_width * hidden).view(rows, pending_width, hidden).bfloat16(),
        "draft.pending_embedding": torch.arange(rows * pending_width * hidden).view(rows, pending_width, hidden).bfloat16() + 32,
    }
    histories = tuple(torch.arange(rows * hidden).view(rows, hidden).bfloat16() + 64 * index
                      for index in range(3))
    proposal = PreparedProposals.__new__(PreparedProposals)
    proposal.batch, proposal.pending_width, proposal.depth, proposal.pitch = batch, pending_width, 1, 5
    proposal.pool = SimpleNamespace(reserved_region=regions.__getitem__)
    proposal.histories = histories
    proposal.embedding = lambda *, token_ids: token_ids[:, None].expand(-1, hidden).bfloat16()
    observed = []

    class Predictor:
        cache_only = False

        def __init__(self):
            self.layer = SimpleNamespace(attention=SimpleNamespace(commit=self.commit))

        def commit(self, counts, records, metadata):
            assert torch.equal(counts, torch.tensor([2, 0], dtype=torch.int32)[:batch])
            assert metadata.state_rows.tolist() == [1, 0][:batch]
            assert records is not None

        def __call__(self, source, metadata, *, embedded):
            assert metadata.requests.is_contiguous()
            observed.append((source.clone(), metadata, embedded.clone()))
            output = torch.where((metadata.positions >= 0)[:, None], source + embedded, 0)
            return output, output[metadata.sample_indices], (source, embedded)

    proposal.predictors = tuple((None, Predictor()) for _ in histories)
    states, active = torch.tensor([1, 2])[:batch], torch.tensor([True, False])[:batch]
    before = torch.tensor([10, 12])[:batch]
    state = proposal.begin(states)
    independent = proposal.begin(states)
    assert state.ids.data_ptr() != independent.ids.data_ptr()
    assert state.ids[:, :pending_width].tolist() == regions["draft.pending_tokens"][[1, 2][:batch]].tolist()
    inactive_history = tuple(value[2].clone() for value in histories)
    logits, state = proposal.step(state, 0, torch.tensor([77, 78])[:batch], before, states, active)
    source, metadata, embedded = observed[0]
    assert state.ids[0, 2] == 77
    if batch > 1:
        assert state.ids[1, 1] == 0
    assert independent.ids[0, 2] != 77
    assert metadata.positions.tolist() == [8, 9, 10, -1, -1] + [-1] * (5 * (batch - 1))
    torch.testing.assert_close(source[0], histories[0][1], rtol=0, atol=0)
    torch.testing.assert_close(source[1:3], regions["draft.pending_hidden"][1, :2], rtol=0, atol=0)
    torch.testing.assert_close(logits[0], source[2] + embedded[2], rtol=0, atol=0)
    proposal.finish(state, before, states, active)
    assert regions["draft.pending_count"].flatten().tolist() == [0, 0, 1]
    for history, untouched in zip(histories, inactive_history, strict=True):
        assert torch.equal(history[2], untouched)
    torch.testing.assert_close(histories[0][1], regions["draft.pending_hidden"][1, 1], rtol=0, atol=0)


def test_minimax_grouped_routes_pass_runtime_weights_and_ids_to_fused_reduction():
    from ignition.components.collectives import WeightedGroupSource
    from ignition.recipes.minimax_m3.decoder import PreparedDecoderLayer

    source = torch.arange(8.).view(2, 4).bfloat16()
    grouped = WeightedGroupSource(torch.arange(24.).view(2, 3, 4).bfloat16(),
        torch.ones(2, 3), torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int32), 3)
    shared = source + 4
    layer = PreparedDecoderLayer.__new__(PreparedDecoderLayer)
    layer.fused = True
    layer.attention_reduce = layer.final_reduce = None
    layer.attention = lambda source, **kwargs: source
    layer.post_norm = lambda source, residual: source
    layer.feed_forward = lambda source, **kwargs: (grouped, shared)
    seen = []

    def reduce(values, residual, *, input_addend, group_weights, group_ids):
        assert values is grouped.values and input_addend is shared
        assert group_weights is grouped.weights and group_ids is grouped.ids
        seen.append(values)
        return values.float().sum(1).bfloat16() + input_addend

    layer.output_norm = reduce
    output, update = layer(source, source.clone(), metadata=None, lengths=None, schedule_storage=None)
    assert update is None and seen
    assert torch.equal(output, grouped.values.float().sum(1).bfloat16() + shared)


def test_minimax_verification_commit_uses_returned_rows_and_accepted_counts():
    from ignition.recipes.minimax_m3.decoder import PreparedDecoder as MiniMaxDecoder

    decoder = MiniMaxDecoder.__new__(MiniMaxDecoder)
    decoder.taps = {1: 0, 2: 1}
    backing = torch.zeros(4, 1)
    history = torch.full((4, 2, 3), -19., dtype=torch.bfloat16)
    decoder.pool = SimpleNamespace(backing=backing, region=lambda name: history)
    records = torch.arange(36.).view(6, 2, 3).bfloat16()
    metadata = SimpleNamespace(query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
                               state_rows=torch.tensor([1, 2]))
    decoder.commit(torch.tensor([2, 0], dtype=torch.int32), records, metadata)
    assert torch.equal(history[1], records[1])
    assert torch.all(history[[0, 2, 3]] == -19)
    decoder.commit(torch.tensor([0, 3], dtype=torch.int32), records + 32, metadata)
    assert torch.equal(history[2], (records + 32)[5])
    assert torch.equal(history[1], records[1])


@pytest.mark.parametrize("recipe", ["step5", "minimax_m3"])
def test_vision_attention_materializes_contiguous_runtime_qkv(recipe):
    import importlib

    module = importlib.import_module(f"ignition.recipes.{recipe}.vision")
    operation = module.VisionAttention.__new__(module.VisionAttention)
    nn.Module.__init__(operation)
    operation.query_shape = (2, 1, 80)
    operation.qkv = nn.Linear(80, 240, dtype=torch.bfloat16)
    operation.output = nn.Linear(80, 80, dtype=torch.bfloat16)
    operation.bias = torch.zeros(80, dtype=torch.bfloat16)
    operation.reduce = operation.reduction = lambda source: source
    operation.inverse = torch.arange(13).float() / 13
    cu = torch.tensor([0, 2], dtype=torch.int32)
    observed = []

    def attention(query, key, value, boundaries):
        assert boundaries is cu
        assert all(tensor.is_contiguous() for tensor in (query, key, value))
        observed.append((query, key, value))
        return value.clone()

    operation.attention = attention
    source = torch.randn(2, 80).bfloat16()
    if recipe == "step5":
        output, auxiliary = operation(source, position_embeddings=(torch.ones(2, 1, 80), torch.zeros(2, 1, 80)), cu=cu)
    else:
        output, auxiliary = operation(source, positions=torch.tensor([[0, 0], [1, 1]]), cu=cu)
    assert auxiliary is None and output.shape == source.shape and observed
    expected_value = operation.qkv(source).chunk(3, -1)[2].reshape(2, 1, 80)
    torch.testing.assert_close(observed[0][2], expected_value, rtol=0, atol=0)
