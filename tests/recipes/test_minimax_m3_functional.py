# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""MiniMax M3 prepared forwards preserve explicit runtime operands."""

from types import SimpleNamespace

import torch
from torch import nn


class Norm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, source, update=None):
        if update is not None:
            source.add_(update)
        return torch.nn.functional.rms_norm(
            source.float(), (source.shape[-1],), self.weight.float(), self.eps
        ).bfloat16()


class FeedForward(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, source, *, metadata=None):
        return (source @ self.weight).bfloat16()


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


def test_minimax_vision_attention_materializes_contiguous_runtime_qkv():
    from ignition.recipes.minimax_m3 import vision as module

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
    output, auxiliary = operation(source, positions=torch.tensor([[0, 0], [1, 1]]), cu=cu)
    assert auxiliary is None and output.shape == source.shape and observed
    expected_value = operation.qkv(source).chunk(3, -1)[2].reshape(2, 1, 80)
    torch.testing.assert_close(observed[0][2], expected_value, rtol=0, atol=0)
