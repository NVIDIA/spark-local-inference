# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi MLP declarations share equivalent planes while live operands remain disjoint."""

import pytest
import torch

from ignition.recipes.kimi3 import decoder


class Projection:
    def __init__(self, source, weight, output, **kwargs):
        self.source, self.weight, self.output = source, weight, output

    def functional(self, **kwargs):
        return self


@pytest.mark.parametrize("shared", [False, True])
def test_mlp_declarations_share_only_matching_operand_planes(monkeypatch, shared):
    monkeypatch.setattr(decoder, "prepare_projection", Projection)
    source = torch.zeros((2, 4), dtype=torch.bfloat16)
    weights = dict(gate_up=torch.zeros((8, 4), dtype=source.dtype),
                   down=torch.zeros((4, 4), dtype=source.dtype))
    examples = {} if shared else None

    def prepare(value):
        return decoder.PreparedMLP(weights, value, session=None, name="mlp", examples=examples)

    first, second = prepare(source), prepare(source)
    for left, right in ((first.gate_up.output, second.gate_up.output),
                        (first.down.source, second.down.source),
                        (first.down.output, second.down.output)):
        assert (left is right) is shared
    first.gate_up.output.fill_(7)
    assert first.down.source.eq(0).all()
    assert first.down.output.eq(0).all()
    first.down.source.fill_(11)
    assert first.down.output.eq(0).all()
    other_shape = prepare(torch.zeros((3, 4), dtype=source.dtype))
    assert other_shape.down.output.shape == (3, 4)
    assert other_shape.down.output.eq(0).all()
