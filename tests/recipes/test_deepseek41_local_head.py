# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed target head construction exposes expanded local rows to its consumers."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.recipes.deepseek41.decoder import PreparedDecoder
from ignition.recipes.deepseek41.bootstrap import PreparationConfig


class CpuProjection:
    """Replace only GPU operator declaration in CPU geometry checks."""
    def __init__(self, source, weight, output, **kwargs):
        self.weight, self.output = weight, output

    def functional(self, **kwargs):
        self.output = None
        return self

    def __call__(self, source):
        value = torch.nn.functional.linear(source.float(), self.weight.float())
        return value if self.output is None else self.output.copy_(value)


@pytest.mark.parametrize("batch,physical", PreparationConfig(max_active=4,
    speculative_widths=tuple(range(1, 9)), speculative_token_capacity=32).packed_speculative_shapes)
def test_packed_head_factory_uses_disjoint_logical_local_output(monkeypatch, batch, physical):
    monkeypatch.setattr("ignition.components.vocabulary.UnquantizedProjection", CpuProjection)
    decoder = PreparedDecoder.__new__(PreparedDecoder)
    logical = batch * 8
    decoder.scratch = SimpleNamespace(hidden=torch.arange(logical * 4).reshape(logical, 4).bfloat16(),
        local_logits=torch.full((logical, 3), float("nan"), dtype=torch.float32),
        logits=torch.full((logical, 12), float("nan"), dtype=torch.float32))
    decoder.weights = SimpleNamespace(head=torch.arange(3 * 4).reshape(3, 4).bfloat16())
    indices = torch.arange(logical, dtype=torch.int64).flip(0) % physical
    decoder.output_metadata = SimpleNamespace(token_ids=torch.zeros(physical, dtype=torch.int64), sample_indices=indices)
    decoder._prepare_head(vocabulary_parallel=True, vocabulary_runtime=None,
                          coordinator=None, session=None, name="target")
    observed = decoder.head(decoder.head_hidden, row_indices=indices if physical < logical else None)
    expected = torch.nn.functional.linear(decoder.head_hidden.float(), decoder.weights.head.float())
    if physical < logical:
        expected = expected[indices]
    assert decoder.scratch.local_logits.isnan().all()
    assert observed.shape == (logical, 3) and observed.is_contiguous()
    torch.testing.assert_close(observed, expected)



@pytest.mark.parametrize("invalid", ["dtype", "rows", "columns", "overlap"])
def test_local_expansion_rejects_invalid_geometry_and_aliases(monkeypatch, invalid):
    monkeypatch.setattr("ignition.components.vocabulary.UnquantizedProjection", CpuProjection)
    source, weight = torch.zeros(2, 4), torch.zeros(3, 4)
    local = torch.empty(8, 3)
    output = torch.empty(8, 3)
    indices = torch.zeros(8, dtype=torch.int64)
    if invalid == "dtype":
        indices = indices.int()
    elif invalid == "rows":
        indices = indices[:4]
    elif invalid == "columns":
        output = torch.empty(8, 6)
    else:
        output = local
    with pytest.raises(ValueError):
        PreparedVocabularyProjection(source, weight, local[:2], output, runtime=None,
            session=None, coordinator=None, name="local", row_indices=indices, gather=False)


@pytest.mark.parametrize("gather", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_functional_head_uses_runtime_indices_and_preserves_bound_defaults(monkeypatch, gather, dynamic):
    import weakref

    monkeypatch.setattr("ignition.components.vocabulary.UnquantizedProjection", CpuProjection)
    source = torch.arange(8).reshape(2, 4).float()
    weight = torch.arange(12).reshape(3, 4).float()
    indices = torch.tensor([0, 1, 0, 1])
    reference = weakref.ref(indices)
    operation = PreparedVocabularyProjection(source, weight, torch.empty(2, 3), torch.empty(4, 3),
        runtime=None, session=None, coordinator=None, name="indices", row_indices=indices, gather=gather)
    operation.functional(session=None, dynamic_indices=dynamic)
    if dynamic:
        del indices
        assert reference() is None
    else:
        torch.testing.assert_close(operation(source), torch.nn.functional.linear(source, weight)[indices])
    compiled = torch.compile(lambda values, rows: operation(values, row_indices=rows), backend="eager", fullgraph=True)
    for rows in (torch.tensor([1, 0, 1, 0]), torch.tensor([0, 0, 1, 1])):
        torch.testing.assert_close(compiled(source, rows), torch.nn.functional.linear(source, weight)[rows])
