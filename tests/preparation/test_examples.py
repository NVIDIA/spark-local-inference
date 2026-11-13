# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation capacities reuse storage without aliasing simultaneous operands."""

import torch

from ignition.preparation.examples import declaration_tensor


def test_descending_capacities_share_one_contiguous_allocation():
    examples = {}
    shapes = [(16, 4), (8, 4), (2, 4), (1, 4)]
    values = [declaration_tensor(examples, "input", shape, torch.bfloat16, "cpu") for shape in shapes]
    assert {value.untyped_storage().data_ptr() for value in values} == {values[0].data_ptr()}
    assert all(value.shape == shape and value.is_contiguous() for value, shape in zip(values, shapes))
    assert declaration_tensor(examples, "input", shapes[1], torch.bfloat16, "cpu") is values[1]


def test_distinct_roles_and_precisions_have_independent_storage():
    examples = {}
    source = declaration_tensor(examples, "input", (8, 4), torch.bfloat16, "cpu")
    output = declaration_tensor(examples, "output", (2, 4), torch.bfloat16, "cpu")
    scores = declaration_tensor(examples, "input", (2, 4), torch.float32, "cpu")
    source.fill_(3)
    output.fill_(7)
    scores.fill_(11)
    assert torch.all(source == 3)
    assert torch.all(output == 7)
    assert torch.all(scores == 11)


def test_growth_preserves_existing_declaration_operands():
    examples = {}
    small = declaration_tensor(examples, "input", (2, 4), torch.bfloat16, "cpu")
    small.fill_(3)
    large = declaration_tensor(examples, "input", (8, 4), torch.bfloat16, "cpu")
    large.fill_(7)
    assert torch.all(small == 3)
    assert declaration_tensor(examples, "input", (2, 4), torch.bfloat16, "cpu") is small
    assert torch.all(large == 7)
