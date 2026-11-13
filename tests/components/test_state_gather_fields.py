# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed state-field gathers preserve bytes, row selection, and destination bounds."""

import pytest
import torch

from ignition.components.state import gather_rows_three


@pytest.mark.parametrize("rows", [1, 3, 4])
@pytest.mark.parametrize("width", [512, 513, 1025])
def test_three_fields_preserve_mutable_rows_and_typed_bytes(rows, width):
    backing = torch.arange(7 * (2 * width + 16), dtype=torch.int64).view(7, -1)
    first = backing[:, :width].to(torch.int32).view(torch.float32)
    second = backing[:, width:2 * width].to(torch.int32).view(torch.float32)
    # Retain strided source rows and int64 tags outside the exact FP32 range.
    first = torch.stack((first, first), dim=1)[:, 0]
    second = torch.stack((second, second), dim=1)[:, 0]
    third = backing[:, -1:].clone().mul_(2**40)
    sources = first, second, third
    owners = [torch.full((rows * value.shape[1] + 16,), 91, dtype=value.dtype)
              for value in sources]
    output = tuple(owner[:-16].view(rows, value.shape[1])
                   for owner, value in zip(owners, sources, strict=True))
    indices = torch.empty(rows, dtype=torch.int64)
    for step in range(4):
        indices.copy_((torch.arange(rows) + step * 2) % 7)
        gather_rows_three(sources, indices, output)
        for value, actual, owner in zip(sources, output, owners, strict=True):
            expected = value[indices].contiguous()
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert owner[-16:].eq(91).all()
