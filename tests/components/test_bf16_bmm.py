# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Batched projection declarations admit aligned strided stacks and size tiles by row count."""

import pytest
import torch

from ignition.components import bf16_bmm


def glm_stacks(rows, heads, major):
    width_in, width_out = (256, 512) if major == "n" else (512, 256)
    source = torch.zeros((rows, heads, width_in), dtype=torch.bfloat16).transpose(0, 1)
    output = torch.zeros((rows, heads, width_out), dtype=torch.bfloat16).transpose(0, 1)
    weight = torch.zeros((heads, 256, 512), dtype=torch.bfloat16)
    return source, weight if major == "n" else weight.transpose(1, 2), output


@pytest.mark.parametrize("rows,row_tiles", [(1, {16}), (16, {16, 32}), (64, {16, 32, 64, 128}),
                                            (256, {32, 64, 128}), (8192, {32, 64, 128})])
def test_tactics_start_with_cublas_and_size_row_tiles_by_capacity(rows, row_tiles):
    query = bf16_bmm.query_from_call(*glm_stacks(rows, 32, "n"))
    tactics = bf16_bmm.CONTRACT.parameters(query, None)["tactic"]
    assert tactics[0] == "cublas"
    assert {tactic[1] for tactic in tactics[1:]} == row_tiles
    assert all(512 % tactic[2] == 0 and 256 % tactic[3] == 0 for tactic in tactics[1:])


def test_expansion_stacks_exclude_column_tiles_wider_than_the_output():
    query = bf16_bmm.query_from_call(*glm_stacks(4096, 16, "k"))
    assert all(tactic[2] <= 256 for tactic in bf16_bmm.CONTRACT.parameters(query, None)["tactic"][1:])


def test_declarations_reject_strided_rows_and_unaligned_stacks():
    source, weight, output = glm_stacks(64, 16, "n")
    with pytest.raises(ValueError, match="contiguous"):
        bf16_bmm.query_from_call(source[:, :, ::2], weight[:, ::2], output)
    with pytest.raises(ValueError, match="aligned"):
        bf16_bmm.query_from_call(source, weight, torch.zeros((64, 16, 513), dtype=torch.bfloat16)[:, :, 1:]
                                 .transpose(0, 1))
    with pytest.raises(ValueError, match="matching"):
        bf16_bmm.query_from_call(source, weight[:, :128], output)
