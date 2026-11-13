# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen DFlash2 draft embedding storage bound by every tensor-parallel rank's compiled forward."""

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.aot import _signature
from ignition.recipes.qwen38.dflash import draft_embedding


def _table():
    generator = torch.Generator().manual_seed(0)
    return torch.randn(16, 2560, generator=generator).to(torch.bfloat16)


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_every_rank_binds_one_draft_embedding_signature(world_size):
    table, columns = _table(), 2560 // world_size
    signatures = set()
    for rank in range(world_size):
        embedding = draft_embedding(table, TensorParallelShard(rank, world_size))
        assert embedding.shape == (16, columns)
        assert embedding.storage_offset() == 0
        assert torch.equal(embedding, table[:, rank * columns:(rank + 1) * columns])
        signatures.add(_signature((embedding,)))
    assert len(signatures) == 1


def test_single_rank_uses_the_target_table():
    table = _table()
    assert draft_embedding(table, TensorParallelShard(0, 1)).data_ptr() == table.data_ptr()
