# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A row-sharded prefill keeps one capture layout and falls back to full rows when it cannot."""

from types import SimpleNamespace

import pytest
import torch


def _surfaces(monkeypatch, *, abi=11, pipe_bytes=1 << 26):
    import b12x.comm.roce._proxy as proxy
    from ignition.components import collectives

    monkeypatch.setattr(proxy, "ABI_VERSION", abi)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: True)
    return SimpleNamespace(world_size=2, pipe_bytes=pipe_bytes, _group=object(),
        should_reduce_scatter=lambda inp: True, should_gather_shards=lambda inp: True)


def test_row_shard_takes_contiguous_rank_rows_and_gates_on_world_and_call_rows():
    from ignition.recipes.deepseek41.decoder import MHC_CALL_ROWS, RowShard

    shard = RowShard.allocate(1024, 2, 1, "cpu")
    assert (shard.world, shard.rows, MHC_CALL_ROWS) == (2, 512, 512)
    assert shard.rows_index.tolist() == list(range(512, 1024))
    assert shard.residual.shape == (512, 4, 5120) and shard.residual.dtype is torch.bfloat16
    assert shard.pre_mix.shape == (512, 4) and shard.pre_mix.dtype is torch.float32
    assert shard.attention_hc.residual.shape == shard.ffn_hc.residual.shape == (512, 4, 5120)
    assert shard.attention_hc.normalized.shape == (512, 5120)
    assert RowShard.allocate(1024, 1, 0, "cpu") is None
    assert RowShard.allocate(1025, 2, 0, "cpu") is None
    # One rank's rows must hold whole mHC calls so its call shape matches the full-row pin.
    assert RowShard.allocate(1024, 4, 0, "cpu") is None


def test_row_shard_graph_shrinks_capacity_and_rejects_rows_that_do_not_divide():
    from ignition.recipes.deepseek41.decoder import RowShard

    shard = RowShard.allocate(2048, 2, 0, "cpu")
    smaller = shard.graph(1024)
    assert smaller.rows == 512 and smaller.rows_index.tolist() == list(range(512))
    assert smaller.residual.shape == (512, 4, 5120) and smaller.attention_hc.normalized.shape == (512, 5120)
    assert shard.graph(2048).rows == 1024
    # Every rank's block of a narrower graph starts at rank * rows of that graph, not of the
    # allocated layout, and stays inside the graph's rows.
    for rank in range(4):
        wide = RowShard.allocate(8192, 4, rank, "cpu")
        for tokens in (2048, 4096, 6144, 8192):
            narrow = wide.graph(tokens)
            rows = tokens // 4
            assert narrow.rows_index.tolist() == list(range(rank * rows, (rank + 1) * rows))
        assert wide.rows_index.tolist() == list(range(rank * 2048, (rank + 1) * 2048))
    assert shard.graph(1023) is None
    assert shard.graph(1536) is None
    assert shard.graph(1024 + 2) is None


def test_decoder_scratch_carries_a_row_shard_only_when_the_bootstrap_asks():
    from ignition.recipes.deepseek41.decoder import Scratch

    plain = Scratch.allocate(1024, 1, 2, "cpu", candidate_tokens=16)
    assert plain.row_shard is None
    asked = Scratch.allocate(1024, 1, 2, "cpu", candidate_tokens=16,
                             row_shard=SimpleNamespace(rank=1))
    assert asked.row_shard.rows == 512 and asked.row_shard.rows_index[0].item() == 512
    assert asked.rows(1024, 1).row_shard.rows == 512
    # A shard keeps whole 512-row mHC calls, so a narrower graph has no shard.
    assert asked.rows(512, 1).row_shard is None


@pytest.mark.parametrize("abi,pipe_bytes,ready", [
    (11, 1 << 26, True), (10, 1 << 26, False), (11, 1 << 22, False)])
def test_row_shard_ready_requires_the_abi_and_capacity_for_every_surface(monkeypatch, abi, pipe_bytes, ready):
    from ignition.recipes.deepseek41.decoder import RowShard, row_shard_ready

    runtime = _surfaces(monkeypatch, abi=abi, pipe_bytes=pipe_bytes)
    rows = 512
    scratch = SimpleNamespace(
        attention=SimpleNamespace(output=torch.zeros(1024, 5120, dtype=torch.bfloat16)),
        attention_hc=SimpleNamespace(normalized=torch.zeros(1024, 5120, dtype=torch.bfloat16)),
        ffn_hc=SimpleNamespace(normalized=torch.zeros(1024, 5120, dtype=torch.bfloat16)),
        mlp_partial=torch.zeros(1024, 5120, dtype=torch.bfloat16),
        pre_mix=torch.zeros(1024, 4, dtype=torch.float32),
        residual=torch.zeros(1024, 4, 5120, dtype=torch.bfloat16))
    shard = SimpleNamespace(
        attention_output=torch.zeros(rows, 5120, dtype=torch.bfloat16),
        mlp_output=torch.zeros(rows, 5120, dtype=torch.bfloat16),
        attention_hc=SimpleNamespace(normalized=torch.zeros(rows, 5120, dtype=torch.bfloat16)),
        ffn_hc=SimpleNamespace(normalized=torch.zeros(rows, 5120, dtype=torch.bfloat16)),
        pre_mix=torch.zeros(rows, 4, dtype=torch.float32),
        residual=torch.zeros(rows, 4, 5120, dtype=torch.bfloat16))
    assert shard.pre_mix.view(torch.bfloat16).shape == (rows, 8)
    assert row_shard_ready(scratch, shard, runtime) is ready


def test_every_row_count_a_graph_can_use_is_one_segment_of_the_table():
    """One allocation holds every segment, so no narrower graph needs storage of its own.

    The table is int64, and each graph's row numbers are a view at that row count's segment
    offset: 512 rows at 0, then 1,024 at 512, 1,536 at 1,536 and 2,048 at 3,072.
    """
    from ignition.recipes.deepseek41.decoder import RowShard

    captured = RowShard.allocate(8192, 4, 1, "cpu")
    assert captured.rows_table.dtype is torch.int64
    assert captured.rows_index.data_ptr() == captured.rows_table.data_ptr() + 3072 * 8
    assert captured.rows_index.tolist() == list(range(2048, 4096))
    offsets = {8192: 3072, 6144: 1536, 4096: 512, 2048: 0}
    for tokens, first in offsets.items():
        graph = captured.graph(tokens)
        assert graph.rows_table is captured.rows_table
        assert graph.rows_index.storage_offset() == first
        rows = tokens // 4
        assert graph.rows_index.tolist() == list(range(rows, 2 * rows))
        assert graph.rows_index.data_ptr() == captured.rows_table.data_ptr() + first * 8


def test_the_row_table_is_written_once_into_persistent_storage(monkeypatch):
    """Prefill scratch exports as zeros, so the table takes the persistent arena instead."""
    from ignition.recipes.deepseek41 import decoder

    calls = []

    def persistent(shape, dtype, device, **kwargs):
        calls.append((tuple(shape), dtype, device, kwargs))
        return torch.zeros(shape, dtype=dtype)

    monkeypatch.setattr(decoder, "persistent_zeros", persistent)
    shard = decoder.RowShard.allocate(4096, 4, 1, "cpu")
    assert calls == [((1536,), torch.int64, "cpu", {})]
    assert shard.rows_table.tolist() == [*range(512, 1024), *range(1024, 2048)]
    assert shard.residual.dtype is torch.bfloat16


def test_every_rank_binds_row_shard_tables_at_one_storage_offset():
    """One compiled forward binds every rank, and binding compares storage offsets.

    Each graph's row numbers therefore sit at the same offset on every rank; only their contents
    differ.
    """
    from ignition.recipes.deepseek41.decoder import RowShard

    shards = [RowShard.allocate(8192, 4, rank, "cpu") for rank in range(4)]
    for tokens in (8192, 6144, 4096, 2048):
        graphs = [shard.graph(tokens) for shard in shards]
        rows = tokens // 4
        assert len({graph.rows_index.storage_offset() for graph in graphs}) == 1
        for rank, graph in enumerate(graphs):
            assert graph.rows_index.tolist() == list(range(rank * rows, (rank + 1) * rows))
