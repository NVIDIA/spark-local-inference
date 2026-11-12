"""Row-shard declaration scratch must not survive prepared-owner finalization."""

import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from ignition.recipes.deepseek41.decoder import PreparedDecoder, PreparedLayer, RowShard


@pytest.mark.parametrize("owner_type,release", [
    (PreparedLayer, "release_examples"), (PreparedDecoder, "release_outputs")])
def test_prepared_owner_releases_row_shard_scratch(owner_type, release):
    shard = RowShard.allocate(1024, 2, 1, "cpu")
    examples = [weakref.ref(value) for value in (
        shard.attention_output, shard.mlp_output, shard.residual, shard.pre_mix, shard.attn_mix,
        *shard.attention_hc.bind_args().values(), *shard.ffn_hc.bind_args().values())]
    rows_index = shard.rows_index
    prepared = object.__new__(owner_type)
    prepared.row_shard = shard
    prepared.scratch = SimpleNamespace(row_shard=shard)
    if owner_type is PreparedDecoder:
        prepared.output_scratch = prepared.scratch
    prepared.concurrent_shared = None
    getattr(prepared, release)()
    del shard
    gc.collect()
    assert all(example() is None for example in examples)
    assert prepared.row_shard.rows_index is rows_index
    assert torch.equal(rows_index, torch.arange(512, 1024))
    for name in ("attention_output", "mlp_output", "residual", "pre_mix", "attn_mix",
                 "attention_hc", "ffn_hc"):
        assert getattr(prepared.row_shard, name) is None
