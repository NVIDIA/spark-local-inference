# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Row-sharded prefill attention replicates every head on every rank."""

def test_attention_weight_extents_partition_per_rank_and_replicate_every_group():
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.deepseek41.weights import attention_weight_extents

    text = {"head_dim": 512, "num_attention_heads": 64, "o_groups": 8, "o_lora_rank": 1024}
    extents = [attention_weight_extents(text, TensorParallelShard(rank, 4)) for rank in range(4)]
    # The tensor-parallel partition keeps 16 heads and 2 output groups per rank: disjoint and whole.
    assert [extent.heads for extent in extents] == [
        (0, 8192), (8192, 16384), (16384, 24576), (24576, 32768)]
    assert [extent.groups for extent in extents] == [
        (0, 2048), (2048, 4096), (4096, 6144), (6144, 8192)]
    assert [extent.sink for extent in extents] == [(0, 16), (16, 32), (32, 48), (48, 64)]
    # A row-sharded rank replicates all 64 heads, all 8 output groups, and all 64 sinks.
    assert all(extent.all_heads == (0, 32768) for extent in extents)
    assert all(extent.all_groups == (0, 8192) for extent in extents)
    assert all(extent.all_sink == (0, 64) for extent in extents)


def test_replicated_attention_weights_cost_the_documented_bytes_per_rank():
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.deepseek41.weights import attention_weight_extents

    text = {"head_dim": 512, "num_attention_heads": 64, "o_groups": 8, "o_lora_rank": 1024}
    hidden, q_lora, layers = 5120, 1280, 40
    extent = attention_weight_extents(text, TensorParallelShard(0, 4))
    group_width = text["num_attention_heads"] * text["head_dim"] // text["o_groups"]
    extra_heads = extent.all_heads[1] - extent.all_heads[0] - extent.heads[1] + extent.heads[0]
    extra_groups = extent.all_groups[1] - extent.all_groups[0] - extent.groups[1] + extent.groups[0]
    # wq_b gains rows, wo_a gains group rows at the fixed 4096-wide group input, and wo_b gains
    # its group columns. MXFP8 holds one value byte per weight and one scale byte per 32 values.
    per_layer = extra_heads * q_lora + extra_groups * group_width + extra_groups * hidden
    delta = per_layer * layers * (1 + 1 / 32)
    assert 3.5e9 < delta < 3.8e9, delta
