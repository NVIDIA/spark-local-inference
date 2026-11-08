# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4 TP2/TP4 host placement and channels preserve the local preparation contract."""

from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.recipes.deepseek4 import bootstrap, recipe


def test_two_hosts_may_each_use_their_own_gpu_zero():
    config = bootstrap.PreparationConfig(devices=(0,) * 2, hosts=("maxwell", "hertz"))
    assert config.devices == (0,) * 2
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 2)
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 2, hosts=("maxwell",) * 2)
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 2, hosts=("maxwell", "hertz", "faraday"))


def test_ring_and_chain_need_hosts_in_rank_order():
    with pytest.raises(ValueError, match="--roce-ring"):
        bootstrap.PreparationConfig(devices=(0, 1), roce_ring=True)
    with pytest.raises(ValueError, match="exclude each other"):
        bootstrap.PreparationConfig(devices=(0,) * 2, hosts=("a", "b"), roce_ring=True, roce_chain=True)
    assert bootstrap.PreparationConfig(devices=(0,) * 2, hosts=("a", "b"), roce_chain=True).roce_chain


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("remote", [False, True])
def test_recipe_dispatches_the_complete_plan_to_the_rank_group(monkeypatch, remote, world):
    from ignition.preparation import rank_group

    gpus = tuple(planning.Gpu(i, "GB10", f"GPU-{i}", 128 << 30, 120 << 30, (12, 1)) for i in range(world))
    plan = planning.fit(recipe.RECIPE, recipe.RECIPE.checkpoints[0], gpus, concurrency=16,
                        reserve_bytes=4 << 30, speculation=recipe.RECIPE.speculation)
    ranks = SimpleNamespace(hosts=("maxwell", "hertz", "faraday", "ampere")[:world]) if remote else None
    calls, result, storage = [], object(), object()
    monkeypatch.setattr(rank_group, "prepare_group", lambda *args, **kwargs: calls.append((args, kwargs)) or result)
    assert recipe.prepare("checkpoint", plan, compile_forwards=True, kernel_cache=None, progress=None,
                          core_storage=storage, ranks=ranks) is result
    (snapshot, config, forward), kwargs = calls[0]
    assert snapshot == "checkpoint" and forward is bootstrap.prepare_rank
    assert config.devices == ((0,) * world if remote else tuple(range(world)))
    assert config.hosts == (ranks.hosts if remote else None)
    assert not config.roce_ring and not config.roce_chain
    assert config.max_active == 16 and config.dspark and config.draft_tokens == 5
    assert kwargs["ranks"] is ranks and kwargs["checkpoint"] is plan.checkpoint
    assert kwargs["core_storage"] is storage


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("dspark", [False, True])
def test_remote_channels_cover_gathers_and_broadcasts_without_pcie(monkeypatch, dspark, world):
    from ignition.components.collectives import packed_broadcast_layout
    from ignition.components.draft_sampling import PreparedDraftBlockBroadcast, PreparedDraftDistribution
    from ignition.components.speculation import SpeculativeAcceptance

    hosts = ("a", "b", "c", "d")[:world]
    config = bootstrap.PreparationConfig(devices=(0,) * world, hosts=hosts, max_active=16, dspark=dspark,
                                         speculative_widths=(1, 2, 3, 4, 5, 6), speculative_token_capacity=128)
    calls, runtime, storage = [], object(), object()
    monkeypatch.setattr(bootstrap, "create_roce_runtime",
                        lambda group, device, **kwargs: calls.append(kwargs) or runtime)
    for name in ("create_all_reduce_runtime", "create_all_gather_runtime", "create_dma_all_reduce_runtime"):
        monkeypatch.setattr(bootstrap, name, lambda *args, **kwargs: pytest.fail("host ranks cannot use CUDA IPC"))
    channels = bootstrap._collective_runtimes(config, None, None,
        coordinator=SimpleNamespace(spans_hosts=True), core_storage=storage)
    assert channels == (runtime, runtime, runtime, runtime if dspark else None)
    (call,) = calls
    assert call["core_storage"] is storage and call["hidden_size"] == 4096
    slots = config.request_capacity
    assert call["pipe_tokens"] == max(config.prefill_chunk_size, config.decode_tokens, slots * config.draft_tokens)
    assert call["max_tokens"] >= config.decode_tokens
    # BF16 vocabulary partitions for every sampled row.
    vocabulary_rows = max(config.decode_samples, slots * config.draft_tokens)
    assert call["gather_bytes"] >= vocabulary_rows * (129280 // world) * 2
    assert call["gather_bytes"] % 16 == 0
    if dspark:
        # FP32 Markov logits travel as two BF16 columns each; sampling broadcasts share the channel.
        assert call["gather_bytes"] >= slots * (129280 // world) * 4
        batch = max(config.batch_capacities)
        for layout in (PreparedDraftDistribution.broadcast_layout(batch),
                       SpeculativeAcceptance.broadcast_layout(batch, 6),
                       PreparedDraftBlockBroadcast.broadcast_layout(batch, 5)):
            assert call["gather_bytes"] >= packed_broadcast_layout(layout.values())[1]
    else:
        assert call["gather_bytes"] == vocabulary_rows * (129280 // world) * 2


def test_local_channels_are_unchanged(monkeypatch):
    config = bootstrap.PreparationConfig(devices=(0, 1), max_active=16, collective_tuning="runtime")
    gathers, reductions = [], []
    monkeypatch.setattr(bootstrap, "create_all_gather_runtime",
                        lambda group, device, **kwargs: gathers.append(kwargs) or object())
    monkeypatch.setattr(bootstrap, "create_all_reduce_runtime",
                        lambda group, device, **kwargs: reductions.append(kwargs) or object())
    monkeypatch.setattr(bootstrap, "create_dma_all_reduce_runtime", lambda *args, **kwargs: object())
    monkeypatch.setattr(bootstrap, "create_roce_runtime",
                        lambda *args, **kwargs: pytest.fail("local ranks share CUDA IPC"))
    ipc = object()
    collective, prefill, vocabulary, markov = bootstrap._collective_runtimes(config, None, None,
        coordinator=SimpleNamespace(spans_hosts=False), core_storage=SimpleNamespace(peer_storage=ipc))
    assert len({id(collective), id(prefill), id(vocabulary), id(markov)}) == 4
    assert reductions[0]["collective_tuning"] == "runtime" and reductions[0]["ipc"] is ipc
    assert gathers[0]["columns"] == 129280
    assert gathers[0]["max_rows"] == max(config.decode_samples, config.request_capacity * config.draft_tokens)
    assert gathers[1] == {"max_rows": config.request_capacity, "columns": 2 * 129280, "ipc": ipc}
