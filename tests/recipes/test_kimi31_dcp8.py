# SPDX-License-Identifier: Apache-2.0

"""Hybrid TP16/DCP8 planning and context geometry preserve the DCP16 default."""

from pathlib import Path
from types import SimpleNamespace

import torch.distributed as dist

import pytest
import torch

from ignition import planning
from ignition.errors import UsageError
from ignition.recipes.kimi3.bootstrap import PreparationConfig, target_tuning_identity
from ignition.recipes.kimi3.dcp import ContextShards
from ignition.preparation.collectives import CollectivePreparation
from ignition.preparation.peer_storage import PeerStorage
from ignition.recipes.kimi31.recipe import RECIPE


def _gpu(index):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index}", 96 << 30, 96 << 30, (12, 0))


def test_plan_supports_only_hybrid_dcp8_and_preserves_dcp16_default():
    gpus = tuple(_gpu(index) for index in range(16))
    default = planning.fit(RECIPE, RECIPE.checkpoints[0], gpus,
                           concurrency=4, reserve_bytes=0, speculation=None)
    dcp8 = planning.fit(RECIPE, RECIPE.checkpoints[0], gpus,
                        concurrency=4, reserve_bytes=0, speculation=None, context_parallel_size=8)
    assert default.context_parallel_size is None and default.page_tokens == 2048
    assert default.to_dict()["context_parallel_size"] == 16
    assert dcp8.world_size == 16 and dcp8.context_parallel_size == 8 and dcp8.page_tokens == 1024
    assert dcp8.to_dict()["context_parallel_size"] == 8
    assert dcp8.context == 1048576 and dcp8.tokens > 1048576
    assert dcp8.pages - default.pages == 64
    with pytest.raises(UsageError, match="hybrid context parallelism"):
        planning.fit(RECIPE, RECIPE.checkpoints[0], gpus,
                     concurrency=4, reserve_bytes=0, speculation=None, context_parallel_size=4)


def test_preparation_pages_and_tuning_identity_follow_dcp_size():
    dcp16 = PreparationConfig(context=1048576)
    dcp8 = PreparationConfig(context=1048576, context_parallel_size=8)
    assert dcp16.page_tokens == 2048 and dcp16.preparation_pages == 10
    assert dcp8.page_tokens == 1024 and dcp8.preparation_pages == 12
    config = SimpleNamespace(text={}, vision=False)
    legacy = target_tuning_identity(config, [], 16)
    assert target_tuning_identity(config, [], 16, context_parallel_size=16) == legacy
    assert target_tuning_identity(config, [], 16, context_parallel_size=8) != legacy
    with pytest.raises(ValueError, match="DCP8 or DCP16"):
        PreparationConfig(context_parallel_size=4)


def test_dcp8_reduction_capacity_covers_packed_selected_ids_not_hidden_activations():
    config = PreparationConfig(context_parallel_size=8, draft_checkpoint=Path("draft"),
        speculative_token_capacity=32, speculative_graph_budget=16)
    assert config.decode_tokens == 32
    assert config.dcp_reduction_elements * 2 == 32 * 2048 * 4 * 2 == 512 * 1024
    assert config.dcp_reduction_elements * 2 < 1024 * 7168 * 2
    assert PreparationConfig(context_parallel_size=8).dcp_reduction_elements >= 8


def test_dcp8_context_shards_use_group_local_rank():
    shards = ContextShards.on(3, 8, "cpu")
    positions = torch.arange(19, dtype=torch.int64)
    tables = torch.tensor([[5, 6]], dtype=torch.int32).expand(positions.numel(), -1)
    local = positions // 8
    expected = torch.where(positions % 8 == 3, tables[0, 0] * 128 + local % 128, -1)
    assert torch.equal(shards.token_slots(positions, tables), expected)
    lengths = positions + 1
    assert torch.equal(shards.local_tokens(lengths), lengths // 8 + (lengths % 8 > 3))


def test_peer_regions_use_group_local_addresses(monkeypatch):
    peer = object.__new__(PeerStorage)
    peer.storage = SimpleNamespace(rank=8, world=16)
    peer.dynamic, peer.owned = {}, {}
    peer.cudaIpcGetMemHandleBytes = lambda address: b"region"
    opened = []
    peer.cudaIpcOpenMemHandleBytes = lambda payload: opened.append(payload) or 999
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: list(range(8, 16)))
    def gather(output, payloads, *, group):
        for index in range(8):
            output[index] = [b"region"]
    monkeypatch.setattr(dist, "all_gather_object", gather)
    region = SimpleNamespace(address=123, extent=456)
    addresses = peer.share_regions(object(), [region])
    assert addresses == [[123, 999, 999, 999, 999, 999, 999, 999]]
    assert opened == [b"region"] * 7
    assert peer.dynamic[123] is region and peer.owned[123] == 456


def test_collective_tuning_authorizes_only_local_dcp_group(monkeypatch):
    coordinator = object.__new__(CollectivePreparation)
    coordinator.group = None
    coordinator.ranks = tuple(range(16))
    monkeypatch.setattr(dist, "get_rank", lambda: 8)

    class Job:
        def __init__(self):
            self.calls = []
        def advance(self, **authorization):
            self.calls.append(authorization)
            done = len(self.calls) > 1
            return SimpleNamespace(done=done, ready_collectives=(), ready_cache=None,
                                   ready_tuning=(), ready_collective_tuning=None)
        def result(self):
            return "complete"
        def close(self):
            pass

    job = Job()
    session = SimpleNamespace(begin=lambda *args, **kwargs: job)

    def gather(states, local, *, group):
        if len(job.calls) == 1:
            for rank in range(16):
                group_ranks = tuple(range(rank // 8 * 8, rank // 8 * 8 + 8))
                states[rank] = (False, (), None, None, (), SimpleNamespace(ranks=group_ranks))
        else:
            for rank in range(16):
                states[rank] = (True, (), None, None, (), None)

    monkeypatch.setattr(dist, "all_gather_object", gather)
    assert coordinator.prepare(session, ()) == "complete"
    contributions = job.calls[1]["collective_tuning"]
    assert len(contributions) == 8
    assert {item.ranks for item in contributions} == {tuple(range(8, 16))}


def test_selection_agreement_is_scoped_to_operation_group(monkeypatch):
    coordinator = object.__new__(CollectivePreparation)
    coordinator.group = None
    coordinator.ranks = tuple(range(16))
    coordinator.operation_ranks = {"dcp.op": tuple(range(8, 16))}

    def gather(states, local, *, group):
        for rank in range(16):
            states[rank] = {
                "dcp.op": ("comm", '"group-a"' if rank < 8 else '"group-b"'),
                "global.op": ("comm", '"root"' if rank == 0 else '"other"'),
            }

    monkeypatch.setattr(dist, "all_gather_object", gather)
    coordinator.agree_selections([{"name": "dcp.op", "component": "comm", "config": "group-b"}])
    with pytest.raises(RuntimeError, match="1 kernel selections differ"):
        coordinator.agree_selections([{"name": "global.op", "component": "comm", "config": "root"}])
