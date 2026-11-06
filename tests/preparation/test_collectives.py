# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collective preparation cannot launch ahead of a peer or mismatched declaration."""

import multiprocessing
from types import SimpleNamespace

import pytest

from ignition.preparation.collectives import CollectivePreparation, dist


def coordinator(monkeypatch):
    group = object()
    monkeypatch.setattr(dist, "get_backend", lambda actual: "gloo")
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda actual: [4, 7])
    monkeypatch.setattr(dist, "all_gather_object",
                        lambda states, value, *, group: states.__setitem__(slice(None), [value] * len(states)))
    return CollectivePreparation(group)


def progress(key="reduce", ranks=(4, 7)):
    return SimpleNamespace(ready_cache=None, ready_tuning=(), ready_collective_tuning=None,
        ready_collectives=(SimpleNamespace(key=key, ranks=ranks),))


@pytest.mark.parametrize("batches,offsets,payload_bytes,transport_rows", [
    ((1, 2, 4, 8), (0, 4096, 6144, 6208), 6216, 13),
    ((1, 2, 4, 8, 16, 32), (0, 16384, 24576, 24832), 24864, 49),
])
def test_result_broadcast_channel_uses_captured_payloads(
        monkeypatch, batches, offsets, payload_bytes, transport_rows):
    import torch

    from ignition.components.draft_sampling import PreparedDraftDistribution
    from ignition.components.collectives import packed_broadcast_layout
    from ignition.components.speculation import SpeculativeAcceptance
    from ignition.preparation import collectives

    catalog = SimpleNamespace(batch_capacities=batches, request_capacity=64)
    batch = max(catalog.batch_capacities)
    assert batch < catalog.request_capacity
    proposals = PreparedDraftDistribution.broadcast_layout(batch)
    acceptance = SpeculativeAcceptance.broadcast_layout(batch, 6)
    actual = SpeculativeAcceptance.allocate(batch, 6, "meta")
    assert acceptance == {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in vars(actual).items()}
    assert packed_broadcast_layout(proposals.values()) == (offsets, payload_bytes)
    calls = []
    group, device, ipc, result = object(), object(), object(), object()
    monkeypatch.setattr(dist, "get_world_size", lambda actual: 4)
    monkeypatch.setattr(collectives, "create_all_gather_runtime",
        lambda actual_group, actual_device, **kwargs: calls.append((actual_group, actual_device, kwargs)) or result)
    assert collectives.create_result_broadcast_runtime(group, device, layouts=(proposals, acceptance), ipc=ipc) is result
    assert calls == [(group, device, dict(max_rows=transport_rows, columns=1024, ipc=ipc))]
    transport_bytes = transport_rows * 512
    for rows in catalog.batch_capacities:
        assert packed_broadcast_layout(PreparedDraftDistribution.broadcast_layout(rows).values())[1] <= transport_bytes
        for width in range(1, 7):
            fields = SpeculativeAcceptance.broadcast_layout(rows, width)
            assert packed_broadcast_layout(fields.values())[1] <= transport_bytes
    assert proposals["valid"] == ((batch, 1), torch.bool)


@pytest.mark.parametrize("layouts", [(), ({},)])
def test_result_broadcast_channel_rejects_empty_payloads(layouts):
    from ignition.preparation.collectives import create_result_broadcast_runtime

    with pytest.raises(ValueError, match="nonempty payload layouts"):
        create_result_broadcast_runtime(object(), object(), layouts=layouts)


def test_waits_for_compiling_peer_before_authorizing_kernel(monkeypatch):
    owner = coordinator(monkeypatch)
    peers = [(), (("reduce", (4, 7)),)]

    def gather(states, pending, *, group):
        assert group is owner.group
        states[:] = [pending, (peers.pop(0), None)]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    assert owner(progress()) is None
    assert owner(progress()) == "reduce"


def test_mismatched_rank_operations_fail_before_kernel_entry(monkeypatch):
    owner = coordinator(monkeypatch)

    def gather(states, pending, *, group):
        states[:] = [pending, ((("another-reduction", (4, 7)),), None)]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="declarations differ across ranks"):
        owner(progress())


def test_collective_timing_waits_for_a_compiling_peer(monkeypatch):
    owner = coordinator(monkeypatch)
    contribution = SimpleNamespace(ranks=(4, 7), step="samples/1/0/0")
    state = progress()
    state.ready_collectives = ()
    state.ready_collective_tuning = contribution
    peers = [((), None), ((), contribution)]
    monkeypatch.setattr(dist, "all_gather_object",
        lambda states, local, **kwargs: states.__setitem__(slice(None), [local, peers.pop(0)]))
    assert owner(state) is None
    assert owner(state) == (contribution, contribution)


def test_collective_race_contributions_are_exchanged_without_sharding(monkeypatch):
    owner = coordinator(monkeypatch)
    mine = SimpleNamespace(ranks=(4, 7), step="samples/1/0/0", latency=2.)
    theirs = SimpleNamespace(ranks=(4, 7), step="samples/1/0/0", latency=3.)
    blank = dict(done=False, ready_collectives=(), ready_cache=None, ready_tuning=(), ready_collective_tuning=None)
    states = [SimpleNamespace(**{**blank, "ready_collective_tuning": mine}),
              SimpleNamespace(**{**blank, "done": True})]
    peers = [(False, (), None, None, (), theirs), (True, (), None, None, (), None)]
    advances = []
    job = SimpleNamespace(advance=lambda **kwargs: advances.append(kwargs) or states.pop(0),
                          result=lambda: "prepared")
    monkeypatch.setattr(dist, "all_gather_object",
        lambda states, local, **kwargs: states.__setitem__(slice(None), [local, peers.pop(0)]))
    assert owner.prepare(SimpleNamespace(begin=lambda *args, **kwargs: job), ()) == "prepared"
    assert advances == [{"collective_key": None}, {"collective_tuning": (mine, theirs)}]


def test_entry_barrier_rejects_incomplete_group(monkeypatch):
    owner = coordinator(monkeypatch)
    with pytest.raises(ValueError, match="does not span its control group"):
        owner.barrier("reduce", (4,))


def test_entry_barrier_preserves_timeout_and_waits_for_every_rank(monkeypatch):
    owner = coordinator(monkeypatch)
    calls = []
    monkeypatch.setattr(dist, "monitored_barrier", lambda **kwargs: calls.append(kwargs))
    owner.barrier("reduce", (4, 7))
    assert calls == [{"group": owner.group, "timeout": owner.timeout, "wait_all_ranks": True}]


def test_finished_rank_keeps_participating_until_peers_finish(monkeypatch):
    owner = coordinator(monkeypatch)
    result = object()
    advances, exchanges = [], []
    done = SimpleNamespace(done=True, ready_cache=None, ready_tuning=(), ready_collective_tuning=None, ready_collectives=())
    job = SimpleNamespace(advance=lambda **kwargs: advances.append(kwargs) or done, result=lambda: result)
    session = SimpleNamespace(begin=lambda requests, **kwargs: job)
    peers = [(False, (), None, None, (), None), (False, (), None, None, (), None), (True, (), None, None, (), None)]

    def gather(states, local, *, group):
        exchanges.append(local)
        states[:] = [local, peers.pop(0)]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    assert owner.prepare(session, ()) is result
    assert len(advances) == 1 and len(exchanges) == 3


def test_mixed_job_authorizes_collective_only_after_every_rank_is_ready(monkeypatch):
    owner = coordinator(monkeypatch)
    ready = progress()
    ready.done = False
    done = SimpleNamespace(done=True, ready_cache=None, ready_tuning=(), ready_collective_tuning=None, ready_collectives=())
    local_states = [ready, ready, done]
    peers = [(False, (), None, None, (), None), (False, (("reduce", (4, 7)),), None, None, (), None),
             (False, (), None, None, (), None), (True, (), None, None, (), None)]
    keys = []
    job = SimpleNamespace(advance=lambda **kwargs: keys.append(kwargs["collective_key"]) or local_states.pop(0),
                          result=lambda: "prepared")
    monkeypatch.setattr(dist, "all_gather_object", lambda states, local, **kwargs: states.__setitem__(slice(None), [local, peers.pop(0)]))
    assert owner.prepare(SimpleNamespace(begin=lambda *args, **kwargs: job), ()) == "prepared"
    assert keys == [None, None, "reduce"]


@pytest.mark.parametrize("peer", [(False, (("reduce", (4, 7)),), None, None, (), None),
                                  (False, (), "ValueError: bad weight geometry", None, (), None)])
def test_finished_rank_releases_trial_resources_on_peer_failure(monkeypatch, peer):
    owner = coordinator(monkeypatch)
    closed = []
    result = SimpleNamespace(close=lambda: closed.append(True))
    done = SimpleNamespace(done=True, ready_cache=None, ready_tuning=(), ready_collective_tuning=None, ready_collectives=())
    job = SimpleNamespace(advance=lambda **kwargs: done, result=lambda: result)
    monkeypatch.setattr(dist, "all_gather_object", lambda states, local, **kwargs: states.__setitem__(slice(None), [local, peer]))
    with pytest.raises(RuntimeError, match="rank (finished|preparation failed)"):
        owner.prepare(SimpleNamespace(begin=lambda *args, **kwargs: job), ())
    assert closed == [True]


def test_local_preparation_failure_is_exchanged_before_cleanup(monkeypatch):
    owner = coordinator(monkeypatch)
    closed, exchanged = [], []

    def advance(**kwargs):
        raise ValueError("bad weight geometry")

    def gather(states, local, **kwargs):
        exchanged.append(local)
        states[:] = [local, (False, (), None, None, (), None)]

    job = SimpleNamespace(advance=advance, close=lambda: closed.append(True))
    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="bad weight geometry") as failure:
        owner.prepare(SimpleNamespace(begin=lambda *args, **kwargs: job), ())
    assert isinstance(failure.value.__cause__, ValueError)
    assert exchanged == [(False, (), "ValueError: bad weight geometry", None, (), None)]
    assert closed == [True]


def test_reports_ranks_on_this_host_and_whether_the_group_spans_hosts(monkeypatch):
    owner = coordinator(monkeypatch)
    assert owner.local_ranks == 2 and not owner.spans_hosts
    monkeypatch.setattr(dist, "all_gather_object",
                        lambda states, value, *, group: states.__setitem__(slice(None), [value, "another-boot"]))
    owner = CollectivePreparation(object())
    assert owner.local_ranks == 1 and owner.spans_hosts


def test_peers_adopt_rank_zero_selections_and_disagreement_fails(monkeypatch):
    owner = coordinator(monkeypatch)
    cache = SimpleNamespace(records={"key": "peer"}, _agreed_records=None)
    monkeypatch.setattr(dist, "get_rank", lambda: 7)
    monkeypatch.setattr(dist, "broadcast_object_list",
                        lambda values, *, src, group: values.__setitem__(0, {"key": "zero"}))
    owner.share_selections(cache)
    assert cache._agreed_records == {"key": "zero"} and cache.records == {"key": "peer"}

    def selection(config):
        return [{"name": "layer0.mlp", "component": "gemm", "config": config}]

    monkeypatch.setattr(dist, "all_gather_object", lambda states, value, *, group: states.__setitem__(
        slice(None), [{"layer0.mlp": ("gemm", '{"tile": 64}')}, value]))
    owner.agree_selections(selection({"tile": 64}))
    with pytest.raises(RuntimeError, match="1 kernel selections differ"):
        owner.agree_selections(selection({"tile": 128}))


def race(key, latency, index, assignment=True):
    return SimpleNamespace(key=key, latency_us=None if not assignment else latency,
                           candidate_index=None if not assignment else index,
                           assignment={"index": index} if assignment else None)


def test_sharded_tuning_reconciles_every_cache_and_adopts_each_race_fastest_result(monkeypatch):
    owner = coordinator(monkeypatch)
    mine, theirs = object(), object()
    local, peer = race("gemm", 9.0, 0), race("gemm", 7.5, 3)
    blank = dict(done=False, ready_collectives=(), ready_cache=None, ready_tuning=(), ready_collective_tuning=None)
    local_states = [SimpleNamespace(**{**blank, "ready_cache": mine}), SimpleNamespace(**{**blank, "ready_cache": mine}),
                    SimpleNamespace(**{**blank, "ready_tuning": (local,)}),
                    SimpleNamespace(**{**blank, "done": True})]
    # The peer offers its cache one exchange late, then its share of the race.
    peers = [(False, (), None, None, (), None), (False, (), None, theirs, (), None), (False, (), None, None, (peer,), None),
             (True, (), None, None, (), None)]
    advances = []
    job = SimpleNamespace(advance=lambda **kwargs: advances.append(kwargs) or local_states.pop(0),
                          result=lambda: "prepared")
    monkeypatch.setattr(dist, "all_gather_object",
                        lambda states, value, **kwargs: states.__setitem__(slice(None), [value, peers.pop(0)]))
    assert owner.prepare(SimpleNamespace(begin=lambda *args, **kwargs: job), ()) == "prepared"
    assert advances == [{"collective_key": None}, {"collective_key": None}, {"cache": (mine, theirs)},
                        {"tuning": peer}]


def test_race_winners_prefer_lower_latency_then_lower_candidate_index():
    from ignition.preparation.collectives import _tuning_results

    winners = _tuning_results([race("b", 5.0, 4), race("a", 2.0, 1), race("b", 5.0, 2), race("a", 3.0, 0),
                        race("a", 0, 0, assignment=False)])
    assert [(item.key, item.candidate_index) for item in winners] == [("a", 1), ("b", 2)]
    with pytest.raises(RuntimeError, match="no rank completed"):
        _tuning_results([race("c", 0, 0, assignment=False), race("c", 0, 1, assignment=False)])


def test_sharded_finalists_preserve_empty_shards_and_rank_order():
    from dataclasses import replace
    from b12x.preparation import FinalistPolicy, FinalistRecord, FrozenMapping, TuningRequirement
    from ignition.preparation.collectives import _tuning_results

    policy = FinalistPolicy(family_fields=("width",))
    assignment = FrozenMapping({"width": 1})
    first = TuningRequirement("qkv", (0, 1, 2, 3), assignment, 5., 0,
        finalist_policy=policy,
        finalists=(FinalistRecord(assignment=assignment, family=assignment, candidate_index=0, latency_us=5.),))
    shards = (first, *(replace(first, assignment=None, latency_us=None, candidate_index=None,
                              finalists=()) for _ in range(3)))
    assert _tuning_results(shards) == shards
    with pytest.raises(ValueError, match="policies differ"):
        _tuning_results((first, replace(shards[1], finalist_policy=None)))


def test_ranks_report_whether_a_value_matches_every_rank(monkeypatch):
    owner = coordinator(monkeypatch)
    monkeypatch.setattr(dist, "get_rank", lambda: 4)

    def broadcast(values, *, src, group):
        assert src == 4 and group is owner.group
        assert len(values) == 1

    def reduce(matching, *, op, group):
        import torch

        assert group is owner.group and op == dist.ReduceOp.MIN
        assert matching.device.type == "cpu" and matching.dtype == torch.int32

    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    assert owner.same_on_every_rank("NVIDIA GB10")
    monkeypatch.setattr(dist, "all_reduce", lambda matching, **kwargs: matching.fill_(0))
    assert not owner.same_on_every_rank("NVIDIA GB10")


def _equality_rank(rank, world, store, outcome):
    from datetime import timedelta
    import traceback

    import torch

    try:
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank,
                                world_size=world, timeout=timedelta(seconds=15))
        owner = CollectivePreparation(dist.group.WORLD)
        catalog = ("NVIDIA GB202", (("projection", {"shape": (64, 128), "dtype": "bf16"}, ()),))
        changed = ("NVIDIA GB202", (("projection", {"shape": (65, 128), "dtype": "bf16"}, ()),))
        results = (owner.same_on_every_rank(catalog),
                   owner.same_on_every_rank(changed if rank == world - 1 else catalog),
                   owner.same_on_every_rank(False))
        group = dist.new_group(ranks=[1, 2], backend="gloo", timeout=timedelta(seconds=15))
        subgroup = None
        if rank in (1, 2):
            owner = CollectivePreparation(group)
            subgroup = (owner.same_on_every_rank(catalog),
                        owner.same_on_every_rank(changed if rank == 2 else catalog))
        assert not torch.cuda.is_initialized()
        outcome.put((rank, (*results, subgroup)))
    except BaseException:
        outcome.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_gloo_value_agreement_reaches_every_rank_and_subgroup(tmp_path):
    context = multiprocessing.get_context("spawn")
    world = 3
    outcome = context.Queue()
    processes = [context.Process(target=_equality_rank,
        args=(rank, world, tmp_path / "store", outcome)) for rank in range(world)]
    for process in processes:
        process.start()
    try:
        results = dict(outcome.get(timeout=30) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert results == {0: (True, False, True, None),
                           1: (True, False, True, (True, False)),
                           2: (True, False, True, (True, False))}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


@pytest.mark.parametrize("ring", [False, True])
def test_roce_setup_preserves_routing_and_package_transport_owner(monkeypatch, ring):
    import sys
    from ignition.preparation.collectives import create_roce_runtime

    class Runtime:
        def __init__(self, *, ring=False, **options):
            self.ring, self.options = ring, options

    monkeypatch.setitem(sys.modules, "b12x.comm", SimpleNamespace(roce=SimpleNamespace(AllReduce=Runtime)))
    storage = SimpleNamespace(transports=[])
    runtime = create_roce_runtime(object(), "cuda:0", max_tokens=32, pipe_tokens=4096,
                                  hidden_size=4096, gather_bytes=4096, ring=ring, package_storage=storage)
    assert runtime.ring is ring and storage.transports == [runtime]
    assert runtime.options["max_size"] == 2 << 20
    assert runtime.options["pipe_bytes"] == 4096 * 4096 * 2
    assert callable(runtime.options["region_allocator"])


def test_differing_tuning_races_report_bounded_counts_and_release_job(monkeypatch):
    owner = coordinator(monkeypatch)
    local = tuple(race(f"local-{index:064x}", 1., index) for index in range(1200))
    peer = tuple(race(f"peer-{index:064x}", 1., index) for index in range(1200))
    state = SimpleNamespace(done=False, ready_collectives=(), ready_cache=None, ready_tuning=local, ready_collective_tuning=None)
    closed = []
    job = SimpleNamespace(advance=lambda **kwargs: state, close=lambda: closed.append(True))
    monkeypatch.setattr(dist, "all_gather_object",
        lambda states, value, **kwargs: states.__setitem__(slice(None), [value, (False, (), None, None, peer, None)]))
    with pytest.raises(RuntimeError, match="tuning races differ across ranks") as error:
        owner.prepare(SimpleNamespace(begin=lambda *args, **kwargs: job), ())
    assert "counts={4: 1200, 7: 1200}" in str(error.value)
    assert "'missing': 1200, 'extra': 1200" in str(error.value)
    assert len(str(error.value)) < 500
    assert closed == [True]


@pytest.mark.parametrize("mode", [None, "static", "runtime"])
def test_joint_workspace_is_provisioned_only_for_runtime_selection(monkeypatch, mode):
    from b12x.comm.pcie import AllReduce
    from ignition.preparation.collectives import create_all_reduce_runtime

    calls = []
    result, group, device, ipc = object(), object(), object(), object()
    monkeypatch.setattr(AllReduce, "from_exchange_group", lambda **kwargs: calls.append(kwargs) or result)
    options = {} if mode is None else {"collective_tuning": mode}
    assert create_all_reduce_runtime(group, device, max_tokens=32, hidden_size=2560, ipc=ipc, **options) is result
    assert calls == [dict(exchange_group=group, device=device, single_channel=True,
                         max_size=32 * 2560 * 2, eager_buffer_bytes=32 * 2560 * 2,
                         ipc=ipc, joint=mode == "runtime")]


@pytest.mark.parametrize("configured,widened", [(20_000_000, 600_000_000), (900_000_000, 900_000_000)])
def test_compiled_warmup_widens_the_roce_wait_across_hosts_and_restores_it(monkeypatch, configured, widened):
    from ignition.preparation.collectives import warmup_wait

    runtime = SimpleNamespace(spin_limit=configured)
    with warmup_wait(runtime, SimpleNamespace(spans_hosts=True)):
        assert runtime.spin_limit == widened
    assert runtime.spin_limit == configured
    with pytest.raises(RuntimeError, match="warmup"):
        with warmup_wait(runtime, SimpleNamespace(spans_hosts=True)):
            raise RuntimeError("warmup failed")
    assert runtime.spin_limit == configured


def test_compiled_warmup_leaves_same_host_and_pcie_waits_alone():
    from ignition.preparation.collectives import warmup_wait

    runtime = SimpleNamespace(spin_limit=20_000_000)
    with warmup_wait(runtime, SimpleNamespace(spans_hosts=False)):
        assert runtime.spin_limit == 20_000_000
    pcie = SimpleNamespace()
    with warmup_wait(pcie, SimpleNamespace(spans_hosts=True)):
        assert not hasattr(pcie, "spin_limit")
    with warmup_wait(None, SimpleNamespace(spans_hosts=True)), warmup_wait(runtime, None):
        assert runtime.spin_limit == 20_000_000
