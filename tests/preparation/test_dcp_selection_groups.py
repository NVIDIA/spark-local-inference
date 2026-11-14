# SPDX-License-Identifier: Apache-2.0

"""Independent context groups retain their own collective selection checks."""

from types import SimpleNamespace

import pytest
import torch.distributed as dist

from ignition.preparation.collectives import CollectivePreparation


@pytest.mark.parametrize("rank,mismatch", [(0, 1), (8, 9)])
@pytest.mark.parametrize("event", ["pending", "collective_tuning", "tuning", "callback_pending", "callback_tuning"])
def test_selection_checks_follow_local_group_after_readiness_exchange(monkeypatch, rank, mismatch, event):
    owner = object.__new__(CollectivePreparation)
    owner.group = None
    owner.ranks = tuple(range(16))
    owner.operation_ranks = {}
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    name = "layer.attention.combine"
    states = []
    for peer in owner.ranks:
        ranks = tuple(range(peer // 8 * 8, peer // 8 * 8 + 8))
        pending = ((name, ranks),)
        contribution = SimpleNamespace(key=name, ranks=ranks)
        states.append({
            "pending": (False, pending, None, None, (), None),
            "collective_tuning": (False, (), None, None, (), contribution),
            "tuning": (False, (), None, None, (contribution,), None),
            "callback_pending": (pending, None),
            "callback_tuning": ((), contribution),
        }[event])
    owner._remember_operation_ranks(states)

    def gather(output, local, *, group):
        for peer in owner.ranks:
            output[peer] = {name: ("comm", '"different"' if peer == mismatch else '"same"')}

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="kernel selections differ"):
        owner.agree_selections([{"name": name, "component": "comm", "config": "same"}])


@pytest.mark.parametrize("rank", [0, 8])
def test_selection_checks_allow_different_choices_between_context_groups(monkeypatch, rank):
    owner = object.__new__(CollectivePreparation)
    owner.group = None
    owner.ranks = tuple(range(16))
    owner.operation_ranks = {}
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    name = "layer.attention.combine"
    states = [
        (False, ((name, tuple(range(peer // 8 * 8, peer // 8 * 8 + 8))),), None, None, (), None)
        for peer in owner.ranks
    ]
    owner._remember_operation_ranks(states)

    def gather(output, local, *, group):
        for peer in owner.ranks:
            output[peer] = {name: ("comm", '"group-a"' if peer < 8 else '"group-b"')}

    monkeypatch.setattr(dist, "all_gather_object", gather)
    owner.agree_selections([{"name": name, "component": "comm",
                             "config": "group-a" if rank < 8 else "group-b"}])
