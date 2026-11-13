# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tuning on a unified-memory GPU splits into jobs that ranks enter together."""

from types import SimpleNamespace

from ignition.preparation import tuning
from ignition.preparation.tuning import RecordedPreparationSession


def request(name, *, collective=False, dependencies=()):
    return SimpleNamespace(name=name, collective=object() if collective else None, dependencies=dependencies)


def jobs(requests, *, coordinator=True, unified=True):
    session = SimpleNamespace(_integrated=unified, _coordinator=object() if coordinator else None)
    return [[item.name for item in job] for job in RecordedPreparationSession._jobs(session, tuple(requests))]


def test_one_job_without_unified_memory():
    assert jobs([request("a", collective=True), request("b")], unified=False) == [["a", "b"]]


def test_jobs_end_after_matching_collectives_and_keep_dependency_groups(monkeypatch):
    monkeypatch.setattr(tuning, "_COLLECTIVES_PER_JOB", 2)
    # Rank zero declares local sampling requests that a peer lacks; collectives match.
    rank_zero = [request("scale"), request("r0", collective=True), request("sample"), request("r1", collective=True),
                 request("moe", dependencies=("scale",)), request("r2", collective=True), request("tail")]
    peer = [request("scale"), request("r0", collective=True), request("r1", collective=True),
            request("moe", dependencies=("scale",)), request("r2", collective=True)]
    assert jobs(rank_zero) == [["r0", "sample", "r1"], ["scale", "moe", "r2", "tail"]]
    assert jobs(peer) == [["r0", "r1"], ["scale", "moe", "r2"]]


def test_single_rank_jobs_count_every_request(monkeypatch):
    monkeypatch.setattr(tuning, "_REQUESTS_PER_JOB", 2)
    assert jobs([request(name) for name in "abcde"], coordinator=False) == [["a", "b"], ["c", "d"], ["e"]]
