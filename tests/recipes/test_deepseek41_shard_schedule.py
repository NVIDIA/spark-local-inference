# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row-shard scheduling across the Engram layers of a DeepSeek V4.1 prefill decoder.

An Engram layer reads and writes full rows, so it runs full-row. Because the row dimension
therefore changes twice inside the decoder, the shard has to end before such a layer and resume
after it, and only a layer that runs its own post may change the dimension.
"""

from dataclasses import dataclass

from ignition.recipes.deepseek41.decoder import shard_schedule


@dataclass(frozen=True)
class _Layer:
    engram: object = None


SHARD = object()


def layers(count=40, engram=()):
    return tuple(_Layer("engram" if index in engram else None) for index in range(count))


def flags(schedule, column):
    return {index for index, entry in enumerate(schedule) if entry[column]}


def test_shard_resumes_after_each_engram_layer():
    schedule = shard_schedule(layers(engram=(1, 14)), 19, SHARD)
    assert flags(schedule, 0) == set(range(19)) - {1, 14}


def test_shard_refills_from_full_rows_where_it_resumes():
    schedule = shard_schedule(layers(engram=(1, 14)), 19, SHARD)
    assert flags(schedule, 2) == {2, 15}


def test_only_same_row_layers_defer_their_post():
    schedule = shard_schedule(layers(engram=(1, 14)), 19, SHARD)
    # Layers 19-38 keep the legacy defer: both neighbours are full-row, so nothing changes there.
    assert flags(schedule, 1) == {*range(2, 13), *range(15, 18), *range(19, 39)}


def test_engram_layer_itself_stays_full_row():
    schedule = shard_schedule(layers(engram=(1, 14)), 19, SHARD)
    for index in (1, 14):
        assert schedule[index][0] is False
        assert schedule[index][1] is False


def test_tail_from_stop_on_stays_full_row():
    schedule = shard_schedule(layers(engram=(1, 14)), 19, SHARD)
    assert not any(entry[0] for entry in schedule[19:])


def test_engram_first_layer_does_not_disable_the_rest():
    schedule = shard_schedule(layers(engram=(0,)), 19, SHARD)
    assert flags(schedule, 0) == set(range(1, 19))
    assert flags(schedule, 2) == {1}
    assert schedule[0][0] is False


def test_aux_layer_runs_full_row_and_restarts_the_shard():
    schedule = shard_schedule(layers(), 19, SHARD, aux_layers=(10,))
    assert 10 not in flags(schedule, 0)
    assert 9 not in flags(schedule, 1)
    assert 11 in flags(schedule, 2)


def test_stop_below_one_shards_nothing():
    schedule = shard_schedule(layers(engram=(1, 14)), 0, SHARD)
    assert not any(entry[0] for entry in schedule)
    assert not any(entry[2] for entry in schedule)


def test_without_a_shard_every_layer_keeps_the_legacy_defer():
    schedule = shard_schedule(layers(engram=(1, 14)), 19, None)
    expected = [index + 1 not in (1, 14) for index in range(39)] + [False]
    assert [entry[1] for entry in schedule] == expected
    assert not any(entry[0] or entry[2] for entry in schedule)
