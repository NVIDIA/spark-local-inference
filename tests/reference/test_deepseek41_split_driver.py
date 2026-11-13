# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native diagnostic timing retains all participating TP rank intervals."""

from types import SimpleNamespace

import pytest

from qualify.deepseek41_split import Driver


def executor_completion(intervals):
    value = SimpleNamespace(fence=7, graph_index=3, records=[[0] * 32],
        gpu_ns=intervals, wall_ns=95, queued=False)
    return SimpleNamespace(submit=lambda index, uploads: 7, poll=lambda: value)


def test_native_rank_intervals_are_reduced_without_losing_measurements():
    driver = Driver.__new__(Driver)
    driver.indices = {'verification': 3}
    intervals = [17, 23, 19, 21] + [None] * 12
    driver.executor = executor_completion(intervals)
    completion, _ = driver.execute('verification', [])
    assert completion.gpu_ns == 23
    assert completion.rank_gpu_ns == intervals and completion.wall_ns == 95
    assert completion.fence == driver.last_fence == 7


@pytest.mark.parametrize('intervals', [[17, 23, 19, None] + [None] * 12,
                                      [17, 23, 19, 0] + [None] * 12])
def test_missing_tp_intervals_do_not_become_performance_results(intervals):
    driver = Driver.__new__(Driver)
    driver.indices = {'verification': 3}
    driver.executor = executor_completion(intervals)
    with pytest.raises(AssertionError, match='four positive GPU intervals'):
        driver.execute('verification', [])
