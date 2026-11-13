# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared graph capture orders indivisible allocations consistently across ranks."""

from datetime import timedelta
import multiprocessing
from types import SimpleNamespace

import pytest

from ignition.preparation.capture import _capture_order


def test_capture_orders_largest_request_before_largest_total_and_preserves_ties():
    names = ("init", "vision", "prefill", "decode", "decode-other", "restore")
    compiled = {name: SimpleNamespace(capture_allocation_sizes=sizes) for name, sizes in (
        ("vision", (1600, 128)), ("prefill", (1603,)), ("decode", (16,)), ("decode-other", (16,)))}
    assert _capture_order(names, compiled, None) == (
        "prefill", "vision", "decode", "decode-other", "init", "restore")
    assert _capture_order(names, {}, None) == names


def _rank_order(rank, store, output, mismatched):
    import torch.distributed as dist

    try:
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2,
                                timeout=timedelta(seconds=15))
        names = ("first", "second", "init")
        if mismatched and rank == 1:
            names = ("first", "second")
        compiled = {name: SimpleNamespace(capture_allocation_sizes=(size,))
                    for name, size in zip(names[:2], ((64, 32) if rank == 0 else (32, 128)))}
        coordinator = SimpleNamespace(ranks=(0, 1), group=dist.group.WORLD)
        try:
            result = _capture_order(names, compiled, coordinator)
        except ValueError as error:
            result = str(error)
        output.put((rank, result))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("mismatched", [False, True])
def test_capture_order_agrees_across_tensor_parallel_ranks(tmp_path, mismatched):
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [context.Process(target=_rank_order, args=(rank, tmp_path / "store", output, mismatched))
                 for rank in range(2)]
    for process in processes:
        process.start()
    try:
        results = dict(output.get(timeout=30) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        expected = ("tensor-parallel capture requires matching forward catalogs" if mismatched
                    else ("second", "first", "init"))
        assert results == {0: expected, 1: expected}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
