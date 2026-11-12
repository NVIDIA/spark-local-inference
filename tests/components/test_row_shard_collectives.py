# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row-sharded reductions and shard gathers keep the all-reduce path's layout and eligibility."""

from types import SimpleNamespace

import pytest
import torch


def _surfaces(monkeypatch, *, abi=11, roce=True):
    """Enable the ABI 11 row-shard surfaces on the module the wrappers test at run time."""
    import b12x.comm.roce._proxy as proxy
    from ignition.components import collectives

    monkeypatch.setattr(proxy, "ABI_VERSION", abi)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: roce)


def _runtime(*, world=4, pipe_bytes=1 << 26, gather=None):
    calls = []

    def reduce_scatter(inp, *, out, plan):
        calls.append(("reduce_scatter", tuple(inp.shape), tuple(out.shape)))

    def gather_shards(inp, *, out, plan):
        calls.append(("gather_shards", tuple(inp.shape), tuple(out.shape)))

    return SimpleNamespace(world_size=world, pipe_bytes=pipe_bytes, _group=object(),
        should_reduce_scatter=lambda inp: True, should_gather_shards=lambda inp: True,
        reduce_scatter=reduce_scatter, gather_shards=gather or gather_shards, calls=calls)


@pytest.fixture
def prepared(monkeypatch):
    """Prepare row-shard operations without a device or a preparation session."""
    from ignition.components import collectives

    _surfaces(monkeypatch)
    surfaces, plan = [], SimpleNamespace(request=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(collectives, "_roce_plan",
                        lambda runtime, coordinator, surface: surfaces.append(surface) or plan)
    monkeypatch.setattr(collectives, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1, 2, 3])
    return surfaces, plan


@pytest.mark.parametrize("abi,roce,drop,expected", [
    (11, True, None, True), (10, True, None, False), (11, False, None, False),
    (11, True, "should_reduce_scatter", False), (11, True, "should_gather_shards", False)])
def test_row_shard_surfaces_require_abi_eleven_and_both_runtime_surfaces(monkeypatch, abi, roce, drop, expected):
    from ignition.components import collectives

    _surfaces(monkeypatch, abi=abi, roce=roce)
    runtime = _runtime()
    if drop is not None:
        delattr(runtime, drop)
    assert collectives.row_shard_surfaces(runtime) is expected


def test_reduce_scatter_fits_splits_whole_row_packs_into_the_shard(monkeypatch):
    from ignition.components.collectives import PreparedReduceScatter

    _surfaces(monkeypatch)
    runtime = _runtime()
    shard = torch.zeros(2, 5120, dtype=torch.bfloat16)
    assert PreparedReduceScatter.fits(runtime, torch.zeros(8, 5120, dtype=torch.bfloat16), shard)
    # A four-stream residual shards along its rows, not its trailing dims.
    assert PreparedReduceScatter.fits(runtime, torch.zeros(8, 4, 5120, dtype=torch.bfloat16),
                                      torch.zeros(2, 4, 5120, dtype=torch.bfloat16))
    assert not PreparedReduceScatter.fits(runtime, torch.zeros(8, 5120, dtype=torch.bfloat16), torch.zeros(4, 5120, dtype=torch.bfloat16))
    assert not PreparedReduceScatter.fits(runtime, torch.zeros(6, 5120, dtype=torch.bfloat16), shard)
    assert not PreparedReduceScatter.fits(runtime, torch.zeros(8, 5120, dtype=torch.bfloat16)[:, :1], torch.zeros(2, 1, dtype=torch.bfloat16))
    assert not PreparedReduceScatter.fits(runtime, torch.zeros(8, 7, dtype=torch.bfloat16), torch.zeros(2, 7, dtype=torch.bfloat16))
    assert not PreparedReduceScatter.fits(runtime, torch.zeros(8, 5120), torch.zeros(2, 5120))
    assert not PreparedReduceScatter.fits(_runtime(pipe_bytes=1 << 15), torch.zeros(8, 5120, dtype=torch.bfloat16), shard)


def test_gather_shards_fits_gathers_every_shard_row_and_chunked_gathers_accept_one_row_per_rank(monkeypatch):
    from ignition.components.collectives import PreparedChunkedGatherShards, PreparedGatherShards

    _surfaces(monkeypatch)
    runtime = _runtime()
    shard = torch.zeros(2, 4, 5120, dtype=torch.bfloat16)
    assert PreparedGatherShards.fits(runtime, shard, torch.zeros(8, 4, 5120, dtype=torch.bfloat16))
    assert not PreparedGatherShards.fits(runtime, shard, torch.zeros(4, 4, 5120, dtype=torch.bfloat16))
    assert not PreparedGatherShards.fits(_runtime(pipe_bytes=1 << 18), shard,
                                         torch.zeros(8, 4, 5120, dtype=torch.bfloat16))
    assert PreparedChunkedGatherShards.fits(runtime, shard, torch.zeros(8, 4, 5120, dtype=torch.bfloat16))
    # One shard row per rank is the smallest chunk, so nothing below its message fits.
    assert not PreparedChunkedGatherShards.fits(_runtime(pipe_bytes=4 * 40960 - 1), shard,
                                                torch.zeros(8, 4, 5120, dtype=torch.bfloat16))


@pytest.mark.parametrize("wrapper", ["PreparedReduceScatter", "PreparedGatherShards"])
def test_single_rank_row_shard_operations_copy_the_source(monkeypatch, wrapper):
    from ignition.components import collectives

    operation = getattr(collectives, wrapper)
    source = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    output = torch.empty_like(source)
    prepared = operation(source, output, runtime=None, session=None, coordinator=None, name="row_shard")
    assert prepared.single
    assert prepared(source) is output
    assert torch.equal(source, output)
    with pytest.raises(ValueError, match="single-rank"):
        operation(source, torch.empty(4, 4, dtype=torch.bfloat16), runtime=None, session=None,
                  coordinator=None, name="row_shard")


def test_prepared_row_shard_operations_request_their_surface_and_launch_the_runtime(prepared):
    from ignition.components.collectives import PreparedGatherShards, PreparedReduceScatter

    surfaces, plan = prepared
    runtime = _runtime()
    source = torch.zeros(8, 5120, dtype=torch.bfloat16)
    shard = torch.zeros(2, 5120, dtype=torch.bfloat16)
    reduce = PreparedReduceScatter(source, shard, runtime=runtime, session=object(),
                                   coordinator=SimpleNamespace(hosts=()), name="layer.attention_reduce")
    gather_output = torch.zeros(8, 5120, dtype=torch.bfloat16)
    gather = PreparedGatherShards(shard, gather_output, runtime=runtime, session=object(),
                                  coordinator=SimpleNamespace(hosts=()), name="layer.attention_gather")
    assert surfaces == ["AllReduce.reduce_scatter", "AllReduce.gather_shards"]
    assert reduce.plan is plan and gather.plan is plan
    assert reduce(source) is shard and gather(shard) is gather_output
    assert runtime.calls == [("reduce_scatter", (8, 5120), (2, 5120)),
                             ("gather_shards", (2, 5120), (8, 5120))]


def test_chunked_gather_scatters_each_chunk_into_its_rank_major_window(prepared):
    from ignition.components.collectives import PreparedChunkedGatherShards

    rows, world, columns = 5, 2, 8
    row_bytes = columns * 2
    # Two shard rows per gathered message, so the five shard rows arrive in three chunks.
    runtime = _runtime(world=world, pipe_bytes=row_bytes * 2 * world)

    def gather(inp, *, out, plan):
        # A gather publishes rank-major rows, so a single-rank test writes the shard to each rank.
        out.view(world, -1, columns)[:] = inp
    runtime.gather_shards = gather
    shard = torch.arange(rows * columns, dtype=torch.bfloat16).reshape(rows, columns)
    output = torch.zeros(rows * world, columns, dtype=torch.bfloat16)
    operation = PreparedChunkedGatherShards(shard, output, runtime=runtime, session=object(),
                                            coordinator=SimpleNamespace(hosts=()), name="layer.residual_gather")
    assert operation.bounds == [(0, 2), (2, 4), (4, 5)]
    assert operation(shard) is output
    for rank in range(world):
        assert torch.equal(output.view(world, rows, columns)[rank], shard)


def test_a_functional_chunked_gather_allocates_its_staging_and_output_in_the_forward(prepared):
    """A captured graph owns every operand, so the wrapper declares shapes and allocates on call."""
    from ignition.components.collectives import PreparedChunkedGatherShards

    rows, world, columns = 2, 2, 8
    row_bytes = columns * 2

    def gather(inp, *, out, plan):
        out.view(world, -1, columns)[:] = inp

    runtime = _runtime(world=world, pipe_bytes=row_bytes * rows * world)
    runtime.gather_shards = gather
    shard = torch.arange(rows * columns, dtype=torch.bfloat16).reshape(rows, columns)
    operation = PreparedChunkedGatherShards(shard, torch.zeros(rows * world, columns, dtype=torch.bfloat16),
                                            runtime=runtime, session=object(),
                                            coordinator=SimpleNamespace(hosts=()), name="layer.residual_gather")
    assert operation.functional(session=object()) is operation
    assert operation.output is None
    first = operation(shard)
    second = operation(shard)
    assert first.shape == second.shape == (rows * world, columns) and first.dtype is torch.bfloat16
    assert first.data_ptr() != second.data_ptr()
    for rank in range(world):
        assert torch.equal(second.view(world, rows, columns)[rank], shard)
