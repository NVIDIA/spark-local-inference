# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision graph storage shares caller-owned pools across serialized capacities."""

from dataclasses import fields, is_dataclass
from importlib import import_module

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.scratch import GraphBufferPool


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name))


@pytest.mark.parametrize("module,large,small,sharded,options", [
    ("recipes.qwen38.vision", 1024, 256, False, {}),
    ("components.deepseek_vision", 576, 144, False, {"hidden": 4096}),
    ("components.deepseek_vision", 576, 144, False, {"hidden": 5120}),
    ("recipes.mimo_v2.vision", 1024, 256, True, {"hidden": 4096}),
    ("recipes.mimo_v2.vision", 1024, 256, True, {"hidden": 6144}),
    ("recipes.kimi3.vision", 1024, 256, False, {}),
    ("recipes.nemotron_h.vision", 256, 64, False, {}),
    ("recipes.step5.vision", 256, 64, True, {}),
])
def test_vision_capacities_reuse_pool_storage(module, large, small, sharded, options):
    scratch_type = import_module(f"ignition.{module}").VisionScratch
    pool = GraphBufferPool("cpu", chunk_bytes=1 << 20)
    args = (TensorParallelShard(0, 4), "cpu") if sharded else ("cpu",)
    first = scratch_type.allocate(large, *args, buffers=pool.cursor(), **options)
    capacity = pool.nbytes
    storage = {chunk.untyped_storage()._cdata for chunks in pool.chunks.values() for chunk in chunks}
    assert all(t.untyped_storage()._cdata in storage for t in tensors(first))
    for size in (small, large, small):
        for chunks in pool.chunks.values():
            for chunk in chunks:
                chunk.fill_(float("nan"))
        scratch = scratch_type.allocate(size, *args, buffers=pool.cursor(), **options)
        assert pool.nbytes == capacity
        assert all(t.untyped_storage()._cdata in storage for t in tensors(scratch))
        assert all(torch.count_nonzero(t) == 0 for t in tensors(scratch))


def test_qwen_decoder_and_vision_share_pool_with_disjoint_live_operands():
    from ignition.recipes.qwen38.decoder import DecoderScratch
    from ignition.recipes.qwen38.vision import VisionMetadata, VisionScratch

    pool = GraphBufferPool("cpu", chunk_bytes=1 << 20)
    decoder = DecoderScratch.allocate(128, "cpu", sample_rows=4, world_size=4, buffers=pool.cursor())
    decoder_storage = {t.untyped_storage()._cdata for t in tensors(decoder)}
    cursor = pool.cursor()
    vision = VisionScratch.allocate(256, "cpu", buffers=cursor)
    metadata = VisionMetadata.allocate(256, 1, "cpu", buffers=cursor)
    assert decoder_storage & {t.untyped_storage()._cdata for t in tensors(vision)}
    common = (vision.hidden, vision.projected, *tensors(metadata))
    for stage in ((vision.patches,), (vision.qkv, vision.query, vision.key, vision.value),
                  (vision.query, vision.key, vision.value, vision.attention),
                  (vision.intermediate,), (vision.merger_intermediate, vision.output)):
        live = (*common, *stage)
        intervals = sorted((t.data_ptr(), t.data_ptr() + t.nbytes) for t in live)
        assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
        for index, t in enumerate(live, 1):
            t.fill_(index)
        for index, t in enumerate(live, 1):
            assert torch.all(t == index)
    from ignition.preparation.tensor_metadata import merge_ranges

    occupied = merge_ranges((t.data_ptr(), t.data_ptr() + t.nbytes) for t in tensors(vision))
    assert sum(end - start for start, end in occupied) < sum(t.nbytes for t in tensors(vision))
