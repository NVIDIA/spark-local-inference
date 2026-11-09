# SPDX-License-Identifier: Apache-2.0
"""Serial text/vision workspaces preserve live ranges without avoidable slabs."""

import pytest
import torch

from ignition.preparation.scratch import GraphBufferPool, scratch_views
from ignition.recipes.kimi3.bootstrap import allocate_serial_workspace
from ignition.recipes.kimi3.vision import VisionScratch


def disjoint(values):
    def span(value):
        if value.is_meta:
            start = value.storage_offset()*value.element_size()
            return value.untyped_storage()._cdata, start, start+value.nbytes
        return 0, value.data_ptr(), value.data_ptr()+value.nbytes
    spans = sorted(span(value) for value in values)
    assert all(left[0] != right[0] or left[2] <= right[1]
               for left, right in zip(spans, spans[1:]))


@pytest.mark.parametrize("rows", [128, 1024, 4096, 8192])
@pytest.mark.parametrize("auxiliary", [0, 5, 6])
@pytest.mark.parametrize("patches", [0, 65536])
def test_live_text_and_vision_operands_remain_disjoint(rows, auxiliary, patches):
    pool = GraphBufferPool("meta", chunk_bytes=2 << 20)
    cursor, residual, shared, shapes = allocate_serial_workspace(pool,
        text_rows=rows, hidden_size=7168, residual_count=8,
        auxiliary_count=auxiliary, vision_patches=patches)
    assert shared.numel() == sum(rows * columns for rows, columns in shapes)
    disjoint((residual, shared))
    extra = cursor.zeros((32, 7168), torch.bfloat16)
    disjoint((residual, shared, extra))
    if patches:
        vision_cursor = pool.cursor()
        vision = VisionScratch.allocate(patches, "meta", buffers=vision_cursor)
        for live in ((vision.hidden, vision.normalized, vision.qkv),
                     (vision.hidden, vision.qkv, vision.query, vision.key, vision.value),
                     (vision.hidden, vision.query, vision.key, vision.value, vision.attended),
                     (vision.hidden, vision.normalized, vision.mlp_up),
                     (vision.mlp_up, vision.mlp_down),
                     (vision.merged, vision.activated), (vision.activated, vision.output)):
            disjoint(live)
        features = vision_cursor.zeros((patches // 4, 448), torch.bfloat16)
        disjoint((vision.output, features))


def test_c4_prefill4096_vision_uses_three_large_slabs():
    pool = GraphBufferPool("meta", chunk_bytes=2 << 20)
    allocate_serial_workspace(pool, text_rows=4096, hidden_size=7168,
                             residual_count=8, auxiliary_count=6, vision_patches=65536)
    assert [chunk.nbytes for chunk in pool.chunks[torch.bfloat16]] == [576 << 20, 128 << 20, 576 << 20]
    # Publication keeps its local feature copy live beside the encoder output.
    cursor = pool.cursor()
    vision = VisionScratch.allocate(65536, "meta", buffers=cursor)
    features = cursor.zeros((16384, 448), torch.bfloat16)
    disjoint((vision.output, features))
    assert pool.nbytes == 1294 << 20


def test_text_larger_than_vision_reserves_the_text_slabs_first():
    pool = GraphBufferPool("meta", chunk_bytes=2 << 20)
    allocate_serial_workspace(pool, text_rows=8192, hidden_size=7168,
                             residual_count=8, auxiliary_count=6, vision_patches=65536)
    assert [chunk.nbytes for chunk in pool.chunks[torch.bfloat16]] == [1120 << 20, 896 << 20]


def test_real_text_views_preserve_simultaneously_live_values():
    pool = GraphBufferPool("cpu", chunk_bytes=2 << 20)
    _, residual, shared, shapes = allocate_serial_workspace(pool, text_rows=128,
        hidden_size=7168, residual_count=8, auxiliary_count=6, vision_patches=256)
    text = scratch_views(shared, shapes)
    disjoint((residual, *text))
    residual.fill_(17)
    for index, value in enumerate(text, 1):
        value.fill_(index)
    assert bool((residual == 17).all())
    for index, value in enumerate(text, 1):
        assert bool((value == index).all())
