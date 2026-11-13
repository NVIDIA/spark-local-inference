# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision declaration tensors preserve capacity and row geometry."""

from dataclasses import fields, is_dataclass
from importlib import import_module

import pytest
import torch

from ignition.components.parallel import TensorParallelShard


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
def test_vision_capacities_preserve_row_geometry(module, large, small, sharded, options):
    scratch_type = import_module(f"ignition.{module}").VisionScratch
    args = (TensorParallelShard(0, 4), "cpu") if sharded else ("cpu",)
    full = scratch_type.allocate(large, *args, **options)
    expected = scratch_type.allocate(small, *args, **options)
    for actual, reference in zip(tensors(full), tensors(expected), strict=True):
        assert actual.shape[0] >= reference.shape[0]
        assert actual.shape[1:] == reference.shape[1:]
        assert actual.dtype == reference.dtype
        assert torch.count_nonzero(actual) == 0
        assert torch.count_nonzero(reference) == 0
    if hasattr(full, "rows"):
        for actual, reference in zip(tensors(full.rows(small)), tensors(expected), strict=True):
            assert actual.shape == reference.shape


def test_qwen_decoder_and_vision_examples_are_disjoint():
    from ignition.recipes.qwen38.decoder import DecoderScratch
    from ignition.recipes.qwen38.vision import VisionMetadata, VisionScratch

    decoder = DecoderScratch.allocate(128, "cpu", sample_rows=4, world_size=4)
    decoder_storage = {t.untyped_storage()._cdata for t in tensors(decoder)}
    vision = VisionScratch.allocate(256, "cpu")
    metadata = VisionMetadata.allocate(256, 1, "cpu")
    assert not decoder_storage & {t.untyped_storage()._cdata for t in tensors(vision)}
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
