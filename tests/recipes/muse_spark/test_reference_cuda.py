# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete checkpoint text reference repeatability and bounded weight residency."""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned CUDA_VISIBLE_DEVICES")


@torch.no_grad()
def test_checkpoint_text_reference_bos_repeatability():
    from ignition.reference.muse_spark import text_logits

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    checkpoint = Path(os.environ.get("FLARE_MUSE_SPARK_CHECKPOINT",
        "/data/cache/huggingface/hub/models--meta-models--Muse-Spark-Open-816B-A42B-NVFP4/snapshots/4239e4db19376904f2e82c167e78f70abb5455fe"))
    if not checkpoint.is_dir():
        pytest.skip("Muse Spark checkpoint is unavailable")
    torch.cuda.reset_peak_memory_stats()
    hidden, logits = text_logits(checkpoint, [200000])
    repeated_hidden, repeated_logits = text_logits(checkpoint, [200000])
    assert hidden.shape == (1, 8192) and logits.shape == (1, 202048)
    assert hidden.dtype == logits.dtype == torch.bfloat16
    assert torch.isfinite(hidden).all() and torch.isfinite(logits).all()
    assert logits.abs().max() <= 20
    assert torch.equal(hidden, repeated_hidden)
    assert torch.equal(logits, repeated_logits)
    assert torch.cuda.max_memory_allocated() < 8 << 30
