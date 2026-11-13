# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Twenty-four-head MLA preserves packed-cache arithmetic and graph replay."""

import os

import pytest
import torch
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference, pack_deepseek_v41_cache_reference,
)

from ignition.components.compressed_mla import PreparedCompressedMLA
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("mode,rows", [("decode", 4), ("extend", 17), ("verify", 7), ("draft_extend", 7)])
@torch.no_grad()
def test_twenty_four_head_bf16_attention_matches_packed_reference_under_replay(mode, rows):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(41019)
    source = torch.randn((128, 512), device=device, generator=generator).bfloat16()
    cache = pack_deepseek_v41_cache_reference(source, page_size=128, cache_kind="swa")
    query = torch.randn((rows, 24, 512), device=device, generator=generator).bfloat16()
    indices = torch.arange(128, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
    lengths = torch.full((rows,), 128, dtype=torch.int32, device=device)
    lengths[-1] = 0
    sink = torch.zeros(24, device=device)
    output = torch.empty_like(query)
    arena = ScratchArena(device)
    graph = torch.cuda.CUDAGraph()
    with RecordedPreparationSession(device=device, checkpoint="compressed-mla-head-partition",
            workload_schema="heads24-bf16", numerical_policy="checkpoint", compile_workers=2) as session:
        session.register_scratch(arena)
        operation = PreparedCompressedMLA(query, cache, indices, lengths, sink, output,
            mode=mode, session=session, name="attention", workspace=arena, compute_mode="bf16")
        session.tune(show_progress=False)
        operation()
        torch.cuda.synchronize()
        with session.capture(), torch.cuda.graph(graph):
            operation()
        try:
            for width in (128, 31, 64):
                lengths[:-1].fill_(width)
                output.fill_(float("nan"))
                before = torch.cuda.memory_stats(device)
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                expected = compressed_sparse_mla_reference(query, cache, indices, lengths,
                    sm_scale=512**-.5, attn_sink=sink, swa_page_size=128, cache_format="deepseek_v41")
                assert expected[:-1].abs().amax(-1).min() > .035
                torch.testing.assert_close(output, expected, atol=.035, rtol=.035)
                assert not output[-1].count_nonzero()
        finally:
            graph.reset()
