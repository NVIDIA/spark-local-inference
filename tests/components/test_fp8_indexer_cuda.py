# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""FP8 key pages and query scaling agree with independent arithmetic under replay."""

import os

import pytest
import torch
from b12x.attention.dsa_indexer.reference import unpack_index_k_cache_reference

from ignition.components.fp8_indexer import PreparedFP8Indexer
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@pytest.mark.parametrize("mode,rows", [("decode", 4), ("prefill", 513), ("prefill", 4096)])
@torch.no_grad()
def test_rotated_fp8_selection_preserves_cache_and_replay_storage(mode, rows):
    device = torch.device("cuda", 0)
    torch.manual_seed(2513 + rows)
    heads, records = 64, 1024
    query = torch.randn((rows, heads, 128), device=device, dtype=torch.bfloat16)
    weights = torch.randn((rows, heads), device=device, dtype=torch.bfloat16)
    keys = torch.randn((records, 128), device=device, dtype=torch.bfloat16)
    slots = torch.arange(records, device=device, dtype=torch.int64)
    cache = torch.zeros((records // 64, 64 * 132), device=device, dtype=torch.uint8)
    table = torch.arange(records // 64 - 1, -1, -1, device=device, dtype=torch.int32)[None].expand(rows, -1)
    lengths = torch.full((rows,), records, device=device, dtype=torch.int32)
    lengths[0] = 0
    width = torch.tensor([records], device=device, dtype=torch.int32)
    output = torch.empty((rows, 512), device=device, dtype=torch.int32)
    arena, buffers = ScratchArena(device), GraphBufferPool(device)
    matrix = torch.ones((1, 1), device=device)
    for _ in range(7):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)

    def rotated_quantized(value):
        rotated = (value.float() @ matrix * 128**-0.5).bfloat16().float()
        scale = torch.exp2(torch.ceil(torch.log2(rotated.abs().amax(-1).clamp_min(1e-4) / 448)))
        quantized = (rotated / scale[..., None]).to(torch.float8_e4m3fn)
        return quantized.float(), scale

    with RecordedPreparationSession(device=device, checkpoint="fp8-index-component",
            workload_schema="fp8-rotated-index-v1", numerical_policy="power-of-two") as session:
        session.register_scratch(arena)
        indexer = PreparedFP8Indexer(query, weights, cache, table, lengths, width, output,
            mode=mode, session=session, name="index", workspace=arena, buffers=buffers.cursor(),
            prefill_records_per_call=131072)
        session.tune()

        def forward():
            indexer.write_keys(keys, slots)
            indexer(query, weights)
            arena.tensor.fill_(255)

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for key in ("allocation.all.allocated", "allocation.all.freed",
                    "segment.all.allocated", "segment.all.freed"):
            assert before[key] == after[key]
        expected_k, key_scale = rotated_quantized(keys)
        actual_k = unpack_index_k_cache_reference(cache, num_tokens=records)
        torch.testing.assert_close(actual_k.float(), expected_k * key_scale[:, None], rtol=0, atol=0)
        expected_q, query_scale = rotated_quantized(query)
        torch.testing.assert_close(indexer.quantized.float(), expected_q, rtol=0, atol=0)
        torch.testing.assert_close(indexer.scaled_weights,
            weights.float() * query_scale * (128 * heads)**-0.5, rtol=2e-6, atol=1e-7)
        permuted = actual_k.view(-1, 64, 128).flip(0).flatten(0, 1).float()
        scores = torch.einsum("rhd,kd->rhk", expected_q[1:4], permuted).relu()
        scores = (scores * indexer.scaled_weights[1:4, :, None]).sum(1)
        selected = scores.gather(1, output[1:4].long())
        cutoff = scores.topk(512, dim=1).values[:, -1]
        assert (selected.amin(1) >= cutoff - 0.001).all()
        assert (output[0] == -1).all()
        saved = cache.clone()
        slots.fill_(-1)
        keys.normal_()
        graph.replay()
        torch.testing.assert_close(cache, saved, rtol=0, atol=0)
        graph.reset()
