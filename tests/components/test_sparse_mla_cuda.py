# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse latent attention survives arena replacement and native graph replay."""

import os

import pytest
import torch
from b12x.attention import sparse_mla

from ignition.components.sparse_mla import PreparedLatentCacheWriter, PreparedSparseMLA
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("mode,rows,heads", [("decode", 2, 32), ("extend", 4096, 16)])
@torch.no_grad()
def test_native_sparse_attention_rebinds_selected_arena(mode, rows, heads, functional):
    device = torch.device("cuda:0")
    records = ((rows + 192 + 63) // 64) * 64
    source = torch.zeros((rows, 512), dtype=torch.bfloat16, device=device)
    query = torch.zeros((rows, heads, 512), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((records // 64, 64, 528), dtype=torch.uint8, device=device)
    slots = torch.arange(rows, dtype=torch.int64, device=device)
    selected = torch.full((rows, 2051), -1, dtype=torch.int32, device=device)
    visible = min(rows, 2051)
    selected[:, :visible] = slots[:visible].to(torch.int32)
    lengths = torch.arange(1, rows + 1, dtype=torch.int32, device=device).clamp_max(2051)
    output = torch.empty_like(query)
    observed = torch.empty_like(query)
    arena = ScratchArena(device)
    worker = None
    graph = torch.cuda.CUDAGraph()
    with RecordedPreparationSession(device=device, checkpoint="sparse-mla-component",
            workload_schema="glm-latent-v1", numerical_policy="default") as session:
        try:
            session.register_scratch(arena)
            writer = PreparedLatentCacheWriter(source, cache, slots, session=session, name="cache-write")
            caps = sparse_mla.Caps(device=device, num_q_heads=heads, max_q_rows=rows, max_width=2051,
                softmax_scale=256**-.5, max_batch=rows, max_kv_rows=records, kv_dtype=torch.uint8,
                head_dim=512, v_head_dim=512, page_size=64, model_type=sparse_mla.ModelType.GLM_NEXT,
                cache_record_bytes=528, mode=mode)
            attention = PreparedSparseMLA(query, cache, selected, lengths, lengths, output,
                caps=caps, session=session, name="attention", workspace=arena)
            previous = arena.tensor
            arena.reserve(arena.nbytes + 1024)
            if functional:
                writer.functional(session=session)
                attention.functional(session=session)
            session.tune()
            if functional:
                assert attention.binding is attention.output is None
            else:
                assert attention.binding.runtime.scratch.shared_scratch.data_ptr() == arena.tensor.data_ptr()
            assert arena.tensor.data_ptr() != previous.data_ptr()
            del previous
            if functional:
                arena.release_preparation()

            def forward():
                if functional:
                    writer(source, slots)
                    result = attention(query, selected, lengths, lengths)
                else:
                    writer()
                    result = attention()
                arena.tensor.fill_(255)
                observed.copy_(result)

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            session.freeze()
            bundle = PreparedGraphBundle(device=0, graphs={mode: graph},
                buffers={"source": source, "query": query, "cache": cache, "output": observed},
                resources=(writer, attention, compiled, arena, session))
            worker = bundle.start_worker()
            for offset in (0, 64, 192):
                source.normal_(std=.1)
                query.normal_(std=.1)
                slots.copy_(torch.arange(rows, dtype=torch.int64, device=device) + offset)
                if mode == "decode":
                    slots[-1] = -1
                    lengths[-1] = 0
                selected[:, :visible] = slots[:visible].to(torch.int32)
                cache.zero_()
                forward()
                expected_output, expected_cache = observed.clone(), cache.clone()
                observed.fill_(42)
                cache.zero_()
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                ticket = worker.submit(0, repetitions=3)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                torch.testing.assert_close(observed, expected_output, rtol=0, atol=0)
                torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
                if mode == "decode":
                    assert not observed[-1].count_nonzero()
        finally:
            if worker is not None:
                worker.close()
            graph.reset()
