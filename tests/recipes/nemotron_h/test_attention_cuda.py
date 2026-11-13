# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unrotated checkpoint attention across paged cache boundaries."""

import gc
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("width,mode,batch,packed,autotune", [(1, "decode", 1, None, False), (1, "decode", 4, None, False), (1, "decode", 8, None, False),
    (8, "verify", 1, None, False), (8, "verify", 4, None, False), (8, "verify", 8, None, False),
    (129, "prefill", 1, None, False), (4096, "prefill", 1, None, False),
    pytest.param(8, "verify", 4, 9, False, id="ragged-p9-w8-notune"),
    pytest.param(8, "verify", 4, 9, True, id="ragged-p9-w8-tune")])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@torch.no_grad()
def test_checkpoint_paged_attention(width, mode, batch, packed, autotune, kv_dtype):
    from b12x.preparation import PreparationSession
    from b12x.testing.q8_0_reference import dequantize_blocks
    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.nemotron_h.attention import PreparedAttention
    from ignition.recipes.nemotron_h.weights import NemotronWeightLoader
    from ignition.reference.qwen38_quant import CheckpointTensors

    path = Path(os.environ.get("FLARE_NEMOTRON_CHECKPOINT",
        "/data/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"))
    if not path.is_dir():
        pytest.skip("set FLARE_NEMOTRON_CHECKPOINT to the Bolt checkpoint")
    manifest = inspect_indexed_checkpoint(path)
    prefix = "language_model.backbone.layers.7.mixer"
    with NemotronWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                              prefixes=("language_model.backbone.layers.7.",)) as loader:
        weights, config = loader.load_layer(7), loader.config
    torch.manual_seed(19)
    tokens = width * batch if packed is None else packed
    source = torch.randn((tokens, config.hidden), device="cuda", dtype=torch.bfloat16) * .2
    start, heads, groups, dim = 127, 32, 2, 128
    length = start + width
    pages = (length + 127) // 128
    key, value = (torch.randn((batch * pages, 128, groups, dim), device="cuda").mul_(.2).to(kv_dtype) for _ in range(2))
    initial_key, initial_value = key.clone(), value.clone()
    scale = torch.ones(1, device="cuda") if kv_dtype == torch.float8_e4m3fn else None
    m = Metadata.allocate(tokens, batch, pages, source.device, max_query_tokens=width)
    starts = torch.full((batch,), start, device="cuda", dtype=torch.int64)
    sizes = ([width] if batch == 1 else [width, max(1, width // 2), 0, width] * (batch // 4)) if packed is None else [6, 1, 0, 2]
    lengths = torch.tensor(sizes, device="cuda", dtype=torch.int32)
    rows = torch.arange(1, batch + 1, device="cuda", dtype=torch.int64)
    tables = torch.arange(batch * pages, device="cuda", dtype=torch.int32).view(batch, pages)
    ids = torch.zeros((batch, width), device="cuda", dtype=torch.int64)
    live = torch.tensor([batch], device="cuda", dtype=torch.int32)
    m.set(starts, lengths, rows, tables, ids, live)
    with PreparationSession(device="cuda:0", autotune=autotune) as session:
        arena = ScratchArena("cuda:0")
        block = PreparedAttention(weights, source, m, key, value, config, mode=mode,
            session=session, name="attention", workspace=arena,
            key_scale=scale, value_scale=scale)
        block.functional(session=session)
        arena.finalize()
        def operation():
            return block(source, m, key, value)
        forward = compile_forward(operation)
        forward()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            actual = forward()
        key.copy_(initial_key)
        value.copy_(initial_value)
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        with CheckpointTensors(manifest, byte_budget=128 << 20) as reader:
            def linear(x, suffix):
                weight = dequantize_blocks(reader.read(prefix + "." + suffix + ".weight")).bfloat16().cuda().float()
                return (x.float() @ weight.T).bfloat16()
            q = linear(source, "q_proj").view(tokens, heads, dim)
            k = linear(source, "k_proj").view(tokens, groups, dim).to(kv_dtype).bfloat16()
            v = linear(source, "v_proj").view(tokens, groups, dim).to(kv_dtype).bfloat16()
            attended, offset = [], 0
            written = torch.zeros((batch, pages * 128), device="cuda", dtype=torch.bool)
            for request, count in enumerate(lengths.tolist()):
                if not count:
                    continue
                history_k = initial_key.view(batch, -1, groups, dim)[request, :start].bfloat16()
                history_v = initial_value.view(batch, -1, groups, dim)[request, :start].bfloat16()
                keys = torch.cat((history_k, k[offset:offset + count])).transpose(0, 1)
                values = torch.cat((history_v, v[offset:offset + count])).transpose(0, 1)
                mask = torch.arange(start + count, device="cuda")[None] <= torch.arange(start, start + count, device="cuda")[:, None]
                result = torch.nn.functional.scaled_dot_product_attention(
                    q[offset:offset + count].transpose(0, 1)[None], keys[None], values[None],
                    attn_mask=mask, enable_gqa=True)
                attended.append(result[0].transpose(0, 1))
                written[request, start:start + count] = True
                offset += count
            expected = linear(torch.cat(attended).flatten(1), "o_proj")
        observed = actual[:expected.shape[0]].clone()
        error = (observed.float() - expected.float()).norm() / expected.float().norm()
        assert error < .02, error.item()
        assert (observed.float() - expected.float()).abs().max() < .05 * expected.float().abs().max()
        for cache, initial in ((key, initial_key), (value, initial_value)):
            torch.testing.assert_close(cache.view(batch, -1, groups, dim).float()[~written],
                initial.view(batch, -1, groups, dim).float()[~written], rtol=0, atol=0)
        cached = key.clone(), value.clone()
        m.set(starts, torch.zeros_like(lengths), rows, tables, ids, live)
        graph.replay()
        for cache, initial in zip((key, value), cached, strict=True):
            torch.testing.assert_close(cache.float(), initial.float(), rtol=0, atol=0)
        m.set(starts, lengths, rows, tables, ids, live)
        graph.replay()
        torch.testing.assert_close(actual[:expected.shape[0]], observed, rtol=0, atol=0)
