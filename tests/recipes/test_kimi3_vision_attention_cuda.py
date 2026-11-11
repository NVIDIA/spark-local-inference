"""MoonViT attention consumes packed QKV through compiled CUDA graph replay."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3.vision import PATCH_CAPACITIES, _Attention


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU",
)
@pytest.mark.parametrize("rows", PATCH_CAPACITIES)
@torch.no_grad()
def test_packed_qkv_attention_compiled_replay(rows):
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(572)
    source = torch.randn((rows, 1024), device=device, dtype=torch.bfloat16, generator=generator)
    output = torch.empty_like(source)
    weights = {
        "wqkv.weight": torch.randn((4608, 1024), device=device, dtype=torch.bfloat16, generator=generator) / 32,
        "wo.weight": torch.randn((1024, 1536), device=device, dtype=torch.bfloat16, generator=generator) / 1536 ** .5,
    }
    packed = torch.empty((rows, 4608), device=device, dtype=torch.bfloat16)
    query, key, value = packed.view(rows, 3, 12, 128).unbind(1)
    scratch = SimpleNamespace(hidden=source, qkv=packed, query=query, key=key, value=value,
                              attended=torch.empty((rows, 1536), device=device, dtype=torch.bfloat16),
                              projected=output)
    positions = torch.arange(rows, device=device, dtype=torch.int32)
    angles = torch.randn((rows, 64), device=device, generator=generator)
    embeddings = torch.cat((angles.cos(), angles.sin()), dim=1)
    cu = torch.tensor([0, rows], device=device, dtype=torch.int32)
    with RecordedPreparationSession(
        device=device, checkpoint="synthetic-moonvit-attention",
        workload_schema="moonvit-attention", numerical_policy="default", autotune_effort="off",
        compile_workers=1,
    ) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = _Attention(weights, scratch, cu, positions, session=session,
                               workspace=arena, name="vision.attention")
        session.tune()
        del packed, query, key, value, scratch
        arena.release_preparation()
        forward = compile_forward(lambda: output.copy_(attention(
            source, cu_seqlens=cu, position_embeddings=embeddings)))
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        for active_rows in (rows, rows // 2, rows):
            source.normal_(generator=generator)
            cu[1] = active_rows
            projected = torch.nn.functional.linear(source, weights["wqkv.weight"])
            reference_qkv = projected.view(rows, 3, 12, 128)
            query, key, value = reference_qkv.unbind(1)

            def rotate(tensor):
                pairs = tensor.float().view(rows, 12, 64, 2)
                cosine = embeddings[:, None, :64]
                sine = embeddings[:, None, 64:]
                even, odd = pairs[..., 0], pairs[..., 1]
                return torch.stack((even * cosine - odd * sine,
                                    odd * cosine + even * sine), dim=-1).flatten(-2).bfloat16()

            query, key = rotate(query), rotate(key)
            sampled = torch.tensor([0, active_rows // 2, active_rows - 1], device=device)
            scores = torch.einsum("qhd,khd->hqk", query[sampled].float(), key[:active_rows].float()) / 128 ** .5
            attended = torch.einsum("hqk,khd->qhd", scores.softmax(-1), value[:active_rows].float()).bfloat16()
            expected = torch.nn.functional.linear(attended.flatten(1), weights["wo.weight"])
            eager = attention(source, cu_seqlens=cu, position_embeddings=embeddings)
            torch.testing.assert_close(eager[sampled].float(), expected.float(), atol=.02, rtol=.02)
            output.fill_(float("nan"))
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output[sampled].float(), expected.float(), atol=.02, rtol=.02)
            torch.testing.assert_close(output.float(), eager.float(), atol=.02, rtol=.02)
        graph.reset()
