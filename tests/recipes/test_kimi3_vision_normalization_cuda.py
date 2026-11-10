"""Prepared vision normalization preserves BF16 image features with local output."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.recipes.kimi3.vision import VisionScratch, _RMSNorm


def test_prepared_operator_examples_are_disjoint(monkeypatch):
    from types import SimpleNamespace
    from ignition.recipes.kimi3 import vision

    operands = []

    class Projection:
        def __init__(self, source, weight, output, **kwargs):
            operands.append((kwargs["name"], (source, output)))

        def functional(self, **kwargs):
            return self

    class Attention(Projection):
        def __init__(self, query, key, value, cu, output, **kwargs):
            operands.append((kwargs["name"], (query, key, value, output)))

    monkeypatch.setattr(vision, "UnquantizedProjection", Projection)
    monkeypatch.setattr(vision, "PreparedTorchRMSNorm", Projection)
    monkeypatch.setattr(vision, "PreparedVarlenAttention", Attention)
    monkeypatch.setattr(vision, "prepare_cached_rotary", lambda session: None)
    names = ["vision_tower.patch_embed.pos_emb.weight", "vision_tower.patch_embed.proj.weight",
             "vision_tower.encoder.final_layernorm.weight", "mm_projector.proj.0.weight",
             "mm_projector.proj.2.weight", "mm_projector.post_norm.weight"]
    for layer in range(27):
        names.extend(f"vision_tower.encoder.blocks.{layer}.{name}" for name in (
            "norm0.weight", "norm1.weight", "wqkv.weight", "wo.weight", "mlp.fc0.weight", "mlp.fc1.weight"))
    weights = SimpleNamespace(tensors={name: torch.empty((1, 1024), device="meta") for name in names})
    examples = {}
    for capacity in (256, 128):
        scratch = VisionScratch.allocate(capacity, "cpu", examples=examples)
        vision.PreparedVision(weights, scratch, None, session=None, workspace=None, name=f"vision.{capacity}")
    assert operands
    for name, tensors in operands:
        spans = sorted((tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes) for tensor in tensors)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), name
    assert len({tensor.untyped_storage().data_ptr() for tensor in examples.values()}) == 4


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned CUDA GPU",
)
@pytest.mark.parametrize(
    "rows,width,eps", [(256, 1024, None), (65536, 1024, None), (16384, 7168, 1e-5)]
)
@pytest.mark.parametrize("inplace", [False, True])
@torch.no_grad()
def test_prepared_norm_is_exact_under_compiled_replay(rows, width, eps, inplace):
    from ignition.preparation.compilation import compile_forward

    generator = torch.Generator(device="cuda").manual_seed(315)
    source = torch.randn(
        (rows, width), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(
        width, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    output = source if inplace else torch.empty_like(source)
    with PreparationSession(device=source.device, autotune=False) as session:
        operation = _RMSNorm(
            weight, source, output, eps, session=session, name="vision.norm"
        )
        forward = compile_forward(lambda: output.copy_(operation(source)))
        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        for scale in (0.0, 0.001, 1.0, 1000.0):
            source.copy_(
                torch.randn(source.shape, device="cuda", generator=generator) * scale
            )
            expected = torch.nn.functional.rms_norm(source, [width], weight, eps)
            if not inplace:
                output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
        graph.reset()
