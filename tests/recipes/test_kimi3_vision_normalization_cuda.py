"""Prepared vision normalization preserves BF16 image features and scratch aliases."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.recipes.kimi3.vision import VisionScratch, _RMSNorm


def test_normalization_reuses_attention_storage_but_not_projection_inputs():
    scratch = VisionScratch.allocate(256, "cpu")
    assert scratch.normalized.shape == (256, 1024)
    assert scratch.normalized.data_ptr() == scratch.query.data_ptr()
    assert (
        scratch.normalized.untyped_storage().data_ptr()
        == scratch.value.untyped_storage().data_ptr()
    )
    assert scratch.normalized.data_ptr() != scratch.qkv.data_ptr()
    assert scratch.normalized.data_ptr() != scratch.hidden.data_ptr()
    assert scratch.rows(128).normalized.shape == (128, 1024)


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
        forward = compile_forward(lambda: operation(source))
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
