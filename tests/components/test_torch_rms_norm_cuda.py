"""Torch-compatible RMSNorm retains exact bits and real compiled graph mutations."""

import os

import pytest
import torch

from b12x.norm import torch_rmsnorm
from b12x.preparation import PreparationSession
from ignition.components.rms_norm import PreparedTorchRMSNorm
from ignition.preparation.compilation import compile_forward


@pytest.mark.parametrize("width", [0, -4, 3, True, 2**24 + 4])
def test_query_rejects_unsupported_vectorized_geometry(width):
    with pytest.raises(ValueError, match="multiple of four"):
        torch_rmsnorm.RMSNormQuery(hidden_size=width)


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU"
)
@pytest.mark.parametrize(
    "rows,width",
    [(1, 1024), (17, 1024), (256, 1024), (65536, 1024), (17, 7168), (16384, 7168)],
)
@pytest.mark.parametrize("inplace", [False, True])
@torch.no_grad()
def test_compiled_graph_matches_native_torch(rows, width, inplace):
    torch.cuda.set_device(0)
    rng = torch.Generator(device="cuda").manual_seed(9241)
    source = torch.randn(
        (rows, width), device="cuda", dtype=torch.bfloat16, generator=rng
    )
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16, generator=rng)
    output = source if inplace else torch.empty_like(source)
    epsilon = None if width == 1024 else 1e-5
    with PreparationSession(device=source.device, autotune=False) as session:
        operation = PreparedTorchRMSNorm(
            source, weight, output, eps=epsilon, session=session, name="norm"
        )
        forward = compile_forward(lambda: operation(source))
        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        for case in ("zero", "small", "normal", "large", "exponents", "signed_zero"):
            values = torch.randn(source.shape, device="cuda", generator=rng)
            if case == "zero":
                values.zero_()
            elif case == "small":
                values.mul_(0.001)
            elif case == "large":
                values.mul_(1000)
            elif case == "exponents":
                exponent = torch.arange(width, device="cuda").remainder(37) - 18
                values.mul_(torch.exp2(exponent.float()))
            elif case == "signed_zero":
                values.zero_()
                values[:, ::2].mul_(-1.0)
            source.copy_(values)
            expected = torch.nn.functional.rms_norm(source, (width,), weight, epsilon)
            if not inplace:
                output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert bool(torch.isfinite(output).all())
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16)), (
                case
            )
        graph.reset()


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU"
)
def test_partial_output_alias_and_affine_weight_overlap_are_rejected():
    storage = torch.zeros(2 * 1024 + 4, dtype=torch.bfloat16, device="cuda")
    source, shifted = storage[:-4].view(2, 1024), storage[4:].view(2, 1024)
    weight = torch.ones(1024, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="not partially"):
        torch_rmsnorm.query_from_call(source, weight, out=shifted, eps=1e-5)
    with pytest.raises(ValueError, match="overlap weights"):
        torch_rmsnorm.query_from_call(source, source[0], out=source, eps=1e-5)
