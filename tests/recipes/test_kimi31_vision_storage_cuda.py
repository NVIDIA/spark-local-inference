# SPDX-License-Identifier: Apache-2.0
"""Complete BF16 MoonViT replay preserves features after declaration storage is poisoned."""

import os

import pytest
import torch

from .test_kimi31 import CHECKPOINT
from .kimi_vision_support import VisionCall


pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA GPU")


class _Features:
    def __init__(self, call, pixels, output):
        self.call, self.pixels, self.output = call, pixels, output

    def __call__(self):
        self.call.patches.copy_(self.pixels)
        self.output.copy_(self.call())


@pytest.mark.parametrize("side", [16, 64, 256])
@torch.no_grad()
def test_full_encoder_matches_torch_norm_after_scratch_reuse(side):
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.vision import PreparedVision, VisionScratch, VisionWeights

    if not (CHECKPOINT / "config.json").is_file():
        pytest.skip("Kimi K3.1 checkpoint is not installed")
    device, rows = torch.device("cuda:0"), side * side
    torch.cuda.set_device(device)
    weights = VisionWeights.load(inspect_indexed_checkpoint(CHECKPOINT), device)
    scratch = VisionScratch.allocate(rows, device)
    pixels = torch.empty_like(scratch.patches)
    output = torch.empty((rows // 4, 7168), device=device, dtype=torch.bfloat16)
    with RecordedPreparationSession(device=device, checkpoint=str(CHECKPOINT),
            workload_schema="kimi31-vision-storage", numerical_policy="bf16-torch-rmsnorm",
            autotune_effort="off", compile_workers=8) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        encoder = PreparedVision(weights, scratch, None, session=session, workspace=arena,
                                 name="vision")
        call = VisionCall(encoder, device)
        positions = torch.cartesian_prod(torch.arange(side), torch.arange(side)).to(device)
        mapped = ((positions.float() + .5) * (64 / side) - .5).clamp_min(0)
        base, fractions = mapped.long(), mapped - mapped.long()
        upper = (base + 1).clamp_max(63)
        fy, fx = fractions.unbind(1)
        call.positions.copy_(positions)
        call.interpolation_indices.copy_(torch.stack((base[:, 0]*64+base[:, 1],
            base[:, 0]*64+upper[:, 1], upper[:, 0]*64+base[:, 1], upper[:, 0]*64+upper[:, 1]), 1))
        call.interpolation_weights.copy_(torch.stack(((1-fy)*(1-fx), (1-fy)*fx, fy*(1-fx), fy*fx), 1))
        call.merge_indices.copy_(torch.arange(rows, device=device).view(side//2, 2, side//2, 2)
                                   .permute(0, 2, 1, 3).reshape(-1, 4))
        call.cu.copy_(torch.tensor([0, rows], device=device, dtype=torch.int32))
        session.tune(show_progress=False)
        forward = _Features(call, pixels, output)
        saved = [(obj, field, getattr(obj, field)) for obj, field in
                 [(encoder, "norm"), (encoder, "post_norm"),
                  *((block, field) for block in encoder.blocks for field in ("norm1", "norm2"))]]
        samples = []
        generator = torch.Generator(device=device).manual_seed(3321)
        for obj, field, operation in saved:
            with torch.device("meta"):
                reference = torch.nn.RMSNorm(operation.operation.weight.numel(), eps=operation.operation.eps)
            reference.weight = torch.nn.Parameter(operation.operation.weight, requires_grad=False)
            setattr(obj, field, reference)
        for _ in range(2):
            pixels.normal_(generator=generator)
            forward()
            samples.append((pixels.clone(), output.cpu()))
        for obj, field, operation in saved:
            setattr(obj, field, operation)
        compiled = compile_forward(forward)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        try:
            for source, expected in (*samples, samples[0]):
                for name in scratch.__dataclass_fields__:
                    getattr(scratch, name).fill_(float("nan"))
                pixels.copy_(source)
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))
        finally:
            graph.reset()
