# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared trigonometric tables preserve rotary arithmetic and replay lifetimes."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.rotary import CompressedRotary, PreparedRotaryAngles
from ignition.preparation.compile_pool import CompilePool
from ignition.preparation.graphs import PreparedGraphBundle
from tests.tensor_examples import TensorExamples
from ignition.recipes.deepseek4.attention import normalize_rotate

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


class RotaryForward:
    def __init__(self, seed, rotated, normalized, angles, inverse):
        self.seed, self.rotated, self.normalized = seed, rotated, normalized
        self.angles, self.inverse = angles, inverse

    def __call__(self):
        self.rotated.copy_(self.seed)
        self.normalized.copy_(self.seed)
        self.angles()
        self.angles.apply(self.rotated, inverse=self.inverse)
        normalize_rotate(self.normalized, self.angles.positions, self.angles.frequencies, 1e-6,
                         angles=self.angles.values)
        self.angles.values.fill_(float("nan"))


@pytest.mark.parametrize("rows,start,heads,width,inverse", [
    (1, 0, 1, 512, False), (4, 123000, 32, 512, False), (8, 1048560, 32, 512, True),
    (4096, 0, 64, 128, False), (4096, 123000, 32, 512, False), (4, 1048560, 64, 128, True),
])
@torch.no_grad()
def test_shared_angles_are_exact_through_metadata_compilation_and_native_replay(
        rows, start, heads, width, inverse, tmp_path):
    device = torch.device("cuda", 0)
    torch.manual_seed(4919)
    seed = torch.randn((rows, heads, width), device=device, dtype=torch.bfloat16)
    rotated, normalized = torch.empty_like(seed), torch.empty_like(seed)
    positions = torch.arange(start, start + rows, device=device)
    positions[0] = -1
    config = SimpleNamespace(text={"rope_theta":10000., "compress_rope_theta":160000.,
        "rope_scaling":{"original_max_position_embeddings":4096, "beta_fast":32.,
                        "beta_slow":1., "factor":16.}})
    rotary = CompressedRotary(config, device)
    buffers = TensorExamples(device)
    angles = PreparedRotaryAngles(positions, rotary.frequencies[True], buffers=buffers)
    forward = RotaryForward(seed, rotated, normalized, angles, inverse)
    path = tmp_path / "rotary.bin"
    with CompilePool(device) as compiler:
        compiler.submit(forward, path).result()
    with path.open("rb") as file:
        compiled = torch.compiler.load_compiled_function(file)
    compiled(forward)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(forward)
    bundle = PreparedGraphBundle(device=0, graphs={"rotary":graph},
        buffers={"seed":seed, "rotated":rotated, "normalized":normalized, "positions":positions},
        resources=(forward, compiled, buffers, angles))
    worker = bundle.start_worker()
    try:
        for offset in (0, 128, -64):
            positions.add_(offset)
            seed.mul_(-.5)
            expected_rotated, expected_normalized = seed.clone(), seed.clone()
            rotary.apply(expected_rotated, positions.clamp_min(0), compressed=True, inverse=inverse)
            normalize_rotate(expected_normalized, positions.clamp_min(0), rotary.frequencies[True], 1e-6)
            rotated.fill_(float("nan"))
            normalized.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(0, repetitions=3)
            completion = worker.poll(timeout_ms=30000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            torch.testing.assert_close(rotated, expected_rotated, rtol=0, atol=0)
            torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=0)
            assert angles.values.isnan().all()
    finally:
        worker.close()
        graph.reset()
