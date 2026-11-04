# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original Inkling W4A4 weights through SM103 grouped GEMMs and captured raw routes."""

import os

import pytest
import torch

from ignition.components.moe import PreparedRoutedExperts
from ignition.components.moe_sm103 import CuteNvfp4Experts
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.inkling.model import mix_routes
from ignition.recipes.inkling.weights import InklingWeightLoader
from tests.recipes.inkling_helpers import checkpoint
from .inkling_nvfp4_helpers import nvfp4_routes, quantize_activation, unpack_weight

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned SM103 GPU")


@pytest.mark.parametrize("amax", [1.8359375, 187., 4.])
def test_sm103_activation_calibration_and_fp4_ties(amax):
    from ignition.components.moe_sm103 import _fp4_quantizer

    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("SM103 NVFP4 quantizer")
    generator = torch.Generator(device="cuda").manual_seed(103)
    source = torch.randn((4, 256), generator=generator, device="cuda", dtype=torch.bfloat16)
    source[0].fill_(1.6875)
    source[0, 0] = 1.
    source[1].zero_()
    amax = torch.tensor([amax], device="cuda", dtype=torch.bfloat16)
    reference, scale = quantize_activation(source, amax, quant_scale_math="flashinfer")
    values = torch.empty((4, 128), device="cuda", dtype=torch.uint8)
    scales = torch.empty(16 * 16, device="cuda", dtype=torch.uint8)
    _fp4_quantizer().fp4_quantize(source, scale.reshape(1), values, scales,
                                 16, False, False, False, False, True, 0)
    actual = unpack_weight(values, scales[:64].view(torch.float8_e4m3fn).reshape(4, 16))
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


class RawRoutes:
    def __init__(self, source, experts, ids, weights, live, session, workspace):
        self.source, self.ids, self.weights, self.live = source, ids, weights, live
        self.active_ids = torch.empty_like(ids)
        self.routes = torch.empty((*ids.shape, source.shape[1]), dtype=source.dtype, device=source.device)
        self.output = torch.empty_like(source)
        self.experts = PreparedRoutedExperts(source, experts, self.active_ids, weights, self.routes,
            session=session, workspace=workspace, name="nvfp4")

    def __call__(self):
        live = torch.arange(self.ids.shape[0], device=self.ids.device)[:, None] < self.live[0]
        self.active_ids.copy_(torch.where(live, self.ids, -1))
        self.experts()
        self.output.copy_(mix_routes(self.routes, self.weights, self.active_ids))


@pytest.mark.parametrize("tokens", [1, 129, 4096])
@torch.no_grad()
def test_sm103_calibrated_raw_routes_compiled_replay(tmp_path, tokens):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("SM103 grouped NVFP4 execution")
    config, tensors = checkpoint(tmp_path, moe_intermediate=256, routed_experts=8, top_k=6)
    prefix = "model.llm.layers.2.mlp.experts"
    manifest = inspect_indexed_checkpoint(tmp_path)
    with InklingWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1)) as loader:
        experts = loader.load_layer(2).experts
    assert isinstance(experts, CuteNvfp4Experts)

    # Native CuTe FC1 is [up64;gate64], derived only by row permutation.
    def interleave(tensor):
        up_gate = torch.stack((tensor[:, 1::2], tensor[:, ::2]), 2)
        return up_gate.reshape(8, 4, 64, 2, -1).transpose(2, 3).reshape_as(tensor)

    assert torch.equal(experts.w13.cpu(), interleave(tensors[prefix + ".w13_weight"]))
    assert torch.equal(experts.w2.cpu(), tensors[prefix + ".w2_weight"])
    for storage, name, rows, columns, reorder in (
        (experts.scale_storage[0], "w13_weight", 512, 16, interleave),
        (experts.scale_storage[1], "w2_weight", 256, 16, lambda x: x),
    ):
        actual = (storage.cpu().view(torch.uint8).reshape(8, rows // 128, columns // 4, 32, 4, 4)
                  .permute(0, 1, 4, 3, 2, 5).contiguous().reshape(8, rows, columns))
        assert torch.equal(actual, reorder(tensors[prefix + "." + name + ".scale"].view(torch.uint8)))
    for name, calibration, alpha in (
        ("w13_weight", experts.input_global_scale, experts.w13_alpha),
        ("w2_weight", experts.intermediate_global_scale, experts.w2_alpha),
    ):
        amax = float(tensors[prefix + "." + name + ".input_amax"].item())
        scale = torch.tensor([amax / 2688.], dtype=torch.float32).reciprocal()
        assert torch.equal(calibration.cpu(), scale)
        assert torch.equal(alpha.cpu(), tensors[prefix + "." + name + ".scale2"] / scale)

    torch.manual_seed(4517 + tokens)
    device = torch.device("cuda:0")
    source = torch.randn((tokens, config.hidden), device=device, dtype=torch.bfloat16)
    ids = ((torch.arange(tokens, device=device)[:, None] + torch.arange(6, device=device)) % 8).int()
    weights = torch.rand((tokens, 6), device=device) * 1.1387381553649902
    live = torch.tensor([tokens], dtype=torch.int32, device=device)
    with RecordedPreparationSession(device=device, checkpoint="inkling-nvfp4-sm103",
            workload_schema="inkling", numerical_policy="default", compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        forward = RawRoutes(source, experts, ids, weights, live, session, arena)
        session.tune()
        run = compile_forward(forward)
        run()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        session.freeze()
        for count, factor in ((tokens, 1.), (max(1, tokens // 2), .3125), (0, 1.), (tokens, -1.5)):
            live.fill_(count)
            source.mul_(factor)
            ids[0, 1] = -1
            active = torch.where(torch.arange(tokens, device=device)[:, None] < count, ids, -1)
            expected = nvfp4_routes(source, active, tensors, prefix, quant_scale_math="flashinfer")
            forward.routes.fill_(float("nan"))
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            torch.testing.assert_close(forward.routes, expected, atol=0, rtol=0)
            before = forward.routes.clone()
            weights.mul_(.375)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(forward.routes, before)
            result = torch.zeros_like(source)
            for expert in range(8):
                route = ((forward.routes.float() * weights[..., None])
                         * (active == expert)[..., None]).sum(1).bfloat16()
                result += route
            assert torch.equal(forward.output, result)
            assert not forward.routes[active < 0].count_nonzero()
        graph.reset()


@pytest.mark.skipif(not os.environ.get("FLARE_INKLING_SNAPSHOT"), reason="requires the official NVFP4 checkpoint")
@torch.no_grad()
def test_original_checkpoint_sm103_production_geometry():
    from pathlib import Path
    from safetensors import safe_open

    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("SM103 grouped NVFP4 execution")
    manifest = inspect_indexed_checkpoint(Path(os.environ["FLARE_INKLING_SNAPSHOT"]))
    with InklingWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1)) as loader:
        layer = loader.config.quantized_routed_layers[0]
        prefix = f"model.llm.layers.{layer}.mlp.experts"
        experts = loader._nvfp4_experts(prefix, loader.config.text["moe_intermediate_size"])
    assert (experts.num_experts, experts.hidden_size, experts.intermediate_size) == (256, 4096, 2048)
    selected = [255, 7, 33, 100, 127, 0]
    original = {}
    # Read original bytes through safetensors, independently of the recipe's packed layout.
    for projection in ("w13_weight", "w2_weight"):
        for suffix in ("", ".scale", ".scale2", ".input_amax"):
            name = prefix + "." + projection + suffix
            with safe_open(manifest.snapshot / manifest.tensors[name].shard, framework="pt", device="cpu") as reader:
                if suffix in ("", ".scale"):
                    original[name] = torch.stack([reader.get_slice(name)[expert] for expert in selected])
                elif suffix == ".scale2":
                    original[name] = reader.get_tensor(name)[selected]
                else:
                    original[name] = reader.get_tensor(name)

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(103276)
    source = torch.randn((1, 4096), generator=generator, device=device, dtype=torch.bfloat16)
    ids = torch.tensor([selected], dtype=torch.int32, device=device)
    weights = torch.tensor([[.7, .1, .6, .9, .3, .5]], dtype=torch.float32, device=device)
    live = torch.ones(1, dtype=torch.int32, device=device)
    reference_ids = torch.arange(6, device=device, dtype=torch.int32)[None]
    with RecordedPreparationSession(device=device, checkpoint="inkling-nvfp4-sm103-production-geometry",
            workload_schema="inkling", numerical_policy="default", compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        forward = RawRoutes(source, experts, ids, weights, live, session, arena)
        session.tune()
        run = compile_forward(forward)
        run()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        session.freeze()
        reference = nvfp4_routes(source, reference_ids, original, prefix, quant_scale_math="flashinfer")
        assert reference.count_nonzero()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        difference = forward.routes.float() - reference.float()
        print({"max_absolute_error": difference.abs().max().item(),
               "relative_l2_error": (difference.norm() / reference.float().norm().clamp_min(1e-12)).item()})
        torch.testing.assert_close(forward.routes, reference, atol=0, rtol=0)
        live.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert not forward.routes.count_nonzero()
        graph.reset()
