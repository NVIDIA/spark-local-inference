# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original Inkling W4A4 weights through SM103 grouped GEMMs and captured raw routes."""

import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from ignition.components.moe import PreparedRoutedExperts
from ignition.components.moe_sm103 import CuteNvfp4Experts, _fp4_quantizer
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.inkling.model import mix_routes
from ignition.recipes.inkling.weights import InklingWeightLoader
from ignition.reference.qwen38_quant import decode_nvfp4
from tests.recipes.inkling_helpers import checkpoint
from .inkling_nvfp4_helpers import nvfp4_routes, quantize_activation

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned SM103 GPU")


@pytest.mark.parametrize("amax", [1.8359375, 187., 4.])
def test_sm103_activation_calibration_and_fp4_ties(amax):
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
    actual = decode_nvfp4(values, scales[:64].view(torch.float8_e4m3fn).reshape(4, 16), torch.ones((), device="cuda"))
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
        torch.testing.assert_close(forward.routes, reference, atol=0, rtol=0)
        live.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert not forward.routes.count_nonzero()
        graph.reset()
