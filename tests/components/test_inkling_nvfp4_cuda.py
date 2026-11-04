# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original packed Inkling experts through prepared, compiled NVFP4 A4 graphs."""

import os

import pytest
import torch

from ignition.components.moe import PreparedRoutedExperts
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.inkling.model import mix_routes
from ignition.recipes.inkling.weights import InklingWeightLoader
from tests.recipes.inkling_helpers import checkpoint
from .inkling_nvfp4_helpers import nvfp4_routes

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA GPU")


class ExpertForward:
    def __init__(self, source, experts, ids, weights, live, *, session, workspace):
        self.source, self.ids, self.weights, self.live = source, ids, weights, live
        self.active_ids = torch.empty_like(ids)
        self.routes = [torch.empty((*ids.shape, source.shape[1]), dtype=source.dtype, device=source.device)
                       for _ in experts]
        self.reduced = torch.empty_like(self.routes[0])
        self.output = torch.empty_like(source)
        self.operations = [PreparedRoutedExperts(source, value, self.active_ids, weights, output,
            session=session, workspace=workspace, name=f"experts{rank}")
            for rank, (value, output) in enumerate(zip(experts, self.routes, strict=True))]

    def __call__(self):
        live = torch.arange(self.ids.shape[0], device=self.ids.device)[:, None] < self.live[0]
        self.active_ids.copy_(torch.where(live, self.ids, -1))
        for operation, output in zip(self.operations, self.routes, strict=True):
            operation()
            output.copy_(torch.where(self.active_ids[..., None] >= 0, output, 0.))
        # Both TP shards run on this device to check the intermediate partition
        # and reduce-before-weight arithmetic; this is not a transport test.
        self.reduced.copy_(self.routes[0].float() + self.routes[1].float())
        self.output.copy_(mix_routes(self.reduced, self.weights, self.active_ids))


@pytest.mark.parametrize("tokens", [1, 129, 4096])
@torch.no_grad()
def test_calibrated_packed_experts_tp_shards_compiled_replay(tmp_path, tokens):
    expert_count, top_k = (8, 6) if tokens == 129 else (4, 2)
    config, tensors = checkpoint(tmp_path, moe_intermediate=256, routed_experts=expert_count, top_k=top_k)
    prefix = "model.llm.layers.2.mlp.experts"
    manifest = inspect_indexed_checkpoint(tmp_path)
    experts = []
    def unswizzle(scales, rows, columns):
        # Invert the native 128x4 scale atom independently of the load helper.
        return (scales.view(torch.uint8).reshape(rows // 128, columns // 4, 32, 4, 4)
                .permute(0, 3, 2, 1, 4).contiguous().reshape(rows, columns))

    for rank in range(2):
        with InklingWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(rank, 2)) as loader:
            expert = loader.load_layer(2).experts
        impl = expert._impl
        first, last = rank * 128, (rank + 1) * 128
        original = tensors[prefix + ".w13_weight"]
        expected = torch.cat((original[:, 2*first:2*last:2], original[:, 2*first+1:2*last:2]), 1)
        assert torch.equal(impl.w1_fp4.cpu(), expected)
        scales = tensors[prefix + ".w13_weight.scale"]
        expected_scales = torch.cat((scales[:, 2*first:2*last:2].view(torch.uint8),
                                     scales[:, 2*first+1:2*last:2].view(torch.uint8)), 1)
        for index in range(config.text["n_routed_experts"]):
            actual = unswizzle(impl.w1_blockscale[index], 256, config.hidden // 16)
            assert torch.equal(actual.cpu(), expected_scales[index])
        assert torch.equal(impl.w2_fp4.cpu(), tensors[prefix + ".w2_weight"][:, :, first//2:last//2])
        for index in range(config.text["n_routed_experts"]):
            actual = unswizzle(impl.w2_blockscale[index], config.hidden, 128 // 16)
            assert torch.equal(actual.cpu(),
                tensors[prefix + ".w2_weight.scale"][index, :, first//16:last//16].view(torch.uint8))
        for attribute, alpha, projection in (("a1_gscale", "w1_alphas", "w13_weight"),
                                             ("a2_gscale", "w2_alphas", "w2_weight")):
            amax = tensors[prefix + "." + projection + ".input_amax"].float().item()
            scale = torch.full((config.text["n_routed_experts"],), amax / 2688.).reciprocal()
            assert scale[0] != torch.tensor(2688. / amax)
            assert torch.equal(getattr(impl, attribute).cpu(), scale)
            assert torch.equal(getattr(impl, alpha).cpu(), tensors[prefix + "." + projection + ".scale2"] / scale)
        experts.append(expert)

    torch.manual_seed(4517 + tokens)
    device = torch.device("cuda:0")
    source = torch.randn((tokens, config.hidden), device=device, dtype=torch.bfloat16)
    offsets = torch.arange(top_k, device=device) * (1 if top_k == 6 else 2)
    ids = ((torch.arange(tokens, device=device)[:, None] + offsets) % expert_count).int()
    # Distinct route weights include the router's global gain; kernels must leave
    # every expert output unweighted until the TP shards have been combined.
    weights = torch.rand((tokens, top_k), device=device) * .14234226942062378 * 8.
    live = torch.tensor([tokens], dtype=torch.int32, device=device)
    with RecordedPreparationSession(device=device, checkpoint="inkling-nvfp4-experts",
            workload_schema="inkling", numerical_policy="default", compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        forward = ExpertForward(source, experts, ids, weights, live, session=session, workspace=arena)
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
            if tokens > 1:
                ids[0, 1] = -1
            active = torch.where(torch.arange(tokens, device=device)[:, None] < count, ids, -1)
            expected = [nvfp4_routes(source, active, tensors, prefix, intermediate_range=(rank*128, (rank+1)*128))
                        for rank in range(2)]
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            for actual, reference in zip(forward.routes, expected, strict=True):
                # This reduced checkpoint is exactly repeatable across the
                # captured capacities and the reference's BF16 boundaries.
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            # Test reduction and routing separately from quantized GEMM error.
            reduced = (forward.routes[0].float() + forward.routes[1].float()).bfloat16()
            result = torch.zeros_like(source)
            for expert_id in range(expert_count):
                route = ((reduced.float() * weights[..., None])
                         * (active == expert_id)[..., None]).sum(1).bfloat16()
                result += route
            assert torch.equal(forward.output, result)
            assert not forward.output[count:].count_nonzero()
        graph.reset()
