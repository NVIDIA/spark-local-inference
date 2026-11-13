# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint MoE arithmetic across eight sequential intermediate shards on one assigned GPU."""

from contextlib import ExitStack
from pathlib import Path
import gc
import os

import pytest
import torch
from safetensors import safe_open

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned CUDA_VISIBLE_DEVICES")

CHECKPOINT = Path('/data/cache/huggingface/hub/models--meta-models--Muse-Spark-Open-816B-A42B-NVFP4/snapshots/4239e4db19376904f2e82c167e78f70abb5455fe')


def metrics(actual, expected):
    error = actual.float() - expected.float()
    return dict(max_absolute_error=error.abs().max().item(),
        relative_l2_error=(error.norm() / expected.float().norm().clamp_min(1e-12)).item())


class CheckpointSlices:
    def __init__(self, manifest, stack):
        self.manifest, self.stack, self.files = manifest, stack, {}

    def read(self, name, extent=None):
        metadata = self.manifest.tensors[name]
        if metadata.shard not in self.files:
            self.files[metadata.shard] = self.stack.enter_context(safe_open(
                self.manifest.snapshot / metadata.shard, framework="pt", device="cpu"))
        file = self.files[metadata.shard]
        value = file.get_tensor(name) if extent is None else file.get_slice(name)[extent]
        return value.cuda()

    def expert(self, prefix, expert, *, rank=None, quantized=False, source=None, scale_math=None):
        from ignition.reference.qwen38_quant import Nvfp4Projection, nvfp4_expert_reference

        first, last = (0, 4096) if rank is None else (rank * 512, (rank + 1) * 512)
        if not quantized:
            gate = self.read(prefix + ".gate_up_proj", (expert, slice(first, last), slice(None)))
            up = self.read(prefix + ".gate_up_proj", (expert, slice(4096 + first, 4096 + last), slice(None)))
            down = self.read(prefix + ".down_proj", (expert, slice(None), slice(first, last)))
            return torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(source, gate))
                                              * torch.nn.functional.linear(source, up), down)
        gains = self.read(prefix + ".gate_up_proj_weight_scale_2", expert)
        activations = self.read(prefix + ".gate_up_proj_input_scale", expert)
        def projection(name, rows, columns, gain, activation):
            return Nvfp4Projection(self.read(prefix + "." + name, (expert, rows, columns)),
                self.read(prefix + "." + name + "_weight_scale", (expert, rows,
                    slice(columns.start // 8 if columns.start is not None else None,
                          columns.stop // 8 if columns.stop is not None else None))), gain, activation)
        gate = projection("gate_up_proj", slice(first, last), slice(None), gains[0], activations[0])
        up = projection("gate_up_proj", slice(4096 + first, 4096 + last), slice(None), gains[1], activations[1])
        down = projection("down_proj", slice(None), slice(first // 2, last // 2),
            self.read(prefix + ".down_proj_weight_scale_2", expert),
            self.read(prefix + ".down_proj_input_scale", expert))
        return nvfp4_expert_reference(source, gate, up, down, quant_scale_math=scale_math).down.bfloat16()


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("tokens,valid", [(1, 1), (17, 13)])
@torch.no_grad()
def test_checkpoint_moe_e256_tp8_arithmetic(layer, tokens, valid):
    from b12x.preparation import PreparationSession
    from ignition.components.bf16_experts import PreparedBF16Experts
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.route_finalizer import PreparedRouteFinalizer
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.muse_spark.model import norm
    from ignition.recipes.muse_spark.weights import BF16Experts, MuseSparkWeightLoader

    checkpoint = Path(os.environ.get("FLARE_MUSE_SPARK_CHECKPOINT", CHECKPOINT))
    if not checkpoint.is_dir():
        pytest.skip("Muse Spark checkpoint is unavailable")
    manifest = inspect_indexed_checkpoint(checkpoint)
    prefix = f"model.language_model.layers.{layer}.mlp"
    torch.manual_seed(1703)
    with ExitStack() as files:
        reader = CheckpointSlices(manifest, files)
        hidden = torch.randn((tokens, 8192), device="cuda", dtype=torch.bfloat16)
        pre = reader.read(prefix + ".pre_expert_proj.weight")
        pre_norm = reader.read(prefix + ".pre_expert_norm.weight")
        source = norm("centered", pre_norm, eps=1e-5)(torch.nn.functional.linear(hidden, pre))
        logits = torch.nn.functional.linear(hidden.float(), reader.read(prefix + ".gate.weight")) * .17677669529663687
        probabilities = logits.sigmoid()
        ids = (probabilities + reader.read(prefix + ".gate.e_score_correction_bias")).topk(8, dim=-1, sorted=False).indices.int()
        weights = probabilities.gather(1, ids.long())
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-15)
        ids[valid:], weights[valid:] = -1, 0
        count = torch.tensor([valid], device="cuda", dtype=torch.int32)
        reduced = torch.zeros((tokens, 8, 4096), device="cuda", dtype=torch.float32)
        raw_reference = torch.zeros_like(reduced, dtype=torch.bfloat16)
        quantized = layer == 1
        scale_math = "dynamic_fast"
        for rank in range(8):
            with MuseSparkWeightLoader(manifest, torch.device("cuda:0"), shard=TensorParallelShard(rank, 8),
                    prefixes=(prefix + ".",)) as loader:
                experts = loader.load_experts(layer)
            output = torch.empty_like(raw_reference)
            with ExitStack() as resources:
                if isinstance(experts, BF16Experts):
                    op = PreparedBF16Experts(source, experts.gate_up, experts.down, ids, weights, output, count,
                        buffers=GraphBufferPool("cuda").cursor())
                else:
                    session = resources.enter_context(PreparationSession(device="cuda:0", autotune=False, compile_workers=0))
                    arena = ScratchArena("cuda")
                    op = PreparedRoutedExperts(source, experts, ids, weights, output, session=session,
                        name="routes", workspace=arena)
                    arena.finalize()
                    scale_math = "dynamic_fast" if op.binding.fast_math else "dynamic_precise"
                op()
                run = compile_forward(op) if rank == 0 else op
                run()
                if quantized:
                    session.freeze()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
                torch.cuda.synchronize()
                saved = output.clone()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                repeat = metrics(output, saved)
                assert repeat["relative_l2_error"] < .004, repeat
                if rank == 0:
                    local = torch.zeros_like(output)
                    for expert in ids[ids >= 0].unique().tolist():
                        rows, slots = torch.where(ids == expert)
                        local[rows, slots] = reader.expert(prefix + ".experts", expert, rank=rank,
                            quantized=quantized, source=source[rows], scale_math=scale_math)
                    error = metrics(output, local)
                    print("local", layer, tokens, error, "repeat", repeat)
                    assert error["relative_l2_error"] < .008, error
                reduced.add_(output.float())
                graph.reset()
            del op, experts, output, graph, run, saved
            gc.collect()
            torch.cuda.empty_cache()
        for expert in ids[ids >= 0].unique().tolist():
            rows, slots = torch.where(ids == expert)
            raw_reference[rows, slots] = reader.expert(prefix + ".experts", expert, quantized=quantized,
                source=source[rows], scale_math=scale_math)
        raw_error = metrics(reduced.bfloat16(), raw_reference)
        assert raw_error["relative_l2_error"] < .008, raw_error
        gain = reader.read(prefix + ".experts.post_expert_norm.weight")
        actual = torch.empty_like(source)
        finalize = PreparedRouteFinalizer(reduced.bfloat16(), gain, weights, ids, count, actual,
                                          eps=1e-8, num_experts=256)
        finalize()
        contributions = (norm("input_scaled", gain, eps=1e-8)(raw_reference) * weights[..., None]).bfloat16()
        expected = torch.zeros_like(actual)
        order = ids.argsort(-1)
        for slot in range(8):
            expected.add_(contributions.gather(1, order[:, slot, None, None].expand(-1, 1, 4096)).squeeze(1))
        post = reader.read(prefix + ".post_expert_proj.weight")
        actual, expected = torch.nn.functional.linear(actual, post), torch.nn.functional.linear(expected, post)
        final_error = metrics(actual, expected)
        print("MoE", layer, tokens, "raw", raw_error, "final", final_error)
        assert final_error["relative_l2_error"] < .006, final_error
        assert torch.count_nonzero(actual[valid:]) == 0
