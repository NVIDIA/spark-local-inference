# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded exact-checkpoint router, shared-expert, and routed-expert components."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from b12x.preparation import PreparationSession
from b12x.preparation.types import require_prepared

from ignition.components.linear import UnquantizedProjection
from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.components.parallel import TensorParallelShard
from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.qwen38 import REVISION, TEXT_PREFIX
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.moe import (
    PreparedQwenMoe,
    PreparedSharedExpert,
    QwenMoeScratch,
    QwenMoeWeights,
    SharedExpertWeights,
)
from ignition.recipes.qwen38.weights import QwenWeightLoader, allocate_expert_scale_scratch
from ignition.recipes.qwen38.tuning import preparation_session
from ignition.reference.qwen38_quant import (
    CheckpointTensors,
    decode_mxfp8,
    nvfp4_expert_reference,
    quantize_mxfp8_activation,
    read_nvfp4_projection,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)


@pytest.fixture(scope="module")
def manifest():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    assert torch.cuda.get_device_capability(0) in ((12, 0), (12, 1))
    default = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    return inspect_qwen38_checkpoint(Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", default)))


def _metrics(actual, expected):
    difference = actual.float().cpu() - expected.float().cpu()
    return {
        "max_absolute_error": difference.abs().max().item(),
        "relative_l2_error": (difference.norm() / expected.float().cpu().norm().clamp_min(1e-12)).item(),
    }


def _assert_values(actual, expected):
    metrics = _metrics(actual, expected)
    assert metrics["relative_l2_error"] < .008, metrics
    torch.testing.assert_close(actual.float().cpu(), expected.float().cpu(),
                               rtol=.015, atol=expected.float().abs().max().item() * .01 + 1e-6)
    return metrics


def _quant_scale_math(experts):
    binding = experts.binding
    if binding.implementation == "micro":
        return "micro"
    assert binding.implementation == "dynamic"
    return "dynamic_fast" if binding.fast_math else "dynamic_precise"


@torch.no_grad()
@pytest.mark.parametrize("combine_routed", [False, True])
@pytest.mark.parametrize("pooled", [False, True])
def test_compiled_shared_gate_preserves_bf16_rounding(combine_routed, pooled):
    from ignition.preparation.compilation import compile_forward

    gates = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    gates = gates[torch.isfinite(gates)].view(-1, 1)
    projected = torch.randn(gates.shape[0], 16, device="cuda", dtype=torch.bfloat16,
                            generator=torch.Generator(device="cuda").manual_seed(581))
    output = torch.empty_like(projected)
    shared = SimpleNamespace(gate_logits=gates, projected=projected, output=output)
    routed = torch.randn_like(projected) if combine_routed else None
    combined = torch.empty_like(projected)
    if pooled:
        from ignition.preparation.scratch import GraphBufferPool

        buffers = GraphBufferPool("cuda:0", chunk_bytes=8 << 20).cursor()
        def store(value):
            return buffers.zeros(value.shape, value.dtype).copy_(value)
        shared.projected, shared.gate_logits, shared.output = map(store, (projected, gates, output))
        combined = store(combined)
        routed = None if routed is None else store(routed)

    def finish():
        return PreparedSharedExpert.finish(shared, routed, out=combined)

    expected = projected * torch.sigmoid(gates)
    if routed is not None:
        expected = routed + expected
    compile_forward(finish)()
    torch.testing.assert_close(combined.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 4, 16])
@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("compiled", [False, True])
def test_actual_router_and_shared_expert_graph(manifest, tokens, world, compiled):
    prefix = f"{TEXT_PREFIX}.layers.0.mlp"
    with torch.no_grad(), QwenWeightLoader(manifest, "cuda:0", prefixes=(
        f"{prefix}.gate.", f"{prefix}.shared_expert.", f"{prefix}.shared_expert_gate.",
    ), shard=TensorParallelShard(0, world)) as loader:
        weights = SharedExpertWeights.load(loader, 0)
        router_weight = loader.load_tensor(f"{prefix}.gate.weight")
    with CheckpointTensors(manifest, byte_budget=16 << 20) as reader:
        reference_weights = {
            projection: (reader.read(f"{prefix}.shared_expert.{projection}.weight"),
                         reader.read(f"{prefix}.shared_expert.{projection}.weight_scale"))
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        reference_gate = reader.read(f"{prefix}.shared_expert_gate.weight")
    source = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device="cuda")
    logits = torch.zeros((tokens, 512), dtype=torch.bfloat16, device="cuda")
    route = RouterScratch.allocate(tokens, source.device, experts=512, top_k=10)
    with torch.no_grad(), PreparationSession(device=source.device, autotune=True, compile_workers=0) as session:
        shared = PreparedSharedExpert(source, weights, session=session, name=f"shared:{tokens}")
        router = UnquantizedProjection(source, router_weight, logits, session=session, name=f"router:{tokens}")
        routing = PreparedRouter(logits, route, session=session, name=f"route:{tokens}")
        session.freeze()

        def project(values, name, projection):
            decoded = decode_mxfp8(*reference_weights[name])
            if name == "down_proj":
                decoded = decoded[:, :640 // world]
            else:
                decoded = decoded[:640 // world]
            mode = require_prepared(projection.plan, projection.plan.contract.component_id).config.mode
            inputs = quantize_mxfp8_activation(values) if mode == "quantized" else values.float()
            return (inputs @ decoded.T).to(torch.bfloat16)

        def run():
            shared(source)
            router(source)
            routing(logits)

        run()
        if compiled:
            from ignition.preparation.compilation import compile_forward

            run = compile_forward(run)
            run()
            torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        generator = torch.Generator().manual_seed(804)
        for iteration in range(2):
            cpu_source = (torch.randn((tokens, 2560), generator=generator) * .2).to(torch.bfloat16)
            source.copy_(cpu_source)
            shared.output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            gate, up = (project(cpu_source, name, shared.up_projection)
                        for name in ("gate_proj", "up_proj"))
            intermediate = F.silu(gate) * up
            projected = project(intermediate, "down_proj", shared.down_projection)
            shared_gate = torch.sigmoid(F.linear(cpu_source, reference_gate))
            expected = projected * shared_gate
            metrics = _assert_values(shared.output, expected)
            expected_ids = logits.shape[-1] - 1 - torch.argsort(
                logits.flip(-1), dim=-1, descending=True, stable=True)[:, :10]
            expected_weights = torch.softmax(logits.float().gather(1, expected_ids), dim=-1)
            assert torch.equal(route.ids, expected_ids)
            torch.testing.assert_close(route.weights, expected_weights, rtol=2e-6, atol=1e-7)
            canonical_logits = F.linear(source, router_weight)
            torch.testing.assert_close(logits, canonical_logits, rtol=.01, atol=.001)
            print(json.dumps({"component": "actual_shared_expert", "tokens": tokens,
                              "iteration": iteration, **metrics}))


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_actual_two_expert_component_reduction_graph(manifest, tokens):
    ids = (0, 511)
    prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts"
    scratch = allocate_expert_scale_scratch("cuda:0", 2)
    with QwenWeightLoader(manifest, "cuda:0", prefixes=tuple(f"{prefix}.{i}." for i in ids)) as loader:
        experts = loader.load_experts(0, expert_ids=ids, scale_scratch=scratch)
    with CheckpointTensors(manifest, byte_budget=16 << 20) as reader:
        projections = [tuple(read_nvfp4_projection(reader, f"{prefix}.{expert}.{name}")
                             for name in ("gate_proj", "up_proj", "down_proj")) for expert in ids]
    source = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device="cuda")
    logits = torch.zeros((tokens, 2), dtype=torch.bfloat16, device="cuda")
    route = RouterScratch.allocate(tokens, source.device, experts=2, top_k=2)
    output = torch.empty_like(source)
    with torch.inference_mode(), PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        routing = PreparedRouter(logits, route, session=session, name=f"two_experts:{tokens}.route")
        prepared = PreparedRoutedExperts(source, experts, route.ids, route.weights, output,
                                        session=session, name=f"two_experts:{tokens}")
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            routing(logits)
            prepared()
        generator = torch.Generator().manual_seed(63)
        for iteration in range(2):
            cpu_source = (torch.randn((tokens, 2560), generator=generator) * .1).to(torch.bfloat16)
            source.copy_(cpu_source)
            logits.copy_(torch.randn((tokens, 2), generator=generator).to(torch.bfloat16))
            for buffer in scratch:
                buffer.view(torch.uint8).fill_(0x7F)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            cpu_ids, cpu_weights = route.ids.cpu(), route.weights.cpu()
            native_sum = torch.zeros_like(cpu_source, dtype=torch.float32)
            canonical_sum = torch.zeros_like(cpu_source)
            for expert in range(2):
                down = nvfp4_expert_reference(cpu_source, *projections[expert],
                    quant_scale_math=_quant_scale_math(prepared)).down
                route_index = (cpu_ids == expert).to(torch.int64).argmax(-1, keepdim=True)
                probability = cpu_weights.gather(1, route_index)
                native_sum += down * probability
                canonical_sum += (down.to(torch.bfloat16) * probability.to(torch.bfloat16))
            metrics = _assert_values(output, native_sum.to(torch.bfloat16))
            canonical_metrics = _metrics(output, canonical_sum)
            assert canonical_metrics["relative_l2_error"] < .015, canonical_metrics
            print(json.dumps({"component": "actual_expert_ids_0_511_top2", "tokens": tokens,
                              "iteration": iteration, "native_reduction": metrics,
                              "canonical_bf16_reduction": canonical_metrics}))


@pytest.fixture(scope="module")
def complete_layer(manifest, request):
    prefix = f"{TEXT_PREFIX}.layers.0.mlp"
    rank, world_size, scale_storage = request.param
    shard = TensorParallelShard(rank, world_size)
    scratch = (allocate_expert_scale_scratch("cuda:0", shard=shard)
               if scale_storage == "compressed" else None)
    with torch.no_grad(), QwenWeightLoader(manifest, "cuda:0", prefixes=(prefix + ".",), shard=shard) as loader:
        weights = QwenMoeWeights.load(loader, 0, scale_scratch=scratch)
        assert loader._session.torch_copy_bytes == 0
    assert weights.experts.num_experts == 512
    assert weights.experts.has_csf_blocks == (scale_storage == "compressed")
    with CheckpointTensors(manifest, byte_budget=16 << 20) as reader:
        shared = {
            projection: (reader.read(f"{prefix}.shared_expert.{projection}.weight"),
                         reader.read(f"{prefix}.shared_expert.{projection}.weight_scale"))
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        shared_gate = reader.read(f"{prefix}.shared_expert_gate.weight")
    return (weights, () if scratch is None else scratch, shared, shared_gate,
            QwenMoeScratch.allocate(16, torch.device("cuda:0"), intermediate=640 // shard.world_size), shard)


@pytest.mark.parametrize("complete_layer,tokens", [
    ((0, 1, "compressed"), 1), ((0, 1, "compressed"), 4), ((0, 1, "compressed"), 16),
    ((0, 2, "expanded"), 3), ((0, 2, "expanded"), 8),
    ((0, 2, "compressed"), 3), ((0, 2, "compressed"), 8),
], indirect=["complete_layer"], scope="module",
    ids=["tp1-m1", "tp1-m4", "tp1-m16", "tp2-m3", "tp2-m8", "tp2-csf-m3", "tp2-csf-m8"])
def test_actual_complete_layer_512_experts_graph(manifest, complete_layer, tokens):
    weights, scale_scratch, shared_weights, shared_gate, moe_scratch, shard = complete_layer
    first, last = shard.partition(640)
    workspace = ScratchArena("cuda:0")
    source = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device="cuda")
    prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts"
    if shard.world_size == 1:
        session = PreparationSession(device=source.device, autotune=False, compile_workers=0)
        gradient_mode = torch.inference_mode
    else:
        session = preparation_session(device=source.device, checkpoint=REVISION, compile_workers=4,
            autotune_effort="default", lock_gpu_clocks=True, headroom_bytes=1 << 30,
            namespace={"config": manifest.metadata_sha256["config.json"],
                       "headers": manifest.headers_sha256, "world_size": shard.world_size})
        session.register_scratch(workspace)
        gradient_mode = torch.no_grad
    with gradient_mode(), session, ExitStack() as resources:
        prepared = PreparedQwenMoe(source, weights, session=session, name=f"complete_moe:{tokens}",
            scratch=moe_scratch, workspace=workspace, projection_workspace=workspace)
        workspace_bytes = workspace.nbytes
        repeated = PreparedQwenMoe(source, weights, session=session, name=f"complete_moe_repeated:{tokens}",
            scratch=moe_scratch, workspace=workspace, projection_workspace=workspace)
        assert workspace.nbytes == workspace_bytes
        assert repeated.output.data_ptr() == prepared.output.data_ptr() == moe_scratch.output.data_ptr()
        assert repeated.shared.gate_up.data_ptr() == prepared.shared.gate_up.data_ptr()
        if shard.world_size == 1:
            workspace.finalize()
            workspace.freeze()
        else:
            session.tune()
            if not scale_scratch:
                assert prepared.experts.binding.implementation == "dynamic"
                assert repeated.experts.binding.implementation == "dynamic"
        assert prepared.experts.scratch.data_ptr() == repeated.experts.scratch.data_ptr() == workspace.tensor.data_ptr()
        for operator in (prepared, repeated):
            for projection in (operator.shared.up_projection, operator.shared.down_projection):
                assert projection.workspace.data_ptr() == workspace.tensor.data_ptr()
        first_output = torch.empty_like(source)
        first_shared = torch.empty_like(source)
        if shard.world_size > 1:
            prepared(source)
            repeated(source)
            torch.cuda.synchronize()
        session.freeze()
        prepared(source)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            workspace.tensor.fill_(0xA5)
            prepared(source)
            first_output.copy_(prepared.output)
            first_shared.copy_(prepared.shared.finish())
            workspace.tensor.fill_(0x5A)
            repeated(source)
            repeated.shared.finish()
        bundle = PreparedGraphBundle(device=0, graphs={"qwen_moe": graph},
            buffers={"source": source, "output": prepared.output},
            resources=(weights, prepared, repeated, session, scale_scratch, moe_scratch, workspace, first_output))
        worker = bundle.start_worker()
        resources.callback(worker.close)
        generator = torch.Generator().manual_seed(1681)
        for iteration in range(2):
            cpu_source = (torch.randn((tokens, 2560), generator=generator) * .1).to(torch.bfloat16)
            source.copy_(cpu_source)
            for buffer in scale_scratch:
                buffer.view(torch.uint8).fill_(0x7F)
            prepared.output.fill_(float("nan"))
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            ticket = worker.submit(0)
            completion = worker.poll(timeout_ms=30000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(first_shared, repeated.shared.output, rtol=0, atol=0)
            cpu_ids = prepared.router.ids.cpu()
            cpu_probabilities = prepared.router.weights.cpu()
            canonical_logits = F.linear(source, weights.router)
            torch.testing.assert_close(prepared.router_logits, canonical_logits, rtol=.01, atol=.001)
            canonical_ids = 511 - torch.argsort(
                prepared.router_logits.flip(-1), dim=-1, descending=True, stable=True)[:, :10]
            assert torch.equal(prepared.router.ids, canonical_ids)
            canonical_probabilities = torch.softmax(
                prepared.router_logits.float().gather(1, canonical_ids), dim=-1)
            torch.testing.assert_close(prepared.router.weights, canonical_probabilities, rtol=2e-6, atol=1e-7)
            native_sum = torch.zeros_like(cpu_source, dtype=torch.float32)
            canonical_sum = torch.zeros_like(cpu_source)
            with CheckpointTensors(manifest, byte_budget=512 << 20) as reader:
                for expert in cpu_ids.unique(sorted=True).tolist():
                    rows, slots = (cpu_ids == expert).nonzero(as_tuple=True)
                    projections = tuple(read_nvfp4_projection(reader, f"{prefix}.{expert}.{name}")
                                        for name in ("gate_proj", "up_proj", "down_proj"))
                    gate, up, down = projections
                    projections = (
                        replace(gate, packed=gate.packed[first:last], scales=gate.scales[first:last]),
                        replace(up, packed=up.packed[first:last], scales=up.scales[first:last]),
                        replace(down, packed=down.packed[:, first // 2:last // 2],
                                scales=down.scales[:, first // 16:last // 16]),
                    )
                    down = nvfp4_expert_reference(cpu_source[rows], *projections,
                        quant_scale_math=_quant_scale_math(prepared.experts)).down
                    probability = cpu_probabilities[rows, slots, None]
                    native_sum[rows] += down * probability
                    canonical_sum[rows] += down.to(torch.bfloat16) * probability.to(torch.bfloat16)
                read_bytes = reader.bytes_read
            def project(values, name, projection):
                mode = require_prepared(projection.plan, projection.plan.contract.component_id).config.mode
                inputs = quantize_mxfp8_activation(values) if mode == "quantized" else values.float()
                decoded = decode_mxfp8(*shared_weights[name])
                decoded = decoded[:, first:last] if name == "down_proj" else decoded[first:last]
                return (inputs @ decoded.T).to(torch.bfloat16)

            gate, up = (project(cpu_source, name, prepared.shared.up_projection)
                        for name in ("gate_proj", "up_proj"))
            intermediate = F.silu(gate) * up
            shared = project(intermediate, "down_proj", prepared.shared.down_projection)
            shared *= torch.sigmoid(F.linear(cpu_source, shared_gate))
            expected_native = native_sum.to(torch.bfloat16) + shared
            expected_canonical = canonical_sum + shared
            metrics = _assert_values(prepared.output, expected_native)
            _assert_values(first_output, expected_native)
            canonical_metrics = _metrics(prepared.output, expected_canonical)
            assert canonical_metrics["relative_l2_error"] < .015, canonical_metrics
            print(json.dumps({"component": "actual_layer0_512_experts_top10", "tokens": tokens,
                              "tp_rank": shard.rank, "tp_size": shard.world_size,
                              "scale_storage": "compressed" if scale_scratch else "expanded",
                              "expert_config": prepared.experts.plan.contract.config_payload(
                                  prepared.experts.plan.selection.config).to_dict(),
                              "iteration": iteration, "selected_experts": cpu_ids.unique().numel(),
                              "reference_payload_bytes": read_bytes, "native_reduction": metrics,
                              "canonical_bf16_reduction": canonical_metrics}))


@torch.no_grad()
def test_actual_packed_a16_decode_and_calibrated_a4_prefill_graphs(manifest):
    from ignition.preparation.compilation import compile_forward
    from ignition.reference.qwen38_quant import nvfp4_linear_reference

    prefix = f"{TEXT_PREFIX}.layers.0.mlp"
    with QwenWeightLoader(manifest, "cuda:0", prefixes=(prefix + ".",),
                          force_moe_decode_a16=True) as loader:
        weights = QwenMoeWeights.load(loader, 0)
    experts = weights.experts
    assert experts.num_experts == 512 and experts.intermediate_size == 640
    assert experts.plan.activation.mode.value == "a16"
    assert experts.plan.activation.a4_prefill is True
    assert experts._impl.plan.quant_modes == {"w4a16"}
    assert weights.experts_a4_prefill and not experts.has_csf_blocks
    representation = experts._impl.representation_for("w4a16")
    assert representation is not None
    input_scale = experts._impl.a1_gscale.cpu().reciprocal().reshape(())
    generator = torch.Generator().manual_seed(713)
    cases = {}
    with PreparationSession(device=experts.device, autotune=False, compile_workers=0) as session:
        for name, tokens, prefill in (("decode", 1, False), ("verify", 128, False), ("prefill", 4096, True)):
            source = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=experts.device)
            operator = PreparedQwenMoe(source, weights, session=session, name=f"packed_precision.{name}",
                                       prefill=prefill)
            cases[name] = SimpleNamespace(source=source, operator=operator, prefill=prefill)
        session.freeze()
        for case in cases.values():
            operator, source = case.operator, case.source
            assert operator.experts.binding.experts is experts._impl
            assert (operator.experts.binding.a4_prefill_launches is not None) is case.prefill
            assert operator.experts.plan.query.a4_prefill is case.prefill
            assert experts._impl.representation_for("w4a16") is representation

            def run(operator=operator, source=source):
                operator(source)

            run()
            case.run = compile_forward(run)
            case.run()
            torch.cuda.synchronize()
            case.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(case.graph):
                case.run()
        for name in ("decode", "verify", "prefill", "verify", "decode", "prefill"):
            case = cases[name]
            cpu_source = (torch.randn(case.source.shape, generator=generator) * .1).to(torch.bfloat16)
            case.source.copy_(cpu_source)
            case.operator.routed_output.fill_(float("nan"))
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            case.graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert torch.isfinite(case.operator.output).all()
            rows = min(3, len(cpu_source))
            ids, probabilities = case.operator.router.ids[:rows].cpu(), case.operator.router.weights[:rows].cpu()
            expected = torch.zeros((rows, 2560), dtype=torch.float64)
            with CheckpointTensors(manifest, byte_budget=128 << 20) as reader:
                for expert in ids.unique(sorted=True).tolist():
                    selected, slots = (ids == expert).nonzero(as_tuple=True)
                    gate, up, down = (read_nvfp4_projection(reader, f"{prefix}.experts.{expert}.{projection}")
                                      for projection in ("gate_proj", "up_proj", "down_proj"))
                    if case.prefill:
                        gate, up = (replace(projection, input_scale=input_scale) for projection in (gate, up))
                    else:
                        gate, up, down = (replace(projection, input_scale=None) for projection in (gate, up, down))
                    values = cpu_source[selected]
                    gate_output = nvfp4_linear_reference(values, gate)
                    up_output = nvfp4_linear_reference(values, up)
                    if case.prefill:
                        gate_output, up_output = (value.to(torch.bfloat16) for value in (gate_output, up_output))
                        intermediate = F.silu(gate_output) * up_output
                    else:
                        intermediate = (F.silu(gate_output) * up_output).to(torch.bfloat16)
                    projected = nvfp4_linear_reference(intermediate, down).to(torch.bfloat16)
                    expected[selected] += projected.double() * probabilities[selected, slots, None].double()
            actual = case.operator.routed_output[:rows].double().cpu()
            relative = ((actual - expected).norm() / expected.norm()).item()
            assert relative < (.004 if case.prefill else .008), (name, relative)
            print(json.dumps({"component": "actual_layer0_packed_precision_512_experts_top10",
                              "role": name, "tokens": len(cpu_source), "reference_rows": rows,
                              "relative_l2_error": relative}))
