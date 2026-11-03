"""Stored MXFP8 projection grouping for the dense QAD checkpoint."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes import resolve
from ignition.recipes.qwen38_dense.config import TEXT_PREFIX
from ignition.reference.qwen38_quant import CheckpointTensors

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU")


@pytest.mark.parametrize("names", [("in_proj_qkv", "in_proj_z"), ("in_proj_b", "in_proj_a"),
                                 ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")])
def test_qad_packed_groups_preserve_serialized_rows(names):
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshot = resolve("qwen3.8-27b-qad").snapshot(hub)
    manifest = inspect_indexed_checkpoint(snapshot)
    prefixes = tuple(f"{TEXT_PREFIX}.layers.0.linear_attn.{name}" for name in names)
    with DirectWeightLoader(manifest, "cuda:0", prefixes=prefixes) as loader:
        packed = loader.load_linear_group(prefixes)
    values = packed.weight.values.view(torch.uint8).cpu().reshape(packed.out_features, packed.padded_in_features)
    scales = packed.weight.scale_rows.view(torch.uint8).cpu().reshape(packed.out_features, -1)
    offset = 0
    with CheckpointTensors(manifest, byte_budget=128 << 20) as reader:
        for prefix in prefixes:
            original = reader.read(prefix + ".weight").view(torch.uint8)
            original_scales = reader.read(prefix + ".weight_scale").view(torch.uint8)
            rows, columns = original.shape
            assert torch.equal(values[offset:offset + rows, :columns], original)
            assert torch.equal(scales[offset:offset + rows, :columns // 32], original_scales[:rows, :columns // 32])
            offset += rows
    assert offset == packed.out_features


@pytest.mark.parametrize("rows", [1, 16, 128, 4096])
@torch.no_grad()
def test_qad_combined_gdn_projection_captured_numerics(rows):
    from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch, dequantize_mxfp8_rows_torch
    from ignition.components.linear import QuantizedProjection
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshot = resolve("qwen3.8-27b-qad").snapshot(hub)
    manifest = inspect_indexed_checkpoint(snapshot)
    names = tuple(f"{TEXT_PREFIX}.layers.0.linear_attn.in_proj_{part}" for part in ("qkv", "z", "b", "a"))
    with DirectWeightLoader(manifest, "cuda:0", prefixes=names) as loader:
        weight = loader.load_linear_group(names)
    source = torch.empty((rows, weight.in_features), device="cuda:0", dtype=torch.bfloat16)
    output = torch.empty((rows, weight.out_features), device="cuda:0", dtype=torch.bfloat16)
    decoded = (weight.weight.values.reshape(weight.out_features, weight.padded_in_features).float()
               * torch.exp2(weight.weight.scale_rows.reshape(weight.out_features, -1).view(torch.uint8).float()
                            - 127).repeat_interleave(32, dim=1))
    with RecordedPreparationSession(device="cuda:0", checkpoint="qad-combined-gdn-projection",
            workload_schema="mxfp8-gdn-inputs", numerical_policy="default", autotune_effort="default") as session:
        arena = ScratchArena("cuda:0")
        session.register_scratch(arena)
        projection = QuantizedProjection(source, weight, output, session=session, name="combined",
                                         workspace=arena, activation_mode="auto",
                                         allow_bf16_atomic_reduction=False)
        assert projection.plan.query.allow_bf16_atomic_reduction is False
        session.tune(show_progress=False)
        selected = projection.plan.selection.config
        print(dict(rows=rows, config=projection.plan.contract.config_payload(selected).to_dict()), flush=True)
        compiled = compile_forward(lambda: projection(source))
        source.normal_()
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for factor in (.5, 1., 1.5):
            source.normal_(0, factor)
            source[:, ::13].zero_()
            if rows > 1:
                source[-1].zero_()
            if selected.mode == "quantized":
                encoded = quantize_mxfp8_rows_torch(source)
                reference_source = dequantize_mxfp8_rows_torch(encoded.values, encoded.scale_rows).float()
            else:
                reference_source = source.float()
            reference = reference_source @ decoded.T
            output.fill_(float("nan"))
            projection.workspace.fill_(255)
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            if rows > 1:
                assert torch.all(output[-1] == 0)
            for actual, expected in zip(output.split((10240, 6144, 48, 48), dim=1),
                                        reference.split((10240, 6144, 48, 48), dim=1), strict=True):
                error = torch.linalg.vector_norm(actual.float() - expected)
                denominator = torch.linalg.vector_norm(expected).clamp_min(1e-12)
                assert float(error / denominator) < .004
                torch.testing.assert_close(actual.float(), expected,
                    atol=float(expected.abs().max()) * .008 + 1e-6, rtol=.008)
        graph.reset()


@pytest.mark.parametrize("tile_k", [128, 256])
@torch.no_grad()
def test_qad_gdn_output_split_reduction_live_replay(tile_k):
    from dataclasses import asdict, replace
    from b12x.gemm import blockscaled
    from b12x.gemm._tuning import TUNING as DENSE_TUNING
    from b12x.gemm.blockscaled._tuning import BlockscaledConfig, quantized_query
    from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch, dequantize_mxfp8_rows_torch
    from b12x.preparation import PreparedCall
    from b12x.preparation.device import detect_device
    from ignition.components.gated_norm import PreparedGatedRMSNorm
    from ignition.components.linear import QuantizedProjection
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession, declare

    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    manifest = inspect_indexed_checkpoint(resolve("qwen3.8-27b-qad").snapshot(hub))
    prefix = f"{TEXT_PREFIX}.layers.1.linear_attn"
    with DirectWeightLoader(manifest, "cuda:0", prefixes=(prefix,)) as loader:
        weight = loader.load_linear(prefix + ".out_proj")
        norm_weight = loader.load_tensor(prefix + ".norm.weight", dtype=torch.bfloat16)
    hidden = torch.empty((16, 48, 128), device="cuda", dtype=torch.bfloat16)
    packed = torch.empty((16, 16480), device="cuda", dtype=torch.bfloat16)
    gate = packed[:, 10240:16384].view_as(hidden)
    source = torch.empty((16, 6144), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((16, 5120), device="cuda", dtype=torch.bfloat16)
    forced_output = torch.empty_like(output)
    live = torch.tensor([16], device="cuda", dtype=torch.int32)
    norm = PreparedGatedRMSNorm(hidden, gate, norm_weight, source.view_as(hidden), live,
        eps=1e-6, round_normalized=True, activation="silu")
    decoded = weight.weight.values.reshape(5120, 6144).float() * torch.exp2(
        weight.weight.scale_rows.reshape(5120, -1).view(torch.uint8).float() - 127
    ).repeat_interleave(32, dim=1)
    workspace = torch.empty(16 << 20, device="cuda", dtype=torch.uint8)
    with RecordedPreparationSession(device="cuda:0", checkpoint="qad-gdn-output-reduction",
            workload_schema="mxfp8-output", numerical_policy="default", autotune_effort="off") as session:
        arena = ScratchArena("cuda:0")
        session.register_scratch(arena)
        projection = QuantizedProjection(source, weight, output, session=session, name="output",
            workspace=arena, activation_mode="quantized")
        assert projection.plan.query.allow_bf16_atomic_reduction is False
        query = replace(projection.plan.query, workspace_nbytes=None)
        dense = DENSE_TUNING.default_config(quantized_query(query, expected_m=None),
                                           detect_device(source.device).identity)
        dense = replace(dense, tile_m=64, tile_n=64, tile_k=tile_k, split_k_slices=4,
                        load_path="tma", swap_ab=False, large_m_unroll=False)
        forced = blockscaled.plan(query, override=BlockscaledConfig(mode="quantized", dense_config=asdict(dense)))
        values, scales, gain = projection.parts
        declare(session, (forced.request(name="output.split4", prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, values, scales, gain, out=forced_output,
                                  workspace=workspace, activation_scale=None))),))
        session.tune(show_progress=False)

        def forward():
            norm()
            projection(source)
            blockscaled.mm(source, weight, out=forced_output, workspace=workspace, plan=forced)

        compiled = compile_forward(forward)
        hidden.normal_()
        gate.normal_()
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for seed in (701, 702, 703):
            torch.manual_seed(seed)
            hidden.normal_()
            gate.normal_()
            for count in (16, 13, 0, 1, 16):
                live.fill_(count)
                norm()
                encoded = quantize_mxfp8_rows_torch(source)
                reference = dequantize_mxfp8_rows_torch(encoded.values, encoded.scale_rows).float() @ decoded.T
                previous = None
                for _ in range(8):
                    output.fill_(float("nan"))
                    forced_output.fill_(float("nan"))
                    workspace.fill_(255)
                    arena.tensor.fill_(255)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key]
                    for actual in (output, forced_output):
                        relative = float((actual.float() - reference).norm() / reference.norm().clamp_min(1e-12))
                        assert relative < .004
                        torch.testing.assert_close(actual.float(), reference,
                            atol=float(reference.abs().max()) * .008 + 1e-6, rtol=.008)
                        assert not actual[count:].any()
                    if previous is not None:
                        assert torch.equal(forced_output, previous)
                    previous = forced_output.clone()
        graph.reset()
