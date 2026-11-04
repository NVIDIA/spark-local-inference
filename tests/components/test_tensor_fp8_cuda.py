# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-calibrated FP8 projections and allocation-stable graph replay."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from ignition.components.tensor_fp8 import TensorFP8Projection
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def assert_fp8_projection(actual, expected):
    # FP32 reduction order can cross one BF16 output rounding boundary.
    torch.testing.assert_close(actual.float().cpu(), expected.bfloat16().float(),
                               rtol=torch.finfo(torch.bfloat16).eps, atol=0)


@pytest.mark.parametrize("rows", [1, 17, 128, 4096])
@pytest.mark.parametrize("shared", [False, True])
@torch.no_grad()
def test_tensor_fp8_calibration_survives_loading_and_replay(tmp_path, rows, shared):
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    generator = torch.Generator().manual_seed(2741)
    values = torch.randn((256, 512), generator=generator).mul(20).to(torch.float8_e4m3fn)
    weight_scale, input_scale = .017, .031
    tensors = {"proj.weight": values, "proj.weight_scale": torch.tensor(weight_scale),
               "proj.input_scale": torch.tensor(input_scale)}
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.nbytes for t in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"proj": {"quant_algo": "FP8"}}}}))
    with DirectWeightLoader(inspect_indexed_checkpoint(tmp_path), "cuda:0") as loader:
        weight = loader.load_linear("proj", rows=(128, 256), columns=(32, 320))
        assert loader._session.torch_copy_bytes == 0
    source = torch.randn((rows, 288), generator=generator).mul(10).bfloat16()
    device_source = source.cuda()
    output = torch.empty((rows, 128), dtype=torch.bfloat16, device="cuda")
    staging = torch.empty(rows * 512, dtype=torch.uint8, device="cuda") if shared else None
    with RecordedPreparationSession(device=device_source.device, checkpoint='tensor-fp8',
            workload_schema='tensor-fp8', numerical_policy='default', compile_workers=1) as session:
        projection = TensorFP8Projection(device_source, weight, output, session=session, name="tensor-fp8", staging=staging)
        if shared:
            assert projection.quantized.data_ptr() == staging.data_ptr()
        session.tune()
        compiled = compile_forward(lambda: projection(device_source))
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for factor in (1., -.5, 2.):
            device_source.copy_(source * factor)
            output.fill_(float("nan"))
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            quantized = ((source * factor).float() / tensors['proj.input_scale']).clamp(-448, 448).to(torch.float8_e4m3fn)
            assert torch.equal(projection.quantized.cpu().view(torch.uint8), quantized.view(torch.uint8))
            expected = ((quantized.float() @ values.float()[128:256, 32:320].T)
                        * (tensors['proj.weight_scale'] * tensors['proj.input_scale']))
            assert_fp8_projection(output, expected)
        graph.reset()


@pytest.mark.parametrize('columns,outputs', [(6656, 19968), (19968, 6656)])
@torch.no_grad()
def test_sm103_checkpoint_sized_fp8_state_does_not_allocate(columns, outputs):
    """Glimmer's largest FP8 gate/up and down projections fit caller-owned workspace."""
    from b12x.gemm import blockscaled
    from b12x.preparation import require_prepared
    from ignition.components.tensor_fp8 import TensorFP8Weight
    from ignition.components.tensor_fp8_sm103 import COMPONENT
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip('checks the SM103 native GEMM workspace contract')
    source = torch.full((4096, columns), .031, device='cuda', dtype=torch.bfloat16)
    values = torch.ones((outputs, columns), device='cuda', dtype=torch.float8_e4m3fn)
    input_scale = torch.tensor(.031, device='cuda')
    combined_scale = torch.tensor(.017, device='cuda') * input_scale
    weight = TensorFP8Weight(blockscaled.pack_weight(values, combined_scale, recipe='tensor_fp8'), input_scale)
    output = torch.empty((4096, outputs), device='cuda', dtype=torch.bfloat16)
    with RecordedPreparationSession(device=source.device, checkpoint='glimmer-fp8-capacity',
            workload_schema='tensor-fp8', numerical_policy='default', compile_workers=1) as session:
        arena = ScratchArena(source.device)
        session.register_scratch(arena)
        projection = TensorFP8Projection(source, weight, output, session=session,
            name='checkpoint-sized-fp8', workspace=arena)
        session.tune()
        projection(source)
        assert bool((projection.quantized.float() == 1).all())
        state = require_prepared(projection.plan, COMPONENT, source.device)

        def run():
            state.run(projection.quantized, weight.packed.values, weight.packed.output_scale,
                      output, projection.workspace)

        run()
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()['allocation.all.allocated']
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
        expected = (combined_scale * columns).bfloat16().expand_as(output)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize('rows', [17, 4096])
@torch.no_grad()
def test_mixed_group_retains_distinct_fp8_calibration(tmp_path, rows):
    from ignition.components.linear import prepare_projection
    from ignition.components.weights import GroupedLinearWeights
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.manual_seed(375)
    tensors, declarations = {}, {}
    source = torch.randn((rows, 256), dtype=torch.bfloat16, device='cuda')
    original = source.cpu()
    for name, scale in [('q', .013), ('k', .071)]:
        values = torch.randn((128, 256)).mul(20).to(torch.float8_e4m3fn)
        tensors.update({name + '.weight': values, name + '.weight_scale': torch.tensor(.017),
                        name + '.input_scale': torch.tensor(scale)})
        declarations[name] = {'quant_algo': 'FP8'}
    tensors['v.weight'] = torch.randn((128, 256), dtype=torch.bfloat16) * .03

    def reference(value):
        parts = []
        for name in ('q', 'k'):
            scale = tensors[name + '.input_scale']
            quantized = (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            parts.append((quantized.float() @ tensors[name + '.weight'].float().T)
                         * (scale * tensors[name + '.weight_scale']))
        return torch.cat([*parts, value.float() @ tensors['v.weight'].float().T], -1)

    def assert_reference(value):
        expected = reference(value)
        assert_fp8_projection(output[:, :256], expected[:, :256])
        # The BF16 GEMM's FP32 reduction differs near cancellation: measured
        # excess beyond a BF16 ULP is at most 4.03e-7 on SM103 and SM120.
        torch.testing.assert_close(output[:, 256:].float().cpu(), expected[:, 256:].bfloat16().float(),
                                   rtol=torch.finfo(torch.bfloat16).eps, atol=2 ** -21)
    save_file(tensors, str(tmp_path / 'weights.safetensors'))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(dict(
        metadata={'total_size': sum(t.nbytes for t in tensors.values())},
        weight_map={name: 'weights.safetensors' for name in tensors})))
    (tmp_path / 'config.json').write_text(json.dumps({'quantization_config': {
        'quant_method': 'modelopt', 'quant_algo': 'MIXED_PRECISION', 'quantized_layers': declarations}}))
    with DirectWeightLoader(inspect_indexed_checkpoint(tmp_path), 'cuda:0') as loader:
        weight = loader.load_mixed_linear_group(('q', 'k', 'v'))
    assert isinstance(weight, GroupedLinearWeights)
    assert weight.widths == (128, 128, 128)
    output = torch.empty((rows, 384), dtype=torch.bfloat16, device='cuda')
    with RecordedPreparationSession(device=source.device, checkpoint='mixed-fp8', workload_schema='mixed-fp8',
            numerical_policy='default', compile_workers=1) as session:
        arena = ScratchArena(source.device)
        session.register_scratch(arena)
        staging = torch.empty(source.numel(), dtype=torch.uint8, device=source.device)
        projection = prepare_projection(source, weight, output, session=session, name='mixed', workspace=arena,
                                        quantized_staging=staging)
        assert all(p.quantized.data_ptr() == staging.data_ptr() for p in projection.projections[:2])
        session.tune()
        compiled = compile_forward(lambda: projection(source))
        compiled()
        torch.cuda.synchronize()
        assert_reference(original)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for factor in (1., -.5, 2.):
            source.copy_(original * factor)
            output.fill_(float('nan'))
            allocations = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
            assert_reference(original * factor)
        graph.reset()
