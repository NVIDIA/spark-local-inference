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


@pytest.mark.parametrize("rows", [1, 17, 128])
@pytest.mark.parametrize("shared", [False, True])
@torch.no_grad()
def test_tensor_fp8_calibration_survives_loading_and_replay(tmp_path, rows, shared):
    from b12x.preparation import PreparationSession

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
    with PreparationSession(device=device_source.device, autotune=False, compile_workers=0) as session:
        projection = TensorFP8Projection(device_source, weight, output, session=session, name="tensor-fp8", staging=staging)
        if shared:
            assert projection.quantized.data_ptr() == staging.data_ptr()
        projection(device_source)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            projection(device_source)
        for factor in (1., -.5, 2.):
            device_source.copy_(source * factor)
            output.fill_(float("nan"))
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            quantized = ((source * factor).float() / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            expected = (quantized.float() @ values.float()[128:256, 32:320].T) * (weight_scale * input_scale)
            torch.testing.assert_close(output.float().cpu(), expected, rtol=.015, atol=.03)


@torch.no_grad()
def test_mixed_group_retains_distinct_fp8_calibration(tmp_path):
    from ignition.components.linear import prepare_projection
    from ignition.components.weights import GroupedLinearWeights
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.manual_seed(375)
    tensors, declarations, references = {}, {}, []
    source = torch.randn((17, 256), dtype=torch.bfloat16, device='cuda')
    for name, scale in [('q', .013), ('k', .071)]:
        values = torch.randn((128, 256)).mul(20).to(torch.float8_e4m3fn)
        tensors.update({name + '.weight': values, name + '.weight_scale': torch.tensor(.017),
                        name + '.input_scale': torch.tensor(scale)})
        declarations[name] = {'quant_algo': 'FP8'}
        quantized = (source.cpu().float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        references.append((quantized.float() @ values.float().T) * (scale * .017))
    tensors['v.weight'] = torch.randn((128, 256), dtype=torch.bfloat16) * .03
    references.append(source.cpu().float() @ tensors['v.weight'].float().T)
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
    output = torch.empty((17, 384), dtype=torch.bfloat16, device='cuda')
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
        torch.testing.assert_close(output.float().cpu(), torch.cat(references, -1), rtol=.015, atol=.03)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        output.zero_()
        allocations = torch.cuda.memory_stats()['allocation.all.allocated']
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
        torch.testing.assert_close(output.float().cpu(), torch.cat(references, -1), rtol=.015, atol=.03)
        graph.reset()
