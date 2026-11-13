# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured MoonViT projections and attention against the checkpoint forward."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from tests.components.completion_support import completion_outputs
from .kimi_vision_support import VisionCall

pytestmark = pytest.mark.skipif(os.getenv('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned GPU')
CHECKPOINT = Path('/data/models/Kimi-K3-MXFP4-CSF')


@torch.no_grad()
@pytest.mark.parametrize('n,h,w', [(256, 16, 16), (65536, 256, 256)])
def test_checkpoint_vision(monkeypatch, tmp_path, n, h, w):
    pytest.importorskip('fla')
    pytest.importorskip('flash_attn')
    import transformers.utils.generic as generic
    from transformers.utils.output_capturing import OutputRecorder
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.kimi3.vision import VisionWeights, VisionInputs, VisionScratch, PreparedVision

    if not CHECKPOINT.is_dir():
        pytest.skip('Kimi K3 checkpoint is unavailable')
    # The checkpoint imports the Transformers output recorder from its former module.
    monkeypatch.setattr(generic, 'OutputRecorder', OutputRecorder, raising=False)
    cls = get_class_from_dynamic_module('modeling_kimi_k3.MoonViT3dPretrainedModel', str(CHECKPOINT))
    monkeypatch.setattr(cls, '_supports_flash_attn', cls._supports_flash_attn_2)
    module = sys.modules[cls.__module__]
    raw = json.loads((CHECKPOINT / 'config.json').read_text())['vision_config']
    source_config = SimpleNamespace(**raw)
    config = module.VisionTowerConfig(source_config)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    weights = VisionWeights.load(inspect_indexed_checkpoint(CHECKPOINT), device)
    with torch.device('meta'):
        reference = cls(config)
        projector = module.PatchMergerMLPV2(module.ProjectorConfig(source_config))
    reference.load_state_dict({n.removeprefix('vision_tower.'): v for n,v in weights.tensors.items() if n.startswith('vision_tower.')}, assign=True)
    projector.load_state_dict({n.removeprefix('mm_projector.'): v for n,v in weights.tensors.items() if n.startswith('mm_projector.')}, assign=True)
    reference.eval(); projector.eval()
    inputs, scratch = VisionInputs.allocate(device), VisionScratch.allocate(n, device)
    with RecordedPreparationSession(device=device, checkpoint=str(CHECKPOINT), workload_schema='kimi3-vision', numerical_policy='default') as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        encoder = PreparedVision(weights, scratch, inputs, session=session, workspace=arena, name='vision')
        call = VisionCall(encoder, device)
        session.tune(show_progress=False)
        call.positions.copy_(torch.tensor([[y,x] for y in range(h) for x in range(w)], device=device))
        indices, factors = [], []
        for y in range(h):
            for x in range(w):
                sy, sx = max(0,64/h*(y+.5)-.5), max(0,64/w*(x+.5)-.5)
                iy, ix = int(sy), int(sx); fy, fx = sy-iy, sx-ix
                indices.append([iy*64+ix, iy*64+min(ix+1,63),min(iy+1,63)*64+ix,min(iy+1,63)*64+min(ix+1,63)])
                factors.append([(1-fy)*(1-fx),(1-fy)*fx,fy*(1-fx),fy*fx])
        call.interpolation_indices.copy_(torch.tensor(indices, device=device))
        call.interpolation_weights.copy_(torch.tensor(factors, device=device))
        call.merge_indices.copy_(torch.tensor([[(y+dy)*w+x+dx for dy in range(2) for dx in range(2)] for y in range(0,h,2) for x in range(0,w,2)], device=device))
        call.cu.copy_(torch.tensor([0,n], dtype=torch.int32, device=device))
        torch.manual_seed(41)
        source = torch.randn((n,3,14,14), dtype=torch.bfloat16, device=device)
        grid = torch.tensor([[1,h,w]], device=device)
        hidden = reference.patch_embed(source, grid)
        rope = reference.encoder.rope_2d.get_freqs_cis(grid, device)
        embeddings = encoder.rotary_embeddings(call.positions)
        for block, reference_block in zip(encoder.blocks, reference.encoder.blocks, strict=True):
            expected_block = reference_block(hidden, call.cu, n, rope)
            actual_block = block(hidden, call.cu, position_embeddings=embeddings)
            relative = ((actual_block.float()-expected_block.float()).square().mean()
                        / expected_block.float().square().mean()).sqrt()
            assert relative < .005
            hidden = expected_block
        call.patches.copy_(source.flatten(1))
        expected = projector(reference(source, grid))[0]
        actual = call().clone()
        error = ((actual.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt()
        print('vision relative RMS', error.item())
        # BF16 attention differences propagate through all 27 residual blocks.
        assert error < .05
        compiled = compile_forward(call)
        call.patches.copy_(source.flatten(1))
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        compiled()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        print(f'vision {n} peak intermediate storage: {peak / (1 << 20):.1f} MiB')
        assert peak <= 320 << 20
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            output = compiled()
        call.patches.copy_(source.flatten(1))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, actual, atol=0, rtol=0)
        graph.reset()
        from ignition.components.paged_batch import BatchInputs
        from ignition.components.sampling import DeviceRequestState
        from ignition.components.vision_state import PrefillVisionInputs, VisionState
        from ignition.recipes.kimi3.vision_state import PreparedVisionGeneration

        requests = DeviceRequestState.allocate(device, vocab_size=163840, capacity=2)
        batch = BatchInputs.allocate(n, 2, device, capacity=2)
        state = VisionState(batch, PrefillVisionInputs.allocate(n, device), requests,
                            image_token=163605, features=16384, hidden=448)
        records = torch.zeros((2,2, 32), dtype=torch.int64, device=device)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.ones(1, dtype=torch.int64, device=device)
        publication = PreparedVisionGeneration(encoder, state, completion_outputs(records), frame, sequence, feature_range=(448,896))
        inputs.vision_pixel_values[:n].copy_(source.flatten(1))
        for target, value in ((inputs.vision_position_ids, call.positions),
                              (inputs.vision_interpolation_indices, call.interpolation_indices),
                              (inputs.vision_interpolation_weights, call.interpolation_weights),
                              (inputs.vision_merge_indices, call.merge_indices)):
            target[:value.shape[0]].copy_(value)
        inputs.vision_cu_seqlens.copy_(call.cu)
        inputs.vision_num_patches.fill_(n)
        inputs.vision_num_features.fill_(n//4)
        state.feature_slot.fill_(1)
        inputs.vision_slot.fill_(1)
        inputs.vision_generation.fill_(12)
        requests.status[1] = 1
        requests.generation[1] = 12
        from ignition.preparation.aot import aot_forward

        compiled_publication = aot_forward(publication)
        compiled_publication(publication)
        artifact = tmp_path / 'vision.bin'
        compiled_publication.save_compiled_function(str(artifact))
        with session.capture(), torch.cuda.graph(graph):
            compiled_publication(publication)
        reserved = sum(segment['total_size'] for segment in torch.cuda.memory_snapshot()
                       if tuple(segment['segment_pool_id']) == graph.pool())
        print(f'vision {n} graph pool: {reserved / (1 << 20):.1f} MiB')
        assert reserved <= 640 << 20
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(state.arena[1,:n//4], actual[:,448:896], atol=0, rtol=0)
        assert state.count.tolist() == [0,n//4]
        assert state.generation.tolist() == [0,12]
        for name in scratch.__dataclass_fields__:
            getattr(scratch, name).fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(state.arena[1,:n//4], actual[:,448:896], atol=0, rtol=0)
        if n < 65536:
            inputs.vision_feature_offset.fill_(n//4)
            graph.replay()
            torch.cuda.synchronize()
            assert state.count.tolist() == [0,n//2]
            torch.testing.assert_close(state.arena[1,n//4:n//2], actual[:,448:896], atol=0, rtol=0)
        graph.reset()
        publication = PreparedVisionGeneration(encoder, state, completion_outputs(records), frame, sequence, feature_range=(6720,7168))
        with artifact.open('rb') as file:
            restored = torch.compiler.load_compiled_function(file)
        inputs.vision_feature_offset.zero_()
        restored(publication)
        with session.capture(), torch.cuda.graph(graph):
            restored(publication)
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(state.arena[1,:n//4], actual[:,6720:7168], atol=0, rtol=0)
        before = state.arena.clone()
        inputs.vision_feature_offset.zero_()
        inputs.vision_generation.fill_(11)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(state.arena, before, atol=0, rtol=0)
        assert requests.status[1] == 1
        inputs.vision_generation.fill_(12)
        inputs.vision_merge_indices[0,0] = n
        graph.replay()
        torch.cuda.synchronize()
        assert requests.status[1] == 3
        torch.testing.assert_close(state.arena, before, atol=0, rtol=0)
        graph.reset()
