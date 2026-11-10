# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoonViT image execution with learned positions, axial RoPE, and spatial merging."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.kimi_k25.modeling_kimi_k25 import Kimi_K25VisionEncoderLayer

from ignition.components.linear import UnquantizedProjection
from ignition.components.rms_norm import PreparedTorchRMSNorm
from ignition.components.rotary import apply_cached_rotary, prepare_cached_rotary
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy

PATCH_CAPACITIES = (256, 1024, 4096, 16384, 65536)
FEATURES = 16384
HIDDEN = 7168


@dataclass(frozen=True)
class VisionWeights:
    tensors: dict[str, torch.Tensor]

    @classmethod
    def load(cls, manifest, device, progress=None):
        with DirectWeightLoader(manifest, device, prefixes=('vision_tower.', 'mm_projector.'), progress=progress) as loader:
            names = sorted(loader._selected_names())
            tensors = dict(zip(names, loader.load_tensors(names), strict=True))
        return cls(tensors)


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_interpolation_indices: torch.Tensor
    vision_interpolation_weights: torch.Tensor
    vision_merge_indices: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_num_features: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)
        n = PATCH_CAPACITIES[-1]
        return cls(zeros((n, 588), torch.float32), zeros((n, 2), torch.int64),
            zeros((n, 4), torch.int64), zeros((n, 4), torch.float32), zeros((n // 4, 4)),
            zeros(2), zeros(1), zeros(1), zeros(1) - 1, zeros(1, torch.int64), zeros(1))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    hidden: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    projected: torch.Tensor
    mlp_up: torch.Tensor
    mlp_down: torch.Tensor
    merged: torch.Tensor
    activated: torch.Tensor
    output: torch.Tensor
    normalized: torch.Tensor

    @classmethod
    def allocate(cls, patches, device, *, examples=None):
        shapes = ((patches, 588), (patches, 1024), (patches, 4608),
                  *((patches, 12, 128),) * 3, (patches, 1536), (patches, 1024),
                  (patches, 4096), (patches, 1024), (patches // 4, 4096),
                  (patches // 4, 4096), (patches // 4, HIDDEN), (patches, 1024))
        return cls(*(declaration_tensor(examples, f"vision.{name}", shape, torch.bfloat16, device)
                     for name, shape in zip(cls.__dataclass_fields__, shapes, strict=True)))

    def rows(self, patches):
        return type(self)(*(getattr(self, name)[:patches // 4 if name in ('merged', 'activated', 'output') else patches]
                           for name in self.__dataclass_fields__))


class _Projection(nn.Module):
    def __init__(self, weight, source, output, *, session, name):
        super().__init__()
        self.operation = UnquantizedProjection(source, weight, output, session=session, name=name).functional(session=session)

    def forward(self, source):
        return self.operation(source)


class _InplaceGELU(nn.Module):
    def forward(self, source):
        return torch.ops.aten.gelu.out(source, approximate="tanh", out=source)


class _RMSNorm(nn.Module):
    def __init__(self, weight, source, output, eps=None, *, session, name):
        super().__init__()
        self.operation = PreparedTorchRMSNorm(source, weight, output, eps=eps, session=session, name=name).functional(session=session)

    def forward(self, source):
        return self.operation(source)


def interpolate_positions(weight, indices, factors, output):
    for start in range(0, indices.shape[0], 4096):
        end = start + 4096
        output[start:end].copy_((weight[indices[start:end]].float()
            * factors[start:end, :, None]).sum(1).to(output.dtype))


class _Attention(nn.Module):
    def __init__(self, weights, scratch, cu, positions, *, session, workspace, name):
        super().__init__()
        s = scratch
        self.positions = positions
        self.qkv = _Projection(weights['wqkv.weight'], s.hidden, s.qkv, session=session, name=name + '.qkv')
        self.output = _Projection(weights['wo.weight'], s.attended, s.projected, session=session, name=name + '.output')
        self.attend = PreparedVarlenAttention(s.query, s.key, s.value, cu, s.attended.view_as(s.query),
            session=session, workspace=workspace, name=name + '.attention').functional(session=session)

    def forward(self, source, *, cu_seqlens, position_embeddings):
        q, k, v = self.qkv(source).view(-1, 3, 12, 128).unbind(1)
        query = torch.empty_like(q, memory_format=torch.contiguous_format)
        key = torch.empty_like(k, memory_format=torch.contiguous_format)
        apply_cached_rotary(q, k, position_embeddings, self.positions, query, key, True)
        attended = self.attend(query, key, v.contiguous(), cu_seqlens)
        return self.output(attended.view(-1, 1536))


class PreparedVision:
    def __init__(self, weights, scratch, inputs, *, session, workspace, name):
        self.inputs = inputs
        s, tensors = scratch, weights.tensors
        n, device = s.patches.shape[0], s.patches.device
        self.capacity = n
        cu = torch.zeros((2,), dtype=torch.int32, device=device)
        frequencies = 1. / (10000 ** (torch.arange(0, 128, 4, dtype=torch.float32, device=device) / 128))
        angles = torch.arange(512, dtype=torch.float32, device=device)[:, None] * frequencies
        self.rotary = persistent_copy(torch.view_as_real(torch.polar(torch.ones_like(angles), angles)))
        rotary_positions = persistent_copy(torch.arange(n, dtype=torch.int32, device=device))
        prepare_cached_rotary(session)
        self.position_weight = tensors['vision_tower.patch_embed.pos_emb.weight'].reshape(-1, 1024)
        self.patch = UnquantizedProjection(s.patches, tensors['vision_tower.patch_embed.proj.weight'].flatten(1),
            s.hidden, session=session, name=name + '.patch').functional(session=session)
        self.blocks = []
        config = SimpleNamespace(hidden_size=1024, num_attention_heads=12, intermediate_size=4096, hidden_act='gelu_pytorch_tanh')
        for layer in range(27):
            prefix = f'vision_tower.encoder.blocks.{layer}.'
            w = {key.removeprefix(prefix): value for key, value in tensors.items() if key.startswith(prefix)}
            with torch.device('meta'):
                block = Kimi_K25VisionEncoderLayer(config)
            block.norm1 = _RMSNorm(w['norm0.weight'], s.hidden, s.normalized,
                                  session=session, name=f'{name}.{layer}.norm1')
            block.norm2 = _RMSNorm(w['norm1.weight'], s.hidden, s.normalized,
                                  session=session, name=f'{name}.{layer}.norm2')
            block.attn = _Attention(w, s, cu, rotary_positions,
                session=session, workspace=workspace, name=f'{name}.{layer}')
            block.mlp.fc1 = _Projection(w['mlp.fc0.weight'], s.hidden, s.mlp_up, session=session, name=f'{name}.{layer}.up')
            # The activation is the only consumer of the up projection's output.
            block.mlp.act = _InplaceGELU()
            block.mlp.fc2 = _Projection(w['mlp.fc1.weight'], s.mlp_up, s.mlp_down, session=session, name=f'{name}.{layer}.down')
            block.eval()
            self.blocks.append(block)
        self.norm = _RMSNorm(tensors['vision_tower.encoder.final_layernorm.weight'], s.hidden, s.normalized,
                             session=session, name=name + '.norm')
        self.up = _Projection(tensors['mm_projector.proj.0.weight'], s.merged, s.activated,
                             session=session, name=name + '.merge_up')
        self.down = _Projection(tensors['mm_projector.proj.2.weight'], s.activated, s.output,
                               session=session, name=name + '.merge_down')
        self.post_norm = _RMSNorm(tensors['mm_projector.post_norm.weight'], s.output, s.output, 1e-5,
                                  session=session, name=name + '.post_norm')

    def rotary_embeddings(self, positions):
        rotary = self.rotary[positions.flip((1,))].transpose(1, 2).flatten(1, 2)
        return torch.cat((rotary[..., 0], rotary[..., 1]), dim=-1)

    def __call__(self, patches, positions, interpolation_indices, interpolation_weights, merge_indices, cu):
        position_embedding = torch.empty((self.capacity, 1024), dtype=patches.dtype, device=patches.device)
        interpolate_positions(self.position_weight, interpolation_indices, interpolation_weights, position_embedding)
        hidden = self.patch(patches) + position_embedding
        embeddings = self.rotary_embeddings(positions)
        for block in self.blocks:
            hidden = block(hidden, cu, position_embeddings=embeddings)
        hidden = self.norm(hidden)
        merged = hidden[merge_indices].flatten(1)
        return self.post_norm(self.down(torch.nn.functional.gelu(self.up(merged))))
