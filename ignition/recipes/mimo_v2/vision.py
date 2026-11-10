# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared MiMo image encoder with row/column windows and spatial patch merging."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ignition.components.collectives import PreparedAllReduce
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.preparation.persistent import persistent_copy

PATCH_CAPACITIES = (256, 1024, 4096, 16384)
FEATURES = 16384
HIDDEN = 6144


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_column_indices: torch.Tensor
    vision_row_indices: torch.Tensor
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
        return cls(zeros((n, 1536), torch.float32), zeros((n, 2), torch.int64), zeros(n // 4), zeros(n // 4),
                   zeros(2), zeros(1), zeros(1), zeros(1) - 1, zeros(1, torch.int64), zeros(1))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    residual: torch.Tensor
    normalized: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    projected: torch.Tensor
    reduced: torch.Tensor
    mlp: MLPScratch
    merged: torch.Tensor
    activated: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, shard, device):
        heads, kv_heads = 32 // shard.world_size, 8 // shard.world_size
        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        pixels = zeros(patches, 1536)
        residual, normalized, projected, reduced = (zeros(patches, 1280) for _ in range(4))
        mlp = MLPScratch.allocate(patches, 1280, 4608 // shard.world_size, device)
        qkv = zeros(patches, (heads + 2 * kv_heads) * 64)
        query = zeros(patches, heads, 64)
        key, value = (zeros(patches, kv_heads, 64) for _ in range(2))
        attended = zeros(patches, heads * 64)
        merged = zeros(patches // 4, 5120)
        activated = zeros(patches // 4, 5120 // shard.world_size)
        partial, output = (zeros(patches // 4, HIDDEN) for _ in range(2))
        return cls(pixels, residual, normalized, qkv, query, key, value, attended, projected,
                   reduced, mlp, merged, activated, partial, output)

    def rows(self, patches):
        return type(self)(*(self.mlp.rows(patches) if field == "mlp"
            else getattr(self, field)[:patches // 4 if field in ("merged", "activated", "partial", "output") else patches]
            for field in self.__dataclass_fields__))


class PreparedVision:
    def __init__(self, weights, scratch, inputs, *, runtime, coordinator, session, workspace, name):
        self.inputs, self.config = inputs, weights.config
        self.capacity = scratch.patches.shape[0]
        self.query_shape, self.key_shape, self.value_shape = (tuple(value.shape) for value in (scratch.query, scratch.key, scratch.value))
        c, s = weights.config, scratch
        device = s.patches.device
        self.frequencies = persistent_copy(10000 ** (-torch.arange(0, 32, 2, dtype=torch.float32, device=device) / 32))
        cu = torch.zeros((2,), dtype=torch.int32, device=device)
        self.patch = UnquantizedProjection(s.patches, weights.patch, s.residual, session=session, name=name + ".patch").functional(session=session)
        self.blocks = []
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.{index}"
            norm1 = PreparedRMSNorm(s.residual, w.norm1, s.normalized, eps=1e-6, session=session, name=prefix + ".norm1").functional()
            qkv = UnquantizedProjection(s.normalized, w.qkv, s.qkv, bias=w.qkv_bias, session=session, name=prefix + ".qkv").functional(session=session)
            output = UnquantizedProjection(s.attended, w.output, s.projected, session=session, name=prefix + ".output").functional(session=session)
            norm2 = PreparedRMSNorm(s.residual, w.norm2, s.normalized, eps=1e-6, session=session, name=prefix + ".norm2").functional()
            mlp = PreparedMLP(s.normalized, w.mlp, s.mlp.gate_up, s.mlp.intermediate, s.mlp.output,
                gate_up_bias=w.gate_up_bias, session=session, workspace=workspace, name=prefix + ".mlp").functional(session=session)
            window = None if index in c["fullatt_block_indexes"] else (c["visual_token_window_size"],) * 2
            attention = PreparedVarlenAttention(s.query, s.key, s.value, cu, s.attended.view_as(s.query),
                window=window, sink=w.sink, session=session, workspace=workspace, name=prefix + ".attention").functional(session=session)
            self.blocks.append((norm1, qkv, attention, output, w.output_bias, norm2, mlp, w.down_bias))
        self.reduce = PreparedAllReduce(s.projected, s.reduced, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".reduce").functional(session=session)
        self.norm = PreparedRMSNorm(s.residual, weights.norm, s.normalized, eps=1e-6, session=session, name=name + ".norm").functional()
        self.merge_up = UnquantizedProjection(s.merged, weights.merge_up, s.activated, session=session, name=name + ".merge_up").functional(session=session)
        self.merge_down = UnquantizedProjection(s.activated, weights.merge_down, s.partial, session=session, name=name + ".merge_down").functional(session=session)
        self.merge_reduce = PreparedAllReduce(s.partial, s.output, runtime=runtime, coordinator=coordinator,
                                             session=session, name=name + ".merge_reduce").functional(session=session)

    def __call__(self, patches, positions, columns, rows, cu):
        hidden = self.patch(patches)
        angles = (positions.float()[:, :, None] * self.frequencies).flatten(1)[:, None]
        col_angles = angles.unflatten(0, (-1, 4))[columns.long()].flatten(0, 1)
        def rotate(source, angles):
            left, right = source.float().chunk(2, -1)
            cos, sin = angles.cos(), angles.sin()
            return torch.cat((left * cos - right * sin, right * cos + left * sin), -1).to(source.dtype)

        previous = -1
        for index, (norm1, project, attention, output, output_bias, norm2, mlp, down_bias) in enumerate(self.blocks):
            kind = self.config["vit_window_attn_types"][index]
            if (kind == 1) != (previous == 1):
                order = columns if kind == 1 else rows
                hidden = hidden.unflatten(0, (-1, 4))[order.long()].flatten(0, 1)
            previous = kind
            widths = tuple(shape[1] * shape[2] for shape in (self.query_shape, self.key_shape, self.value_shape))
            q, k, v = project(norm1(hidden)).split(widths, -1)
            selected = col_angles if kind == 1 else angles
            query = rotate(q.view(self.query_shape), selected)
            key = rotate(k.view(self.key_shape), selected)
            attended = attention(query, key, v.view(self.value_shape).contiguous(), cu)
            hidden = hidden + (self.reduce(output(attended.flatten(1))) + output_bias)
            hidden = hidden + (self.reduce(mlp(norm2(hidden))) + down_bias)
        if previous == 1:
            hidden = hidden.unflatten(0, (-1, 4))[rows.long()].flatten(0, 1)
        merged = self.norm(hidden).view(self.capacity // 4, -1)
        activated = F.gelu(self.merge_up(merged))
        return self.merge_reduce(self.merge_down(activated))
