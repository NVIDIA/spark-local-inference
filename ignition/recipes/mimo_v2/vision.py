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

PATCH_CAPACITIES = (256, 1024, 4096, 16384)
FEATURES = 16384
# Encoder block width, reduced across ranks after every attention and MLP block.
VISION_WIDTH = 1280


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
    def allocate(cls, patches, shard, device, *, hidden, buffers=None):
        """Allocate encoder scratch; `hidden` is the text model width the merged features project into."""
        heads, kv_heads = 32 // shard.world_size, 8 // shard.world_size
        def zeros(cursor, *shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=device) if cursor is None
                    else cursor.zeros(shape, torch.bfloat16))
        pixels = zeros(buffers, patches, 1536)
        residual, normalized, projected, reduced = (zeros(buffers, patches, VISION_WIDTH) for _ in range(4))
        stages = [buffers.fork() for _ in range(3)] if buffers is not None else [None] * 3
        attention, feedforward, merger = stages
        mlp = MLPScratch.allocate(patches, 1280, 4608 // shard.world_size, device, buffers=feedforward)
        qkv = zeros(attention, patches, (heads + 2 * kv_heads) * 64)
        query = zeros(attention, patches, heads, 64)
        key, value = (zeros(attention, patches, kv_heads, 64) for _ in range(2))
        attended = zeros(attention, patches, heads * 64)
        merged = zeros(merger, patches // 4, 5120)
        activated = zeros(merger, patches // 4, 5120 // shard.world_size)
        partial, output = (zeros(merger, patches // 4, hidden) for _ in range(2))
        if buffers is not None:
            buffers.join(*stages)
        return cls(pixels, residual, normalized, qkv, query, key, value, attended, projected,
                   reduced, mlp, merged, activated, partial, output)

    def rows(self, patches):
        return type(self)(*(self.mlp.rows(patches) if field == "mlp"
            else getattr(self, field)[:patches // 4 if field in ("merged", "activated", "partial", "output") else patches]
            for field in self.__dataclass_fields__))


class PreparedVision:
    def __init__(self, weights, scratch, inputs, *, runtime, coordinator, session, workspace, name, buffers=None):
        self.scratch, self.inputs, self.config = scratch, inputs, weights.config
        c, s = weights.config, scratch
        tokens, device = s.patches.shape[0], s.patches.device
        self.frequencies = 10000 ** (-torch.arange(0, 32, 2, dtype=torch.float32, device=device) / 32)
        def zeros(shape, dtype=torch.int32):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        self.positions = zeros((tokens, 2), torch.int64)
        self.columns, self.rows = zeros((tokens // 4,)), zeros((tokens // 4,))
        self.columns.copy_(torch.arange(tokens // 4, dtype=torch.int32, device=device))
        self.rows.copy_(self.columns)
        self.cu = zeros((2,))
        self.num_patches, self.num_features = zeros((1,)), zeros((1,))
        self.patch = UnquantizedProjection(s.patches, weights.patch, s.residual, session=session, name=name + ".patch")
        self.blocks = []
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.{index}"
            norm1 = PreparedRMSNorm(s.residual, w.norm1, s.normalized, eps=1e-6, session=session, name=prefix + ".norm1")
            qkv = UnquantizedProjection(s.normalized, w.qkv, s.qkv, bias=w.qkv_bias, session=session, name=prefix + ".qkv")
            output = UnquantizedProjection(s.attended, w.output, s.projected, session=session, name=prefix + ".output")
            norm2 = PreparedRMSNorm(s.residual, w.norm2, s.normalized, eps=1e-6, session=session, name=prefix + ".norm2")
            mlp = PreparedMLP(s.normalized, w.mlp, s.mlp.gate_up, s.mlp.intermediate, s.mlp.output,
                gate_up_bias=w.gate_up_bias, session=session, workspace=workspace, name=prefix + ".mlp")
            window = None if index in c["fullatt_block_indexes"] else (c["visual_token_window_size"],) * 2
            attention = PreparedVarlenAttention(s.query, s.key, s.value, self.cu, s.attended.view_as(s.query),
                window=window, sink=w.sink, session=session, workspace=workspace, name=prefix + ".attention")
            self.blocks.append((norm1, qkv, attention, output, w.output_bias, norm2, mlp, w.down_bias))
        self.reduce = PreparedAllReduce(s.projected, s.reduced, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".reduce")
        self.norm = PreparedRMSNorm(s.residual, weights.norm, s.normalized, eps=1e-6, session=session, name=name + ".norm")
        self.merge_up = UnquantizedProjection(s.merged, weights.merge_up, s.activated, session=session, name=name + ".merge_up")
        self.merge_down = UnquantizedProjection(s.activated, weights.merge_down, s.partial, session=session, name=name + ".merge_down")
        self.merge_reduce = PreparedAllReduce(s.partial, s.output, runtime=runtime, coordinator=coordinator,
                                             session=session, name=name + ".merge_reduce")

    def __call__(self):
        s, c = self.scratch, self.config
        self.patch(s.patches)
        angles = (self.positions.float()[:, :, None] * self.frequencies).flatten(1)[:, None]
        col_angles = angles.unflatten(0, (-1, 4))[self.columns.long()].flatten(0, 1)
        def rotate(source, destination, angles):
            left, right = source.float().chunk(2, -1)
            cos, sin = angles.cos(), angles.sin()
            destination.copy_(torch.cat((left * cos - right * sin, right * cos + left * sin), -1))

        previous = -1
        for index, (norm1, project, attention, output, output_bias, norm2, mlp, down_bias) in enumerate(self.blocks):
            kind = c["vit_window_attn_types"][index]
            if (kind == 1) != (previous == 1):
                order = self.columns if kind == 1 else self.rows
                s.residual.copy_(s.residual.unflatten(0, (-1, 4))[order.long()].flatten(0, 1))
            previous = kind
            q, k, v = project(norm1(s.residual)).split((s.query[0].numel(), s.key[0].numel(), s.value[0].numel()), -1)
            selected = col_angles if kind == 1 else angles
            rotate(q.view_as(s.query), s.query, selected)
            rotate(k.view_as(s.key), s.key, selected)
            s.value.copy_(v.view_as(s.value))
            attention()
            s.residual.add_(self.reduce(output(s.attended)) + output_bias)
            s.residual.add_(self.reduce(mlp(norm2(s.residual))) + down_bias)
        if previous == 1:
            s.residual.copy_(s.residual.unflatten(0, (-1, 4))[self.rows.long()].flatten(0, 1))
        self.norm(s.residual)
        s.merged.copy_(s.normalized.view_as(s.merged))
        s.activated.copy_(F.gelu(self.merge_up(s.merged)))
        return self.merge_reduce(self.merge_down(s.activated))
