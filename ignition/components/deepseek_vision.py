# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared DeepSeek ViT, spatial aligner, and bounded image feature delivery."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ignition.components.vision_delivery import PreparedFeaturePublication
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.components.weights import MLPWeights
from ignition.preparation.persistent import persistent_copy

@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_merge_indices: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_num_features: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device, *, patches):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((patches, 588), torch.float32),
            zeros((patches, 2), torch.int64),
            zeros((patches // 9, 9)),
            zeros(2),
            zeros(1),
            zeros(1),
            zeros(1) - 1,
            zeros(1, torch.int64),
            zeros(1),
        )

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
    mlp: MLPScratch
    merged: torch.Tensor
    aligned: torch.Tensor
    activated: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, device, *, hidden):
        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)

        return cls(zeros(patches, 588), zeros(patches, 1024), zeros(patches, 1024),
            zeros(patches, 3072), *(zeros(patches, 16, 64) for _ in range(3)),
            zeros(patches, 1024), zeros(patches, 1024), MLPScratch.allocate(patches, 1024, 2816, device),
            zeros(patches // 9, 9216), *(zeros(patches // 9, hidden) for _ in range(3)))

    def rows(self, patches):
        return type(self)(
            *(
                self.mlp.rows(patches)
                if name == "mlp"
                else getattr(self, name)[
                    : patches // 9 if name in ("merged", "aligned", "activated", "output") else patches
                ]
                for name in self.__dataclass_fields__
            )
        )


class PreparedVision:
    def __init__(self, weights, scratch, inputs, *, session, workspace, name):
        self.inputs = inputs
        self.capacity = scratch.patches.shape[0]
        tokens, device = scratch.patches.shape[0], scratch.patches.device
        self.frequencies = persistent_copy(10000 ** (-torch.arange(0, 32, 2, dtype=torch.float32, device=device) / 32))
        cu = torch.zeros(2, dtype=torch.int32, device=device)

        def project(key, source, out):
            return UnquantizedProjection(
                source,
                weights[key + ".weight"],
                out,
                bias=weights.get(key + ".bias"),
                session=session,
                name=name + "." + key,
            ).functional(session=session)

        self.patch = project("vision.patch_embed.proj", scratch.patches, scratch.residual)
        self.blocks = []
        for index in range(32):
            prefix = f"vision.blocks.{index}"
            norm1 = PreparedRMSNorm(
                scratch.residual,
                weights[prefix + ".norm1.weight"],
                scratch.normalized,
                eps=1e-6,
                session=session,
                name=f"{name}.{index}.norm1",
            ).functional()
            qkv = project(prefix + ".attn.wqkv", scratch.normalized, scratch.qkv)
            output = project(prefix + ".attn.wo", scratch.attended, scratch.projected)
            norm2 = PreparedRMSNorm(
                scratch.residual,
                weights[prefix + ".norm2.weight"],
                scratch.normalized,
                eps=1e-6,
                session=session,
                name=f"{name}.{index}.norm2",
            ).functional()
            mlp = PreparedMLP(
                scratch.normalized,
                MLPWeights(weights[prefix + ".mlp.w1.weight"], weights[prefix + ".mlp.w2.weight"]),
                scratch.mlp.gate_up,
                scratch.mlp.intermediate,
                scratch.mlp.output,
                session=session,
                name=f"{name}.{index}.mlp",
                workspace=workspace,
                round_silu=True,
            ).functional(session=session)
            self.blocks.append((norm1, qkv, output, norm2, mlp))
        self.attention = PreparedVarlenAttention(
            scratch.query,
            scratch.key,
            scratch.value,
            cu,
            scratch.attended.view(tokens, 16, 64),
            session=session,
            workspace=workspace,
            name=name + ".attention",
        ).functional(session=session)
        self.norm = PreparedRMSNorm(
            scratch.residual,
            weights["vision.norm.weight"],
            scratch.normalized,
            eps=1e-6,
            session=session,
            name=name + ".norm",
        ).functional()
        self.align1 = project("aligner.w1", scratch.merged, scratch.aligned)
        self.align2 = project("aligner.w2", scratch.activated, scratch.output)

    def __call__(self, pixels, positions, cu, indices):
        residual = self.patch(pixels)
        angles = (positions.float()[:, :, None] * self.frequencies).flatten(1)[:, None, :]
        cos, sin = angles.cos(), angles.sin()

        def rotate(source):
            a, b = source.float().chunk(2, dim=-1)
            return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(source.dtype)

        for norm1, qkv, output, norm2, mlp in self.blocks:
            q, k, v = qkv(norm1(residual)).view(-1, 3, 16, 64).unbind(1)
            attended = self.attention(rotate(q), rotate(k), v.contiguous(), cu).flatten(1)
            residual = residual + output(attended)
            residual = residual + mlp(norm2(residual))
        normalized = self.norm(residual)
        indices = indices.long()
        gathered = normalized[indices.clamp(0, normalized.shape[0] - 1)]
        gathered = torch.where(indices[:, :, None] >= 0, gathered, 0)
        merged = gathered.transpose(1, 2).reshape(indices.shape[0], -1)
        return self.align2(F.gelu(self.align1(merged)))


@dataclass(frozen=True)
class PrefillVisionInputs:
    vision_token_indices: torch.Tensor
    vision_left: torch.Tensor
    vision_right: torch.Tensor
    vision_feature_slot: torch.Tensor

    @classmethod
    def allocate(cls, tokens, device):
        return cls(
            torch.full((tokens,), -1, dtype=torch.int32, device=device),
            torch.zeros(tokens, dtype=torch.int32, device=device),
            torch.zeros(tokens, dtype=torch.int32, device=device),
            torch.full((1,), -1, dtype=torch.int32, device=device),
        )

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class VisionState:
    def __init__(self, weights, inputs, prefill, requests, *, features, token_base, sentinel_names, feature_type,
                 image_slots=None, arena=None):
        from ignition.components.capacity import image_capacity

        self.inputs, self.prefill, self.requests = inputs, prefill, requests
        self.token_base, self.feature_token = token_base, token_base + feature_type
        device = inputs.token_ids.device
        self.feature_slot = prefill.vision_feature_slot
        shape = (image_capacity(requests.capacity, image_slots), features, weights["image_start"].numel())
        if arena is None:
            arena = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        elif (arena.shape != shape or arena.dtype != torch.bfloat16 or arena.device != device
              or not arena.is_contiguous()):
            raise ValueError("image feature arena must be contiguous BF16 storage with the prepared shape and device")
        self.arena = arena
        self.generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.count = torch.zeros(requests.capacity, dtype=torch.int32, device=device)
        self.sentinels = persistent_copy(torch.stack([weights[name] for name in sentinel_names]))

    def visible(self, metadata):
        count = metadata.token_ids.numel()
        return self.prefill.vision_left[:count], self.prefill.vision_right[:count]

    def validate_prefill(self, tokens, lengths):
        b, p = self.inputs, self.prefill
        slot = b.row_slot_ids[:1].long().clamp(0, self.requests.capacity - 1)
        index = p.vision_token_indices[: tokens.numel()]
        offsets = torch.arange(tokens.numel(), device=tokens.device)
        live, image = offsets < lengths[0], tokens >= self.token_base
        feature = tokens == self.feature_token
        valid = (p.vision_left[: tokens.numel()] >= 0) & (p.vision_left[: tokens.numel()] <= offsets)
        valid &= (p.vision_right[: tokens.numel()] >= 0) & (
            p.vision_right[: tokens.numel()] + offsets < lengths[0]
        )
        valid &= (~feature & (index == -1)) | (feature & (index >= 0) & (index < self.count[slot]))
        valid &= ~image | (self.generation[slot] == b.row_generations[0])
        valid &= ~image | ((self.feature_slot >= 0) & (self.feature_slot < self.arena.shape[0]))
        return (~live | valid).all()

    def inject(self, metadata, embedding):
        slot = self.feature_slot.long().clamp(0, self.arena.shape[0] - 1)
        tokens = metadata.token_ids
        index = self.prefill.vision_token_indices[: tokens.numel()].long().clamp(0, self.arena.shape[1] - 1)
        value = torch.where(
            (tokens == self.feature_token)[:, None],
            self.arena[slot, index],
            self.sentinels[(tokens - self.token_base).clamp(0, self.sentinels.shape[0] - 1)],
        )
        embedding.copy_(torch.where((tokens >= self.token_base)[:, None], value, embedding))


class PreparedVisionGeneration:
    def __init__(self, encoder, state, completion, frame, sequence):
        self.encoder, self.state = encoder, state
        self.publish = PreparedFeaturePublication(state, encoder.inputs, completion, frame, sequence)

    def __call__(self):
        e, s = self.encoder, self.state
        i, r = e.inputs, s.requests
        capacity = e.capacity
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, features = i.vision_num_patches, i.vision_num_features
        offset = i.vision_feature_offset
        current = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        current &= (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid = current & (count > 0) & (count <= capacity) & (features > 0) & (features <= capacity // 9)
        valid &= (offset >= 0) & (offset + features <= s.arena.shape[1])
        valid &= (offset == 0) | ((s.generation[slot] == i.vision_generation) & (s.count[slot] == offset))
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count)
        patch_live = torch.arange(capacity, device=count.device) < count
        indices = i.vision_merge_indices[: capacity // 9]
        feature_live = torch.arange(capacity // 9, device=count.device) < features
        valid &= (~feature_live[:, None] | ((indices >= -1) & (indices < count))).all()
        valid &= (~patch_live[:, None] | torch.isfinite(i.vision_pixel_values[:capacity])).all()
        valid &= (
            ~patch_live[:, None]
            | ((i.vision_position_ids[:capacity] >= 0) & (i.vision_position_ids[:capacity] < capacity))
        ).all()
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        errors = (~valid).to(torch.int32)
        pixels = torch.where(valid & patch_live[:, None], i.vision_pixel_values[:capacity], 0).to(torch.bfloat16)
        positions = torch.where(valid & patch_live[:, None], i.vision_position_ids[:capacity], 0)
        merge = torch.where(valid & feature_live[:, None], indices, -1)
        cu = torch.where(valid, i.vision_cu_seqlens, 0)
        num_features = torch.where(valid, features, 0)
        output = e(pixels, positions, cu, merge)
        finite = (~feature_live[:, None] | torch.isfinite(output)).all()
        errors = errors | (~finite).int() * 64
        self.publish(output, num_features, errors)
