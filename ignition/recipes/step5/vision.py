# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared perception encoder with canonical transformer blocks and strided convolutions."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.step3p7.modeling_step3p7 import Step3p7VisionEncoderLayer

from ignition.components.collectives import PreparedAllReduce
from ignition.components.linear import UnquantizedProjection
from ignition.components.varlen_attention import PreparedVarlenAttention

PATCH_CAPACITIES = (4096, 8192, 16384, 32768, 65536)
FEATURES = PATCH_CAPACITIES[-1] // 16
MAX_VIEWS = (PATCH_CAPACITIES[-1] - 2704) // 1296 + 1


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_interpolation_indices: torch.Tensor
    vision_interpolation_weights: torch.Tensor
    vision_downsample1_indices: torch.Tensor
    vision_downsample2_indices: torch.Tensor
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
            zeros((n, 4), torch.int64), zeros((n, 4), torch.float32), zeros((n // 4, 9)), zeros((n // 16, 9)),
            zeros(MAX_VIEWS + 1), zeros(1), zeros(1), zeros(1) - 1, zeros(1, torch.int64), zeros(1))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    hidden: torch.Tensor
    normalized: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    projected: torch.Tensor
    reduced: torch.Tensor
    mlp_up: torch.Tensor
    activated: torch.Tensor
    first_input: torch.Tensor
    first_output: torch.Tensor
    second_input: torch.Tensor
    second_output: torch.Tensor
    second_reduced: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, shard, device, *, buffers=None):
        def zeros(cursor, *shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=device) if cursor is None
                    else cursor.zeros(shape, torch.bfloat16))
        world = shard.world_size
        heads = 16 // world
        pixels = zeros(buffers, patches, 588)
        hidden, normalized, projected, reduced = (zeros(buffers, patches, 1536) for _ in range(4))
        stages = [buffers.fork() for _ in range(3)] if buffers is not None else [None] * 3
        attention, feedforward, merger = stages
        mlp_up, activated = (zeros(feedforward, patches, 8960 // world) for _ in range(2))
        qkv = zeros(attention, patches, 3 * heads * 96)
        query, key, value, attended = (zeros(attention, patches, heads, 96) for _ in range(4))
        first_input = zeros(merger, patches // 4, 1536 * 9)
        first_output = zeros(merger, patches // 4, 3072 // world)
        second_input = zeros(merger, patches // 16, 3072 // world * 9)
        second_output, second_reduced = (zeros(merger, patches // 16, 6144) for _ in range(2))
        output = zeros(merger, patches // 16, 4096 // world)
        if buffers is not None:
            buffers.join(*stages)
        return cls(pixels, hidden, normalized, qkv, query, key, value, attended, projected,
                   reduced, mlp_up, activated, first_input, first_output, second_input,
                   second_output, second_reduced, output)

    def rows(self, patches):
        def divisor(name):
            return 4 if name.startswith("first_") else 16 if name.startswith("second_") or name == "output" else 1
        return type(self)(*(getattr(self, name)[:patches // divisor(name)] for name in self.__dataclass_fields__))


def layer_norm(weights):
    weight, bias = weights
    module = nn.LayerNorm(weight.numel(), eps=1e-5, device="meta", dtype=weight.dtype)
    module.weight, module.bias = nn.Parameter(weight, requires_grad=False), nn.Parameter(bias, requires_grad=False)
    return module


def rotate(source, cosine, sine):
    pairs = source.unflatten(-1, (-1, 2))
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
    return source * cosine + rotated * sine


class VisionAttention(nn.Module):
    def __init__(self, weights, scratch, cu, *, runtime, coordinator, session, workspace, name):
        super().__init__()
        s = self.scratch = scratch
        self.qkv = UnquantizedProjection(s.normalized, weights["qkv"][0], s.qkv, bias=weights["qkv"][1],
                                         session=session, name=name + ".qkv")
        self.attention = PreparedVarlenAttention(s.query, s.key, s.value, cu, s.attended, max_seqlen=2704,
                                                 session=session, workspace=workspace, name=name + ".attention")
        self.output = UnquantizedProjection(s.attended.flatten(1), weights["output"][0], s.projected,
                                            session=session, name=name + ".output")
        self.reduction = PreparedAllReduce(s.projected, s.reduced, runtime=runtime, coordinator=coordinator,
                                           session=session, name=name + ".reduce")
        self.bias = weights["output"][1]

    def forward(self, hidden_states, *, position_embeddings, **kwargs):
        s = self.scratch
        s.normalized.copy_(hidden_states)
        q, k, v = self.qkv(s.normalized).chunk(3, -1)
        cosine, sine = position_embeddings
        s.query.copy_(rotate(q.view_as(s.query), cosine, sine))
        s.key.copy_(rotate(k.view_as(s.key), cosine, sine))
        s.value.copy_(v.view_as(s.value))
        self.attention()
        return self.reduction(self.output(s.attended.flatten(1))) + self.bias, None


class VisionMLP(nn.Module):
    def __init__(self, weights, scratch, *, runtime, coordinator, session, name):
        super().__init__()
        s = self.scratch = scratch
        self.up = UnquantizedProjection(s.normalized, weights["up"][0], s.mlp_up, bias=weights["up"][1],
                                        session=session, name=name + ".up")
        self.down = UnquantizedProjection(s.activated, weights["down"][0], s.projected, session=session, name=name + ".down")
        self.reduction = PreparedAllReduce(s.projected, s.reduced, runtime=runtime, coordinator=coordinator,
                                           session=session, name=name + ".reduce")
        self.bias, self.activation = weights["down"][1], ACT2FN["quick_gelu"]

    def forward(self, hidden_states):
        s = self.scratch
        s.normalized.copy_(hidden_states)
        s.activated.copy_(self.activation(self.up(s.normalized)))
        return self.reduction(self.down(s.activated)) + self.bias


@triton.jit
def _gather_neighborhood(Source, Indices, Output, CHANNELS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    neighbor = tl.load(Indices + row * 9 + column % 9, column < CHANNELS * 9, other=-1)
    value = tl.load(Source + neighbor * CHANNELS + column // 9,
                    (column < CHANNELS * 9) & (neighbor >= 0), other=0)
    tl.store(Output + row * CHANNELS * 9 + column, value, column < CHANNELS * 9)


def neighborhood(source, indices, output):
    torch.library.wrap_triton(_gather_neighborhood)[(output.shape[0], triton.cdiv(output.shape[1], 256))](
        source, indices, output, source.shape[1], 256)


class PreparedVision:
    def __init__(self, weights, scratch, inputs, *, runtime, coordinator, session, workspace, name, buffers=None):
        self.weights, self.scratch, self.inputs = weights, scratch, inputs
        s = scratch
        n, device = s.patches.shape[0], s.patches.device
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)
        self.positions = zeros((n, 2), torch.int64)
        self.interpolation_indices = zeros((n, 4), torch.int64)
        self.interpolation_weights = zeros((n, 4), torch.float32)
        self.first_indices = zeros((n // 4, 9))
        self.second_indices = zeros((n // 16, 9))
        self.cu = zeros((MAX_VIEWS + 1,))
        self.num_patches, self.num_features = zeros((1,)), zeros((1,))
        self.inverse = 10000 ** (-torch.arange(0, 48, 2, dtype=torch.float32, device=device) / 48)
        self.patch = UnquantizedProjection(s.patches, weights.patch, s.hidden, session=session, name=name + ".patch")
        self.norm = layer_norm(weights.norm)
        self.blocks = []
        common = {"runtime": runtime, "coordinator": coordinator, "session": session}
        for i, w in enumerate(weights.layers):
            block = Step3p7VisionEncoderLayer.__new__(Step3p7VisionEncoderLayer)
            nn.Module.__init__(block)
            block.layernorm_before, block.layernorm_after = layer_norm(w["norm1"]), layer_norm(w["norm2"])
            block.lambda_1, block.lambda_2 = (nn.Parameter(w[key], requires_grad=False) for key in ("scale1", "scale2"))
            block.self_attn = VisionAttention(w, s, self.cu, workspace=workspace, name=f"{name}.{i}.attention", **common)
            block.mlp = VisionMLP(w, s, name=f"{name}.{i}.mlp", **common)
            self.blocks.append(block)
        self.first = UnquantizedProjection(s.first_input, weights.first[0], s.first_output, bias=weights.first[1],
                                           session=session, name=name + ".downsample1")
        self.second = UnquantizedProjection(s.second_input, weights.second[0], s.second_output,
                                            session=session, name=name + ".downsample2")
        self.reduce_second = PreparedAllReduce(s.second_output, s.second_reduced, name=name + ".downsample_reduce", **common)
        self.projector = UnquantizedProjection(s.second_reduced, weights.projector, s.output,
                                               session=session, name=name + ".projector")

    def __call__(self):
        s, w = self.scratch, self.weights
        self.patch(s.patches)
        position = (w.position[self.interpolation_indices].float() * self.interpolation_weights[..., None]).sum(1).bfloat16()
        hidden = self.norm(s.hidden + position)
        angles = (self.positions.flip(-1).float()[..., None] * self.inverse).repeat_interleave(2, -1).flatten(1).bfloat16()[:, None]
        embeddings = angles.cos(), angles.sin()
        for block in self.blocks:
            hidden = block(hidden, position_embeddings=embeddings)
        s.hidden.copy_(hidden)
        neighborhood(s.hidden, self.first_indices, s.first_input)
        self.first(s.first_input)
        neighborhood(s.first_output, self.second_indices, s.second_input)
        self.reduce_second(self.second(s.second_input)).add_(w.second[1])
        return self.projector(s.second_reduced)

    def scratch_tensors(self):
        return (self.positions, self.interpolation_indices, self.interpolation_weights, self.first_indices,
                self.second_indices, self.cu, self.num_patches, self.num_features)
