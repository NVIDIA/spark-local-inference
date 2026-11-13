# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured Qwen vision encoder using canonical blocks and prepared attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from b12x.preparation import PreparationSession
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)

from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer


@dataclass(frozen=True)
class VisionMetadata:
    interpolation_indices: torch.Tensor
    interpolation_weights: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    num_patches: torch.Tensor

    @classmethod
    def allocate(cls, patches: int, images: int, device: torch.device) -> VisionMetadata:
        if patches <= 0 or patches % 4 or images <= 0:
            raise ValueError("vision requires positive image capacity and a patch capacity divisible by four")
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(zeros((patches, 4), torch.int64), zeros((patches, 4), torch.float32),
                   zeros((patches, 2), torch.int64), zeros((images + 1,), torch.int32),
                   zeros((1,), torch.int32))


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    hidden: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention: torch.Tensor
    projected: torch.Tensor
    intermediate: torch.Tensor
    merger_intermediate: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches: int, device: torch.device) -> VisionScratch:
        if patches <= 0 or patches % 4:
            raise ValueError("vision patch capacity must be positive and divisible by four")
        def zeros(*shape):
            return torch.zeros(shape, device=device, dtype=torch.bfloat16)
        hidden, projected = (zeros(patches, 1152) for _ in range(2))
        query, key, value = (zeros(patches, 16, 72) for _ in range(3))
        return cls(zeros(patches, 1536), hidden, zeros(patches, 3456), query, key, value,
                   zeros(patches, 1152), projected, zeros(patches, 4304),
                   zeros(patches // 4, 4608), zeros(patches // 4, 2560))

    def rows(self, patches: int) -> VisionScratch:
        if patches <= 0 or patches % 4 or patches > self.patches.shape[0]:
            raise ValueError("vision view exceeds its patch capacity")
        return type(self)(*(getattr(self, name)[:patches // 4 if name in ("merger_intermediate", "output") else patches]
                            for name in self.__dataclass_fields__))


@dataclass(frozen=True)
class VisionWeights:
    parameters: dict[str, torch.Tensor]
    projections: dict[str, object]

    @classmethod
    def load(cls, module, loader) -> VisionWeights:
        paths, projections = [], {}
        for path, _ in module.named_parameters():
            projection = path.removesuffix(".weight")
            quantized = (path.endswith(".weight") and not path.startswith("patch_embed.")
                         and path.split(".")[-2] in ("qkv", "proj", "linear_fc1", "linear_fc2"))
            if quantized:
                projections[projection] = loader.load_linear(f"model.visual.{projection}")
            else:
                paths.append(path)
        parameters = dict(zip(paths, loader.load_tensors(tuple(f"model.visual.{path}" for path in paths)), strict=True))
        return cls(parameters, projections)


class _Linear(torch.nn.Module):
    def __init__(self, projection, bias=None):
        super().__init__()
        self.projection = projection
        self.bias = bias

    def forward(self, source):
        output = self.projection(source)
        if self.bias is not None:
            output.add_(self.bias)
        return output


class _Attention(torch.nn.Module):
    def __init__(self, module, scratch, metadata, weights, *, session, name, shared=None,
                 projection_workspace=None, attention_workspace=None, attention_math_mode="flash"):
        super().__init__()
        self.scratch = scratch
        self.qkv = QuantizedProjection(scratch.hidden, weights["qkv"], scratch.qkv,
            session=session, name=f"{name}.qkv", bias=module.qkv.bias, workspace=projection_workspace)
        self.proj = QuantizedProjection(scratch.attention, weights["proj"], scratch.projected,
            session=session, name=f"{name}.proj", bias=module.proj.bias, workspace=projection_workspace)
        self.attention = shared.attention if shared is not None else PreparedVarlenAttention(
            scratch.query, scratch.key, scratch.value, metadata.cu_seqlens,
            scratch.attention.view_as(scratch.query), session=session, workspace=attention_workspace,
            name=name, math_mode=attention_math_mode)

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        projected = self.qkv(hidden_states).view(-1, 3, 16, 72)
        query, key = apply_rotary_pos_emb_vision(projected[:, 0], projected[:, 1], *position_embeddings)
        query = query.contiguous()
        key = key.contiguous()
        value = projected[:, 2].contiguous()
        return self.proj(self.attention(query, key, value, cu_seqlens).view(-1, 1152))


class PreparedVision:
    """Patch one canonical vision tree for a fixed patch-capacity graph.

    Geometry is computed by the native processor and copied into metadata.
    Padding is excluded from attention by cumulative lengths. Canonical block,
    layer normalization, activation, residual, and merger methods run only
    during graph construction. The returned embedding tensor is graph-local.
    Each capacity requires its own canonical module tree because methods are
    bound to capacity-specific prepared objects.
    """

    def __init__(self, module, scratch: VisionScratch, metadata: VisionMetadata,
                 weights: VisionWeights, *, session: PreparationSession, name: str = "vision",
                 projection_workspace: ScratchArena | torch.Tensor | None = None,
                 attention_workspace: ScratchArena | torch.Tensor | None = None,
                 attention_math_mode: str = "flash"):
        if attention_math_mode not in ("flash", "eager_bf16"):
            raise ValueError("vision attention math mode must be flash or eager_bf16")
        config = module.config
        if (config.hidden_size, config.num_heads, config.intermediate_size, config.depth) != (1152, 16, 4304, 27):
            raise ValueError("vision geometry differs from the checkpoint contract")
        self.module, self.scratch, self.metadata = module, scratch, metadata
        capacity, device = scratch.hidden.shape[0], scratch.hidden.device
        if metadata.interpolation_indices.shape[0] != capacity:
            raise ValueError("vision metadata and scratch capacities differ")
        if metadata.num_patches.item() != 0 or bool(metadata.cu_seqlens.any().item()):
            raise ValueError("vision preparation requires inactive metadata")
        for path, tensor in weights.parameters.items():
            parent, _, attribute = path.rpartition(".")
            setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(tensor, requires_grad=False))
        # Conv3d rounds the projection before adding the BF16 bias.
        self.patch_embed = _Linear(UnquantizedProjection(scratch.patches,
            module.patch_embed.proj.weight.flatten(1), scratch.hidden,
            session=session, name=f"{name}.patch_embed"), module.patch_embed.proj.bias)
        module.rotary_pos_emb = Qwen4ExpVisionRotaryEmbedding(config).to(device)
        for i, block in enumerate(module.blocks):
            prefix = f"blocks.{i}"
            block.attn = _Attention(block.attn, scratch, metadata,
                {key: weights.projections[f"{prefix}.attn.{key}"] for key in ("qkv", "proj")},
                session=session, name=f"{name}.blocks.{i}.attn",
                shared=module.blocks[0].attn if i else None, projection_workspace=projection_workspace,
                attention_workspace=attention_workspace, attention_math_mode=attention_math_mode)
            for key, source, destination in (("linear_fc1", scratch.hidden, scratch.intermediate),
                                              ("linear_fc2", scratch.intermediate, scratch.projected)):
                original = getattr(block.mlp, key)
                setattr(block.mlp, key, _Linear(QuantizedProjection(source,
                    weights.projections[f"{prefix}.mlp.{key}"], destination, session=session,
                    name=f"{name}.blocks.{i}.mlp.{key}", activation_mode="a16", bias=original.bias,
                    workspace=projection_workspace)))
        for key, source, destination in (("linear_fc1", scratch.hidden.view(capacity // 4, 4608), scratch.merger_intermediate),
                                         ("linear_fc2", scratch.merger_intermediate, scratch.output)):
            original = getattr(module.merger, key)
            setattr(module.merger, key, _Linear(QuantizedProjection(source,
                weights.projections[f"merger.{key}"], destination, session=session,
                name=f"{name}.merger.{key}", activation_mode="a16", bias=original.bias,
                workspace=projection_workspace)))
        def refresh_attention_workspace():
            if self.scratch is not None:
                self.attention_workspace = module.blocks[0].attn.attention.scratch

        register_finalizer(session, refresh_attention_workspace, workspace=attention_workspace)
        self.output_rows = torch.arange(capacity // 4, device=device)

    def functional(self, *, session):
        self.patch_embed.projection.functional(session=session)
        self.module.blocks[0].attn.attention.functional(session=session)
        for block in self.module.blocks:
            block.attn.qkv.functional(session=session)
            block.attn.proj.functional(session=session)
            block.attn.scratch = None
            block.mlp.linear_fc1.projection.functional(session=session)
            block.mlp.linear_fc2.projection.functional(session=session)
        self.module.merger.linear_fc1.projection.functional(session=session)
        self.module.merger.linear_fc2.projection.functional(session=session)
        self.capacity = self.scratch.hidden.shape[0]
        self.scratch = self.metadata = self.output_rows = None
        return self

    def __call__(self, pixels: torch.Tensor, metadata) -> torch.Tensor:
        module = self.module
        hidden = self.patch_embed(pixels.to(torch.bfloat16))
        positions = (module.pos_embed(metadata.interpolation_indices) * metadata.interpolation_weights[:, :, None]).sum(1)
        hidden = hidden + positions.to(hidden.dtype)
        rope = module.rotary_pos_emb(hidden, metadata.position_ids)
        for block in module.blocks:
            hidden = block(hidden, cu_seqlens=metadata.cu_seqlens, position_embeddings=rope)
        result = module.merger(hidden)
        output_rows = torch.arange(self.capacity // 4, device=pixels.device)
        result.masked_fill_((output_rows >= metadata.num_patches // 4)[:, None], 0)
        return result
