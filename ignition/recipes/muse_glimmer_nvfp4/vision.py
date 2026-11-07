# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured Muse Glimmer vision encoder using canonical blocks and prepared attention.

The frontend packs each image's patches in window order (32x32-patch cells in raster
order, patches in raster order within a cell) and emits two cumulative-length tensors:
``cu_window_seqlens`` (one segment per cell, the 37 window layers) and ``cu_seqlens``
(one per image, the 13 full layers). ``reverse_index`` restores raster order after the
layers; the graph then gathers 2x2 blocks the way ``get_vision_pixel_shuffle_index``
does and folds them dim-major into 4 x hidden features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from b12x.preparation import PreparationSession
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerRMSNorm,
    MuseGlimmerVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)

from ignition.components.linear import UnquantizedProjection
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.preparation.scratch import ScratchArena

HIDDEN, HEADS, HEAD_DIM, INTERMEDIATE = 1536, 16, 96, 8960
PIXEL_WIDTH = 2 * 3 * 14 * 14  # (temporal, channel, y, x)
POSITION_TABLE = 32 * 32
MERGED = 4 * HIDDEN
ADAPTER = 4096
WIDTH = 6656
WINDOW_PATCHES = 32 * 32  # 448 px / 14 px per side
PATCH_CAPACITIES = (1024, 4096, 16384)
PERCEPTION_NORM_EPS = 1e-5
TOWER = "model.vision_tower."
OUTER = ("model.vision_adapter.fc1.weight", "model.vision_adapter.fc2.weight", "model.vision_projection.weight")


def window_capacity(patches: int, images: int) -> int:
    """Cells an image set of ``patches`` even-sided patches can span: ceil(h/32)*ceil(w/32) <= h*w/64 + 1."""
    return patches // 64 + images


@dataclass(frozen=True)
class VisionMetadata:
    interpolation_indices: torch.Tensor
    interpolation_weights: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_window_seqlens: torch.Tensor
    reverse_index: torch.Tensor
    num_patches: torch.Tensor

    @classmethod
    def allocate(cls, patches: int, images: int, device: torch.device, *, buffers=None) -> VisionMetadata:
        if patches <= 0 or patches % 4 or images <= 0:
            raise ValueError("vision requires positive image capacity and a patch capacity divisible by four")
        def zeros(shape, dtype):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        return cls(zeros((patches, 4), torch.int64), zeros((patches, 4), torch.float32),
                   zeros((patches, 2), torch.int64), zeros((images + 1,), torch.int32),
                   zeros((window_capacity(patches, images) + 1,), torch.int32),
                   zeros((patches,), torch.int64), zeros((1,), torch.int32))


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
    raster: torch.Tensor
    merged: torch.Tensor
    adapter_intermediate: torch.Tensor
    adapter: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches: int, device: torch.device, *, buffers=None) -> VisionScratch:
        if patches <= 0 or patches % 4:
            raise ValueError("vision patch capacity must be positive and divisible by four")
        def zeros(cursor, *shape):
            return (torch.zeros(shape, device=device, dtype=torch.bfloat16) if cursor is None
                    else cursor.zeros(shape, torch.bfloat16))
        hidden, projected = (zeros(buffers, patches, HIDDEN) for _ in range(2))
        stages = [buffers.fork() for _ in range(4)] if buffers is not None else [None] * 4
        attention, mlp, patch, merger = stages
        query, key, value = (zeros(attention, patches, HEADS, HEAD_DIM) for _ in range(3))
        projection = attention.fork() if attention is not None else None
        attended = attention.fork() if attention is not None else None
        qkv = zeros(projection, patches, 3 * HIDDEN)
        output = zeros(attended, patches, HIDDEN)
        if attention is not None:
            # QKV is consumed by the rotary and value copies before attention writes its output.
            attention.join(projection, attended)
        intermediate = zeros(mlp, patches, INTERMEDIATE)
        pixels = zeros(patch, patches, PIXEL_WIDTH)
        raster = zeros(merger, patches, HIDDEN)
        merged = zeros(merger, patches // 4, MERGED)
        adapter_intermediate, adapter = (zeros(merger, patches // 4, ADAPTER) for _ in range(2))
        features = zeros(merger, patches // 4, WIDTH)
        if buffers is not None:
            buffers.join(*stages)
        return cls(pixels, hidden, qkv, query, key, value, output, projected, intermediate,
                   raster, merged, adapter_intermediate, adapter, features)

    def rows(self, patches: int) -> VisionScratch:
        if patches <= 0 or patches % 4 or patches > self.patches.shape[0]:
            raise ValueError("vision view exceeds its patch capacity")
        merged = ("merged", "adapter_intermediate", "adapter", "output")
        return type(self)(*(getattr(self, name)[:patches // 4 if name in merged else patches]
                            for name in self.__dataclass_fields__))


@dataclass(frozen=True)
class VisionWeights:
    """BF16 tower parameters for the canonical modules and the projection weights the graph binds."""

    parameters: dict[str, torch.Tensor]
    projections: dict[str, torch.Tensor]

    @classmethod
    def load(cls, module, loader) -> VisionWeights:
        paths = [path for path, _ in module.named_parameters()]
        tensors = loader.load_tensors((*(TOWER + path for path in paths), *OUTER))
        loaded = dict(zip(paths, tensors[:len(paths)], strict=True))
        parameters, projections = {}, dict(zip(("adapter.fc1.weight", "adapter.fc2.weight", "projection.weight"),
                                               tensors[len(paths):], strict=True))
        for path, tensor in loaded.items():
            kind = path.split(".")[-2]
            if kind in ("q_proj", "k_proj", "v_proj"):
                continue
            (projections if kind in ("patch_embedding", "proj", "fc1", "fc2") else parameters)[path] = tensor
        for i in range(len(module.layers)):
            for part in ("weight", "bias"):
                projections[f"layers.{i}.attn.qkv.{part}"] = torch.cat(
                    [loaded[f"layers.{i}.attn.{name}_proj.{part}"] for name in ("q", "k", "v")])
        return cls(parameters, projections)


class _Linear(torch.nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.projection = projection

    def forward(self, source):
        return self.projection(source)


class _Attention(torch.nn.Module):
    def __init__(self, scratch, weights, prefix, attention, *, session, name):
        super().__init__()
        self.scratch, self.attention = scratch, attention
        self.qkv = UnquantizedProjection(scratch.hidden, weights[prefix + ".qkv.weight"], scratch.qkv,
            session=session, name=f"{name}.qkv", bias=weights[prefix + ".qkv.bias"])
        self.proj = UnquantizedProjection(scratch.attention, weights[prefix + ".proj.weight"], scratch.projected,
            session=session, name=f"{name}.proj", bias=weights[prefix + ".proj.bias"])

    def forward(self, hidden_states, cu_seqlens=None, position_embeddings=None, **kwargs):
        s = self.scratch
        projected = self.qkv(hidden_states).view(-1, 3, HEADS, HEAD_DIM)
        query, key = apply_rotary_pos_emb_vision(projected[:, 0], projected[:, 1], *position_embeddings)
        s.query.copy_(query)
        s.key.copy_(key)
        s.value.copy_(projected[:, 2])
        self.attention()
        return self.proj(s.attention)


class PreparedVision:
    """Patch one canonical ``MuseGlimmerVisionModel`` tree for a fixed patch-capacity graph.

    Geometry is computed by the frontend and copied into metadata (one image per
    graph). Padding is excluded from attention by cumulative lengths. Canonical
    layer normalization, activation, residual and rotary methods run only during
    graph construction. The adapter, projection and scale-less RMSNorm of
    ``MuseGlimmerModel.get_image_features`` follow the tower. Rows past
    ``num_patches // 4`` of the persistent output are zeroed.
    """

    def __init__(self, module, scratch: VisionScratch, metadata: VisionMetadata,
                 weights: VisionWeights, *, session: PreparationSession, name: str = "vision",
                 attention_workspace: ScratchArena | torch.Tensor | None = None,
                 attention_math_mode: str = "flash"):
        if attention_math_mode not in ("flash", "eager_bf16"):
            raise ValueError("vision attention math mode must be flash or eager_bf16")
        config = module.config
        if ((config.hidden_size, config.num_attention_heads, config.intermediate_size, config.patch_size,
             config.patch_temporal, config.pos_emb_height * config.pos_emb_width, config.merge_size)
                != (HIDDEN, HEADS, INTERMEDIATE, 14, 2, POSITION_TABLE, 2)
                or len(module.layers) != len(config.layer_types)
                or any(kind not in ("window_attention", "full_attention") for kind in config.layer_types)):
            raise ValueError("vision geometry differs from the checkpoint contract")
        self.module, self.scratch, self.metadata = module, scratch, metadata
        capacity, device = scratch.hidden.shape[0], scratch.hidden.device
        if metadata.interpolation_indices.shape[0] != capacity or metadata.reverse_index.shape[0] != capacity:
            raise ValueError("vision metadata and scratch capacities differ")
        if metadata.num_patches.item() != 0 or bool(metadata.cu_seqlens.any().item()) \
                or bool(metadata.cu_window_seqlens.any().item()):
            raise ValueError("vision preparation requires inactive metadata")
        for path, tensor in weights.parameters.items():
            parent, _, attribute = path.rpartition(".")
            setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(tensor, requires_grad=False))
        p = weights.projections
        self.patch_embed = UnquantizedProjection(scratch.patches, p["patch_embedder.patch_embedding.weight"],
            scratch.hidden, session=session, name=f"{name}.patch_embed")
        module.rotary_emb = MuseGlimmerVisionRotaryEmbedding(config).to(device)
        attention = {}
        for kind, cu, longest in (("window_attention", metadata.cu_window_seqlens, min(WINDOW_PATCHES, capacity)),
                                  ("full_attention", metadata.cu_seqlens, capacity)):
            attention[kind] = PreparedVarlenAttention(scratch.query, scratch.key, scratch.value, cu,
                scratch.attention.view_as(scratch.query), session=session, workspace=attention_workspace,
                name=f"{name}.{kind}", max_seqlen=longest, math_mode=attention_math_mode)
        for i, (layer, kind) in enumerate(zip(module.layers, config.layer_types, strict=True)):
            prefix = f"layers.{i}"
            layer.attn = _Attention(scratch, p, prefix + ".attn", attention[kind],
                                    session=session, name=f"{name}.layers.{i}.attn")
            for key, source, destination in (("fc1", scratch.hidden, scratch.intermediate),
                                             ("fc2", scratch.intermediate, scratch.projected)):
                setattr(layer.mlp, key, _Linear(UnquantizedProjection(source, p[f"{prefix}.mlp.{key}.weight"],
                    destination, session=session, name=f"{name}.layers.{i}.mlp.{key}",
                    bias=p[f"{prefix}.mlp.{key}.bias"])))
        self.adapter_fc1 = UnquantizedProjection(scratch.merged, p["adapter.fc1.weight"], scratch.adapter_intermediate,
                                                 session=session, name=f"{name}.adapter.fc1")
        self.adapter_fc2 = UnquantizedProjection(scratch.adapter_intermediate, p["adapter.fc2.weight"], scratch.adapter,
                                                 session=session, name=f"{name}.adapter.fc2")
        self.projection = UnquantizedProjection(scratch.adapter, p["projection.weight"], scratch.output,
                                                session=session, name=f"{name}.projection")
        self.patch_rows = torch.arange(capacity, device=device)
        self.output_rows = torch.arange(capacity // 4, device=device)
        self.perception_norm = MuseGlimmerRMSNorm(eps=PERCEPTION_NORM_EPS, with_scale=False)

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        s, m, module = self.scratch, self.metadata, self.module
        capacity = s.hidden.shape[0]
        s.patches.copy_(pixels)
        hidden = self.patch_embed(s.patches)
        positions = (module.patch_embedder.position_embedding_table(m.interpolation_indices)
                     * m.interpolation_weights[:, :, None]).sum(1)
        hidden = module.ln_pre(hidden + positions.to(hidden.dtype))
        # The canonical encoder feeds `position_ids.flip(-1) + 1`; the frontend emits `(row, col)`.
        rope = module.rotary_emb(hidden, m.position_ids.flip(-1) + 1)
        for layer in module.layers:
            hidden = layer(hidden, cu_seqlens=None, position_embeddings=rope)
        s.raster.copy_(module.ln_post(hidden)[m.reverse_index])
        # `get_vision_pixel_shuffle_index`: raster row of merged block b, corner i, for a grid `width` wide.
        live = self.patch_rows < m.num_patches
        width = m.position_ids[:, 1].masked_fill(~live, 0).max() + 1
        half = (width // 2).clamp_min(1)
        block, corner = self.patch_rows // 4, self.patch_rows % 4
        source = (((block // half) * 2 + corner // 2) * width + (block % half) * 2 + corner % 2).clamp(max=capacity - 1)
        s.merged.view(capacity // 4, HIDDEN, 4).copy_(s.raster[source].view(capacity // 4, 4, HIDDEN).permute(0, 2, 1))
        activated = torch.nn.functional.gelu(self.adapter_fc1(s.merged))
        activated = torch.nn.functional.gelu(self.adapter_fc2(activated))
        projected = self.projection(activated).float()
        s.output.copy_(self.perception_norm(projected))
        s.output.masked_fill_((self.output_rows >= m.num_patches // 4)[:, None], 0)
        return s.output
