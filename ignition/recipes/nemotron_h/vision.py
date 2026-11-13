# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""C-RADIO image encoding with prepared projections."""

from dataclasses import dataclass
import json

import torch
from torch import nn
from transformers.models.radio.configuration_radio import RadioConfig
from transformers.models.radio.modeling_radio import RadioLayer

from ignition.components.linear import UnquantizedProjection
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.preparation.examples import declaration_tensor
from ignition.components.weights import DirectWeightLoader

PATCH_CAPACITIES = (1024, 4096, 13312)
FEATURES = 13312
HIDDEN = 4096
VISION_HIDDEN = 1280
REGISTER_TOKENS = 10
PIXEL_WIDTH = 3 * 16 * 16
PREFIX = "vision_model.radio_model.model."


@dataclass(frozen=True)
class VisionConfig:
    radio: dict
    projector_hidden: int
    hidden: int

    @classmethod
    def from_dict(cls, raw):
        required = dict(downsample_ratio=0.5, patch_size=16, ps_version="v2",
                        vit_hidden_size=1280,
                        vision_final_layernorm=False)
        for key, expected in required.items():
            if raw.get(key) != expected:
                raise ValueError(f"unsupported C-RADIO {key}: {raw.get(key)!r}")
        radio = raw["vision_config"]
        required = dict(hidden_size=1280, num_attention_heads=16, num_hidden_layers=32,
                        mlp_ratio=4.0, patch_size=16, num_channels=3, max_img_size=2048,
                        num_cls_tokens=3, num_registers=7, hidden_act="gelu", qkv_bias=True,
                        use_swiglu_ffn=False, layerscale_value=1.0, drop_path_rate=0.0,
                        layer_norm_eps=1e-6)
        for key, expected in required.items():
            if radio.get(key) != expected:
                raise ValueError(f"unsupported RADIO {key}: {radio.get(key)!r}")
        if raw.get("projector_hidden_size") != 20480 or raw["llm_config"]["hidden_size"] != HIDDEN:
            raise ValueError("C-RADIO projector requires widths 5120, 20480, and 4096")
        return cls(radio, raw["projector_hidden_size"], HIDDEN)


@dataclass(frozen=True)
class VisionWeights:
    config: VisionConfig
    tensors: dict[str, torch.Tensor]
    patch: torch.Tensor

    @classmethod
    def load(cls, manifest, device, progress=None):
        config = VisionConfig.from_dict(json.loads((manifest.snapshot / "config.json").read_text()))
        with DirectWeightLoader(manifest, device, prefixes=(PREFIX, "mlp1."), progress=progress) as loader:
            names = sorted(name for name in loader._selected_names() if not name.endswith("video_embedder.weight"))
            tensors = dict(zip(names, loader.load_tensors(names), strict=True))
        stem = PREFIX + "patch_generator."
        patch = tensors.pop(stem + "embedder.weight")
        if patch.shape != (VISION_HIDDEN, PIXEL_WIDTH):
            raise ValueError("RADIO image patch projection requires 16 by 16 RGB patches")
        if any(t.dtype != torch.bfloat16 for t in (*tensors.values(), patch)):
            raise ValueError("RADIO projections and normalization parameters require BF16 storage")
        return cls(config, tensors, patch)


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
        return cls(zeros((n, PIXEL_WIDTH), torch.float32), zeros((n, 2), torch.int64),
                   zeros((n, 4), torch.int64), zeros((n, 4), torch.float32), zeros((n // 4, 4)),
                   zeros(2), zeros(1), zeros(1), zeros(1) - 1, zeros(1, torch.int64), zeros(1))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    patch_hidden: torch.Tensor
    hidden: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    projected: torch.Tensor
    mlp_up: torch.Tensor
    merged: torch.Tensor
    activated: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, device, *, examples=None):
        def zeros(role, shape):
            return declaration_tensor(examples, "nemotron.vision." + role, shape, torch.bfloat16, device)
        rows = patches + REGISTER_TOKENS
        pixels = zeros("pixels", (patches, PIXEL_WIDTH))
        patch_hidden = zeros("patch_hidden", (patches, VISION_HIDDEN))
        hidden, projected = (zeros(role, (rows, VISION_HIDDEN)) for role in ("hidden", "projected"))
        mlp_up = zeros("mlp_up", (rows, 4 * VISION_HIDDEN))
        qkv = zeros("qkv", (rows, 3 * VISION_HIDDEN))
        query, key, value = (zeros(role, (rows, 16, 80)) for role in ("query", "key", "value"))
        attended = zeros("attended", (rows, VISION_HIDDEN))
        merged = zeros("merged", (patches // 4, 4 * VISION_HIDDEN))
        activated = zeros("activated", (patches // 4, 20480))
        output = zeros("output", (patches // 4, HIDDEN))
        return cls(pixels, patch_hidden, hidden, qkv, query, key, value, attended, projected,
                   mlp_up, merged, activated, output)


class _Projection(nn.Module):
    def __init__(self, weight, source, output, *, session, name, bias=None):
        super().__init__()
        self.operation = UnquantizedProjection(source, weight, output, session=session, name=name, bias=bias)

    def forward(self, source):
        return self.operation(source)

    def functional(self, *, session):
        self.operation.functional(session=session)
        return self


class _Attention(nn.Module):
    def __init__(self, weights, scratch, cu, *, session, workspace, name):
        super().__init__()
        s = scratch
        self.qkv = _Projection(weights["attn.qkv.weight"], s.hidden, s.qkv, session=session,
                               name=name + ".qkv", bias=weights["attn.qkv.bias"])
        self.output = _Projection(weights["attn.proj.weight"], s.attended, s.projected, session=session,
                                  name=name + ".output", bias=weights["attn.proj.bias"])
        self.attend = PreparedVarlenAttention(s.query, s.key, s.value, cu, s.attended.view_as(s.query),
                                             session=session, workspace=workspace, name=name + ".attention")

    def functional(self, *, session):
        self.qkv.functional(session=session)
        self.output.functional(session=session)
        self.attend.functional(session=session)
        self.cu = None
        return self

    def forward(self, source, attention_mask=None, **kwargs):
        q, k, v = (value.contiguous() for value in self.qkv(source).view(-1, 3, 16, 80).unbind(1))
        attended = self.attend(q, k, v, attention_mask)
        return self.output(attended.reshape(source.shape[0], VISION_HIDDEN)), None


def _norm(weight, bias, eps):
    with torch.device("meta"):
        module = nn.LayerNorm(weight.numel(), eps=eps)
    module.weight = nn.Parameter(weight, requires_grad=False)
    module.bias = nn.Parameter(bias, requires_grad=False)
    return module


class PreparedVision:
    def __init__(self, weights, scratch, inputs, *, session, workspace, name):
        self.session = session
        self.scratch, self.inputs = scratch, inputs
        s = scratch
        tensors = weights.tensors
        n, device = s.patches.shape[0], s.patches.device
        self.patch_capacity = n
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)
        cu = zeros((2,))
        self.position_weight = tensors[PREFIX + "patch_generator.pos_embed"].reshape(-1, VISION_HIDDEN)
        self.registers = tensors[PREFIX + "patch_generator.cls_token.token"]
        self.patch = UnquantizedProjection(s.patches, weights.patch, s.patch_hidden,
                                          session=session, name=name + ".patch")
        config = RadioConfig(**weights.config.radio)
        self.blocks = []
        for layer in range(config.num_hidden_layers):
            prefix = PREFIX + f"blocks.{layer}."
            w = {key.removeprefix(prefix): value for key, value in tensors.items() if key.startswith(prefix)}
            with torch.device("meta"):
                block = RadioLayer(config)
            block.norm1 = _norm(w["norm1.weight"], w["norm1.bias"], config.layer_norm_eps)
            block.norm2 = _norm(w["norm2.weight"], w["norm2.bias"], config.layer_norm_eps)
            block.layer_scale1 = block.layer_scale2 = nn.Identity()
            block.attention = _Attention(w, s, cu, session=session, workspace=workspace, name=f"{name}.{layer}")
            block.mlp.fc1 = _Projection(w["mlp.fc1.weight"], s.hidden, s.mlp_up,
                session=session, name=f"{name}.{layer}.up", bias=w["mlp.fc1.bias"])
            block.mlp.fc2 = _Projection(w["mlp.fc2.weight"], s.mlp_up, s.projected,
                session=session, name=f"{name}.{layer}.down", bias=w["mlp.fc2.bias"])
            block.eval()
            self.blocks.append(block)
        self.projector_norm = tensors["mlp1.0.weight"]
        self.up = UnquantizedProjection(s.merged, tensors["mlp1.1.weight"], s.activated,
                                       session=session, name=name + ".projector_up")
        self.down = UnquantizedProjection(s.activated, tensors["mlp1.3.weight"], s.output,
                                         session=session, name=name + ".projector_down")
        self.release_outputs()

    def release_outputs(self):
        self.patch.functional(session=self.session)
        for block in self.blocks:
            block.attention.functional(session=self.session)
            block.mlp.fc1.functional(session=self.session)
            block.mlp.fc2.functional(session=self.session)
        self.up.functional(session=self.session)
        self.down.functional(session=self.session)
        self.scratch = None
        return self

    def __call__(self, patches, interpolation_indices, interpolation_weights, merge_indices, cu):
        patches = patches.to(self.registers.dtype)
        positions = (self.position_weight[interpolation_indices].float()
                     * interpolation_weights[:, :, None]).sum(1).to(patches.dtype)
        hidden = torch.cat((self.registers, self.patch(patches) + positions), dim=0)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=cu)
        merged = hidden[REGISTER_TOKENS:][merge_indices].flatten(1).float()
        merged = merged * torch.rsqrt(merged.square().mean(-1, keepdim=True) + 1e-5)
        merged = (merged * self.projector_norm.float()).to(torch.bfloat16)
        activated = torch.relu(self.up(merged))
        activated = activated * activated
        return self.down(activated)
