# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct loading of Inkling's interleaved NVFP4 and unquantized checkpoint tensors."""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights
from .config import ModelConfig

PREFIX = "model.llm"


@dataclass(frozen=True)
class Experts:
    gate_up: torch.Tensor
    down: torch.Tensor


@dataclass(frozen=True)
class LayerWeights:
    qkv_relative: torch.Tensor
    output: torch.Tensor
    relative: torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    key_conv: torch.Tensor
    value_conv: torch.Tensor
    attention_conv: torch.Tensor
    mlp_conv: torch.Tensor
    scale: torch.Tensor
    dense: MLPWeights | None
    router: torch.Tensor | None
    correction: torch.Tensor | None
    experts: object | None
    shared: tuple[MLPWeights, ...]


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    embedding_norm: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    layers: tuple[LayerWeights, ...]
    allocated_bytes: int
    draft: object | None = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, package_storage=None):
        before = torch.cuda.memory_allocated(device)
        loading = nullcontext if package_storage is None else package_storage.transient
        with loading(), InklingWeightLoader(manifest, device, shard=shard, progress=progress) as loader:
            config = loader.config
            first, last = shard.partition(config.hidden)
            embedding = loader.load_tensor(PREFIX + ".embed.weight", extent=(slice(None), slice(first, last)))
            embed_norm = loader.load_tensor(PREFIX + ".embed_norm.weight")
            norm = loader.load_tensor(PREFIX + ".norm.weight")
            first, last, width = config.vocabulary_partition(shard)
            head = torch.zeros((width, config.hidden), dtype=torch.bfloat16, device=device)
            loader._copy(head[:last-first], loader._sources[PREFIX + ".unembed.weight"][first:last])
            loader._session.flush()
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
        return cls(config, shard, embedding, embed_norm, norm, head, layers,
                   torch.cuda.memory_allocated(device) - before)


class InklingWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None):
        super().__init__(manifest, device, progress=progress)
        self.config, self.shard = ModelConfig.load(manifest.snapshot), shard
        self.config.validate_shard(shard)

    @cached_property
    def quantization(self):
        return {}

    def _selected_names(self):
        return {n for n in super()._selected_names() if n.startswith(PREFIX + ".")}

    def _interleaved(self, name, down_name, inter, *, expert=None, destination=None):
        first, last = self.shard.partition(inter, alignment=64)
        source, down = self._sources[name], self._sources[down_name]
        if expert is not None:
            source, down = source[expert], down[expert]
        if (source.dtype != torch.bfloat16 or down.dtype != torch.bfloat16
                or source.shape != (2 * inter, self.config.hidden)
                or down.shape != (self.config.hidden, inter)):
            raise ValueError(f"{name}: expected interleaved BF16 gate/up and BF16 down weights")
        if destination is None:
            destination = MLPWeights(
                torch.empty((2 * (last-first), self.config.hidden), dtype=torch.bfloat16, device=self.device),
                torch.empty((self.config.hidden, last-first), dtype=torch.bfloat16, device=self.device))
        target, output = destination.gate_up, destination.down
        self._copy(target[:last-first], source[2*first:2*last:2])
        self._copy(target[last-first:], source[2*first+1:2*last:2])
        self._copy(output, down[:, first:last])
        self._session.flush()
        return MLPWeights(target, output)

    def _nvfp4_experts(self, prefix, inter):
        from b12x._lib.intrinsics.reference import swizzle_block_scale
        from b12x.moe import fused_moe as moe
        from ignition.preparation.tensor_metadata import record_numeric_tensor

        first, last = self.shard.partition(inter, alignment=64)
        local, count, hidden = last - first, self.config.text["n_routed_experts"], self.config.hidden
        packed, blocks, globals_, inputs = [], [], [], []
        for projection, rows, columns in (("w13_weight", 2 * inter, hidden), ("w2_weight", hidden, inter)):
            name = prefix + "." + projection
            if any(name + suffix not in self._sources for suffix in ("", ".scale", ".scale2", ".input_amax", ".original_shape")):
                raise ValueError(f"{name}: incomplete NVFP4 checkpoint tensors")
            source, scale = self._sources[name], self._sources[name + ".scale"]
            shape = self._session.materialize(self._sources[name + ".original_shape"])
            global_scale = self.load_tensor(name + ".scale2")
            amax = self._session.materialize(self._sources[name + ".input_amax"])
            if (source.dtype != torch.uint8 or source.shape != (count, rows, columns // 2)
                    or scale.dtype != torch.float8_e4m3fn or scale.shape != (count, rows, columns // 16)
                    or shape.dtype != torch.int64 or shape.shape != (3,) or shape.tolist() != [count, rows, columns]):
                raise ValueError(f"{name}: invalid packed NVFP4 weight or block-scale geometry")
            if (global_scale.dtype != torch.float32 or global_scale.shape != (count,)
                    or not bool((torch.isfinite(global_scale) & (global_scale > 0)).all())
                    or amax.dtype != torch.bfloat16 or amax.shape != (1,)
                    or not bool((torch.isfinite(amax) & (amax > 0)).all())):
                raise ValueError(f"{name}: expected positive FP32 per-expert scale2 and BF16 input_amax")
            down = projection == "w2_weight"
            output_shape = (count, hidden, local // 2) if down else (count, 2 * local, hidden // 2)
            scale_shape = (count, hidden, local // 16) if down else (count, 2 * local, hidden // 16)
            values = torch.empty(output_shape, dtype=torch.uint8, device=self.device)
            scales = torch.empty(scale_shape, dtype=torch.float8_e4m3fn, device=self.device)
            for expert in range(count):
                if down:
                    self._copy(values[expert], source[expert, :, first // 2:last // 2])
                    self._copy(scales[expert], scale[expert, :, first // 16:last // 16])
                else:
                    for offset in (0, 1):
                        target = slice(offset * local, (offset + 1) * local)
                        self._copy(values[expert, target], source[expert, 2 * first + offset:2 * last:2])
                        self._copy(scales[expert, target], scale[expert, 2 * first + offset:2 * last:2])
            globals_.append(global_scale)
            # Preserve ModelOpt's FP32 amax/(448*6) boundary before b12x's reciprocal.
            activation_scale = (amax.float() / 2688.).reciprocal().to(self.device).expand(count).clone()
            record_numeric_tensor(activation_scale)
            inputs.append(activation_scale)
            self._session.flush()
            packed.append(values)
            blocks.append(swizzle_block_scale(scales))
            del scales
        plan = moe.plan_weights(source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w31"),
            activation=moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
            geometry=moe.MoEGeometry(num_experts=count, hidden_size=hidden, intermediate_size=local))
        return moe.prepare_weights(plan=plan, weights=moe.PackedWeights(
            w13=packed[0], w2=packed[1], w13_block_scales=blocks[0], w2_block_scales=blocks[1],
            w13_global_scales=globals_[0], w2_global_scales=globals_[1],
            input_scale=inputs[0], intermediate_scale=inputs[1], immutable_input_scales=True))

    def load_layer(self, layer):
        c, shard = self.config.text, self.shard
        prefix = f"{PREFIX}.layers.{layer}"
        a = self.config.attention(layer)
        q0, q1 = shard.partition(a.heads * a.key_dim, alignment=a.key_dim)
        r0, r1 = shard.partition(a.heads * c["d_rel"], alignment=c["d_rel"])
        k0, k1 = shard.replicated_partition(a.kv_heads)
        kv = (k0 * a.key_dim, k1 * a.key_dim)
        attn = prefix + ".attn"
        qkv = self.load_linear_group(tuple(attn + "." + name for name in ("wq_du", "wk_dv", "wv_dv", "wr_du")),
                                     row_ranges=((q0, q1), kv, kv, (r0, r1)))
        def tensor(name, **kwargs):
            return self.load_tensor(prefix + "." + name, **kwargs)
        def conv(name, extent=None):
            value = tensor(name + ".weight", dtype=torch.float32,
                extent=None if extent is None else (slice(*extent), slice(None), slice(None)))
            return value.reshape(value.shape[0], c["conv_kernel_size"])
        dense, router, correction, experts, shared = None, None, None, None, ()
        mlp = prefix + ".mlp"
        if c["mlp_layer_types"][layer] == "dense":
            dense = self._interleaved(mlp + ".w13_dn.weight", mlp + ".w2_md.weight", c["intermediate_size"])
            scale = tensor("mlp.global_scale")
        else:
            router, correction, scale = (tensor("mlp.gate." + n) for n in ("weight", "bias", "global_scale"))
            if layer in self.config.quantized_routed_layers:
                experts = self._nvfp4_experts(mlp + ".experts", c["moe_intermediate_size"])
            else:
                count, local = c["n_routed_experts"], c["moe_intermediate_size"] // shard.world_size
                experts = Experts(
                    torch.empty((count, 2 * local, self.config.hidden), dtype=torch.bfloat16, device=self.device),
                    torch.empty((count, self.config.hidden, local), dtype=torch.bfloat16, device=self.device))
                for expert in range(count):
                    self._interleaved(mlp + ".experts.w13_weight", mlp + ".experts.w2_weight",
                        c["moe_intermediate_size"], expert=expert,
                        destination=MLPWeights(experts.gate_up[expert], experts.down[expert]))
            shared = tuple(self._interleaved(mlp + ".shared_experts.shared_w13_weight",
                mlp + ".shared_experts.shared_w2_weight", c["moe_intermediate_size"], expert=i)
                for i in range(c["n_shared_experts"]))
        return LayerWeights(qkv, self.load_linear(attn + ".wo_ud", columns=(q0, q1)),
            tensor("attn.rel_logits_proj.proj").T.contiguous(), tensor("attn.q_norm.weight"),
            tensor("attn.k_norm.weight"), tensor("attn_norm.weight"), tensor("mlp_norm.weight"),
            conv("attn.k_sconv", kv), conv("attn.v_sconv", kv), conv("attn_sconv"), conv("mlp_sconv"),
            scale, dense, router, correction, experts, shared)
