# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 checkpoint transfers preserving BF16, FP8, and NVFP4 weight formats."""

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.tensor_fp8 import TensorFP8Weight
from ignition.components.weights import DirectWeightLoader, _aligned_range
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.persistent import persistent_copy, persistent_empty, persistent_zeros, allocated_tensor_bytes
from ignition.preparation.tensor_metadata import record_numeric_tensor

from .config import DraftConfig, ModelConfig


def _linear_shape(weight) -> tuple[int, int]:
    if isinstance(weight, torch.Tensor):
        return tuple(weight.shape)
    packed = weight.packed if isinstance(weight, TensorFP8Weight) else weight
    return int(packed.out_features), int(packed.in_features)


@dataclass(frozen=True)
class BF16ExpertsWeight:
    gate_up_proj: torch.Tensor
    down_proj: torch.Tensor


@dataclass(frozen=True)
class NVFP4ExpertsWeight:
    gate_up_weight: torch.Tensor
    gate_up_scale: torch.Tensor
    gate_up_scale_2: torch.Tensor
    down_weight: torch.Tensor
    down_scale: torch.Tensor
    down_scale_2: torch.Tensor


@dataclass(frozen=True)
class PLEWeights:
    embedding: torch.Tensor
    projection: object
    norm: torch.Tensor


@dataclass(frozen=True)
class DraftWeights:
    pre_projection: torch.Tensor
    post_projection: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    layers: tuple[dict, ...]
    centroids: torch.Tensor | None = None
    token_ordering: torch.Tensor | None = None

    @classmethod
    def load(cls, snapshot, config: DraftConfig, device, *, shard: TensorParallelShard, progress=None) -> "DraftWeights":
        manifest = inspect_indexed_checkpoint(Path(snapshot).resolve())
        with DraftWeightLoader(manifest, device, config=config, shard=shard, progress=progress) as loader:
            return loader.load_all()


class DraftWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, config: DraftConfig, shard: TensorParallelShard, progress=None):
        self.config = config
        self.shard = shard
        super().__init__(manifest, device, progress=progress, progress_label="Load draft weights")

    def _matrix(self, name: str, expected_shape: tuple[int, int], *, rows=None, columns=None) -> torch.Tensor:
        extent = (slice(*(rows or (None, None))), slice(*(columns or (None, None))))
        tensor = self.load_tensor(name, extent=extent)
        if tensor.dtype != torch.bfloat16 or tensor.shape != expected_shape:
            raise ValueError(f"Gemma 4 Assistant {name}: expected BF16 {expected_shape}, got {tensor.dtype} {tuple(tensor.shape)}")
        return tensor

    def load_all(self) -> DraftWeights:
        c = self.config
        bh = c.backbone_hidden_size
        pre_proj = self._matrix("pre_projection.weight", (c.hidden, 2 * bh))
        post_proj = self._matrix("post_projection.weight", (bh, c.hidden))
        norm = self.load_tensor("model.norm.weight")
        if norm.dtype != torch.bfloat16 or norm.shape != (c.hidden,):
            raise ValueError("Gemma 4 Assistant model.norm.weight shape or dtype differs from configuration")
        if c.use_ordered_embeddings:
            head = self._matrix("model.embed_tokens.weight", (c.vocabulary, c.hidden))
            centroids = self._matrix("masked_embedding.centroids.weight", (c.num_centroids, c.hidden))
            ordering = self.load_tensor("masked_embedding.token_ordering", dtype=torch.int64)
            if ordering.shape != (c.vocabulary,):
                raise ValueError("Gemma 4 Assistant token_ordering shape differs from vocabulary")
        else:
            v0, v1 = self.shard.partition(c.vocabulary, alignment=64)
            head = self._matrix("model.embed_tokens.weight", (v1 - v0, c.hidden), rows=(v0, v1))
            centroids = None
            ordering = None

        layers = []
        for index in range(c.num_layers):
            prefix = f"model.layers.{index}"
            dim = c.layer_head_dim(index)
            q0, q1 = c.q_range(index, self.shard)
            local_q = q1 - q0
            r0, r1 = c.intermediate_range(index, self.shard)
            local_inter = r1 - r0
            norms = self.load_tensors((
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.pre_feedforward_layernorm.weight",
                f"{prefix}.post_feedforward_layernorm.weight",
                f"{prefix}.layer_scalar",
                f"{prefix}.self_attn.q_norm.weight",
            ))
            in_norm, post_attn_norm, pre_ff_norm, post_ff_norm, scalar, q_norm = norms
            for t in (in_norm, post_attn_norm, pre_ff_norm, post_ff_norm):
                if t.shape != (c.hidden,):
                    raise ValueError(f"{prefix}: layernorm shape differs from hidden_size")
            if scalar.shape != (1,) or q_norm.shape != (dim,):
                raise ValueError(f"{prefix}: scalar or q_norm shape differs from configuration")
            layers.append({
                "input_layernorm": in_norm,
                "post_attention_layernorm": post_attn_norm,
                "pre_feedforward_layernorm": pre_ff_norm,
                "post_feedforward_layernorm": post_ff_norm,
                "layer_scalar": scalar,
                "q_norm": q_norm,
                "q_proj": self._matrix(f"{prefix}.self_attn.q_proj.weight", (local_q, c.hidden), rows=(q0, q1)),
                "o_proj": self._matrix(f"{prefix}.self_attn.o_proj.weight", (c.hidden, local_q), columns=(q0, q1)),
                "gate_proj": self._matrix(f"{prefix}.mlp.gate_proj.weight", (local_inter, c.hidden), rows=(r0, r1)),
                "up_proj": self._matrix(f"{prefix}.mlp.up_proj.weight", (local_inter, c.hidden), rows=(r0, r1)),
                "down_proj": self._matrix(f"{prefix}.mlp.down_proj.weight", (c.hidden, local_inter), columns=(r0, r1)),
            })
        return DraftWeights(
            pre_projection=pre_proj,
            post_projection=post_proj,
            norm=norm,
            head=head,
            layers=tuple(layers),
            centroids=centroids,
            token_ordering=ordering,
        )


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    embedding: torch.Tensor
    norm: torch.Tensor
    head: object
    layers: tuple[dict, ...]
    allocated_bytes: int
    ple: PLEWeights | None = None
    shard: TensorParallelShard = TensorParallelShard(0, 1)
    draft: DraftWeights | None = None

    @classmethod
    def load(cls, manifest, device, *, shard: TensorParallelShard = TensorParallelShard(0, 1), progress=None, draft_checkpoint=None):
        config = ModelConfig.load(manifest.snapshot)
        if draft_checkpoint is not None:
            config = replace(config, draft=DraftConfig.load(draft_checkpoint, config))
        config.validate_shard(shard)
        prefix = config.text_prefix
        v0, v1, local_vocab = config.vocabulary_partition(shard)
        before = allocated_tensor_bytes(device)
        with Gemma4WeightLoader(manifest, device, config=config, shard=shard, progress=progress) as loader:
            embedding, norm = loader.load_tensors((f"{prefix}.embed_tokens.weight", f"{prefix}.norm.weight"))
            if embedding.dtype != torch.bfloat16 or embedding.shape != (config.vocabulary, config.hidden):
                raise ValueError("Gemma 4 requires a BF16 embedding matching the configured vocabulary and hidden_size")
            if norm.shape != (config.hidden,):
                raise ValueError("Gemma 4 final norm weight shape differs from hidden_size")
            if "lm_head.weight" in manifest.tensors or "lm_head" in config.quantization:
                head = loader._checked_linear("lm_head", (local_vocab, config.hidden), rows=(v0, v1))
            elif shard.world_size == 1:
                head = embedding
            else:
                head = persistent_copy(embedding[v0:v1])
                record_numeric_tensor(head)
            if _linear_shape(head) != (local_vocab, config.hidden):
                raise ValueError("Gemma 4 vocabulary projection shape differs from the model")
            ple = loader.load_ple() if config.ple_dim > 0 else None
            layers = tuple(loader.load_layer(index) for index in range(config.num_layers))
        draft = (
            None
            if config.draft is None
            else DraftWeights.load(draft_checkpoint, config.draft, device, shard=shard, progress=progress)
        )
        return cls(
            config=config,
            embedding=embedding,
            norm=norm,
            head=head,
            layers=layers,
            allocated_bytes=allocated_tensor_bytes(device) - before,
            ple=ple,
            shard=shard,
            draft=draft,
        )


class Gemma4WeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, config: ModelConfig, shard: TensorParallelShard, progress=None):
        self.config = config
        self.shard = shard
        self.quantization = config.quantization
        prefixes = [config.text_prefix + "."]
        if any(name.startswith("lm_head.") for name in manifest.tensors):
            prefixes.append("lm_head.")
        super().__init__(manifest, device, prefixes=tuple(prefixes), progress=progress)

    def _selected_names(self):
        prefix = self.config.text_prefix + "."
        return {
            name
            for name in super()._selected_names()
            if name.startswith(prefix) or name.startswith("lm_head.")
        }

    def _load_fp8_channel_linear(self, prefix: str, *, rows=None, columns=None) -> torch.Tensor:
        source = self._sources[f"{prefix}.weight"]
        scale = self._sources[f"{prefix}.weight_scale"]
        if source.ndim != 2 or source.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{prefix}: FP8_CHANNEL weight must be a 2D float8_e4m3fn tensor")
        if scale.ndim != 2 or scale.shape != (source.shape[0], 1) or scale.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"{prefix}: FP8_CHANNEL weight_scale must have shape ({source.shape[0]}, 1)")
        r0, r1 = _aligned_range(rows, source.shape[0], 1)
        c0, c1 = _aligned_range(columns, source.shape[1], 1)
        raw_w = torch.empty((r1 - r0, c1 - c0), dtype=source.dtype, device=self.device)
        raw_s = torch.empty((r1 - r0, 1), dtype=scale.dtype, device=self.device)
        self._copy(raw_w, source[r0:r1, c0:c1])
        self._copy(raw_s, scale[r0:r1, :])
        self._session.flush()
        dequantized = persistent_empty(raw_w.shape, dtype=torch.bfloat16, device=self.device)
        dequantized.copy_(raw_w.float() * raw_s.float())
        record_numeric_tensor(dequantized)
        return dequantized

    def _load_compressed_nvfp4_linear(self, prefix: str, *, rows=None, columns=None):
        from b12x._lib.intrinsics.reference import swizzle_block_scale
        from b12x.gemm.blockscaled import pack_weight

        source = self._sources[f"{prefix}.weight_packed"]
        scale = self._sources[f"{prefix}.weight_scale"]
        w_global = self._sources[f"{prefix}.weight_global_scale"]
        in_global = self._sources[f"{prefix}.input_global_scale"]
        if source.ndim != 2 or source.dtype != torch.uint8:
            raise ValueError(f"{prefix}: compressed-tensors NVFP4 weight_packed must be a 2D uint8 tensor")
        n, packed_k = source.shape
        k = packed_k * 2
        if scale.dtype != torch.float8_e4m3fn or scale.shape != (n, k // 16):
            raise ValueError(f"{prefix}: compressed-tensors NVFP4 weight_scale must match 16-value blocks")
        if w_global.dtype != torch.float32 or w_global.numel() != 1:
            raise ValueError(f"{prefix}.weight_global_scale must be an FP32 scalar")
        if in_global.dtype != torch.float32 or in_global.numel() != 1:
            raise ValueError(f"{prefix}.input_global_scale must be an FP32 scalar")
        r0, r1 = _aligned_range(rows, n, 1)
        c0, c1 = _aligned_range(columns, k, 16)
        logical_k = c1 - c0
        stored_k = (logical_k + 127) // 128 * 128
        values = persistent_zeros((r1 - r0, stored_k // 2), dtype=torch.uint8, device=self.device)
        scales = torch.zeros((r1 - r0, stored_k // 16), dtype=torch.float8_e4m3fn, device=self.device)
        raw_gain = torch.empty(w_global.shape, dtype=torch.float32, device=self.device)
        activation_scale = persistent_empty(in_global.shape, dtype=torch.float32, device=self.device)
        self._copy(values[:, : logical_k // 2], source[r0:r1, c0 // 2 : c1 // 2])
        self._copy(scales[:, : logical_k // 16], scale[r0:r1, c0 // 16 : c1 // 16])
        self._copy(raw_gain, w_global)
        self._copy(activation_scale, in_global)
        self._session.flush()
        if not bool((torch.isfinite(raw_gain) & (raw_gain > 0)).all()):
            raise ValueError(f"{prefix}.weight_global_scale must be a positive finite FP32 scalar")
        if not bool((torch.isfinite(activation_scale) & (activation_scale > 0)).all()):
            raise ValueError(f"{prefix}.input_global_scale must be a positive finite FP32 scalar")
        gain = persistent_empty(raw_gain.shape, dtype=raw_gain.dtype, device=self.device)
        torch.reciprocal(raw_gain, out=gain)
        record_numeric_tensor(gain)
        record_numeric_tensor(activation_scale)
        weight = pack_weight(
            values,
            swizzle_block_scale(scales, allocate=persistent_empty),
            recipe="nvfp4",
            global_scale=gain,
            global_scale_kind="multiplier",
            activation_global_scale=activation_scale,
            allocate=persistent_empty,
        )
        return replace(weight, in_features=logical_k)

    def _checked_linear(self, name: str, expected_shape: tuple[int, int], *, rows=None, columns=None):
        spec = self.config.quantization.get(name)
        if spec is None:
            extent = (slice(*(rows or (None, None))), slice(*(columns or (None, None))))
            weight = self.load_tensor(f"{name}.weight", extent=extent)
            if weight.dtype != torch.bfloat16:
                raise ValueError(f"{name}.weight: expected BF16 tensor, got {weight.dtype}")
        elif spec["quant_algo"] == "FP8_CHANNEL":
            weight = self._load_fp8_channel_linear(name, rows=rows, columns=columns)
        elif spec["quant_algo"] == "COMPRESSED_NVFP4":
            weight = self._load_compressed_nvfp4_linear(name, rows=rows, columns=columns)
        else:
            weight = self.load_linear(name, rows=rows, columns=columns)
        if _linear_shape(weight) != expected_shape:
            raise ValueError(f"{name}: expected projection shape {expected_shape}, got {_linear_shape(weight)}")
        return weight

    def load_ple(self) -> PLEWeights:
        c = self.config
        prefix = c.text_prefix
        total_ple = c.num_layers * c.ple_dim
        ple_embed_name = f"{prefix}.embed_tokens_per_layer"
        ple_spec = c.quantization.get(ple_embed_name)
        if ple_spec is not None and ple_spec["quant_algo"] == "INT8_PACK_CHANNEL":
            packed_source = self._sources[f"{ple_embed_name}.weight_packed"]
            scale_source = self._sources[f"{ple_embed_name}.weight_scale"]
            if (
                packed_source.ndim != 2
                or packed_source.dtype != torch.int32
                or packed_source.shape != (c.ple_vocabulary, total_ple // 4)
            ):
                raise ValueError(f"{ple_embed_name}: INT8_PACK_CHANNEL weight_packed shape or dtype mismatch")
            if (
                scale_source.ndim != 2
                or scale_source.shape != (c.ple_vocabulary, 1)
                or scale_source.dtype not in (torch.bfloat16, torch.float32)
            ):
                raise ValueError(f"{ple_embed_name}: INT8_PACK_CHANNEL weight_scale shape or dtype mismatch")
            raw_packed = torch.empty(packed_source.shape, dtype=packed_source.dtype, device=self.device)
            raw_scale = torch.empty(scale_source.shape, dtype=scale_source.dtype, device=self.device)
            norm = self._queue_tensor(f"{prefix}.per_layer_projection_norm.weight")
            self._copy(raw_packed, packed_source)
            self._copy(raw_scale, scale_source)
            self._session.flush()
            embedding = persistent_empty((c.ple_vocabulary, total_ple), dtype=torch.bfloat16, device=self.device)
            embedding.copy_(raw_packed.view(torch.uint8))
            del raw_packed
            embedding.sub_(128.0).mul_(raw_scale.to(torch.bfloat16))
            del raw_scale
            record_numeric_tensor(embedding)
        else:
            embedding, norm = self.load_tensors((
                f"{ple_embed_name}.weight",
                f"{prefix}.per_layer_projection_norm.weight",
            ))
        if embedding.dtype != torch.bfloat16 or embedding.shape != (c.ple_vocabulary, total_ple):
            raise ValueError("Gemma 4 per-layer token embedding shape or dtype differs from configuration")
        if norm.shape != (c.ple_dim,):
            raise ValueError("Gemma 4 per_layer_projection_norm shape differs from configuration")
        projection = self._checked_linear(f"{prefix}.per_layer_model_projection", (total_ple, c.hidden))
        return PLEWeights(embedding=embedding, projection=projection, norm=norm)

    def load_layer(self, index: int) -> dict:
        c = self.config
        prefix = f"{c.text_prefix}.layers.{index}"
        dim = c.layer_head_dim(index)
        q_rows = c.q_range(index, self.shard)
        local_q = q_rows[1] - q_rows[0]
        kv_rows = c.kv_range(index, self.shard)
        local_kv = kv_rows[1] - kv_rows[0]
        inter_range = c.intermediate_range(index, self.shard)
        local_inter = inter_range[1] - inter_range[0]
        shared_kv = c.is_kv_shared(index)

        tensor_names = [
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.pre_feedforward_layernorm.weight",
            f"{prefix}.post_feedforward_layernorm.weight",
            f"{prefix}.layer_scalar",
            f"{prefix}.self_attn.q_norm.weight",
        ]
        if not shared_kv:
            tensor_names.append(f"{prefix}.self_attn.k_norm.weight")
        if c.enable_moe:
            tensor_names.extend((
                f"{prefix}.pre_feedforward_layernorm_2.weight",
                f"{prefix}.post_feedforward_layernorm_1.weight",
                f"{prefix}.post_feedforward_layernorm_2.weight",
                f"{prefix}.router.scale",
                f"{prefix}.router.per_expert_scale",
            ))
        if c.ple_dim > 0:
            tensor_names.append(f"{prefix}.post_per_layer_input_norm.weight")

        loaded = dict(zip(tensor_names, self.load_tensors(tuple(tensor_names)), strict=True))
        for norm_key in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ):
            if loaded[f"{prefix}.{norm_key}"].shape != (c.hidden,):
                raise ValueError(f"{prefix}.{norm_key}: shape differs from hidden_size")
        if loaded[f"{prefix}.layer_scalar"].shape != (1,):
            raise ValueError(f"{prefix}.layer_scalar: expected shape (1,)")
        if loaded[f"{prefix}.self_attn.q_norm.weight"].shape != (dim,):
            raise ValueError(f"{prefix}.self_attn.q_norm.weight: shape differs from head_dim")

        layer = {
            "input_layernorm": loaded[f"{prefix}.input_layernorm.weight"],
            "post_attention_layernorm": loaded[f"{prefix}.post_attention_layernorm.weight"],
            "pre_feedforward_layernorm": loaded[f"{prefix}.pre_feedforward_layernorm.weight"],
            "post_feedforward_layernorm": loaded[f"{prefix}.post_feedforward_layernorm.weight"],
            "layer_scalar": loaded[f"{prefix}.layer_scalar"],
            "q_norm": loaded[f"{prefix}.self_attn.q_norm.weight"],
            "q_proj": self._checked_linear(f"{prefix}.self_attn.q_proj", (local_q, c.hidden), rows=q_rows),
            "o_proj": self._checked_linear(f"{prefix}.self_attn.o_proj", (c.hidden, local_q), columns=q_rows),
            "gate_proj": self._checked_linear(f"{prefix}.mlp.gate_proj", (local_inter, c.hidden), rows=inter_range),
            "up_proj": self._checked_linear(f"{prefix}.mlp.up_proj", (local_inter, c.hidden), rows=inter_range),
            "down_proj": self._checked_linear(f"{prefix}.mlp.down_proj", (c.hidden, local_inter), columns=inter_range),
        }
        if not shared_kv:
            k_norm = loaded[f"{prefix}.self_attn.k_norm.weight"]
            if k_norm.shape != (dim,):
                raise ValueError(f"{prefix}.self_attn.k_norm.weight: shape differs from head_dim")
            layer["k_norm"] = k_norm
            layer["k_proj"] = self._checked_linear(f"{prefix}.self_attn.k_proj", (local_kv, c.hidden), rows=kv_rows)
            if c.layer_has_v_proj(index):
                layer["v_proj"] = self._checked_linear(
                    f"{prefix}.self_attn.v_proj", (local_kv, c.hidden), rows=kv_rows
                )
        if c.ple_dim > 0:
            ple_norm = loaded[f"{prefix}.post_per_layer_input_norm.weight"]
            if ple_norm.shape != (c.hidden,):
                raise ValueError(f"{prefix}.post_per_layer_input_norm.weight: shape differs from hidden_size")
            layer["post_per_layer_input_norm"] = ple_norm
            layer["per_layer_input_gate"] = self._checked_linear(
                f"{prefix}.per_layer_input_gate", (c.ple_dim, c.hidden)
            )
            layer["per_layer_projection"] = self._checked_linear(
                f"{prefix}.per_layer_projection", (c.hidden, c.ple_dim)
            )
        if c.enable_moe:
            for norm_key in (
                "pre_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm_1.weight",
                "post_feedforward_layernorm_2.weight",
                "router.scale",
            ):
                if loaded[f"{prefix}.{norm_key}"].shape != (c.hidden,):
                    raise ValueError(f"{prefix}.{norm_key}: shape differs from hidden_size")
            if loaded[f"{prefix}.router.per_expert_scale"].shape != (c.num_experts,):
                raise ValueError(f"{prefix}.router.per_expert_scale: shape differs from num_experts")
            layer.update(
                pre_feedforward_layernorm_2=loaded[f"{prefix}.pre_feedforward_layernorm_2.weight"],
                post_feedforward_layernorm_1=loaded[f"{prefix}.post_feedforward_layernorm_1.weight"],
                post_feedforward_layernorm_2=loaded[f"{prefix}.post_feedforward_layernorm_2.weight"],
                router_scale=loaded[f"{prefix}.router.scale"],
                per_expert_scale=loaded[f"{prefix}.router.per_expert_scale"],
                router_proj=self._checked_linear(f"{prefix}.router.proj", (c.num_experts, c.hidden)),
                experts=self._load_experts(prefix),
            )
        return layer

    def _load_experts(self, prefix: str) -> BF16ExpertsWeight | NVFP4ExpertsWeight:
        self._require_open()
        c = self.config
        experts, inter, hidden = c.num_experts, c.moe_intermediate_size, c.hidden
        r0, r1 = c.moe_intermediate_range(self.shard)
        local_inter = r1 - r0
        fused_key = f"{prefix}.experts.gate_up_proj"
        if fused_key in self.manifest.tensors:
            source = self._sources[fused_key]
            down_source = self._sources[f"{prefix}.experts.down_proj"]
            gate = source[:, r0:r1]
            up = source[:, inter + r0:inter + r1]
            down = down_source[:, :, r0:r1]
            if (
                gate.dtype != torch.bfloat16
                or up.dtype != torch.bfloat16
                or down.dtype != torch.bfloat16
                or gate.shape != (experts, local_inter, hidden)
                or up.shape != (experts, local_inter, hidden)
                or down.shape != (experts, hidden, local_inter)
            ):
                raise ValueError(f"{prefix}.experts: BF16 expert tensor geometry differs from configuration")
            gate_up = persistent_empty((experts, 2 * local_inter, hidden),
                                       dtype=torch.bfloat16, device=self.device)
            down_proj = persistent_empty(down.shape, dtype=down.dtype, device=self.device)
            self._copy(gate_up[:, :local_inter], gate)
            self._copy(gate_up[:, local_inter:], up)
            self._copy(down_proj, down)
            self._session.flush()
            record_numeric_tensor(gate_up)
            record_numeric_tensor(down_proj)
            return BF16ExpertsWeight(gate_up_proj=gate_up, down_proj=down_proj)

        first_base = f"{prefix}.experts.0.gate_proj"
        compressed = self.config.quantization.get(first_base, {}).get("quant_algo") == "COMPRESSED_NVFP4"
        w_suffix = "weight_packed" if compressed else "weight"
        g_suffix = "weight_global_scale" if compressed else "weight_scale_2"

        gate_up_weight = persistent_empty((experts, 2 * local_inter, hidden // 2),
                                          dtype=torch.uint8, device=self.device)
        gate_up_scale = persistent_empty((experts, 2 * local_inter, hidden // 16),
                                         dtype=torch.float8_e4m3fn, device=self.device)
        gate_up_scale_2 = persistent_empty((experts, 2), dtype=torch.float32, device=self.device)
        down_weight = persistent_empty((experts, hidden, local_inter // 2),
                                       dtype=torch.uint8, device=self.device)
        down_scale = persistent_empty((experts, hidden, local_inter // 16),
                                      dtype=torch.float8_e4m3fn, device=self.device)
        down_scale_2 = persistent_empty((experts,), dtype=torch.float32, device=self.device)
        for expert in range(experts):
            for proj, w_out, s_out, g_out, out_dim, in_dim, w_ext, s_ext in (
                (
                    "gate_proj",
                    gate_up_weight[expert, :local_inter],
                    gate_up_scale[expert, :local_inter],
                    gate_up_scale_2[expert, :1],
                    local_inter,
                    hidden,
                    (slice(r0, r1), slice(None)),
                    (slice(r0, r1), slice(None)),
                ),
                (
                    "up_proj",
                    gate_up_weight[expert, local_inter:],
                    gate_up_scale[expert, local_inter:],
                    gate_up_scale_2[expert, 1:],
                    local_inter,
                    hidden,
                    (slice(r0, r1), slice(None)),
                    (slice(r0, r1), slice(None)),
                ),
                (
                    "down_proj",
                    down_weight[expert],
                    down_scale[expert],
                    down_scale_2[expert:expert + 1],
                    hidden,
                    local_inter,
                    (slice(None), slice(r0 // 2, r1 // 2)),
                    (slice(None), slice(r0 // 16, r1 // 16)),
                ),
            ):
                base = f"{prefix}.experts.{expert}.{proj}"
                w = self._sources[f"{base}.{w_suffix}"][w_ext]
                s = self._sources[f"{base}.weight_scale"][s_ext]
                g = self._sources[f"{base}.{g_suffix}"]
                if (
                    w.dtype != torch.uint8
                    or w.shape != (out_dim, in_dim // 2)
                    or s.dtype != torch.float8_e4m3fn
                    or s.shape != (out_dim, in_dim // 16)
                    or g.dtype != torch.float32
                    or g.numel() != 1
                ):
                    raise ValueError(f"{base}: NVFP4 expert tensor geometry differs from configuration")
                self._copy(w_out, w)
                self._copy(s_out, s)
                self._copy(g_out, g.reshape(1))
        self._session.flush()
        if compressed:
            gate_up_scale_2.reciprocal_()
            down_scale_2.reciprocal_()
        for tensor in (gate_up_weight, gate_up_scale, gate_up_scale_2,
                       down_weight, down_scale, down_scale_2):
            record_numeric_tensor(tensor)
        return NVFP4ExpertsWeight(
            gate_up_weight=gate_up_weight,
            gate_up_scale=gate_up_scale,
            gate_up_scale_2=gate_up_scale_2,
            down_weight=down_weight,
            down_scale=down_scale,
            down_scale_2=down_scale_2,
        )
