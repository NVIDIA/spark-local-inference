# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming canonical text-model reference for the exact mixed-precision checkpoint.

Transformers owns the model, attention, recurrence, PLE, routing, and residual
semantics. Checkpoint adapters decode stored weights and quantize activations
with independent PyTorch references. Dense weights and selected experts are
materialized only while their operation runs; neither the full expert bank nor
the PLE table is expanded. This module never imports b12x.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType, MethodType

import torch
import torch.nn.functional as F

from ignition.preparation.checkpoint import CheckpointManifest
from ignition.preparation.qwen38 import PLE_PREFIX, TEXT_PREFIX
from ignition.recipes.qwen38.model import create_model_tree

from .qwen38_quant import (
    CheckpointTensors,
    Nvfp4Projection,
    mxfp8_linear_reference,
    nvfp4_expert_reference,
    read_nvfp4_projection,
    read_ple_rows,
)


def _reference_gdn_forward(module, convolution_output=None) -> None:
    """Bind canonical Python functions without package or hub-kernel dispatch."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as canonical

    original = inspect.unwrap(type(module).forward)
    namespace = dict(original.__globals__)
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    ):
        namespace[name] = inspect.unwrap(getattr(canonical, name))
        if convolution_output is not None and name.startswith("causal_conv1d"):
            function = namespace[name]

            def observed(*args, _function=function, **kwargs):
                result = _function(*args, **kwargs)
                convolution_output(result)
                return result

            namespace[name] = observed
    function = FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    function.__kwdefaults__ = original.__kwdefaults__
    module.forward = MethodType(function, module)


class _StreamLinear(torch.nn.Module):
    def __init__(self, owner, prefix: str, *, row_block: int | None = None):
        super().__init__()
        self.owner, self.prefix, self.row_block = owner, prefix, row_block
        descriptor = owner.manifest.tensors[prefix + ".weight"]
        self.out_features, self.in_features = descriptor.shape
        self.quantized = descriptor.dtype == "F8_E4M3"
        if descriptor.dtype not in ("BF16", "F8_E4M3"):
            raise ValueError(
                f"unsupported reference projection {prefix}: {descriptor.dtype}"
            )
        self.register_buffer(
            "weight", torch.empty(0, dtype=torch.bfloat16, device="meta")
        )

    def forward(self, source):
        reader = self.owner.reader
        if self.quantized:
            weight = reader.read(self.prefix + ".weight").to(source.device)
            scales = reader.read(self.prefix + ".weight_scale").to(source.device)
            return mxfp8_linear_reference(source, weight, scales).to(torch.bfloat16)
        if self.row_block is None:
            weight = reader.read(self.prefix + ".weight").to(source.device)
            return F.linear(source, weight)
        output = torch.empty(
            (*source.shape[:-1], self.out_features),
            dtype=source.dtype,
            device=source.device,
        )
        for start in range(0, self.out_features, self.row_block):
            stop = min(start + self.row_block, self.out_features)
            weight = reader.read(self.prefix + ".weight", rows=(start, stop)).to(
                source.device
            )
            output[..., start:stop] = F.linear(source, weight)
        return output


class _SelectedEmbedding(torch.nn.Module):
    def __init__(self, owner, prefix: str, *, ple: bool = False):
        super().__init__()
        self.owner, self.prefix, self.ple = owner, prefix, ple
        self.register_buffer(
            "weight", torch.empty(0, dtype=torch.bfloat16, device="meta")
        )

    def forward(self, ids):
        cpu_ids = ids.detach().to(device="cpu", dtype=torch.int64)
        reader = self.owner.reader
        if self.ple:
            flat = cpu_ids.flatten()
            parts = [read_ple_rows(reader, part) for part in flat.split(4096)]
            values = torch.cat(parts).view(*ids.shape, 160)
        else:
            unique, inverse = cpu_ids.flatten().unique(sorted=True, return_inverse=True)
            selected = torch.cat(
                [
                    reader.read(self.prefix + ".weight", rows=(index, index + 1))
                    for index in unique.tolist()
                ]
            )
            values = selected[inverse].reshape(*ids.shape, -1)
        return values.to(ids.device)


def _projection_to(projection: Nvfp4Projection, device) -> Nvfp4Projection:
    return Nvfp4Projection(
        projection.packed.to(device),
        projection.scales.to(device),
        projection.weight_scale_2.to(device),
        None if projection.input_scale is None else projection.input_scale.to(device),
    )


class _SelectedExperts(torch.nn.Module):
    """Checkpoint NVFP4 arithmetic with canonical expert order and BF16 reduction."""

    def __init__(self, owner, prefix):
        super().__init__()
        self.owner, self.prefix = owner, prefix

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        for expert in top_k_index.unique(sorted=True).tolist():
            top_k_pos, token_idx = torch.where(top_k_index.T == expert)
            projections = tuple(
                _projection_to(
                    read_nvfp4_projection(
                        self.owner.reader, f"{self.prefix}.{expert}.{name}"
                    ),
                    hidden_states.device,
                )
                for name in ("gate_proj", "up_proj", "down_proj")
            )
            down = nvfp4_expert_reference(
                hidden_states[token_idx], *projections
            ).down.to(hidden_states.dtype)
            weighted = down * top_k_weights[token_idx, top_k_pos, None]
            output.index_add_(0, token_idx, weighted.to(output.dtype))
        return output


@dataclass(frozen=True)
class DecoderReferenceOutput:
    """CPU logits/hidden vectors and the canonical continuation cache."""

    logits: torch.Tensor
    hidden_states: torch.Tensor
    past_key_values: object
    checkpoint_bytes_read: int


class CanonicalTextReference:
    """Single-sequence exact-checkpoint reference with bounded device weights.

    Canonical GDN chunk size64 and FP32 intermediates remain unchanged. Expert
    W4A4 projections use checkpoint activation scales, FP32 gate/up arithmetic,
    a BF16 SwiGLU boundary, and canonical BF16 routing accumulation. Floating
    matmuls use IEEE FP32 rather than TF32. This is a numerical reference, not a
    serving backend; execution synchronizes and streams checkpoint payloads.

    ``forward`` returns all-token logits by default. ``logits_to_keep=1`` keeps
    only the last token while still executing every input token through all 48
    decoder layers. The caller retains the returned cache for continuation.
    Instances are deliberately non-reentrant and cannot run concurrently with
    code changing PyTorch matmul precision settings.
    """

    def __init__(
        self,
        manifest: CheckpointManifest,
        *,
        device="cpu",
        lm_head_rows: int = 4096,
        byte_budget: int = 128 << 30,
        progress: Callable[[str], None] | None = None,
        layer_output: Callable[[int, torch.Tensor], None] | None = None,
        stage_output: Callable[[str, torch.Tensor], None] | None = None,
    ):
        if type(lm_head_rows) is not int or lm_head_rows <= 0:
            raise ValueError("LM-head row block must be positive")
        if type(byte_budget) is not int or byte_budget <= 0:
            raise ValueError("reference read budget must be positive")
        self.manifest, self.device = manifest, torch.device(device)
        self.byte_budget, self.progress, self.layer_output = (
            byte_budget,
            progress,
            layer_output,
        )
        self._reader = None
        self.stage_output = stage_output
        tree = create_model_tree(manifest)
        self.model = tree.model.language_model
        self.model.config._attn_implementation = "eager"
        self.model.config.use_cache = True
        for layer_id, layer in enumerate(self.model.layers):
            layer.mlp.experts = _SelectedExperts(
                self, f"{TEXT_PREFIX}.layers.{layer_id}.mlp.experts"
            )
            if hasattr(layer, "linear_attn"):
                _reference_gdn_forward(layer.linear_attn)
            layer.register_forward_hook(self._layer_hook(layer_id))
        for path, module in list(self.model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                parent, _, name = path.rpartition(".")
                setattr(
                    self.model.get_submodule(parent),
                    name,
                    _StreamLinear(self, f"{TEXT_PREFIX}.{path}"),
                )
        self.model.embed_tokens = _SelectedEmbedding(
            self, f"{TEXT_PREFIX}.embed_tokens"
        )
        ple_embedding = self.model.layers[1].ple.ple_embedding
        ple_embedding.ngram_embedding = _SelectedEmbedding(
            self, f"{PLE_PREFIX}.ngram_embedding", ple=True
        )
        self.lm_head = _StreamLinear(self, "lm_head", row_block=lm_head_rows)
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpTextRotaryEmbedding,
        )

        self.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(self.model.config).to(
            self.device
        )
        with CheckpointTensors(manifest, byte_budget=1 << 30) as reader:
            for path, parameter in list(self.model.named_parameters()):
                prefix = f"{TEXT_PREFIX}.{path}"
                value = reader.read(prefix).to(self.device)
                if value.shape != parameter.shape:
                    raise ValueError(
                        f"canonical parameter shape disagrees with checkpoint: {prefix}"
                    )
                parent, _, name = path.rpartition(".")
                setattr(
                    self.model.get_submodule(parent),
                    name,
                    torch.nn.Parameter(value, requires_grad=False),
                )
            for name in (
                "layer_multipliers",
                "ngram_heads_vocab_sizes",
                "ngram_heads_offsets",
            ):
                setattr(
                    ple_embedding,
                    name,
                    reader.read(f"{PLE_PREFIX}.{name}").to(self.device),
                )
        self.model.eval().requires_grad_(False)
        if stage_output is not None:
            self._install_stage_hooks()

    def _stage(self, name, value, width=None):
        value = value.reshape(-1, value.shape[-1] if width is None else width)
        self.stage_output(name, value.detach().cpu())

    def _install_stage_hooks(self):
        """Observe layer-zero canonical values without changing tensor arithmetic."""
        layer = self.model.layers[0]
        self.model.embed_tokens.register_forward_hook(
            lambda m, a, value: self._stage("embedding", value)
        )
        layer.attn_hyper_connection.register_forward_hook(
            lambda m, a, value: self._stage("attn_mix", value[0])
        )
        layer.mlp_hyper_connection.register_forward_pre_hook(
            lambda m, a: self._stage("attn_residual", a[0])
        )
        layer.mlp_hyper_connection.register_forward_hook(
            lambda m, a, value: self._stage("mlp_mix", value[0])
        )
        layer.linear_attn.register_forward_hook(
            lambda m, a, value: self._stage("attn_output", value)
        )
        layer.linear_attn.in_proj_qkv.register_forward_hook(
            lambda m, a, value: self._stage("gdn_projected_qkv", value)
        )
        layer.linear_attn.norm.register_forward_pre_hook(
            lambda m, a: self._stage("gdn_recurrence", a[0], 6144)
        )
        layer.linear_attn.norm.register_forward_hook(
            lambda m, a, value: self._stage("gdn_gated", value, 6144)
        )
        _reference_gdn_forward(
            layer.linear_attn,
            lambda value: self._stage(
                "gdn_convolved_qkv", value[..., -self._input_tokens :].transpose(1, 2)
            ),
        )
        layer.mlp.register_forward_hook(
            lambda m, a, value: self._stage("mlp_output", value)
        )
        layer.mlp.experts.register_forward_hook(
            lambda m, a, value: self._stage("moe_routed", value)
        )

        def router(module, args, value):
            logits, weights, ids = value
            self._stage("moe_router_logits", logits)
            self._stage("moe_ids", ids)
            self._stage("moe_weights", weights.float())

        def shared(module, args, value):
            self._shared_output = value

        def shared_gate(module, args, value):
            self._stage("moe_shared", self._shared_output * torch.sigmoid(value))

        layer.mlp.gate.register_forward_hook(router)
        layer.mlp.shared_expert.register_forward_hook(shared)
        layer.mlp.shared_expert_gate.register_forward_hook(shared_gate)

    @property
    def reader(self):
        if self._reader is None:
            raise RuntimeError(
                "checkpoint reference operations require an active forward"
            )
        return self._reader

    def _layer_hook(self, layer):
        def hook(module, inputs, output):
            if self.layer_output is not None:
                self.layer_output(layer, output.detach().cpu())
            if self.progress is not None:
                self.progress(f"canonical decoder layer {layer + 1}/48")

        return hook

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, *, past_key_values=None, logits_to_keep: int = 0
    ):
        if self._reader is not None:
            raise RuntimeError("canonical reference forward is not reentrant")
        if (
            input_ids.dtype != torch.int64
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or input_ids.shape[1] <= 0
        ):
            raise ValueError(
                "canonical reference requires nonempty int64 [1,tokens] input"
            )
        if (
            type(logits_to_keep) is not int
            or not 0 <= logits_to_keep <= input_ids.shape[1]
        ):
            raise ValueError("logits_to_keep must fit the input token count")
        if ((input_ids < 0) | (input_ids >= 248320)).any():
            raise ValueError("reference token ID lies outside checkpoint vocabulary")
        precision = torch.get_float32_matmul_precision()
        self._input_tokens = input_ids.shape[1]
        try:
            torch.set_float32_matmul_precision("highest")
            with CheckpointTensors(
                self.manifest, byte_budget=self.byte_budget
            ) as reader:
                self._reader = reader
                output = self.model(
                    input_ids=input_ids.to(self.device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                hidden = output.last_hidden_state
                selected = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
                logits = self.lm_head(selected).float().cpu()
                return DecoderReferenceOutput(
                    logits, hidden.cpu(), output.past_key_values, reader.bytes_read
                )
        finally:
            self._reader = None
            torch.set_float32_matmul_precision(precision)


__all__ = ["CanonicalTextReference", "DecoderReferenceOutput"]
