# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validated Inkling 2 text, vision and drafter geometry and checkpoint tensor precision.

`ModelConfig.load` reads `config.json` and `hf_quant_config.json`. Every field that
selects arithmetic must be present; values the recipe does not implement, such as
projection biases, pre-affine normalization, shared experts or another routing
fan-out, are rejected instead of being ignored.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

ARCHITECTURE = "Inkling2ForConditionalGeneration"
TEXT_PREFIX = "model.llm"
VISION_PREFIX = "model.visual"
DRAFT_FILE = "mtp.safetensors"
QUANTIZATION_FILE = "hf_quant_config.json"
WORLD_SIZE = 4
VOCABULARY_ALIGNMENT = 64
# Activation blocks of both expert formats (NVFP4 16, MXFP8 32) must stay inside one rank's slice.
EXPERT_ALIGNMENT = 32

_TOP_LEVEL = {"architectures", "model_type", "eos_token_id", "text_config", "vision_config", "mtp_config",
              "transformers_version"}
_TEXT = {
    "model_max_length", "torch_dtype", "hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
    "num_key_value_heads", "head_dim", "d_rel", "rel_extent", "q_bias", "o_bias", "log_scaling_n_floor",
    "log_scaling_alpha", "rms_norm_eps", "pre_norm_pre_affine", "final_norm_pre_affine",
    "use_value_norm_no_gamma", "qk_l2_clip_threshold", "rel_l2_clip_threshold", "local_layer_ids",
    "sconv_kernel_size", "unpadded_vocab_size", "final_logit_softcapping", "swa_head_dim",
    "swa_num_attention_heads", "swa_num_key_value_heads", "sliding_window_size", "n_layers_per_attn_res_block",
    "attn_res_normalize_values", "n_routed_experts", "num_experts_per_tok", "n_shared_experts",
    "intermediate_size", "mlp_gate_softcap", "mlp_up_softcap", "route_scale", "use_gate_bias",
}
_VISION = {"temporal_patch_size", "native_patch_size", "adapter_merge_size", "projection_hidden_dim",
           "projector_bias", "hidden_size", "num_hidden_layers", "num_attention_heads", "mlp_ratio",
           "posemb_grid_size", "layer_norm_eps"}
_DRAFT = {"hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
          "intermediate_size", "max_position_embeddings", "torch_dtype", "q_bias", "o_bias", "rms_norm_eps",
          "rope_theta", "local_layer_ids", "sliding_window_size", "block_size", "markov_rank",
          "markov_head_type", "target_layer_ids"}

# Unquantized precision groups by (compute dtype, output dtype, forward output dtype, weight transform).
_PLAIN_GROUPS = {
    ("float32", "float32", None, "cast"): "float32",
    ("bfloat16", "bfloat16", None, "cast"): "bfloat16",
    ("bfloat16", "bfloat16", "float32", "cast"): "bfloat16_float32_output",
}
_NVFP4_ACTIVATION = {"amax_type": "per_token", "block_size": 16, "format": "nvfp4_e2m1", "rounding": "nearest",
                     "scale_dtype": "float8_e4m3fn", "scale_selection_rule": "fouroversix_mae"}
_MXFP8_ACTIVATION = {"block_size": 32, "format": "mxfp8_e4m3", "rounding": "nearest", "scale_dtype": "ue8m0"}


def _integer(section, name, *, minimum=1):
    value = section.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _real(section, name, *, positive=True):
    value = section.get(name)
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}finite number")
    return float(value)


def _flag(section, name, required):
    if section.get(name) is not required:
        raise ValueError(f"Inkling 2 preparation requires {name}={required}")


def _layer_ids(section, name, layers):
    ids = section.get(name)
    if (not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < layers for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError(f"{name} must list unique layer indices below {layers}")
    return frozenset(ids)


def _known(section, keys, label):
    if not isinstance(section, dict):
        raise ValueError(f"{label} must be an object")
    unknown = section.keys() - keys
    if unknown:
        raise ValueError(f"{label} declares unsupported fields: {', '.join(sorted(unknown))}")


@dataclass(frozen=True)
class TextConfig:
    """Decoder geometry. Local layers attend a sliding window; the others attend the full context."""

    hidden: int
    layers: int
    vocabulary: int
    valid_vocabulary: int
    heads: int
    kv_heads: int
    head_dim: int
    relative_width: int
    relative_extent: int
    window: int
    local_layers: frozenset[int]
    eps: float
    qk_clip: float
    relative_clip: float
    log_scaling_floor: int
    log_scaling_alpha: float
    convolution_width: int
    softcap: float
    residual_block: int
    experts: int
    experts_per_token: int
    intermediate: int
    gate_softcap: float
    up_softcap: float
    route_scale: float
    max_positions: int
    eos_token: int

    @classmethod
    def from_dict(cls, c, eos_token):
        _known(c, _TEXT, "text_config")
        missing = _TEXT - c.keys()
        if missing:
            raise ValueError(f"text_config is missing {', '.join(sorted(missing))}")
        if c["torch_dtype"] != "bfloat16":
            raise ValueError("Inkling 2 preparation requires a bfloat16 text model")
        for name, value in (("q_bias", False), ("o_bias", False), ("pre_norm_pre_affine", False),
                            ("final_norm_pre_affine", False), ("use_value_norm_no_gamma", True),
                            ("attn_res_normalize_values", True), ("use_gate_bias", True)):
            _flag(c, name, value)
        if c["n_shared_experts"] != 0:
            raise ValueError("Inkling 2 preparation implements routed experts without shared experts")
        layers = _integer(c, "num_hidden_layers")
        config = cls(
            hidden=_integer(c, "hidden_size"), layers=layers, vocabulary=_integer(c, "vocab_size"),
            valid_vocabulary=_integer(c, "unpadded_vocab_size"), heads=_integer(c, "num_attention_heads"),
            kv_heads=_integer(c, "num_key_value_heads"), head_dim=_integer(c, "head_dim"),
            relative_width=_integer(c, "d_rel"), relative_extent=_integer(c, "rel_extent"),
            window=_integer(c, "sliding_window_size"), local_layers=_layer_ids(c, "local_layer_ids", layers),
            eps=_real(c, "rms_norm_eps"), qk_clip=_real(c, "qk_l2_clip_threshold"),
            relative_clip=_real(c, "rel_l2_clip_threshold"), log_scaling_floor=_integer(c, "log_scaling_n_floor"),
            log_scaling_alpha=_real(c, "log_scaling_alpha", positive=False),
            convolution_width=_integer(c, "sconv_kernel_size", minimum=2),
            softcap=_real(c, "final_logit_softcapping"), residual_block=_integer(c, "n_layers_per_attn_res_block"),
            experts=_integer(c, "n_routed_experts"), experts_per_token=_integer(c, "num_experts_per_tok"),
            intermediate=_integer(c, "intermediate_size"), gate_softcap=_real(c, "mlp_gate_softcap"),
            up_softcap=_real(c, "mlp_up_softcap"), route_scale=_real(c, "route_scale"),
            max_positions=_integer(c, "model_max_length"), eos_token=eos_token)
        if (c["swa_head_dim"], c["swa_num_attention_heads"], c["swa_num_key_value_heads"]) != (
                config.head_dim, config.heads, config.kv_heads):
            raise ValueError("Inkling 2 preparation requires sliding-window layers with the global head geometry")
        if config.heads % config.kv_heads:
            raise ValueError("query heads must divide evenly among KV heads")
        if not config.valid_vocabulary <= config.vocabulary or not 0 <= eos_token < config.valid_vocabulary:
            raise ValueError("the valid vocabulary and EOS token must lie inside the embedding table")
        if config.residual_block > layers:
            raise ValueError("the attention-residual block size must not exceed the layer count")
        # Routing is implemented and tested for the checkpoint's eight experts per token.
        if config.experts_per_token != 8 or config.experts < config.experts_per_token:
            raise ValueError("Inkling 2 preparation implements top-8 routing over at least eight experts")
        for name, width, block in (("hidden_size", config.hidden, EXPERT_ALIGNMENT),
                                   ("intermediate_size", config.intermediate, EXPERT_ALIGNMENT)):
            if width % block:
                raise ValueError(f"{name} must be a multiple of {block} for block-scaled experts")
        return config

    def local(self, layer: int) -> bool:
        if not 0 <= layer < self.layers:
            raise ValueError(f"decoder layer {layer} is outside 0..{self.layers - 1}")
        return layer in self.local_layers

    def extent(self, layer: int) -> int:
        """Relative-bias distances of a layer: the window for local layers, `rel_extent` otherwise."""
        return self.window if self.local(layer) else self.relative_extent

    @property
    def global_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.layers) if i not in self.local_layers)

    def prefix(self, layers: int) -> "TextConfig":
        """The geometry of the first `layers` decoder layers, which development preparation runs alone."""
        if type(layers) is not int or not 1 <= layers <= self.layers:
            raise ValueError(f"a decoder-layer prefix holds 1 through {self.layers} layers")
        return replace(self, layers=layers, local_layers=frozenset(i for i in self.local_layers if i < layers))


@dataclass(frozen=True)
class VisionConfig:
    """Image tower geometry: native patches, merged `merge x merge` and projected to the text width."""

    hidden: int
    layers: int
    heads: int
    mlp_hidden: int
    patch: int
    temporal_patch: int
    merge: int
    projection_hidden: int
    position_grid: int
    eps: float

    @classmethod
    def from_dict(cls, c):
        _known(c, _VISION, "vision_config")
        missing = _VISION - c.keys()
        if missing:
            raise ValueError(f"vision_config is missing {', '.join(sorted(missing))}")
        _flag(c, "projector_bias", False)
        hidden, ratio = _integer(c, "hidden_size"), _real(c, "mlp_ratio")
        if not float(hidden * ratio).is_integer():
            raise ValueError("vision MLP width must be an integer multiple of the hidden size")
        config = cls(hidden=hidden, layers=_integer(c, "num_hidden_layers"), heads=_integer(c, "num_attention_heads"),
            mlp_hidden=int(hidden * ratio), patch=_integer(c, "native_patch_size"),
            temporal_patch=_integer(c, "temporal_patch_size"), merge=_integer(c, "adapter_merge_size"),
            projection_hidden=_integer(c, "projection_hidden_dim"), position_grid=_integer(c, "posemb_grid_size"),
            eps=_real(c, "layer_norm_eps"))
        if config.hidden % config.heads:
            raise ValueError("vision heads must divide the vision hidden size")
        return config


@dataclass(frozen=True)
class DraftConfig:
    """Block drafter: a non-causal stack over target taps with a recurrent Markov head."""

    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    eps: float
    rope_theta: float
    local_layers: frozenset[int]
    window: int
    block: int
    markov_rank: int
    taps: tuple[int, ...]
    max_positions: int

    @classmethod
    def from_dict(cls, c, text: TextConfig):
        _known(c, _DRAFT, "mtp_config")
        missing = _DRAFT - c.keys()
        if missing:
            raise ValueError(f"mtp_config is missing {', '.join(sorted(missing))}")
        if c["torch_dtype"] != "bfloat16":
            raise ValueError("the Inkling 2 drafter must be bfloat16")
        _flag(c, "q_bias", False)
        _flag(c, "o_bias", False)
        if c["markov_head_type"] != "rnn_hidden":
            raise ValueError("Inkling 2 preparation implements the rnn_hidden Markov head")
        layers = _integer(c, "num_hidden_layers")
        taps = c["target_layer_ids"]
        # The last tap is the sentinel for the target's final normalized hidden state.
        if (not isinstance(taps, list) or len(taps) < 2 or any(type(i) is not int for i in taps)
                or taps != sorted(set(taps)) or taps[-1] != text.layers or not 0 <= taps[0]):
            raise ValueError("target_layer_ids must be sorted unique target layers ending with the layer count")
        config = cls(hidden=_integer(c, "hidden_size"), layers=layers, heads=_integer(c, "num_attention_heads"),
            kv_heads=_integer(c, "num_key_value_heads"), head_dim=_integer(c, "head_dim"),
            intermediate=_integer(c, "intermediate_size"), eps=_real(c, "rms_norm_eps"),
            rope_theta=_real(c, "rope_theta"), local_layers=_layer_ids(c, "local_layer_ids", layers),
            window=_integer(c, "sliding_window_size"), block=_integer(c, "block_size", minimum=2),
            markov_rank=_integer(c, "markov_rank"), taps=tuple(taps),
            max_positions=_integer(c, "max_position_embeddings"))
        if config.hidden != text.hidden or config.heads % config.kv_heads:
            raise ValueError("the drafter must share the target width and group its query heads evenly")
        return config


@dataclass(frozen=True)
class Quantization:
    """Checkpoint-declared precision of every listed tensor.

    Values are `float32`, `bfloat16`, `bfloat16_float32_output` (BF16 weights whose
    projection output is FP32), `nvfp4` or `mxfp8`. Block-scaled groups name the
    expert weight tensors; their scale tensors take the weight's precision.
    """

    formats: Mapping[str, str]
    draft_formats: Mapping[str, str]

    @classmethod
    def from_dict(cls, q):
        if not isinstance(q, dict) or q.get("quant_method") != "inkling2" or q.keys() != {"quant_method", "model", "mtp"}:
            raise ValueError("hf_quant_config.json must declare inkling2 model and mtp precision groups")
        return cls(_groups(q["model"], "model"), _groups(q["mtp"], "mtp"))

    def expert_format(self, layer: int) -> str:
        prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp.experts."
        formats = {self.formats.get(prefix + name) for name in ("w13_weight", "w2_weight")}
        if len(formats) != 1 or not formats <= {"nvfp4", "mxfp8"}:
            raise ValueError(f"layer {layer} experts must declare one NVFP4 or MXFP8 format for w13 and w2")
        return formats.pop()


def _groups(section, label):
    if not isinstance(section, dict) or section.keys() != {"parameter_groups"}:
        raise ValueError(f"{label} precision must list parameter_groups")
    formats = {}
    for group in section["parameter_groups"]:
        kind = _group_kind(group)
        tensors = group.get("tensors")
        if not isinstance(tensors, list) or any(not isinstance(name, str) for name in tensors):
            raise ValueError(f"{label} precision groups must list tensor names")
        for name in tensors:
            if name in formats:
                raise ValueError(f"{name} appears in more than one precision group")
            formats[name] = kind
    return MappingProxyType(formats)


def _group_kind(group):
    if not isinstance(group, dict):
        raise ValueError("precision groups must be objects")
    keys = group.keys() - {"tensors"}
    compute, output = group.get("compute_dtype"), group.get("output_dtype")
    activation, shape = group.get("required_activation_quantization"), group.get("weight_block_shape")
    transform = group.get("weight_transform")
    if compute in ("float32", "bfloat16"):
        allowed = {"compute_dtype", "output_dtype", "required_activation_quantization", "weight_block_shape",
                   "weight_transform", "fwd_output_dtype"}
        kind = _PLAIN_GROUPS.get((compute, output, group.get("fwd_output_dtype"), transform))
        if keys - allowed or kind is None or activation is not None or shape not in (None, []):
            raise ValueError(f"unsupported {compute} precision group")
        return kind
    if compute == "nvfp4":
        if (keys != {"compute_dtype", "output_dtype", "required_activation_quantization", "weight_block_shape",
                     "weight_scale_format", "weight_transform"}
                or output != "uint8" or activation != _NVFP4_ACTIVATION or shape != [1, 16]
                or group.get("weight_scale_format") != "amax_denominator" or transform != "nvfp4_nearest"):
            raise ValueError("NVFP4 experts must use per-token four-over-six activations and amax/denominator scales")
        return "nvfp4"
    if compute == "mxfp8":
        # The stored scales hold one UE8M0 exponent per output row and 32 inputs, with distinct rows
        # inside 32-row groups, so the stored tensor shape, not this block shape, fixes their layout.
        if (keys != {"compute_dtype", "output_dtype", "required_activation_quantization", "weight_block_shape",
                     "weight_transform"}
                or output != "float8_e4m3fn" or activation != _MXFP8_ACTIVATION or shape not in ([1, 32], [32, 32])
                or transform != "mxfp8_nearest"):
            raise ValueError("MXFP8 experts must use 32-element UE8M0 activation and weight scales")
        return "mxfp8"
    raise ValueError(f"unsupported precision group compute dtype {compute!r}")


@dataclass(frozen=True)
class ModelConfig:
    text: TextConfig
    vision: VisionConfig
    draft: DraftConfig
    quantization: Quantization

    @classmethod
    def load(cls, snapshot):
        snapshot = Path(snapshot)
        return cls.from_dict(json.loads((snapshot / "config.json").read_text()),
                             json.loads((snapshot / QUANTIZATION_FILE).read_text()))

    @classmethod
    def from_dict(cls, config, quantization):
        _known(config, _TOP_LEVEL, "config.json")
        if config.get("model_type") != "inkling2" or config.get("architectures") != [ARCHITECTURE]:
            raise ValueError(f"Inkling 2 preparation requires {ARCHITECTURE}")
        eos = config.get("eos_token_id")
        if type(eos) is not int:
            raise ValueError("eos_token_id must be one token id")
        text = TextConfig.from_dict(config.get("text_config"), eos)
        result = cls(text, VisionConfig.from_dict(config.get("vision_config")),
                     DraftConfig.from_dict(config.get("mtp_config"), text), Quantization.from_dict(quantization))
        for layer in range(text.layers):
            result.quantization.expert_format(layer)
        return result

    def expert_format(self, layer: int) -> str:
        self.text.local(layer)
        return self.quantization.expert_format(layer)

    def validate_shard(self, shard):
        """Require the four-rank layout: whole query and KV heads, aligned expert slices and vocabulary."""
        if shard.world_size != WORLD_SIZE:
            raise ValueError(f"Inkling 2 preparation requires tensor parallelism of {WORLD_SIZE}")
        t, d = self.text, self.draft
        shard.partition(t.heads)
        shard.partition(t.kv_heads)
        shard.partition(t.intermediate, alignment=EXPERT_ALIGNMENT)
        shard.partition(d.heads)
        shard.partition(d.kv_heads)
        shard.partition(d.intermediate, alignment=EXPERT_ALIGNMENT)
        shard.padded_partition(t.vocabulary, alignment=VOCABULARY_ALIGNMENT)
