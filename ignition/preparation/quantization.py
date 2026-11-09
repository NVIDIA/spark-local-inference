# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Supported ModelOpt projection storage declarations."""

from collections.abc import Mapping
import fnmatch
import re


def matches_pattern(name: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return bool(re.search(pattern[3:], name))
    return fnmatch.fnmatchcase(name, pattern)


def modelopt_nvfp4_layers(config, candidates):
    """Expand a ModelOpt NVFP4 ``quantization_config`` over the candidate linear names.

    Requires ``config_groups.group_0`` to declare 4-bit group-16 weights and ``ignore``
    to be a list of glob strings; every candidate no ignore pattern matches is returned
    as an NVFP4 group-16 entry.
    """
    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping) or quant.get("quant_method") != "modelopt" or quant.get("quant_algo") != "NVFP4":
        raise ValueError("expected modelopt NVFP4 quantization_config")
    groups = quant.get("config_groups")
    group_0 = groups.get("group_0") if isinstance(groups, Mapping) else None
    if not isinstance(group_0, Mapping):
        raise ValueError("modelopt NVFP4 quantization requires config_groups.group_0")
    weights = group_0.get("weights")
    if not isinstance(weights, Mapping) or weights.get("num_bits") != 4 or weights.get("group_size") != 16:
        raise ValueError("modelopt NVFP4 quantization requires 4-bit group_size=16 weights")
    ignore = quant.get("ignore", ())
    if not isinstance(ignore, list) or any(not isinstance(pattern, str) for pattern in ignore):
        raise ValueError("modelopt NVFP4 ignore list must contain glob strings")
    entry = {"quant_algo": "NVFP4", "group_size": 16}
    return {
        name: entry
        for name in candidates
        if not any(matches_pattern(name, pattern) for pattern in ignore)
    }


def quantized_layers(config):
    quant = config.get("quantization_config")
    if (not isinstance(quant, Mapping) or quant.get("quant_method") != "modelopt"
            or quant.get("quant_algo") != "MIXED_PRECISION"
            or quant.get("kv_cache_quant_algo") is not None):
        raise ValueError("expected modelopt MIXED_PRECISION with unquantized KV")
    layers = quant.get("quantized_layers")
    if not isinstance(layers, Mapping):
        raise ValueError("missing quantized_layers mapping")
    return layers


def linear_format(layers, prefix):
    entry = layers.get(prefix)
    if not isinstance(entry, Mapping):
        raise ValueError(f"{prefix}: missing quantized_layers declaration")
    algo = entry.get("quant_algo")
    if algo == "FP8_PB_WO":
        if (entry.get("weight_block_size") != [128, 128] or entry.get("group_size") is not None
                or entry.get("weight_scale_encoding") is not None):
            raise ValueError(f"{prefix}: unsupported block-FP8 quantization {entry!r}")
        return "block_fp8"
    if algo in ("IQ2_XXS", "Q8_0"):
        group, payload = (256, 66) if algo == "IQ2_XXS" else (32, 34)
        if (entry.get("group_size") != group or entry.get("block_payload_bytes") != payload
                or entry.get("packing") != "ggml" or entry.get("weight_scale_encoding") is not None):
            raise ValueError(f"{prefix}: invalid {algo} block encoding {entry!r}")
        return algo.lower()
    if algo == "FP8" and entry.get("group_size") is None and entry.get("weight_scale_encoding") is None:
        return "tensor_fp8"
    group = {"MXFP8": 32, "MXFP4": 32, "NVFP4": 16, "W4A16_NVFP4": 16}.get(algo)
    declared_group = entry.get("group_size")
    block = entry.get("weight_block_size")
    if algo == "MXFP8" and block == [1, 32] and declared_group is None:
        declared_group = 32
    if (group is None or declared_group != group
            or (block is not None and block != [1, group])):
        raise ValueError(f"{prefix}: unsupported linear quantization {entry!r}")
    if entry.get("weight_scale_encoding") is not None:
        raise ValueError(f"{prefix}: dense projections require ordinary block scales")
    return algo.lower() if algo in ("MXFP8", "MXFP4") else "nvfp4"


def projection_group_format(layers, prefixes):
    if not prefixes:
        raise ValueError("projection group must be nonempty")
    formats = [linear_format(layers, prefix) for prefix in prefixes]
    if "tensor_fp8" in formats and len(prefixes) > 1:
        raise ValueError("tensor-FP8 projections retain separate checkpoint activation and weight scales")
    if len(set(formats)) != 1:
        raise ValueError("fused projections must use the same quantization format")
    return formats[0]


def expert_format(layers, prefix):
    entry = layers.get(prefix)
    if not isinstance(entry, Mapping):
        raise ValueError(f"{prefix}: missing expert quantization declaration")
    if entry.get("weight_scale_encoding") in ("csf", "csf_32x2_v1"):
        if entry.get("quant_algo") == "NVFP4" and entry.get("group_size") == 16:
            return "nvfp4_csf", "a4"
        raise ValueError(f"{prefix}: unsupported compressed expert scales")
    fmt = linear_format(layers, prefix)
    if fmt in ("iq2_xxs", "q8_0"):
        return fmt, "a16"
    if fmt not in ("nvfp4", "mxfp4"):
        raise ValueError(f"{prefix}: expected FP4 or GGML block-quantized experts")
    mode = "a16" if entry["quant_algo"] == "W4A16_NVFP4" else "a8" if fmt == "mxfp4" else "a4"
    return fmt, mode
