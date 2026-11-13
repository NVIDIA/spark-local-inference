# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned DFlash2 checkpoint and target feature geometry for Qwen3.8-27B."""

from ignition.components.dflash2_geometry import DFlash2Config

REPOSITORY = "incoai/Qwen3.8-27B-DFlash2"
REVISION = "015e795645c74b1a0eeef3b570031fb62e769bc5"
PROFILE = dict(hidden_size=5120, intermediate_size=17408, num_hidden_layers=5,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128, vocab_size=248320,
    num_target_layers=64, rms_norm_eps=1e-6, sliding_window=2048,
    rope_parameters=dict(rope_type="default", rope_theta=10000000),
    dflash_config=dict(block_size=8, conv_kernel_size=2, conv_group_size=16, selector_rank=256,
        selector_top_k=16, mask_token_id=248070, target_layer_ids=[5, 19, 33, 47, 61]))


def from_dict(text, target):
    return DFlash2Config.from_dict(text, target_hidden=target.hidden, target_vocabulary=target.vocabulary,
        target_layers=len(target.layers), expected=PROFILE)


def load(snapshot, target):
    return DFlash2Config.load(snapshot, target_hidden=target.hidden, target_vocabulary=target.vocabulary,
        target_layers=len(target.layers), expected=PROFILE)
