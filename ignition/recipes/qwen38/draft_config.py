# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned DFlash2 draft and the Flash Next target feature geometry."""

REPOSITORY = "incoai/Qwen3.8-Flash-Next-DFlash2"
REVISION = "a22a1744e85c17364dcc54a6a420af7135d4c7f6"
WEIGHT_BYTES = 1_206_140_008

EXPECTED = dict(hidden_size=2560, intermediate_size=7680, num_hidden_layers=5,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128, vocab_size=248320,
    num_target_layers=48, rms_norm_eps=1e-6, sliding_window=2048, rope_parameters=dict(rope_type="default", rope_theta=10000),
    dflash_config=dict(block_size=8, conv_kernel_size=2, conv_group_size=16,
        selector_rank=256, selector_top_k=16, mask_token_id=248077,
        target_layer_ids=[5, 15, 25, 35, 45]))


def load(snapshot):
    from ignition.components.dflash2_geometry import DFlash2Config

    return DFlash2Config.load(snapshot, target_hidden=2560, target_vocabulary=248320,
                              target_layers=48, expected=EXPECTED)
