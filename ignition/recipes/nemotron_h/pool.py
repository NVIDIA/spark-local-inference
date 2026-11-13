# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba continuations and paged FP8 key/value histories."""

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.recurrent import DeltaState
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS

REPLAY_RING = 16


def page_specs(config):
    width = config.text["num_key_value_heads"] * config.text["head_dim"]
    for index, kind in enumerate(config.layers):
        if kind == "attention":
            for field in ("key", "value"):
                yield f"layer{index}.{field}", (PAGE_TOKENS * width,)


def page_bytes(config):
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config):
    c, regions = config.text, []
    for index, kind in enumerate(config.layers):
        if kind == "mamba":
            columns = c["conv_kernel"] - 1 + (config.draft.block_size - 1 if config.draft is not None else 0)
            regions.extend(((f"layer{index}.convolution", (config.convolution_width, columns), torch.bfloat16),
                (f"layer{index}.recurrent", (c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"]), torch.bfloat16)))
            if config.draft is not None:
                regions.extend(((f"layer{index}.replay_x", (c["mamba_num_heads"], REPLAY_RING, c["mamba_head_dim"]), torch.bfloat16),
                    (f"layer{index}.replay_b", (c["n_groups"], REPLAY_RING, c["ssm_state_size"]), torch.bfloat16),
                    (f"layer{index}.replay_dt", (c["mamba_num_heads"], REPLAY_RING), torch.float32)))
    if config.draft is not None:
        regions.extend((("mamba_ring_start", (1,), torch.int32), ("mamba_pending_count", (1,), torch.int32),
                        ("draft_anchor", (config.hidden,), torch.bfloat16)))
        draft = config.draft.text
        for index in range(config.draft.layers):
            for field in ("key", "value"):
                regions.append((f"draft{index}.{field}",
                    (draft["sliding_window"], draft["num_key_value_heads"], draft["head_dim"]), torch.float8_e4m3fn))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, slots, pages, device, **kwargs):
        return cls.allocate(continuation_layout(config), page_specs(config), slots, pages, device, **kwargs)

    def recurrent(self, layer):
        return DeltaState(self.reserved_region(f"layer{layer}.convolution"), self.reserved_region(f"layer{layer}.recurrent"))

    def attention(self, layer, config):
        c = config.text
        return tuple(self.pages[f"layer{layer}.{field}"].view(torch.float8_e4m3fn).view(
            -1, PAGE_TOKENS, c["num_key_value_heads"], c["head_dim"]) for field in ("key", "value"))

    def replay_records(self, layer):
        return tuple(self.reserved_region(f"layer{layer}.replay_{field}") for field in ("x", "b", "dt"))
