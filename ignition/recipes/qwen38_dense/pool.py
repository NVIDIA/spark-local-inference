# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident Gated DeltaNet histories and paged full-attention KV."""

import torch

from ignition.components.gated_delta import GDNState
from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS


def page_specs(config, *, mtp=False):
    size = PAGE_TOKENS * config.text["num_key_value_heads"] * config.text["head_dim"]
    for index, kind in enumerate(config.layers):
        if kind == "full_attention":
            for field in ("key", "value"):
                yield f"layer{index}.{field}", (size,)


def page_bytes(config, *, mtp=False):
    return sum(shape[0] for _, shape in page_specs(config, mtp=mtp))


def continuation_layout(config, *, mtp=False, verification_width=8):
    if type(verification_width) is not int or not 1 <= verification_width <= 16:
        raise ValueError("Qwen verification capacity must be an integer in [1,16]")
    regions = []
    for index, kind in enumerate(config.layers):
        if kind == "linear_attention":
            regions.extend(((f"layer{index}.convolution", (10240, 2 + verification_width if mtp else 3), torch.bfloat16),
                            (f"layer{index}.recurrent", (48, 128, 128), torch.float32)))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config, mtp=mtp))
    if mtp:
        from ignition.components.dflash2_geometry import draft_context_specs
        from ignition.components.parallel import TensorParallelShard

        regions.extend(draft_context_specs(config.draft, TensorParallelShard(0, 1)))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16),
                    ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, slots, pages, device, *, mtp=False, verification_width=8, **kwargs):
        return cls.allocate(continuation_layout(config, mtp=mtp, verification_width=verification_width), page_specs(config, mtp=mtp), slots, pages, device, **kwargs)

    def partial_pages(self):
        return tuple(self.pages.items())

    def recurrent(self, layer):
        return GDNState(self.reserved_region(f"layer{layer}.convolution"),
                        self.reserved_region(f"layer{layer}.recurrent"))

    def attention(self, layer, config):
        shape = (-1, PAGE_TOKENS, config.text["num_key_value_heads"], config.text["head_dim"])
        prefix = f"layer{layer}"
        return tuple(self.pages[f"{prefix}.{field}"].view(torch.float8_e4m3fn).view(shape)
                     for field in ("key", "value"))
