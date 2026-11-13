# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged GQA and region summaries, with sliding and partial-region continuation state."""

from dataclasses import dataclass

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS


def page_specs(config, shard):
    for layer in range(config.layers):
        a = config.attention(layer)
        if a.window < 0:
            _, heads = a.local(shard)
            for kind in ("key", "value"):
                yield f"layer{layer}.{kind}", (PAGE_TOKENS * heads * a.dimension,)
            yield f"layer{layer}.index", (PAGE_TOKENS // config.sparse["region_block_size"] * config.sparse["proxy_dim"],)


def page_bytes(config, shard):
    return sum(shape[0] for _, shape in page_specs(config, shard))


def continuation_layout(config, shard, *, mtp=True):
    regions = []
    for layer in range(config.layers):
        a = config.attention(layer)
        if a.window > 0:
            _, heads = a.local(shard)
            regions.extend((f"layer{layer}.{kind}", (a.window, heads, a.dimension), torch.float8_e4m3fn)
                           for kind in ("key", "value"))
        else:
            regions.extend((f"layer{layer}.pending_{kind}",
                            (config.sparse["region_block_size"], 1, config.sparse["proxy_dim"]), torch.bfloat16)
                           for kind in ("key", "gate"))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config, shard))
    if mtp:
        regions.append(("draft.target_hidden", (config.hidden,), torch.bfloat16))
        width = config.text["num_nextn_predict_layers"] + 1
        regions.extend((("draft.pending_count", (1,), torch.int32),
                        ("draft.pending_tokens", (width,), torch.int64),
                        ("draft.pending_hidden", (width, config.hidden), torch.bfloat16),
                        ("draft.pending_embedding", (width, config.hidden), torch.bfloat16)))
        for index in range(config.text["num_nextn_predict_layers"]):
            a = config.attention(config.layers + index)
            _, heads = a.local(shard)
            regions.extend((f"draft{index}.{kind}", (a.window, heads, a.dimension), torch.float8_e4m3fn)
                           for kind in ("key", "value"))
            if index + 1 < config.text["num_nextn_predict_layers"]:
                regions.append((f"draft{index}.hidden", (config.hidden,), torch.bfloat16))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


@dataclass(frozen=True)
class ContinuationPool(PagedStateStorage):
    config: object
    shard: object

    @classmethod
    def for_model(cls, config, shard, slots, pages, device, *, mtp=True, **kwargs):
        backing = PagedStateStorage.allocate(continuation_layout(config, shard, mtp=mtp), page_specs(config, shard),
                                             slots, pages, device, **kwargs)
        return cls(backing.backing, backing.layout, backing.memory_capacity, backing.pages,
                   backing.reserved_backing, config, shard)

    def cache(self, layer, kind):
        a = self.config.attention(layer)
        name = f"layer{layer}.{kind}" if layer < self.config.layers else f"draft{layer - self.config.layers}.{kind}"
        if a.window > 0 or kind.startswith("pending_"):
            return self.reserved_region(name)
        if kind == "index":
            return self.pages[name].view(torch.float8_e4m3fn).view(-1,
                PAGE_TOKENS // self.config.sparse["region_block_size"], self.config.sparse["proxy_dim"])
        _, heads = a.local(self.shard)
        return self.pages[name].view(torch.float8_e4m3fn).view(-1, PAGE_TOKENS, heads, a.dimension)
