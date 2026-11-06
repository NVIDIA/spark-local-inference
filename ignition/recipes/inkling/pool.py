# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 paged KV, local rings, and all four FP32 convolution histories per layer."""

from dataclasses import dataclass
import math

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

PAGE_TOKENS = 128
KV_DTYPE = torch.bfloat16


def page_specs(config, shard):
    for layer in range(config.layers):
        geometry = config.attention(layer)
        if geometry.window < 0:
            _, heads = geometry.local(shard)
            for kind, dim in (("key", geometry.key_dim), ("value", geometry.value_dim)):
                yield f"layer{layer}.{kind}", (PAGE_TOKENS * heads * dim * KV_DTYPE.itemsize,)


def page_bytes(config, shard):
    return sum(shape[0] for _, shape in page_specs(config, shard))


def continuation_layout(config, shard):
    regions = []
    for layer in range(config.layers):
        geometry = config.attention(layer)
        if geometry.window > 0:
            _, heads = geometry.local(shard)
            for kind, dim in (("key", geometry.key_dim), ("value", geometry.value_dim)):
                regions.append((f"layer{layer}.{kind}", (geometry.window, heads, dim), KV_DTYPE))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config, shard))
    for layer in range(config.layers):
        geometry = config.attention(layer)
        _, heads = geometry.local(shard)
        for kind, channels in (("key_conv", heads * geometry.key_dim),
                               ("value_conv", heads * geometry.value_dim),
                               ("attention_conv", config.hidden), ("mlp_conv", config.hidden)):
            regions.append((f"layer{layer}.{kind}", (channels, config.text["conv_kernel_size"] - 1), torch.float32))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16),
                    ("consumed_tokens", (1,), torch.int64)))
    layout = StateLayout.from_specs(regions)
    # Rows align to whole KV pages so window_pages can view rings across adjacent rows.
    alignment = math.lcm(256, *(math.prod(shape) * dtype.itemsize
        for name, shape, dtype in regions if name.startswith("layer") and dtype == KV_DTYPE))
    return StateLayout(layout.regions, (layout.row_bytes + alignment - 1) // alignment * alignment)


@dataclass(frozen=True)
class ContinuationPool(PagedStateStorage):
    config: object
    shard: object

    @classmethod
    def allocate(cls, config, shard, slots, pages, device, *, storage=None, maximum_pages=None, maximum_rows=None,
                 backed_slots=None):
        layout = continuation_layout(config, shard)
        backing = PagedStateStorage.allocate(layout, page_specs(config, shard), slots, pages, device,
            storage=storage, maximum_pages=maximum_pages, maximum_rows=maximum_rows, backed_slots=backed_slots)
        return cls(backing.backing, backing.layout, backing.memory_capacity,
                   backing.pages, backing.reserved_backing, config, shard)

    def cache(self, layer, kind):
        geometry = self.config.attention(layer)
        name = f"layer{layer}.{kind}"
        if geometry.window > 0:
            return self.reserved_region(name)
        _, heads = geometry.local(self.shard)
        dim = geometry.key_dim if kind == "key" else geometry.value_dim
        return self.pages[name].view(KV_DTYPE).view(-1, PAGE_TOKENS, heads, dim)

    def window_page_stride(self, layer):
        geometry = self.config.attention(layer)
        _, heads = geometry.local(self.shard)
        record = PAGE_TOKENS * heads * geometry.key_dim * KV_DTYPE.itemsize
        if self.layout.row_bytes % record:
            raise ValueError("circular continuation rows must align to physical KV pages")
        return self.layout.row_bytes // record

    def window_pages(self, layer, kind):
        """Expose circular regions as pages with the continuation row stride intact."""
        geometry = self.config.attention(layer)
        _, heads = geometry.local(self.shard)
        dim = geometry.key_dim if kind == "key" else geometry.value_dim
        region = self.layout.region(f"layer{layer}.{kind}")
        row_pages = self.window_page_stride(layer)
        pages = (self.reserved_backing.shape[0] - 1) * row_pages + geometry.window // PAGE_TOKENS
        return self.reserved_backing.view(KV_DTYPE).as_strided(
            (pages, PAGE_TOKENS, heads, dim), (PAGE_TOKENS * heads * dim, heads * dim, dim, 1),
            storage_offset=region.offset // KV_DTYPE.itemsize)

    def convolution(self, layer, kind):
        return self.reserved_region(f"layer{layer}.{kind}_conv")
