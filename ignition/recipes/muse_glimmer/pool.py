# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 paged histories, partial-page snapshots, and saved normalized hidden rows."""

from dataclasses import dataclass
import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

PAGE_TOKENS = 128
KV_DTYPE = torch.bfloat16


def page_specs(config):
    c = config.text
    size = PAGE_TOKENS * c["num_key_value_heads"] * c["head_dim"] * KV_DTYPE.itemsize
    for layer in range(config.layers):
        for kind in ("key", "value"):
            yield f"layer{layer}.{kind}", (size,)


def page_bytes(config):
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config):
    return StateLayout.from_specs([*((name, shape, torch.uint8) for name, shape in page_specs(config)),
        ("last_hidden", (config.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)])


@dataclass(frozen=True)
class ContinuationPool(PagedStateStorage):
    config: object

    @classmethod
    def allocate(cls, config, slots, pages, device, *, storage=None, maximum_pages=None,
                 maximum_rows=None, backed_slots=None):
        backing = PagedStateStorage.allocate(continuation_layout(config), page_specs(config), slots, pages, device,
            storage=storage, maximum_pages=maximum_pages, maximum_rows=maximum_rows, backed_slots=backed_slots)
        return cls(backing.backing, backing.layout, backing.memory_capacity, backing.pages, backing.reserved_backing, config)

    def cache(self, layer, kind):
        c = self.config.text
        return self.pages[f"layer{layer}.{kind}"].view(KV_DTYPE).view(
            -1, PAGE_TOKENS, c["num_key_value_heads"], c["head_dim"])
