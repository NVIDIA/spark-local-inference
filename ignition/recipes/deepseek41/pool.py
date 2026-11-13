# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared compressed pages and complete per-request continuation state."""

from dataclasses import dataclass

import torch

from ignition.components.state import StateLayout
from ignition.components.paged_state import PagedStateStorage

PAGE_TOKENS = 128
SWA_RECORD_BYTES = 528
COMPRESSED_RECORD_BYTES = 288
INDEX_RECORD_BYTES = 68


def page_specs(config):
    """One logical page covers the same token span in every source region."""
    for source in config.sources:
        records = source.records(PAGE_TOKENS)
        yield f"layer{source.layer}.latent_page", (records * COMPRESSED_RECORD_BYTES,)
        yield f"layer{source.layer}.index_page", (records * INDEX_RECORD_BYTES,)


def page_bytes(config):
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config, *, dspark=False):
    regions = [(f"layer{i}.window", (config.text["sliding_window"] * SWA_RECORD_BYTES,), torch.uint8)
               for i in range(config.layers)]
    for source in config.sources:
        if source.ratio == 2:
            regions.extend((
                (f"layer{source.layer}.pending_values", (512,), torch.float32),
                (f"layer{source.layer}.pending_gates", (512,), torch.float32),
                (f"layer{source.layer}.pending_position", (1,), torch.int64),
            ))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    if dspark:
        regions.extend((f"draft{i}.window", (128 * SWA_RECORD_BYTES,), torch.uint8)
                       for i in range(config.draft.layers))
    regions.extend((
        ("last_hidden", (config.text["hidden_size"],), torch.bfloat16),
        ("consumed_tokens", (1,), torch.int64),
        ("engram_history", (3,), torch.int64),
    ))
    return StateLayout.from_specs(regions)


@dataclass(frozen=True)
class ContinuationPool(PagedStateStorage):
    config: object

    @classmethod
    def allocate(cls, config, slots, pages, device, *, storage=None, maximum_pages=None, maximum_rows=None,
                 backed_slots=None, dspark=False):
        backing = PagedStateStorage.allocate(continuation_layout(config, dspark=dspark), page_specs(config),
            slots, pages, device, storage=storage, maximum_pages=maximum_pages, maximum_rows=maximum_rows,
            backed_slots=backed_slots)
        return cls(backing.backing, backing.layout, backing.memory_capacity,
                   backing.pages, backing.reserved_backing, config)

    def window(self, layer, *, draft=False):
        """Expose the reserved row extent to cache writers; admission bounds physical rows."""
        return self.reserved_region(f"{'draft' if draft else 'layer'}{layer}.window")

