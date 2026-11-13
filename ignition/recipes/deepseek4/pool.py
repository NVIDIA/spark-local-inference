# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""C4/C128 histories and exact-prefix compressor continuation rows."""

from dataclasses import dataclass

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

PAGE_TOKENS = 256
INDEX_RECORD_BYTES = 132


def latent_page_bytes(records):
    # V4 places FP8/BF16 payloads before a page-local UE8M0 scale footer.
    return ((records * 584 + 575) // 576) * 576


def page_specs(config):
    for layer, ratio in enumerate(config.text["compress_ratios"][: config.layers]):
        if ratio:
            yield f"layer{layer}.latent_page", (latent_page_bytes(PAGE_TOKENS // ratio),)
        if ratio == 4:
            yield f"layer{layer}.index_page", (PAGE_TOKENS // ratio * INDEX_RECORD_BYTES,)


def page_bytes(config):
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config, *, dspark=True):
    regions = [
        (f"layer{i}.window", (latent_page_bytes(config.text["sliding_window"]),), torch.uint8)
        for i in range(config.layers)
    ]
    for compressor in config.compressors:
        regions.extend(
            (f"{compressor.prefix}.{kind}", compressor.state_shape, torch.float32)
            for kind in ("values", "scores")
        )
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    if dspark:
        regions.extend(
            (f"draft{i}.window", (latent_page_bytes(config.text["sliding_window"]),), torch.uint8)
            for i in range(config.draft.layers)
        )
    regions.extend(
        (
            ("last_hidden", (config.text["hidden_size"],), torch.bfloat16),
            ("consumed_tokens", (1,), torch.int64),
        )
    )
    return StateLayout.from_specs(regions)


@dataclass(frozen=True)
class ContinuationPool(PagedStateStorage):
    config: object

    @classmethod
    def allocate(
        cls, config, slots, pages, device, *, storage=None, maximum_pages=None, maximum_rows=None,
        backed_slots=None, dspark=True
    ):
        backing = PagedStateStorage.allocate(
            continuation_layout(config, dspark=dspark),
            page_specs(config),
            slots,
            pages,
            device,
            storage=storage,
            maximum_pages=maximum_pages,
            maximum_rows=maximum_rows,
            backed_slots=backed_slots,
        )
        return cls(
            backing.backing,
            backing.layout,
            backing.memory_capacity,
            backing.pages,
            backing.reserved_backing,
            config,
        )

    def window(self, layer, *, draft=False):
        return self.reserved_region(f"{'draft' if draft else 'layer'}{layer}.window")
