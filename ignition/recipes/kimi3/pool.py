# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""KDA continuation rows, target attention pages, and accepted draft context."""

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.recurrent import DeltaState
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS
from .dcp import RECORD_BYTES
from ignition.recipes.kimi31.dflash_config import DFlash2Config


def page_specs(config):
    """Per-rank target and full-context draft records in one logical page."""
    # E4M3 records with a unit scale, as vLLM's Kimi K3 serving stores its latent cache (--kv-cache-dtype fp8).
    for index, kind in enumerate(config.layers):
        if kind in ("mla", "dsa"):
            yield f"layer{index}.latent", (PAGE_TOKENS * RECORD_BYTES,)
            if kind == "dsa":
                yield f"layer{index}.indexer", (PAGE_TOKENS * (config.dsa.head_dim + 4),)
    if config.draft is not None and config.draft.window is None:
        for index in range(config.draft.layers):
            yield f"draft{index}.latent", (PAGE_TOKENS * RECORD_BYTES * 2,)


def page_bytes(config):
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config, world_size, *, speculative_tokens=0):
    """One continuation row preserves KDA history and each cache region's partial page.

    Speculation widens KDA convolution history by the verified drafts. Sliding-window drafts
    retain every layer's accepted history here; full-context drafts retain their history in pages.
    """
    heads = config.text["linear_attn_config"]["num_heads"] // world_size
    regions = []
    for index, kind in enumerate(config.layers):
        if kind == "kda":
            regions.extend(((f"layer{index}.convolution", (3 * heads * 128, 3 + speculative_tokens), torch.bfloat16),
                            (f"layer{index}.recurrent", (heads, 128, 128), torch.float32)))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    if isinstance(config.draft, DFlash2Config):
        for layer in range(config.draft.layers):
            regions.append((f"draft{layer}.latent", (config.draft.window, RECORD_BYTES), torch.float8_e4m3fn))
    elif config.draft is not None and config.draft.window is not None:
        d = config.draft
        for layer in range(d.layers):
            for kind in ("key", "value"):
                regions.append((f"draft{layer}.{kind}", (d.window, d.kv_heads // world_size, d.head_dim),
                                torch.bfloat16))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, world_size, slots, pages, device, *, speculative_tokens=0, **kwargs):
        return cls.allocate(continuation_layout(config, world_size, speculative_tokens=speculative_tokens),
                            page_specs(config), slots, pages, device, **kwargs)

    def recurrent(self, layer):
        return DeltaState(self.reserved_region(f"layer{layer}.convolution"), self.reserved_region(f"layer{layer}.recurrent"))

    def attention(self, layer):
        return self.pages[f"layer{layer}.latent"].view(torch.float8_e4m3fn).view(-1, PAGE_TOKENS, RECORD_BYTES)

    def indexer_pages(self, layer):
        """Two 64-record sparse-index pages share each 128-record latent page's allocation."""
        return self.pages[f"layer{layer}.indexer"].view(-1, 64, 132)

    def draft_attention(self, layer):
        return self.pages[f"draft{layer}.latent"].view(torch.bfloat16).view(-1, PAGE_TOKENS, RECORD_BYTES)

    def share_pages(self, storage, group):
        """Every rank's address of each MLA layer's page pool, {layer: [world] int64}, mapped through CUDA IPC.

        Context-parallel prefill reads earlier positions from the ranks that own them.
        """
        names = [name for name in self.pages if name.startswith("layer") and name.endswith(".latent")]
        regions = {region.address: region for region in storage.dynamic_regions}
        addresses = storage.peer_storage.share_regions(group, [regions[self.pages[name].data_ptr()] for name in names])
        return {int(name.split(".")[0].removeprefix("layer")): torch.tensor(pointers, dtype=torch.int64,
                device=self.backing.device) for name, pointers in zip(names, addresses, strict=True)}
