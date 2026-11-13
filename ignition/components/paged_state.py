# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuation rows and heterogeneous page regions with runtime-sized backing."""

from dataclasses import dataclass

import torch

from .state import StateStorage


@dataclass(frozen=True)
class PagedStateStorage(StateStorage):
    pages: dict[str, torch.Tensor]
    reserved_backing: torch.Tensor

    @classmethod
    def allocate(
        cls, layout, page_specs, slots, pages, device, *, storage=None, maximum_pages=None, maximum_rows=None, backed_slots=None
    ):
        if type(pages) is not int or pages <= 0:
            raise ValueError("cache requires positive logical page capacity")
        maximum = pages if storage is None else maximum_pages
        if type(maximum) is not int or maximum < pages:
            raise ValueError("virtual cache extent must cover prepared physical pages")
        backing = StateStorage.allocate(layout, slots, device, storage=storage, maximum_slots=maximum_rows, backed_slots=backed_slots)
        regions = {}
        for name, shape in page_specs:
            regions[name] = (
                torch.zeros((pages, *shape), dtype=torch.uint8, device=device)
                if storage is None
                else storage.device_allocate(
                    name, (maximum, *shape), torch.uint8, group="KV_PAGES", maximum=maximum, units=pages
                )
            )
        backing.memory_capacity[1] = pages
        reserved = backing.backing
        if storage is not None:
            from b12x.sequence._shared.disk_table import _tensor_from_pointer

            region = next(r for r in storage.dynamic_regions if r.address == backing.backing.data_ptr())
            reserved = _tensor_from_pointer(
                region.address,
                shape=(region.maximum, layout.row_bytes),
                dtype=torch.uint8,
                device=backing.backing.device,
                nbytes=region.maximum * layout.row_bytes,
            )
        return cls(backing.backing, backing.layout, backing.memory_capacity, regions, reserved)

    def reserved_region(self, name):
        """Expose reserved rows to kernels whose admission checks bound physical rows."""
        region = self.layout.region(name)
        return (
            self.reserved_backing[:, region.offset : region.offset + region.nbytes]
            .view(region.dtype)
            .view(self.reserved_backing.shape[0], *region.shape)
        )

    @property
    def last_hidden(self):
        return self.region("last_hidden")

    def partial_pages(self):
        return self.pages.items()

    def clear_for_preparation(self):
        self.backing[:int(self.memory_capacity[0].item())].zero_()
        count = int(self.memory_capacity[1].item())
        for value in self.pages.values():
            value[:count].zero_()
