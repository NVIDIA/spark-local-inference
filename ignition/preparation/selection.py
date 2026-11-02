# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit round and prefill geometry for diagnostic graph preparation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSelection:
    batches: tuple[int, ...] = (1, 2, 4)
    contexts: tuple[int, ...] = (512, 2048, 8192, 16384)
    width: int = 8
    prefill_chunk: int = 0

    def __post_init__(self):
        for name, values in (("batches", self.batches), ("contexts", self.contexts)):
            if not isinstance(values, tuple) or not values or any(type(v) is not int for v in values):
                raise ValueError(f"benchmark {name} must be a nonempty tuple of integers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"benchmark {name} must be distinct and increasing")
        if any(batch not in (1, 2, 4) for batch in self.batches):
            raise ValueError("benchmark batches must be selected from 1, 2, 4")
        if type(self.prefill_chunk) is not int or self.prefill_chunk < 0 or self.prefill_chunk % 128:
            raise ValueError("benchmark prefill chunk must be zero or a positive multiple of 128")
        if self.prefill_chunk and self.batches != (1,):
            raise ValueError("prefill benchmark requires batch one")
        if any(context < (0 if self.prefill_chunk else 128) or context % 128 for context in self.contexts):
            raise ValueError("benchmark contexts must be positive multiples of 128, or zero for prefill")
        if type(self.width) is not int or self.width != 8:
            raise ValueError("benchmark verification width must be 8")

    @property
    def max_batch(self):
        return self.batches[-1]

    @property
    def max_context(self):
        return self.contexts[-1]

    @property
    def speculative_round_shapes(self):
        if self.prefill_chunk:
            return ()
        return tuple((batch, self.width, batch * self.width) for batch in self.batches)

    @property
    def required_context(self):
        return self.max_context + (self.prefill_chunk + 2 if self.prefill_chunk else 2 * self.width)

    def prefill_capacities(self, chunk):
        if self.prefill_chunk:
            if chunk != self.prefill_chunk:
                raise ValueError("benchmark chunk must match the prepared prefill chunk")
            return tuple(sorted({chunk, *(context % chunk for context in self.contexts if context % chunk)}))
        return tuple(sorted({min(context, chunk) for context in self.contexts}))

    def physical_pages(self, page_tokens):
        extent = self.max_context + (self.prefill_chunk if self.prefill_chunk else self.width)
        return 1 + self.max_batch * ((extent + page_tokens - 1) // page_tokens)

    def to_dict(self):
        return dict(batches=self.batches, contexts=self.contexts, width=self.width,
                    prefill_chunk=self.prefill_chunk,
                    speculative_round_shapes=self.speculative_round_shapes)
