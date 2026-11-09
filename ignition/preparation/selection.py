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
    speculation_study: bool = False
    study_widths: tuple[int, ...] = ()
    verification_widths: tuple[int, ...] = ()

    def __post_init__(self):
        for name, values in (("batches", self.batches), ("contexts", self.contexts)):
            if not isinstance(values, tuple) or not values or any(type(v) is not int for v in values):
                raise ValueError(f"benchmark {name} must be a nonempty tuple of integers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"benchmark {name} must be distinct and increasing")
        if any(batch not in (1, 2, 4, 8) for batch in self.batches):
            raise ValueError("benchmark batches must be selected from 1, 2, 4, 8")
        if type(self.prefill_chunk) is not int or self.prefill_chunk < 0 or self.prefill_chunk % 128:
            raise ValueError("benchmark prefill chunk must be zero or a positive multiple of 128")
        if self.prefill_chunk and self.batches != (1,):
            raise ValueError("prefill benchmark requires batch one")
        if any(context < (0 if self.prefill_chunk else 128) or context % 128 for context in self.contexts):
            raise ValueError("benchmark contexts must be positive multiples of 128, or zero for prefill")
        if self.speculation_study and (self.batches != (1,) or self.prefill_chunk):
            raise ValueError("speculation study requires one request and no prefill timing")
        if type(self.width) is not int or self.width not in ((8, 13, 16) if self.speculation_study else (8, 16)):
            raise ValueError("benchmark verification width must be 8 or 16 (also 13 for speculation studies)")

        if self.study_widths:
            if (not self.speculation_study or not isinstance(self.study_widths, tuple)
                    or tuple(sorted(set(self.study_widths))) != self.study_widths
                    or any(type(w) is not int or w not in (8, 13, 16) for w in self.study_widths)
                    or self.width != max(self.study_widths)):
                raise ValueError("study widths must be increasing distinct widths in 8,13,16 with width equal to their maximum")
        if self.verification_widths:
            if (self.speculation_study or self.prefill_chunk
                    or not isinstance(self.verification_widths, tuple)
                    or tuple(sorted(set(self.verification_widths))) != self.verification_widths
                    or any(type(w) is not int or not 1 <= w <= self.width for w in self.verification_widths)
                    or self.verification_widths[-1] != self.width):
                raise ValueError("verification widths must be distinct increasing serving widths through the prepared maximum")

    @property
    def selected_verification_widths(self):
        return self.verification_widths or (self.width,)

    @property
    def selected_study_widths(self):
        return (self.study_widths or (self.width,)) if self.speculation_study else ()

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
        return tuple((batch, width, batch * width) for batch in self.batches
                     for width in self.selected_verification_widths)

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
                    prefill_chunk=self.prefill_chunk, speculation_study=self.speculation_study, study_widths=self.selected_study_widths,
                    verification_widths=self.selected_verification_widths,
                    speculative_round_shapes=self.speculative_round_shapes)
