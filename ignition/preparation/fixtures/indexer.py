# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite private paged-index operands for b12x sparse-selection tuning."""

from functools import partial

import torch
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import planned_scratch
from ignition.preparation.tuning import benchmark_resource


def bind(state, arguments, storage):
    caps = state.layout.caps
    return state.bind(scratch=storage, real_page_table=arguments["page_table"],
        cache_seqlens_int32=arguments["cache_lengths"], active_width=arguments["active_width"],
        expected_num_q_heads=caps.num_q_heads, shared_page_table=caps.shared_page_table,
        output_physical_slots=caps.output_physical_slots)


def run(state, runtime, arguments):
    return state.run(runtime, **{name: arguments[name] for name in
        ("q_fp8", "query_weights", "index_k_cache", "output_indices", "output_scores")})


class IndexerBenchmarkInputs:
    def __init__(self, original):
        self.arguments = {}
        pages = original["page_table"].shape[1]
        for name, value in original.items():
            if value is None:
                self.arguments[name] = None
                continue
            shape = (pages, *value.shape[1:]) if name == "index_k_cache" else value.shape
            self.arguments[name] = torch.empty_strided(shape, value.stride(),
                                                       dtype=value.dtype, device=value.device)
        a = self.arguments
        device = a["q_fp8"].device
        generator = torch.Generator(device=device).manual_seed(530128)
        a["q_fp8"].copy_(torch.randn(a["q_fp8"].shape, device=device, generator=generator).to(a["q_fp8"].dtype))
        a["query_weights"].normal_(std=1 / 64, generator=generator)
        values = torch.randn((pages, 64 * 128), device=device, generator=generator).to(torch.float8_e4m3fn)
        a["index_k_cache"][:, :64 * 128].copy_(values.view(torch.uint8))
        a["index_k_cache"][:, 64 * 128:].view(torch.float32).fill_(.125)
        table = torch.arange(pages, dtype=torch.int32, device=device)
        if a["page_table"].stride(0) == 0:
            a["page_table"][0].copy_(table)
        else:
            a["page_table"].copy_(table.expand_as(a["page_table"]))
        if a["active_width"] is not None:
            a["active_width"].fill_(pages * 64)
        counts = sorted({min(pages * 64, 512), pages * 64})
        self.producers = tuple(partial(a["cache_lengths"].fill_, count) for count in counts)
        self.producers[-1]()

    @property
    def nbytes(self):
        return sum(value.untyped_storage().nbytes() for value in self.arguments.values() if value is not None)


def benchmark_call(state, *, arguments, session, workspace):
    identity = tuple((name, None if value is None else
        (value.shape if name != "index_k_cache" else (1, *value.shape[1:]),
         value.stride(), value.dtype, value.device)) for name, value in arguments.items())
    inputs = benchmark_resource(session, ("sparse_indexer", identity),
                                lambda: IndexerBenchmarkInputs(arguments))
    storage = planned_scratch(state.layout.scratch_specs(), workspace)
    runtime = bind(state, inputs.arguments, storage)
    return PreparedCall(run=lambda: run(state, runtime, inputs.arguments),
        produce=inputs.producers[0], benchmark_producers=inputs.producers,
        owners=(inputs, storage, runtime))
