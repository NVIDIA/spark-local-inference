# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small serving workloads and a synthetic prepared graph timing table."""

from .model import Graph, Request


def catalog(*, mixed: bool = False) -> tuple[Graph, ...]:
    graphs = [Graph(f"prefill-{p}", 2 + p / 64, prefill_capacity=p)
              for p in (64, 128, 256, 512, 1024, 2048, 4096)]
    for batch in (1, 2, 4, 8, 16):
        for width, singleton_ms in ((1, 15.0), (3, 17.3), (8, 30.0)):
            decode_ms = singleton_ms + 2 * (batch - 1)
            graphs.append(Graph(f"decode-b{batch}-w{width}", decode_ms,
                                decode_capacity=batch, width=width))
            if mixed:
                for p in (64, 256, 512, 1024, 4096):
                    graphs.append(Graph(f"mixed-p{p}-b{batch}-w{width}",
                                        max(2 + p / 64, decode_ms) + 1 + 0.15 * (p / 64 + batch * width),
                                        prefill_capacity=p, decode_capacity=batch, width=width))
    return tuple(graphs)


def workloads() -> dict[str, list[Request]]:
    def prefill(name, tokens=32_768, **kwargs):
        return Request(name, tokens, 128, **kwargs)

    def decode(name, accepted=7, output=128, **kwargs):
        probabilities = tuple(float(i < accepted) for i in range(7))
        return Request(name, 0, output, acceptance=probabilities, **kwargs)

    return {
        "solo_prefill": [prefill("p")],
        "two_prefills": [prefill("p0"), prefill("p1")],
        "sixteen_prefills": [prefill(f"p{i}") for i in range(16)],
        "tail_64": [prefill("p", 64)],
        "tail_576": [prefill("p", 576)],
        "two_decodes_low": [decode("d0", 0), decode("d1", 0)],
        "two_decodes_high": [decode("d0"), decode("d1")],
        "two_decodes_heterogeneous": [decode("high"), decode("low", 0)],
        "sixteen_decodes": [decode(f"d{i}", i % 8) for i in range(16)],
        "prefill_decode_low": [prefill("p"), decode("d", 0)],
        "prefill_decode_high": [prefill("p"), decode("d")],
        "prefill_four_decodes": [prefill("p"), *(decode(f"d{i}") for i in range(4))],
        "prefill_five_decodes": [prefill("p"), *(decode(f"d{i}") for i in range(5))],
        "short_arrival": [prefill("long"), prefill("short", 64, arrival_ms=1)],
        "short_decode": [decode("long", output=2048), decode("short", output=3)],
        "continuous_decode": [prefill("p"), decode("d", output=4096)],
    }
