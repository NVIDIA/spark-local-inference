# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured paged attention ignores uncommitted cache rows."""

import os

import pytest
import torch

from ignition.components.paged_attention import PreparedPagedAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU",
)


def _poison_tail(cache, lengths, capacity, poison):
    rows = cache.view(len(lengths), capacity, *cache.shape[-2:])
    for row, length in enumerate(lengths):
        if cache.dtype == torch.float8_e4m3fn:
            rows[row, length:].view(torch.uint8).fill_(0x7F)
        else:
            rows[row, length:].fill_(poison)


@pytest.mark.parametrize(
    "mode,batch,query_capacity,query_count,qk_dim,value_dim,kv_dtype,route",
    [
        pytest.param("decode", 1, 1, 1, 256, 256, torch.float8_e4m3fn, "paged_decode", id="qwen-decode-b1"),
        pytest.param("decode", 4, 1, 1, 256, 256, torch.float8_e4m3fn, "paged_decode", id="qwen-decode-b4"),
        pytest.param("verify", 4, 8, 8, 256, 256, torch.float8_e4m3fn, "paged_decode", id="qwen-verify-b4-w8"),
        pytest.param("verify", 4, 16, 16, 256, 256, torch.float8_e4m3fn, "paged_decode", id="qwen-verify-b4-w16"),
        pytest.param("prefill", 1, 128, 17, 256, 256, torch.float8_e4m3fn, "tma", id="qwen-prefill-b1-q128-live17"),
        pytest.param("prefill", 1, 128, 17, 256, 256, torch.float8_e4m3fn, "tma-gated", id="qwen-prefill-b1-q128-live17-gated"),
        pytest.param("decode", 1, 1, 1, 128, 128, torch.bfloat16, "paged_decode", id="fast-decode-d128-bf16"),
        pytest.param("decode", 1, 1, 1, 128, 128, torch.float8_e4m3fn, "paged_decode", id="fast-decode-d128-fp8"),
        pytest.param("verify", 4, 8, 8, 128, 128, torch.bfloat16, "paged_decode", id="fast-verify-d128-bf16"),
        pytest.param("verify", 4, 8, 8, 128, 128, torch.float8_e4m3fn, "paged_decode", id="fast-verify-d128-fp8"),
        pytest.param("decode", 1, 1, 1, 64, 64, torch.bfloat16, "paged_decode", id="fast-decode-d64-bf16"),
        pytest.param("decode", 1, 1, 1, 64, 64, torch.float8_e4m3fn, "paged_decode", id="fast-decode-d64-fp8"),
        pytest.param("decode", 1, 1, 1, 192, 128, torch.bfloat16, "paged_decode", id="fast-decode-d192-bf16"),
        pytest.param("decode", 1, 1, 1, 192, 128, torch.float8_e4m3fn, "paged_decode", id="fast-decode-d192-fp8"),
        pytest.param("decode", 1, 1, 1, 256, 256, torch.bfloat16, "paged_decode", id="fast-decode-d256-bf16"),
        pytest.param("verify", 4, 8, 8, 256, 256, torch.bfloat16, "paged_decode", id="fast-verify-d256-bf16"),
    ],
)
@torch.no_grad()
def test_paged_attention_ignores_uncommitted_cache_rows(
    mode, batch, query_capacity, query_count, qk_dim, value_dim, kv_dtype, route,
):
    import b12x
    import flashinfer

    device = torch.device("cuda", 0)
    torch.manual_seed(311)
    capacity, page_size = 512, 128
    counts = [query_count] * batch
    if batch > 1:
        counts[1] = 0
    maximum_counts = counts.copy()
    live_queries = sum(counts)
    query_dtype = torch.float8_e4m3fn if route.startswith("tma") else torch.bfloat16
    query = (torch.randn((batch * query_capacity, 24, qk_dim), device=device) * .2).to(query_dtype)
    key = (torch.randn((batch * capacity // page_size, page_size, 4, qk_dim), device=device) * .2).to(kv_dtype)
    value = (torch.randn((*key.shape[:-1], value_dim), device=device) * .2).to(kv_dtype)
    clean_key, clean_value = key.clone(), value.clone()
    output = torch.zeros((batch * query_capacity, 24, value_dim), dtype=torch.bfloat16, device=device)
    output_gate = torch.randn(output.shape, dtype=torch.bfloat16, device=device) if route == "tma-gated" else None
    pages = torch.arange(key.shape[0], dtype=torch.int32, device=device).view(batch, -1)
    lengths = torch.tensor([257 if count else 0 for count in counts], dtype=torch.int32, device=device)
    cu = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32, device=device)
    failures = []

    with RecordedPreparationSession(
        device=device, checkpoint="qwen-attention-cache-tail",
        workload_schema="qwen-attention-cache-tail", numerical_policy="default",
    ) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(
            query, key, value, pages, lengths, cu, output,
            mode=mode, width=query_capacity if mode == "verify" else 1,
            session=session, workspace=arena, name=f"attention-{mode}-{batch}-{query_capacity}",
            output_gate=output_gate,
        )
        if route.startswith("tma"):
            assert attention.tma is not None
        elif route == "paged_decode":
            assert attention.tma is None and attention.decode
        else:
            assert attention.tma is None and not attention.decode
        session.tune()
        compiled = compile_forward(attention)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        print(
            f"route={attention.plan.component_id} mode={mode} batch={batch} "
            f"query_capacity={query_capacity} live_queries={live_queries} "
            f"qk_dim={qk_dim} value_dim={value_dim} kv_dtype={kv_dtype} "
            f"gated={output_gate is not None} "
            f"device={torch.cuda.get_device_name(device)} "
            f"b12x={b12x.__file__} flashinfer={flashinfer.__file__}",
            flush=True,
        )

        def replay():
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == allocated
            return output[:live_queries]

        try:
            for length in (1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 257):
                counts = [min(count, length) for count in maximum_counts]
                live_queries = sum(counts)
                active_lengths = [length if count else 0 for count in counts]
                lengths.copy_(torch.tensor(active_lengths, dtype=torch.int32, device=device))
                cu.copy_(torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32, device=device))
                key.copy_(clean_key)
                value.copy_(clean_value)
                clean = replay().clone()
                assert torch.isfinite(clean).all(), f"clean {mode} length={length} is non-finite"
                assert torch.equal(replay(), clean), f"clean {mode} length={length} is not repeatable"
                tail_values = (float("nan"), float("inf")) if kv_dtype == torch.bfloat16 else (float("nan"),)
                for tail_value in tail_values:
                    for poison in ("key", "value", "both"):
                        key.copy_(clean_key)
                        value.copy_(clean_value)
                        if poison in ("key", "both"):
                            _poison_tail(key, active_lengths, capacity, tail_value)
                        if poison in ("value", "both"):
                            _poison_tail(value, active_lengths, capacity, tail_value)
                        observed = replay()
                        finite = bool(torch.isfinite(observed).all())
                        unchanged = bool(torch.equal(observed, clean))
                        case = (f"mode={mode} batch={batch} width={query_capacity} length={length} "
                                f"live_queries={live_queries} kv_dtype={kv_dtype} poison={poison} tail_value={tail_value}")
                        print(f"{case} finite={finite} unchanged={unchanged}", flush=True)
                        if not finite or not unchanged:
                            failures.append(f"{case}: finite={finite}, unchanged={unchanged}")

            for target, cache in (("key", key), ("value", value)):
                key.copy_(clean_key)
                value.copy_(clean_value)
                live_row = cache.view(batch, capacity, *cache.shape[-2:])[0, 256]
                if kv_dtype == torch.float8_e4m3fn:
                    live_row.view(torch.uint8).fill_(0x7F)
                else:
                    live_row.fill_(float("nan"))
                live_nan_visible = not bool(torch.isfinite(replay()[counts[0] - 1]).all())
                print(f"mode={mode} live_{target}_nan_visible={live_nan_visible}", flush=True)
                if not live_nan_visible:
                    failures.append(f"attention sanitized a committed NaN {target}")
        finally:
            graph.reset()

    assert not failures, "\n".join(failures)
