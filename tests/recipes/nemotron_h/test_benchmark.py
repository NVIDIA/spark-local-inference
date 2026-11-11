# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron graph benchmarks retain complete request and model continuation state."""

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ignition.preparation.selection import BenchmarkSelection
from ignition.recipes.nemotron_h.bootstrap import PreparationConfig, _benchmark_state


@pytest.mark.parametrize("mtp_only", [False, True])
def test_selected_nemotron_round_preserves_serving_configuration(mtp_only):
    ordinary = PreparationConfig(max_active=8, draft_checkpoint=Path("lilicorr"),
        speculative_token_capacity=64, mtp_only=mtp_only)
    selection = BenchmarkSelection(batches=(8,), contexts=(512, 2048, 8192), width=8)
    selected = replace(ordinary, benchmark_selection=selection)
    assert selected.speculative_round_shapes == ((8, 8, 64),)
    assert selected.decode_tokens == selected.decode_samples == 64
    assert selected.completion_token_capacity == 8
    assert selected.preparation_pages == 521
    assert ordinary.preparation_pages == 48
    assert selection.physical_pages(128) == 521
    assert selected.max_active == ordinary.max_active == 8
    assert selected.speculative_token_capacity == ordinary.speculative_token_capacity == 64
    assert selected.stream_overlap and selected.compile_decoder
    assert selected.mtp_only is mtp_only
    assert (8, 1, 8) in ordinary.speculative_round_shapes
    assert (8, 8, 64) in ordinary.speculative_round_shapes


@pytest.mark.parametrize("prefixes,widths,shapes", [
    ((1,), (1,), ((1, 1, 1),)),
    ((1, 4, 8), (1, 8), ((1, 1, 1), (1, 8, 4), (1, 8, 8))),
])
def test_nemotron_verification_prefixes_preserve_logical_width(prefixes, widths, shapes):
    selected = PreparationConfig(max_active=1, draft_checkpoint=Path("lilicorr"),
        speculative_widths=widths, speculative_token_capacity=8,
        benchmark_selection=BenchmarkSelection(batches=(1,), contexts=(4096,),
            verification_prefixes=prefixes))
    assert selected.speculative_round_shapes == shapes
    with pytest.raises(ValueError, match="request or width"):
        replace(selected, speculative_widths=(8,) if widths == (1,) else (1,))


@pytest.mark.parametrize("changes,match", [
    (dict(draft_checkpoint=None), "complete unscored"),
    (dict(prompt_logprobs=True), "complete unscored"),
    (dict(max_active=4), "request or width"),
    (dict(speculative_widths=(1,)), "request or width"),
    (dict(context=2048), "fit the context"),
    (dict(speculative_token_capacity=32), "token capacity"),
])
def test_nemotron_selection_rejects_unavailable_geometry(changes, match):
    options = dict(max_active=8, draft_checkpoint=Path("lilicorr"), speculative_token_capacity=64,
        benchmark_selection=BenchmarkSelection(batches=(8,), contexts=(2048,)))
    options.update(changes)
    with pytest.raises(ValueError, match=match):
        PreparationConfig(**options)


def test_nemotron_benchmark_state_covers_backed_histories_requests_and_rng():
    import torch

    from ignition.components.paged_state import PagedStateStorage
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.speculation import SpeculativeRngState
    from ignition.components.state import StateLayout

    layout = StateLayout.from_specs((("recurrent", (2, 2), torch.bfloat16),
                                    ("convolution", (2, 10), torch.bfloat16),
                                    ("replay_x", (2, 16, 2), torch.bfloat16),
                                    ("replay_b", (2, 16, 2), torch.bfloat16),
                                    ("replay_dt", (2, 16), torch.float32),
                                    ("mamba_ring_start", (1,), torch.int32),
                                    ("mamba_pending_count", (1,), torch.int32),
                                    ("draft_anchor", (4,), torch.bfloat16),
                                    ("draft0.key", (8, 2), torch.float8_e4m3fn),
                                    ("draft0.value", (8, 2), torch.float8_e4m3fn)))
    page_specs = (("layer3.key", (128,)), ("layer3.value", (128,)))
    pool = PagedStateStorage.allocate(layout, page_specs, 4, 6, "cpu")
    pool.memory_capacity.copy_(torch.tensor([3, 5]))
    requests = DeviceRequestState.allocate("cpu", vocab_size=64, capacity=8)
    speculation = SimpleNamespace(rng=SpeculativeRngState.allocate("cpu", capacity=8),
        generation=torch.arange(8), offset=torch.arange(8) * 28)
    buffers = dict(state_pool=pool.backing, submission_sequence=torch.tensor([17]),
                   completion_records=torch.zeros((2, 8, 24), dtype=torch.int64))
    state = _benchmark_state(buffers, pool, requests, speculation)
    assert state["state_pool"].shape == (3, layout.row_bytes)
    assert state["state_pool"].data_ptr() == pool.backing.data_ptr()
    for name, tensor in pool.pages.items():
        assert state[f"cache.{name}"].shape == (5, *tensor.shape[1:])
        assert state[f"cache.{name}"].data_ptr() == tensor.data_ptr()
    for field in fields(requests):
        assert state[f"request.{field.name}"] is getattr(requests, field.name)
    for field in fields(speculation.rng):
        assert state[f"speculation.rng.{field.name}"] is getattr(speculation.rng, field.name)
    assert state["speculation.generation"] is speculation.generation
    assert state["speculation.offset"] is speculation.offset
    assert state["submission_sequence"] is buffers["submission_sequence"]
    assert state["completion_records"] is buffers["completion_records"]
    assert not any(name.startswith("speculation.") for name in _benchmark_state(buffers, pool, requests, None))


@pytest.mark.parametrize("chunk", [4096, 8192])
@pytest.mark.parametrize("draft", [None, Path("lilicorr")])
def test_prefill_selection_backs_exact_chunk_and_prefix(chunk, draft):
    selection = BenchmarkSelection(batches=(1,), contexts=(0, 16384), prefill_chunk=chunk)
    selected = PreparationConfig(max_active=8, draft_checkpoint=draft,
        prefill_chunk_size=chunk, benchmark_selection=selection)
    assert selected.speculative_round_shapes == ()
    assert selected.preparation_pages == 1 + (16384 + chunk) // 128
    assert selection.prefill_capacities(chunk) == (chunk,)
    with pytest.raises(ValueError, match="chunk must match"):
        replace(selected, prefill_chunk_size=chunk // 2)
