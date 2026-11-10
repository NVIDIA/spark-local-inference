# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi graph benchmarks retain complete request and model continuation state."""

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ignition.preparation.selection import BenchmarkSelection
from ignition.recipes.kimi3.bootstrap import PreparationConfig, _benchmark_state


def test_selected_kimi_round_preserves_serving_configuration():
    ordinary = PreparationConfig(max_active=8, draft_checkpoint=Path("dspark"),
        speculative_token_capacity=64, online_mxfp8=True)
    selection = BenchmarkSelection(batches=(8,), contexts=(512, 2048, 8192), width=8)
    selected = replace(ordinary, benchmark_selection=selection)
    assert selected.speculative_round_shapes == ((8, 8, 64),)
    assert selected.decode_tokens == selected.decode_samples == 64
    assert selected.completion_token_capacity == 8 and selected.draft_tokens == 7
    assert selected.page_tokens == 2048 and selected.preparation_pages == 41
    assert ordinary.preparation_pages == 18
    assert selection.physical_pages(selected.page_tokens) == 41
    assert selected.max_active == ordinary.max_active == 8
    assert selected.speculative_token_capacity == ordinary.speculative_token_capacity == 64
    assert selected.vision and selected.online_mxfp8 and selected.compile_decoder
    assert (8, 1, 8) in ordinary.speculative_round_shapes
    assert (8, 8, 64) in ordinary.speculative_round_shapes
    assert len(ordinary.speculative_round_shapes) == 68


def test_packed_prefix_matrix_preserves_k7_owner_and_backing():
    selection = BenchmarkSelection(batches=(1,), contexts=(512, 8192),
                                   verification_prefixes=(1, 4, 6, 8))
    config = PreparationConfig(max_active=4, draft_checkpoint=Path("dflash2"),
                               speculative_token_capacity=32, benchmark_selection=selection)
    assert config.draft_tokens == 7
    assert config.completion_token_capacity == 8
    assert config.speculative_round_shapes == (
        (1, 1, 1), (1, 8, 4), (1, 8, 6), (1, 8, 8))
    assert selection.physical_pages(2048) == 6
    with pytest.raises(ValueError, match="width capacities"):
        replace(config, speculative_widths=(1,))


@pytest.mark.parametrize("changes,match", [
    (dict(draft_checkpoint=None), "complete unscored"),
    (dict(prompt_logprobs=True), "complete unscored"),
    (dict(max_active=4), "request or width"),
    (dict(speculative_widths=(1,)), "request or width"),
    (dict(context=2048), "fit the context"),
    (dict(speculative_token_capacity=32), "token capacity"),
])
def test_kimi_selection_rejects_unavailable_geometry(changes, match):
    options = dict(max_active=8, draft_checkpoint=Path("dspark"), speculative_token_capacity=64,
        benchmark_selection=BenchmarkSelection(batches=(8,), contexts=(2048,)))
    options.update(changes)
    with pytest.raises(ValueError, match=match):
        PreparationConfig(**options)


@pytest.mark.parametrize("alias,default", [("kimi-k3", "all"), ("kimi-k3.1-pro", "kda-input")])
@pytest.mark.parametrize("scope", [None, "all", "kda-input"])
def test_kimi_adapter_passes_benchmark_selection_and_precision(monkeypatch, tmp_path, alias, default, scope):
    from ignition.recipes import resolve
    from .test_declarations import fitted

    model = resolve(alias)
    selection = BenchmarkSelection(batches=(8,), contexts=(2048,), width=8) if model.recipe.name == "kimi31" else None
    plan = replace(fitted(model, 16, concurrency=8), benchmark_selection=selection,
        speculation=replace(model.recipe.speculation, token_capacity=64), online_mxfp8=True, online_mxfp8_scope=scope,
        language_model_only=False, autotune_effort="default")
    (tmp_path / "dspark").mkdir()
    monkeypatch.setattr("ignition.preparation.rank_group.prepare_group",
                        lambda snapshot, config, prepare, **kwargs: config)
    config = model.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert config.benchmark_selection is selection
    assert config.draft_checkpoint == str(tmp_path / "dspark")
    if selection is not None:
        assert config.speculative_round_shapes == ((8, 8, 64),)
    assert config.max_active == 8 and config.speculative_token_capacity == 64
    assert config.autotune_effort == "default" and config.compile_decoder
    assert config.vision and config.online_mxfp8 and config.context_parallel
    assert config.online_mxfp8_scope == (default if scope is None else scope)
    assert config.sampling_defaults == model.recipe.sampling_defaults.as_dict()
    assert Path(config.frontend_template).parts[-3:] == (model.recipe.name, "assets", "chat_template.jinja")


def test_kimi_benchmark_state_covers_backed_histories_requests_and_rng():
    import torch

    from ignition.components.paged_state import PagedStateStorage
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.speculation import SpeculativeRngState
    from ignition.components.state import StateLayout

    layout = StateLayout.from_specs((("recurrent", (2, 2), torch.float32),
                                    ("convolution", (2, 3), torch.bfloat16)))
    page_specs = (("layer3.latent", (128,)), ("layer3.indexer", (32,)), ("draft0.latent", (256,)))
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
