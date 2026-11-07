# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer preparation bounds and the width-16 speculative catalog."""

import pytest
import torch

from ignition.recipes.muse_glimmer_nvfp4.bootstrap import PreparationConfig
from ignition.recipes.muse_glimmer_nvfp4.recipe import RECIPE


def test_defaults_are_the_single_gpu_text_only_declaration():
    config = PreparationConfig()
    assert (config.device, config.context, config.max_active, config.prefill_chunk_size) == (0, 1048576, 4, 4096)
    assert config.language_model_only and config.image_concurrency is None
    assert not config.dflash and not config.speculative_round_shapes
    assert config.speculative_widths == RECIPE.speculation.widths == tuple(range(1, 17))
    assert config.decode_tokens == config.decode_samples == 4 and config.completion_token_capacity == 1
    assert PreparationConfig(prefill_chunk_size=64).prefill_chunk_size == 64
    assert PreparationConfig(context=128, prefill_chunk_size=128).context == 128


def test_image_concurrency_is_bounded_by_concurrency_when_vision_is_prepared():
    assert PreparationConfig(language_model_only=False).image_concurrency == 4
    assert PreparationConfig(language_model_only=False, max_active=2).image_concurrency == 2
    assert PreparationConfig(language_model_only=False, image_concurrency=1).image_concurrency == 1
    assert PreparationConfig(language_model_only=True, image_concurrency=2).image_concurrency is None
    with pytest.raises(ValueError):
        PreparationConfig(language_model_only=False, image_concurrency=8)


@pytest.mark.parametrize("fault", [
    dict(device=-1), dict(context=1048576 + 128), dict(context=131000), dict(context=64),
    dict(prefill_chunk_size=96), dict(prefill_chunk_size=8192, context=4096),
    dict(snapshot_budget_bytes=0), dict(headroom_bytes=-1), dict(max_active=0),
    dict(dflash=True), dict(speculative_widths=()), dict(speculative_widths=tuple(range(1, 18))),
    dict(speculative_fixed_width=16, speculative_widths=(1, 2, 3)),
])
def test_rejects_out_of_bounds_preparation(fault, tmp_path):
    with pytest.raises(ValueError):
        PreparationConfig(**({"draft_snapshot": tmp_path} if "speculative" in str(fault) else {}), **fault)


def test_dflash_catalog_covers_width_one_and_the_full_sixteen_token_block(tmp_path):
    config = PreparationConfig(dflash=True, draft_snapshot=tmp_path, speculative_token_capacity=64)
    shapes = config.speculative_round_shapes
    for batch in (1, 2, 4):
        assert (batch, 1, batch) in shapes
        assert (batch, 16, batch * 16) in shapes
    assert all(width in (1, 16) and packed <= 64 for _, width, packed in shapes)
    assert (4, 16, 64) in shapes and (4, 16, 5) in shapes
    assert config.decode_tokens == config.decode_samples == 64
    assert config.completion_token_capacity == 16
    # The budget bounds the packed width-16 rounds; width-one rounds per batch ride outside it.
    assert len([shape for shape in shapes if shape[1] == 16]) <= config.speculative_graph_budget
    fixed = PreparationConfig(dflash=True, draft_snapshot=tmp_path, speculative_token_capacity=64,
                              speculative_fixed_width=8)
    assert all(width in (1, 8) for _, width, _ in fixed.speculative_round_shapes)
    with pytest.raises(ValueError):
        PreparationConfig(dflash=True, draft_snapshot=tmp_path, speculative_token_capacity=8)


def test_recipe_prepare_runs_in_process_on_one_gpu(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    captured = {}
    monkeypatch.setitem(sys.modules, "ignition.recipes.muse_glimmer_nvfp4.bootstrap", SimpleNamespace(
        PreparationConfig=lambda **options: SimpleNamespace(**options),
        prepare=lambda snapshot, config, **options: captured.update(config=config, **options) or config))
    plan = SimpleNamespace(world_size=1, speculation=None, prompt_logprobs=False, compile_workers=None,
        collective_tuning="static", collective_ladder=None,
        autotune_effort="off", tuning_replica=None, native_dense_activations=False, lock_gpu_clocks=True,
        context=131072, concurrency=4, prefill_chunk=4096, snapshots=2, costs=SimpleNamespace(state_row_bytes=1 << 20),
        headroom_bytes=1 << 30, reserve_bytes=1 << 30, language_model_only=False, image_concurrency=2, hub_cache=tmp_path)
    config = RECIPE.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert config.device == 0 and config.dflash is False and config.language_model_only is False
    assert config.collective_tuning == "static" and config.collective_ladder is None
    assert config.image_concurrency == 2 and config.snapshot_budget_bytes == 2 << 20 and config.headroom_bytes == 2 << 30
    assert not config.lock_gpu_clocks and captured["progress"] is None
    with pytest.raises(ValueError, match="exactly one GPU"):
        RECIPE.prepare(tmp_path, SimpleNamespace(**{**plan.__dict__, "world_size": 2}), compile_forwards=True,
                       kernel_cache=None, progress=None)
    with pytest.raises(ValueError, match="exactly one GPU"):
        RECIPE.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None, ranks=object())


def _stub_decoder(vision_state):
    """A ``PreparedDecoder`` whose text parts are identities; only the vision hook is under test."""
    import inspect
    from types import SimpleNamespace

    from ignition.recipes.muse_glimmer_nvfp4.decoder import PreparedDecoder

    assert inspect.signature(PreparedDecoder.__init__).parameters["vision_state"].default is None
    decoder = object.__new__(PreparedDecoder)
    decoder.embedding = lambda: torch.full((4, 8), 3.)
    decoder.embedding_norm = lambda hidden: hidden * .5
    decoder.layers, decoder.target_layers, decoder.scoring = [], (), None
    decoder.metadata = SimpleNamespace(sample_indices=torch.tensor([0, 3]))
    decoder.norm = lambda hidden: hidden
    decoder.scratch = SimpleNamespace(hidden=torch.zeros(2, 8))
    decoder.head = lambda hidden: hidden.clone()
    decoder.output_multiplier, decoder.softcap = 1., 30.
    decoder.vision_state = vision_state
    return decoder


def test_prefill_decoder_injects_image_features_after_the_embedding_norm():
    class Vision:
        calls = []

        def inject(self, metadata, embedding):
            self.calls.append((metadata, embedding.clone()))
            embedding[0].fill_(7.)

    vision = Vision()
    decoder = _stub_decoder(vision)
    decoder()
    (metadata, normed), = vision.calls
    assert metadata is decoder.metadata and torch.equal(normed, torch.full((4, 8), 1.5))
    assert torch.equal(decoder.scratch.hidden[0], torch.full((8,), 7.))


def test_text_only_decoder_skips_the_vision_hook():
    decoder = _stub_decoder(None)
    decoder()
    assert torch.equal(decoder.scratch.hidden, torch.full((2, 8), 1.5))
