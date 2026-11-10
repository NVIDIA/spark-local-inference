# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation can omit ordinary graph modes while preserving speculative execution."""

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import pytest
from google.protobuf.json_format import ParseDict

from ignition.cli import _parser, display, preparation
from ignition.components.capacity import batch_capacities, prefill_capacities, serving_modes
from ignition.errors import UsageError
from ignition.preparation import rebuild_manifest, runtime_package_pb2 as proto
from ignition.recipes import ResolvedModel, recipes, resolve
from ignition.recipes.declaration import Speculation
from tests.cli.test_planning import derive, gpu


def selected(*flags, command="export"):
    model = resolve("kimi-k3.1-pro")
    args = _parser().parse_args([command, model.checkpoint.alias, *flags])
    return preparation.speculation(args, model)


@pytest.mark.parametrize("command", ["export", "debug-serve", "bench"])
def test_mtp_only_is_explicit_and_preserves_speculative_capacities(command):
    default = selected(command=command)
    assert not default.mtp_only
    assert selected("--mtp-only", command=command) == replace(default, mtp_only=True)
    assert selected("--mtp-only", command=command).widths[0] == 1


@pytest.mark.parametrize("flags,message", [
    (("--no-mtp",), "cannot accompany"),
    (("--mtp-draft-tokens", "0"), "positive learned draft depth"),
])
def test_mtp_only_rejects_disabled_speculation(flags, message):
    with pytest.raises(UsageError, match=message):
        selected("--mtp-only", *flags)
    with pytest.raises(ValueError, match="positive learned draft depth"):
        Speculation(0, mtp_only=True)


def test_mtp_only_rejects_a_recipe_without_speculation():
    model = resolve("kimi-k3.1-pro")
    recipe = replace(model.recipe, speculation=None, max_draft_tokens=0, plain_ranks={})
    args = _parser().parse_args(["export", model.checkpoint.alias, "--mtp-only"])
    with pytest.raises(UsageError, match="has no multi-token speculation; --mtp-only"):
        preparation.speculation(args, ResolvedModel(recipe, model.checkpoint))


def test_mtp_only_plan_display_and_rebuild_round_trip(monkeypatch):
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    for mtp_only in (False, True):
        plan = derive(speculation=Speculation(3, mtp_only=mtp_only))
        assert plan.to_dict()["speculation"]["mtp_only"] is mtp_only
        for brief in (False, True):
            assert ("MTP-only graphs" in display.speculation(plan.speculation, brief=brief)) is mtp_only
        metadata = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
        encoded = ParseDict(metadata, proto.RebuildManifest())
        decoded = proto.RebuildManifest.FromString(encoded.SerializeToString())
        assert decoded.speculation.mtp_only is mtp_only
    assert not proto.RebuildSpeculation.FromString(b"\x08\x07").mtp_only


@pytest.mark.parametrize("enabled,mtp_only,expected", [
    (False, False, (False,)), (True, False, (True, False)), (True, True, (True,)),
])
def test_serving_modes_select_before_graph_construction(enabled, mtp_only, expected):
    assert serving_modes(enabled, mtp_only) == expected
    with pytest.raises(ValueError, match="enabled learned speculation"):
        serving_modes(False, True)


@pytest.mark.parametrize("recipe", recipes(), ids=lambda recipe: recipe.name)
@pytest.mark.parametrize("mtp_only", [False, True])
def test_recipe_adapters_preserve_mtp_only_and_validate_direct_configs(monkeypatch, tmp_path, recipe, mtp_only):
    import huggingface_hub

    name = "kimi3" if recipe.name == "kimi31" else recipe.name
    bootstrap = import_module(f"ignition.recipes.{name}.bootstrap")
    world = recipe.tensor_parallel[0]
    devices = tuple(replace(gpu(i, 4096), total_bytes=4096 << 30) for i in range(world))
    plan = derive(model=recipe, gpus=devices, selected=tuple(range(world)),
                  speculation=replace(recipe.speculation, mtp_only=mtp_only))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: str(tmp_path / "draft"))
    original = bootstrap.PreparationConfig
    captured = []

    class ConfigurationCaptured(Exception):
        pass

    def intercept(**kwargs):
        captured.append(original(**kwargs))
        raise ConfigurationCaptured

    monkeypatch.setattr(bootstrap, "PreparationConfig", intercept)
    with pytest.raises(ConfigurationCaptured):
        recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    config, = captured
    assert config.mtp_only is mtp_only
    assert config.speculative_widths == plan.speculation.widths
    assert original.__dataclass_fields__["mtp_only"].default is False
    if not mtp_only:
        return
    source = (dict(draft_checkpoint=None) if name in ("kimi3", "gemma4", "nemotron_h")
              else dict(dspark=False) if name in ("deepseek4", "deepseek41", "minimax_m3")
              else dict(dflash=False) if name in ("mimo_v2", "muse_spark") else dict(mtp=False))
    for invalid in (source, dict(speculative_widths=(1,), speculative_fixed_width=None),
                    dict(speculative_fixed_width=1)):
        with pytest.raises(ValueError, match="--mtp-only requires enabled learned speculation with positive draft depth"):
            replace(config, **invalid)


@pytest.mark.parametrize("concurrency,normal,only", [(1, 42, 27), (4, 104, 79)])
def test_kimi_catalog_geometry_keeps_width_one_and_vision(concurrency, normal, only):
    from ignition.components.prompt_logprobs import prefill_modes
    from ignition.recipes.kimi3.bootstrap import PreparationConfig
    from ignition.recipes.kimi3.vision import PATCH_CAPACITIES

    catalogs = []
    for mtp_only, expected in ((False, normal), (True, only)):
        config = PreparationConfig(draft_checkpoint="fixture", max_active=concurrency,
            prefill_chunk_size=4096, speculative_widths=tuple(range(1, 9)), speculative_token_capacity=32,
            speculative_graph_budget=64, mtp_only=mtp_only)
        batches = batch_capacities(config.max_active)
        catalog = []
        for enabled in serving_modes(config.drafting, config.mtp_only):
            for kind, _, capacities in prefill_modes(prefill_capacities(config.prefill_chunk_size),
                                                   () if enabled else batches, None):
                catalog.extend((kind, capacity, enabled) for capacity in capacities)
            catalog.extend((kind, batch, enabled) for kind in ("init", "restore", "snapshot", "resume")
                           for batch in batches)
        catalog.extend(("speculative", shape, True) for shape in config.speculative_round_shapes)
        catalog.extend(("vision", capacity, None) for capacity in PATCH_CAPACITIES)
        assert len(catalog) == expected
        assert {batch for batch, width, _ in config.speculative_round_shapes if width == 1} == set(batches)
        assert len([graph for graph in catalog if graph[0] == "vision"]) == len(PATCH_CAPACITIES)
        catalogs.append(catalog)
    assert catalogs[1] == [graph for graph in catalogs[0] if graph[2] is not False]


def test_qwen_mixed_shapes_agree_with_the_plan_for_mtp_only():
    from ignition.recipes.qwen38.bootstrap import PreparationConfig

    recipe = resolve("qwen3.8-flash").recipe
    for mtp_only in (False, True):
        settings = replace(recipe.speculation, mtp_only=mtp_only)
        config = PreparationConfig(mtp=True, mtp_only=mtp_only, mixed_prefill=True,
            speculative_widths=settings.widths, speculative_token_capacity=settings.token_capacity)
        assert config.mixed_shapes == recipe.mixed_shapes(config.max_active, config.prefill_chunk_size, settings)
        assert {shape[0] for shape in config.mixed_shapes} == ({True} if mtp_only else {True, False})


@pytest.mark.parametrize("name", ["kimi3", "qwen38_dense"])
def test_mtp_only_preserves_selected_benchmark_geometry(name):
    from ignition.preparation.selection import BenchmarkSelection

    selection = BenchmarkSelection(batches=(1, 4))
    config_type = import_module(f"ignition.recipes.{name}.bootstrap").PreparationConfig
    options = dict(draft_checkpoint="fixture") if name == "kimi3" else dict(mtp=True, draft_snapshot="fixture")
    config = config_type(benchmark_selection=selection, mtp_only=True, speculative_token_capacity=128, **options)
    assert config.speculative_round_shapes == selection.speculative_round_shapes
