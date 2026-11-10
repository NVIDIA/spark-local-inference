# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed graph-count budgets and their independent preparation settings."""

from dataclasses import replace

import pytest

from ignition.cli import _parser, preparation
from ignition.cli.config import Configuration
from ignition.errors import UsageError
from ignition.recipes import resolve
from ignition.recipes.declaration import Speculation


def selected(*flags, alias="deepseek-v4.1-flash", configuration=None):
    args = _parser().parse_args(["export", alias, *flags])
    return preparation.speculation(args, resolve(alias), configuration)


@pytest.mark.parametrize("command", ["export", "debug-serve"])
def test_graph_budget_flag_parses_independently_of_token_capacity(command):
    args = _parser().parse_args([command, "deepseek-v4.1-flash", "--speculative-graph-budget", "48",
                                 "--speculative-token-capacity", "96"])
    assert (args.speculative_graph_budget, args.speculative_token_capacity) == (48, 96)
    assert _parser().parse_args([command]).speculative_graph_budget is None


def test_graph_budget_precedence_and_token_capacity_are_independent(tmp_path):
    path = tmp_path / "flare.toml"
    default = resolve("deepseek-v4.1-flash").recipe.speculation
    assert default.graph_budget == 64
    assert selected() == default
    path.write_text('[defaults]\nspeculative_graph_budget=48\nspeculative_token_capacity=96\n')
    config = Configuration.load(path)
    assert selected(configuration=config) == replace(default, graph_budget=48, token_capacity=96)
    path.write_text(path.read_text() + '[models."deepseek-v4.1-flash"]\nspeculative_graph_budget=32\n')
    config = Configuration.load(path)
    assert selected(configuration=config) == replace(default, graph_budget=32, token_capacity=96)
    assert selected("--speculative-graph-budget", "24", configuration=config) == replace(
        default, graph_budget=24, token_capacity=96)
    assert selected("--speculative-token-capacity", "192", configuration=config) == replace(
        default, graph_budget=32, token_capacity=192)


@pytest.mark.parametrize("alias", ["qwen3.8-flash", "glm-5.3-flash", "step5-preview-nvfp4"])
def test_explicit_graph_budget_rejects_recipes_without_packed_rounds(alias):
    with pytest.raises(UsageError, match="--speculative-graph-budget"):
        selected("--speculative-graph-budget", "32", alias=alias)


@pytest.mark.parametrize("flags,message", [
    (("--no-mtp",), "cannot accompany"),
    (("--mtp-draft-tokens", "0"), "positive draft depth"),
    (("--no-mtp-adaptive-verification",), "requires adaptive verification"),
])
def test_explicit_graph_budget_rejects_incompatible_modes(flags, message):
    with pytest.raises(UsageError, match=message):
        selected(*flags, "--speculative-graph-budget", "32")


@pytest.mark.parametrize("budget", ["0", "-1"])
def test_graph_budget_requires_a_positive_integer(budget):
    with pytest.raises(UsageError, match="positive integer"):
        selected("--speculative-graph-budget", budget)


def test_global_graph_budget_is_scoped_to_active_packed_speculation(tmp_path):
    path = tmp_path / "flare.toml"
    path.write_text('[defaults]\nspeculative_graph_budget=48\n')
    config = Configuration.load(path)
    assert selected(configuration=config).graph_budget == 48
    assert selected("--mtp-draft-tokens", "0", configuration=config).graph_budget is None
    assert selected("--no-mtp", configuration=config) is None
    assert selected(alias="qwen3.8-flash", configuration=config) == resolve("qwen3.8-flash").recipe.speculation


@pytest.mark.parametrize("alias,flags", [
    ("qwen3.8-flash", ()),
    ("deepseek-v4.1-flash", ("--no-mtp",)),
    ("deepseek-v4.1-flash", ("--mtp-draft-tokens", "0")),
])
def test_model_graph_budget_is_an_explicit_selection(tmp_path, alias, flags):
    path = tmp_path / "flare.toml"
    path.write_text(f'[models."{alias}"]\nspeculative_graph_budget=32\n')
    with pytest.raises(UsageError, match="--speculative-graph-budget"):
        selected(*flags, alias=alias, configuration=Configuration.load(path))


def test_plan_serializes_graph_budget_only_for_applicable_speculation():
    from tests.cli.test_planning import derive

    packed = selected("--speculative-graph-budget", "48")
    plan = derive(speculation=packed)
    assert plan.to_dict()["speculation"]["graph_budget"] == 48
    assert plan.speculative_token_capacity == 128
    assert "graph_budget" not in derive().to_dict()["speculation"]
    assert derive(speculation=None).to_dict()["speculation"] is None


def test_planning_resolves_missing_graph_budget_from_recipe_without_overriding_explicit_values():
    from tests.cli.test_planning import SOLO, derive

    default = selected()
    recipe = replace(SOLO, speculation=default)
    assert derive(model=recipe, speculation=replace(default, graph_budget=None)).speculation.graph_budget == 64
    assert derive(model=recipe, speculation=replace(default, graph_budget=32)).speculation.graph_budget == 32
    assert derive(model=recipe, speculation=Speculation(0)).speculation.graph_budget is None


def test_planning_rejects_graph_budget_below_required_endpoints():
    from tests.cli.test_planning import derive, gpu

    settings = selected("--speculative-graph-budget", "11")
    devices = (gpu(0, 95), gpu(1, 95))
    with pytest.raises(UsageError, match="--speculative-graph-budget: .*at least 12"):
        derive(gpus=devices, concurrency=32, speculation=settings)
    assert derive(gpus=devices, concurrency=32,
                  speculation=replace(settings, graph_budget=12)).speculation.graph_budget == 12


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_speculation_declaration_validates_graph_budget_type_and_range(budget):
    with pytest.raises(ValueError, match="positive integer"):
        Speculation(5, packed_verification=True, graph_budget=budget)


def test_graph_budget_preserves_cpu_fixed_width_mode():
    declaration = Speculation(5, adaptive=False, packed_verification=True, graph_budget=64)
    assert declaration.fixed_width == 6 and declaration.draft_prefix_selection == "cpu"
    with pytest.raises(ValueError, match="positive learned draft depth"):
        Speculation(0, packed_verification=True, graph_budget=64)
    with pytest.raises(ValueError, match="GPU draft prefix selection"):
        Speculation(5, adaptive=False, packed_verification=True, draft_prefix_selection="gpu", graph_budget=64)


@pytest.mark.parametrize("alias", ["deepseek-v4-flash", "mimo-v2.6-pro-mopd", "kimi-k3-mxfp4-csf", "qwen3.8-27b", "qwen3.8-27b-qad"])
def test_cpu_packed_recipe_accepts_fixed_width_and_explicit_graph_budget(alias):
    value = selected("--no-mtp-adaptive-verification", "--speculative-graph-budget", "32", alias=alias)
    assert value.packed_verification and value.draft_prefix_selection == "cpu"
    assert not value.adaptive and value.graph_budget == 32


@pytest.mark.parametrize("budget", [12, 32, 48])
def test_rebuild_metadata_preserves_the_selected_physical_graph_catalog(monkeypatch, budget):
    from types import SimpleNamespace
    from google.protobuf.json_format import ParseDict
    from ignition.components.capacity import packed_speculative_shapes
    from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto
    from tests.cli.test_planning import derive, gpu

    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    plan = derive(gpus=(gpu(0, 95), gpu(1, 95)), concurrency=32,
                  speculation=selected("--speculative-graph-budget", str(budget)))
    metadata = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
    decoded = proto.RebuildManifest.FromString(ParseDict(metadata, proto.RebuildManifest()).SerializeToString())
    assert decoded.speculation.HasField("graph_budget")
    assert decoded.speculation.graph_budget == budget
    arguments = (plan.concurrency, plan.speculation.draft_tokens + 1, plan.speculative_token_capacity)
    catalog = packed_speculative_shapes(*arguments, decoded.speculation.graph_budget)
    assert catalog == packed_speculative_shapes(*arguments, budget)
    assert catalog != packed_speculative_shapes(*arguments, 64)


def test_packed_rebuild_metadata_rejects_an_unresolved_graph_budget(monkeypatch):
    from types import SimpleNamespace
    from ignition.preparation.rebuild_manifest import from_plan
    from tests.cli.test_planning import derive

    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    plan = derive(speculation=selected())
    plan = replace(plan, speculation=replace(plan.speculation, graph_budget=None))
    with pytest.raises(ValueError, match="explicit graph budget"):
        from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)



def test_vocabulary_parallel_sampling_is_an_explicit_preparation_choice():
    from tests.cli.test_planning import derive, gpu

    settings = selected("--vocabulary-parallel-sampling", "--draft-head-quantization", "nvfp4",
                        "--markov-head-quantization", "fp8")
    assert settings.vocabulary_parallel_sampling
    assert not selected().vocabulary_parallel_sampling
    assert derive(model=resolve("deepseek-v4.1-flash").recipe, gpus=tuple(gpu(i, 140) for i in range(4)),
                  speculation=settings).to_dict()["speculation"]["vocabulary_parallel_sampling"]
    for flags in (("--no-mtp",), ("--mtp-draft-tokens", "0")):
        with pytest.raises((UsageError, ValueError), match="cannot accompany|positive learned"):
            selected(*flags, "--vocabulary-parallel-sampling")
    with pytest.raises(UsageError, match="DeepSeek V4.1"):
        derive(speculation=settings)


@pytest.mark.parametrize("alias", ["qwen3.8-27b", "qwen3.8-27b-qad"])
def test_qwen_dense_packed_budget_precedence_preserves_token_capacity(tmp_path, alias):
    default = resolve(alias).recipe.speculation
    assert default.packed_verification and default.graph_budget == 64
    path = tmp_path / "flare.toml"
    path.write_text('[defaults]\nspeculative_graph_budget=48\n')
    config = Configuration.load(path)
    assert selected(alias=alias, configuration=config) == replace(default, graph_budget=48)
    path.write_text(path.read_text() + f'[models."{alias}"]\nspeculative_graph_budget=32\n')
    config = Configuration.load(path)
    assert selected(alias=alias, configuration=config) == replace(default, graph_budget=32)
    assert selected("--speculative-graph-budget", "24", alias=alias, configuration=config) == replace(
        default, graph_budget=24)
    assert selected("--speculative-token-capacity", "64", alias=alias, configuration=config) == replace(
        default, graph_budget=32, token_capacity=64)
