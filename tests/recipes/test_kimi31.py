# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 checkpoint configuration, recipe selection, and native MXFP4 execution."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

CHECKPOINT = Path(os.environ.get("FLARE_KIMI31_CHECKPOINT", str(Path.home() /
    ".cache/huggingface/hub/models--moonshotai--Kimi-K3.1-Pro/snapshots/"
    "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e")))


@pytest.fixture(scope="module")
def raw_config():
    if not (CHECKPOINT / "config.json").is_file():
        pytest.skip("Kimi K3.1 checkpoint configuration is not installed")
    return json.loads((CHECKPOINT / "config.json").read_text())


@pytest.fixture(scope="module")
def config(raw_config):
    from ignition.recipes.kimi3.config import ModelConfig

    return ModelConfig.from_dict(raw_config)


def test_checkpoint_geometry(config):
    from ignition.components.parallel import TensorParallelShard

    config.validate_shard(TensorParallelShard(0, 16))
    assert len(config.layers) == 93
    assert config.layers.count("kda") == 69
    assert config.layers.count("dsa") == 24
    assert config.layers.count("mla") == 0
    assert config.layers[3] == config.layers[92] == "dsa"
    assert config.draft is None
    assert (config.dsa.heads, config.dsa.head_dim, config.dsa.top_k) == (32, 128, 2048)


def test_mla_norm_epsilon_uses_the_k31_checkpoint_value(config):
    from dataclasses import replace

    assert config.text["rms_norm_eps"] == 1e-5
    assert config.mla_norm_eps == 1e-5
    changed = replace(config, text={**config.text, "rms_norm_eps": 1e-4})
    assert changed.mla_norm_eps == 1e-4


@pytest.mark.parametrize("field,value", [
    ("index_num_heads", 16), ("index_head_dim", 64), ("index_topk", 1024),
    ("index_rope_dim", 64), ("index_k_norm_type", "rmsnorm"),
    ("index_topk_freq", 2), ("index_skip_topk_offset", 0),
    ("index_topk", True), ("index_head_dim", 128.0), ("unknown_option", True),
])
def test_unsupported_dsa_configuration_rejected(raw_config, field, value):
    from ignition.recipes.kimi3.config import ModelConfig

    raw = deepcopy(raw_config)
    raw["text_config"]["dsa_config"][field] = value
    with pytest.raises(ValueError, match="DSA|dsa"):
        ModelConfig.from_dict(raw)


@pytest.mark.parametrize("layers", [[], [4], [4, 4], [0, 4], [94],
    list(reversed([*range(4, 93, 4), 93])), [*range(4, 93, 4), 92]])
def test_dsa_layers_must_match_the_attention_partition(raw_config, layers):
    from ignition.recipes.kimi3.config import ModelConfig

    raw = deepcopy(raw_config)
    raw["text_config"]["dsa_config"]["dsa_layers"] = layers
    with pytest.raises(ValueError, match="DSA|dsa"):
        ModelConfig.from_dict(raw)


@pytest.mark.parametrize("mutation", ["missing_dsa", "k3_with_dsa", "architecture_mismatch"])
def test_dsa_identity_cannot_select_dense_attention(raw_config, mutation):
    from ignition.recipes.kimi3.config import ModelConfig

    raw = deepcopy(raw_config)
    if mutation == "missing_dsa":
        del raw["text_config"]["dsa_config"]
    elif mutation == "k3_with_dsa":
        raw.update(model_type="kimi_k3", architectures=["KimiK3ForConditionalGeneration"])
    else:
        raw["architectures"] = ["KimiK3ForConditionalGeneration"]
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw)


def test_k31_does_not_attach_an_unrelated_k3_draft(raw_config, tmp_path):
    from ignition.recipes.kimi3.config import ModelConfig

    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    (tmp_path / "dspark").mkdir()
    (tmp_path / "dspark/config.json").write_text('{"architectures": ["DSparkDraftModel"]}')
    config = ModelConfig.load(tmp_path)
    assert config.draft is None
    assert config.layers.count("dsa") == 24


def test_recipe_registration_and_sampling_defaults(raw_config, tmp_path):
    from ignition.recipes import resolve

    model = resolve("kimi-k3.1-pro")
    assert model.recipe.name == "kimi31"
    assert model.checkpoint.repository == "moonshotai/Kimi-K3.1-Pro"
    assert model.checkpoint.architecture == "KimiK31ForConditionalGeneration"
    assert model.checkpoint.revision == "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e"
    assert resolve(model.checkpoint.repository).recipe is model.recipe
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    assert resolve(str(tmp_path)).recipe is model.recipe
    assert model.recipe.sampling_defaults.as_dict() == {
        "temperature": 1., "top_p": .95, "top_k": -1, "seed": 0, "min_p": 0.,
        "repetition_penalty": 1., "presence_penalty": 0., "frequency_penalty": 0.}
    assert model.recipe.speculation.draft_tokens == model.recipe.max_draft_tokens == 7
    assert model.checkpoint.draft_bytes == 8_558_149_632


@pytest.mark.parametrize("command", ["export", "debug-serve"])
def test_cli_draft_defaults_and_explicit_disabling(command):
    from ignition.cli import _parser, preparation
    from ignition.recipes import resolve

    model = resolve("kimi-k3.1-pro")
    args = _parser().parse_args([command, model.checkpoint.alias])
    assert preparation.speculation(args, model).draft_tokens == 7
    args = _parser().parse_args([command, model.checkpoint.alias, "--no-mtp"])
    assert preparation.speculation(args, model) is None
    args = _parser().parse_args([command, model.checkpoint.alias, "--mtp-draft-tokens", "3",
        "--speculative-token-capacity", "32", "--speculative-graph-budget", "16"])
    chosen = preparation.speculation(args, model)
    assert (chosen.draft_tokens, chosen.token_capacity, chosen.graph_budget) == (3, 32, 16)


@pytest.mark.parametrize("embedded_draft", [False, True])
def test_preparation_adapter_preserves_k31_defaults_without_fetching_a_draft(monkeypatch, tmp_path, embedded_draft):
    from ignition.recipes import resolve
    from .test_declarations import fitted

    model = resolve("kimi-k3.1-pro")
    plan = fitted(model, 16, concurrency=4, speculation=None)
    if embedded_draft:
        (tmp_path / "dspark").mkdir()

    def unexpected_download(*args, **kwargs):
        pytest.fail("Kimi K3.1 preparation fetched a Kimi K3 draft")

    monkeypatch.setattr("huggingface_hub.snapshot_download", unexpected_download)
    monkeypatch.setattr("ignition.preparation.rank_group.prepare_group",
                        lambda snapshot, config, prepare, **kwargs: config)
    config = model.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert config.devices == tuple(range(16)) and config.page_tokens == plan.page_tokens == 2048
    assert config.sampling_defaults == model.recipe.sampling_defaults.as_dict()
    assert Path(config.frontend_template).parts[-3:] == ("kimi31", "assets", "chat_template.jinja")
    assert config.draft_checkpoint is None and not config.drafting
    assert config.snapshot_budget_bytes == plan.snapshots * 30_275_328


@pytest.mark.parametrize("embedded_draft", [False, True])
def test_preparation_selects_the_inferact_draft(monkeypatch, tmp_path, embedded_draft):
    from ignition.recipes import resolve
    from ignition.recipes.kimi31.recipe import DSPARK_REPOSITORY, DSPARK_REVISION
    from .test_declarations import fitted

    model = resolve("kimi-k3.1-pro")
    plan = fitted(model, 16, concurrency=4)
    downloaded = []
    if embedded_draft:
        (tmp_path / "dspark").mkdir()

    def download(repository, *, revision, cache_dir, allow_patterns):
        downloaded.append((repository, revision))
        assert allow_patterns == ("config.json", "model.safetensors")
        return str(tmp_path / "external")

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr("ignition.preparation.rank_group.prepare_group",
                        lambda snapshot, config, prepare, **kwargs: config)
    config = model.recipe.prepare(tmp_path, plan, compile_forwards=False, kernel_cache=None, progress=None)
    assert downloaded == ([] if embedded_draft else [(DSPARK_REPOSITORY, DSPARK_REVISION)])
    assert config.draft_checkpoint == str(tmp_path / ("dspark" if embedded_draft else "external"))
    assert config.speculative_widths == tuple(range(1, 9))
    assert config.snapshot_budget_bytes == plan.snapshots * 33_238_272


def test_preparation_rejects_unsharded_context_before_dispatch(tmp_path):
    from types import SimpleNamespace
    from ignition.recipes import resolve

    with pytest.raises(ValueError, match="requires context parallelism"):
        resolve("kimi-k3.1-pro").recipe.prepare(tmp_path, SimpleNamespace(context_parallel=False),
            compile_forwards=True, kernel_cache=None, progress=None)


@pytest.mark.parametrize("options,reason", [
    ({"context_parallel": False}, "requires context parallelism"),
    ({}, "requires recipe sampling defaults and a frontend template"),
])
def test_rank_preparation_rejects_invalid_k31_inputs_before_cuda(raw_config, tmp_path, options, reason):
    from ignition.recipes.kimi3.bootstrap import PreparationConfig, prepare_rank

    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    with pytest.raises(ValueError, match=reason):
        prepare_rank(tmp_path, PreparationConfig(**options), rank=0, group=None, package_storage=None)


def test_native_quantization_covers_all_routed_experts(config):
    assert config.text["quantization_config"].get("weight_scale_encoding") is None
    for layer in range(config.text["first_k_dense_replace"], len(config.layers)):
        prefix = f"language_model.model.layers.{layer}.block_sparse_moe"
        for expert in range(config.text["num_experts"]):
            for projection in ("w1", "w2", "w3"):
                assert config.projection_format(f"{prefix}.experts.{expert}.{projection}") == "mxfp4"
        assert config.projection_format(prefix + ".shared_experts.gate_proj") == "bf16"
        assert config.projection_format(f"language_model.model.layers.{layer}.self_attn.indexer.wk") == "bf16"


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires CUDA checkpoint tests")
def test_checkpoint_indexer_weights_are_replicated_without_conversion(config):
    import torch
    from safetensors import safe_open
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.kimi3.weights import KimiWeightLoader

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    manifest = inspect_indexed_checkpoint(CHECKPOINT)
    prefix = "language_model.model.layers.3.self_attn."
    names = {"indexer.wq_b": (4096, 1536), "indexer.wk": (128, 7168),
             "indexer.weights_proj": (32, 7168), "indexer.k_norm.weight": (128,),
             "indexer.k_norm.bias": (128,)}
    expected = {}
    for key in names:
        name = prefix + key + ("" if key.endswith((".weight", ".bias")) else ".weight")
        with safe_open(CHECKPOINT / manifest.tensors[name].shard, framework="pt") as file:
            expected[key] = file.get_tensor(name)
        assert expected[key].shape == names[key] and expected[key].dtype == torch.bfloat16
    for rank in (0, 15):
        with KimiWeightLoader(manifest, device, shard=TensorParallelShard(rank, 16),
                prefixes=(prefix,), online_mxfp8=True) as loader:
            actual = loader.load_attention(3)
        for key, tensor in expected.items():
            assert actual[key].dtype == torch.bfloat16
            torch.testing.assert_close(actual[key].cpu(), tensor, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires CUDA checkpoint tests")
@pytest.mark.parametrize("partition", [(0, 192), (2880, 3072)])
def test_native_expert_shards_preserve_checkpoint_arithmetic(partition):
    import torch
    from b12x.moe._shared.kernels.reference import moe_reference_w4a16_fp4_e8m0_k32
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from .test_kimi3_experts_cuda import PREFIXES, _load

    def load(loader, scales):
        experts = loader.load_fp4_experts(PREFIXES, hidden_size=3584, intermediate_size=3072, format="mxfp4",
            activation_mode="a16", intermediate_range=partition, projection_names=("w1", "w3", "w2"),
            weight_name="weight_packed", activation="situ", w13_layout="w31", scale_compression="csf",
            scale_scratch=scales)
        assert experts.plan.scale_compression == "csf"
        return experts

    experts, scratch, packed, scales = _load(CHECKPOINT, partition, load,
        lambda read, name, rows, columns: read(name + ".weight_scale"))
    device = torch.device("cuda:0")
    w13, w2 = (torch.stack(x).to(device) for x in zip(*packed))
    s13, s2 = (torch.stack(x).to(device) for x in zip(*scales))
    generator = torch.Generator(device=device).manual_seed(41)
    source = torch.randn((3, 3584), device=device, dtype=torch.bfloat16, generator=generator)
    ids = torch.tensor([[0, 1], [1, 2], [2, 3]], device=device, dtype=torch.int32)
    probabilities = torch.tensor([[.3, .7], [.6, .4], [.2, .8]], device=device)
    output = torch.empty_like(source)
    with RecordedPreparationSession(device=device, checkpoint=str(CHECKPOINT),
            workload_schema="kimi31-native-experts", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        op = PreparedRoutedExperts(source, experts, ids, probabilities, output,
                                   session=session, name="experts", workspace=arena)
        session.tune(show_progress=False)
        expected = moe_reference_w4a16_fp4_e8m0_k32(source, w13, s13, torch.ones(4, device=device),
            w2, s2, torch.ones(4, device=device), ids, probabilities, 4, 3584, 192,
            activation="situ", w13_layout="w13")
        actual = op().clone()
        relative_rms = (actual.float() - expected).square().mean().sqrt() / expected.square().mean().sqrt()
        assert relative_rms.item() < .02
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op()
        for scale in scratch:
            scale.fill_(255)
        allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
        torch.testing.assert_close(output, actual, rtol=0, atol=0)
