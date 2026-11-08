# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Online weight conversion respects Kimi target and draft projection scopes."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.kimi3 import weights


@dataclass
class Converted:
    shape: tuple
    retain_scale_rows: bool = True


@dataclass
class MetadataConfig:
    text: dict
    layers: tuple = ("kda", "dsa", "mla")
    dsa: object = None
    draft: object = None
    hidden: int = 64
    vocabulary: int = 256

    def validate_shard(self, shard):
        pass

    def projection_format(self, name):
        return "mxfp4"


class MetadataLoader(weights.KimiWeightLoader):
    """Run the loader's composition with tensor metadata in place of checkpoint reads."""

    def __init__(self, manifest, device, **kwargs):
        super().__init__(manifest, device, **kwargs)
        self._sources = {"language_model.model.layers.0.self_attn.A_log":
                         SimpleNamespace(shape=(16,), dtype=torch.float32)}
        for name, shape in {"wq_b.weight": (4096, 16), "wk.weight": (128, 64),
                            "weights_proj.weight": (32, 64), "k_norm.weight": (128,),
                            "k_norm.bias": (128,)}.items():
            self._sources["language_model.model.layers.1.self_attn.indexer." + name] = (
                SimpleNamespace(shape=shape, dtype=torch.bfloat16))
        self.expert_modes = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def linear(self, prefix, *, rows=None, columns=None):
        name = prefix.rsplit(".", 1)[-1]
        output, source = {
            "embed_tokens": (256, 64),
            "q_proj": (2048, 64), "k_proj": (2048, 64), "v_proj": (2048, 64),
            "g_proj": (2048, 64), "b_proj": (16, 64), "f_a_proj": (32, 64),
            "f_b_proj": (2048, 32), "o_proj": (64, 2048), "q_b_proj": (3072, 16),
            "kv_b_proj": (4096, 512), "q_a_proj": (16, 64),
            "kv_a_proj_with_mqa": (576, 64), "wq_b": (4096, 16), "wk": (128, 64),
            "weights_proj": (32, 64), "gate_proj": (1024, 64), "up_proj": (1024, 64),
            "down_proj": (64, 1024), "gate": (256, 64),
            "routed_expert_down_proj": (128, 64), "routed_expert_up_proj": (64, 128),
        }[name]
        if rows is not None:
            output = rows[1] - rows[0]
        if columns is not None:
            source = columns[1] - columns[0]
        return torch.empty((output, source), device="meta", dtype=torch.bfloat16)

    def load_tensor(self, name, *, extent=None):
        if "conv1d.weight" in name:
            shape, dtype = (2048, 1, 4), torch.bfloat16
        elif name.endswith("A_log"):
            shape, dtype = (16,), torch.float32
        elif name == "language_model.lm_head.weight":
            shape, dtype = (256, 64), torch.bfloat16
        elif ".indexer.k_norm." in name:
            shape, dtype = (128,), torch.bfloat16
        else:
            shape, dtype = (2048,), torch.bfloat16
        value = torch.empty(shape, device="meta", dtype=dtype)
        return value if extent is None else value[extent]

    def load_tensors(self, names):
        return tuple(torch.empty(64, device="meta", dtype=torch.bfloat16) for _ in names)

    def load_fp4_experts(self, *args, **kwargs):
        self.expert_modes.append(kwargs["activation_mode"])
        return "experts"

    def _shared_reads(self):
        return None


@pytest.fixture
def metadata_config(monkeypatch):
    config = MetadataConfig(
        text=dict(linear_attn_config=dict(num_heads=16), num_attention_heads=16,
                  q_lora_rank=16, kv_lora_rank=512, qk_rope_head_dim=64,
                  hidden_size=64, routed_expert_hidden_size=128, num_experts=256,
                  moe_intermediate_size=512, num_shared_experts=2,
                  first_k_dense_replace=0),
        dsa=SimpleNamespace(heads=32, head_dim=128),
    )
    monkeypatch.setattr(weights.ModelConfig, "load", lambda snapshot: config)
    monkeypatch.setattr(weights, "quantize_bf16_linear_as_mxfp8",
                        lambda tensor, *, retain_scale_rows=True:
                        Converted(tuple(tensor.shape), retain_scale_rows))
    return config


@pytest.mark.parametrize("scope,online", [
    (None, True), ("all", True), ("kda-input", True), ("all", False), ("kda-input", False),
])
def test_scope_selects_actual_loader_projection_groups(metadata_config, scope, online):
    options = {} if scope is None else {"online_mxfp8_scope": scope}
    loader = MetadataLoader(SimpleNamespace(snapshot="metadata"), torch.device("meta"),
                            shard=TensorParallelShard(0, 16), online_mxfp8=online, **options)
    broad = online and scope in (None, "all")
    assert loader.online_mxfp8_scope == (scope or "all")
    for index, kind in enumerate(loader.config.layers):
        layer = loader.load_layer(index)
        converted = {name for name, value in layer["attention"].items() if isinstance(value, Converted)}
        expected = ({"input", "forget", "output"} if kind == "kda"
                    else {"latent_gate", "q_b", "output"}) if broad else (
                        {"input"} if online and kind == "kda" else set())
        assert converted == expected
        assert all(not layer["attention"][name].retain_scale_rows for name in converted)
        assert all(isinstance(value, Converted) == broad for value in layer["shared"].values())
        assert all(isinstance(layer[name], Converted) == broad
                   for name in ("latent_down", "latent_down_rows", "latent_up"))
        assert layer["router"].dtype == torch.bfloat16
        assert layer["experts"] == "experts"
        if kind == "dsa":
            for name in ("wq_b", "wk", "weights_proj", "k_norm.weight", "k_norm.bias"):
                assert layer["attention"]["indexer." + name].dtype == torch.bfloat16
            latent_gate = layer["attention"]["latent_gate"]
            assert latent_gate.shape == (168, 64)
    assert loader.expert_modes == ["a16"] * len(loader.config.layers)


@pytest.mark.parametrize("scope", ["", "attention", None, 3])
def test_loader_rejects_unknown_scope_before_checkpoint_access(scope):
    with pytest.raises(ValueError, match="online MXFP8 scope must be all or kda-input"):
        weights.KimiWeightLoader(None, "meta", shard=TensorParallelShard(0, 16),
                                 online_mxfp8_scope=scope)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("scope,online", [
    (None, True), ("all", True), ("kda-input", True), ("all", False),
])
def test_model_loader_forwards_scope_without_changing_dspark_family(
        monkeypatch, metadata_config, sparse, scope, online, tmp_path):
    from ignition.recipes.kimi3 import dspark_weights
    from ignition.recipes.kimi31 import dspark_config, dspark_weights as sparse_draft_weights

    metadata_config.dsa = SimpleNamespace(heads=32, head_dim=128) if sparse else None
    metadata_config.layers = ()
    loaders = []

    def load_target(*args, **kwargs):
        loader = MetadataLoader(*args, **kwargs)
        loaders.append(loader)
        return loader

    monkeypatch.setattr(weights, "KimiWeightLoader", load_target)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    draft_config = SimpleNamespace(family="sparse" if sparse else "dense")
    config_class = dspark_config.DSparkConfig if sparse else weights.DSparkConfig
    monkeypatch.setattr(config_class, "load", lambda snapshot, config: draft_config)
    calls = []
    draft_owner = sparse_draft_weights.DraftWeights if sparse else dspark_weights.DraftWeights
    draft = object()

    def load_draft(snapshot, config, device, shard, **kwargs):
        calls.append(kwargs)
        assert config.draft is draft_config
        return draft

    monkeypatch.setattr(draft_owner, "load", load_draft)
    (tmp_path / "config.json").write_text('{"architectures": ["K3DSparkModel"]}' if sparse
                                         else '{"architectures": ["DSparkDraftModel"]}')
    options = {} if scope is None else {"online_mxfp8_scope": scope}
    result = weights.ModelWeights.load(SimpleNamespace(snapshot="metadata"), torch.device("meta"),
        shard=TensorParallelShard(0, 1), online_mxfp8=online, draft_checkpoint=tmp_path,
        vision=False, **options)
    assert loaders[0].online_mxfp8_scope == (scope or "all")
    assert loaders[0].online_mxfp8 == online
    assert result.draft is draft
    assert calls[0]["online_mxfp8"] == (online and scope in (None, "all"))
    assert calls[0]["target_head"] is result.head
    assert ("target_embedding" in calls[0]) == sparse
    if sparse:
        assert calls[0]["target_embedding"] is result.embedding
