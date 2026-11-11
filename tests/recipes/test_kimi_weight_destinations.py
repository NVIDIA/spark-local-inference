# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi checkpoint composition retains only final weight destinations."""

from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import pytest
import torch

from ignition.components import weights as common
from ignition.components.parallel import TensorParallelShard
from ignition.preparation import persistent
from ignition.recipes.kimi3 import weights
from ignition.recipes.kimi3.residual import score_weight
from ignition.recipes.kimi31 import dspark_weights


class CopySession:
    def __call__(self, destination, source):
        destination.copy_(source)
        return True

    def flush(self):
        pass


@pytest.fixture
def destinations(monkeypatch):
    allocated = []
    empty = persistent.persistent_empty

    def allocate(*args, **kwargs):
        value = empty(*args, **kwargs)
        allocated.append(value)
        return value

    for module in (persistent, common, weights, dspark_weights):
        monkeypatch.setattr(module, "persistent_empty", allocate)
    return allocated


def loader_with_sources(sources):
    loader = weights.KimiWeightLoader.__new__(weights.KimiWeightLoader)
    common.DirectWeightLoader.__init__(loader, SimpleNamespace(), "cpu")
    loader._sources, loader._session = sources, CopySession()
    loader.peers = None
    return loader


@pytest.mark.parametrize("temporary", [False, True])
def test_fused_projection_retains_one_padded_destination(destinations, temporary):
    first = torch.arange(32).reshape(4, 8).to(torch.bfloat16)
    second = (100 + torch.arange(24)).reshape(3, 8).to(torch.bfloat16)
    loader = loader_with_sources({"a.weight": first, "b.weight": second})
    result = loader.fused(("a", "b"), ((1, 4), (0, 2)), temporary=temporary, pad_to=8)
    expected = torch.cat((first[1:], second[:2], torch.zeros(3, 8, dtype=torch.bfloat16)))
    assert torch.equal(result, expected)
    assert result.is_contiguous()
    assert len(destinations) == (0 if temporary else 1)
    if not temporary:
        assert destinations[0] is result


def test_residual_score_releases_checkpoint_inputs(destinations):
    norm = torch.tensor([1.25, -0.75, 0.125], dtype=torch.bfloat16)
    projection = torch.tensor([[0.5, 0.625, -2.0]], dtype=torch.bfloat16)
    loader = loader_with_sources({"norm": norm, "projection": projection})
    result = loader.score("norm", "projection")
    assert torch.equal(result, score_weight(norm, projection))
    assert result.dtype == torch.float32
    assert len(destinations) == 1 and destinations[0] is result


@pytest.mark.parametrize("temporary", [False, True])
def test_shared_column_read_fills_selected_destination(destinations, temporary):
    source = torch.arange(32).reshape(4, 8).to(torch.bfloat16)
    loader = loader_with_sources({"projection.weight": source})

    def read_columns(owner, matrix, columns, destination):
        return destination.copy_(matrix[:, columns[0]:columns[1]])

    loader._shared_reads = lambda: SimpleNamespace(read_columns=read_columns)
    result = loader.linear("projection", columns=(2, 6), temporary=temporary)
    assert torch.equal(result, source[:, 2:6])
    assert len(destinations) == (0 if temporary else 1)
    if not temporary:
        assert destinations[0] is result


def tensor_storages(value):
    if isinstance(value, torch.Tensor):
        return {value.untyped_storage().data_ptr()}
    if is_dataclass(value):
        return set().union(*(tensor_storages(getattr(value, field.name)) for field in fields(value)))
    if isinstance(value, dict):
        return set().union(*(tensor_storages(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(tensor_storages(item) for item in value))
    return set()


@pytest.mark.parametrize("online", [False, True])
def test_dspark_mla_retains_only_final_destinations(monkeypatch, destinations, online):
    hidden, vocabulary, heads, layers = 64, 128, 2, 2
    text = dict(hidden_size=hidden, vocab_size=vocabulary, q_lora_rank=16,
                kv_lora_rank=512, qk_rope_head_dim=64, qk_head_dim=192,
                qk_nope_head_dim=128, v_head_dim=128, intermediate_size=64)
    draft = SimpleNamespace(text=text, heads=heads, layers=layers, markov_rank=8,
                            auxiliary_layers=(0, 1), validate_shard=lambda shard: None)
    config = SimpleNamespace(draft=draft, hidden=hidden, vocabulary=vocabulary)
    shapes = {"context_proj.weight": (hidden, hidden * 2),
              "context_norm.weight": (hidden,), "final_norm.weight": (hidden,),
              "markov_head.markov_w1.weight": (vocabulary, 8),
              "markov_head.markov_w2.weight": (vocabulary, 8)}
    for index in range(layers):
        prefix = f"layers.{index}."
        for name, shape in {"self_attn.q_a_proj": (16, hidden),
                "self_attn.q_b_proj": (heads * 192, 16), "self_attn.q_a_layernorm": (16,),
                "self_attn.kv_a_proj_with_mqa": (576, hidden),
                "self_attn.kv_b_proj": (heads * 256, 512), "self_attn.kv_a_layernorm": (512,),
                "self_attn.o_proj": (hidden, heads * 128), "input_layernorm": (hidden,),
                "post_attention_layernorm": (hidden,), "mlp.gate_proj": (64, hidden),
                "mlp.up_proj": (64, hidden), "mlp.down_proj": (hidden, 64)}.items():
            shapes[prefix + name + ".weight"] = shape
    sources = {name: (torch.arange(torch.Size(shape).numel()).reshape(shape) % 17).to(torch.bfloat16)
               for name, shape in shapes.items()}
    manifest = SimpleNamespace(tensors={name: SimpleNamespace(shape=shape, dtype="BF16")
                                       for name, shape in shapes.items()})

    class Loader(common.DirectWeightLoader):
        def __enter__(self):
            self._sources, self._session = sources, CopySession()
            return self

        def __exit__(self, *exc):
            return False

    def convert(source):
        assert source.untyped_storage().data_ptr() not in tensor_storages(destinations)
        return persistent.persistent_empty(source.shape, source.dtype, source.device).copy_(source)

    monkeypatch.setattr(dspark_weights, "DirectWeightLoader", Loader)
    monkeypatch.setattr(dspark_weights, "inspect_indexed_checkpoint", lambda snapshot: manifest)
    monkeypatch.setattr(common, "quantize_bf16_linear_as_mxfp8", convert)
    embedding = torch.empty(vocabulary, hidden, dtype=torch.bfloat16)
    head = torch.empty_like(embedding)
    result = dspark_weights.DraftWeights.load("checkpoint", config, "cpu", TensorParallelShard(0, 1),
        target_embedding=embedding, target_head=head, online_mxfp8=online)
    assert result.embedding is embedding and result.head is None
    assert tensor_storages(destinations) == tensor_storages(result) - tensor_storages(embedding)
    for index, layer in enumerate(result.layers):
        kv = sources[f"layers.{index}.self_attn.kv_b_proj.weight"].view(heads, 256, 512)
        assert torch.equal(layer.attention["key"], kv[:, :128])
        assert torch.equal(layer.attention["value"], kv[:, 128:].transpose(1, 2))
        assert layer.attention["key"].is_contiguous() and layer.attention["value"].is_contiguous()
    expected_context = torch.cat([sources[f"layers.{index}.self_attn.kv_a_proj_with_mqa.weight"]
                                  for index in range(layers)])
    assert torch.equal(result.context_kv, expected_context)
