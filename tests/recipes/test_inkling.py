# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inkling architecture, checkpoint-precision arithmetic, and complete cache contract."""

import pytest
import torch
from google.protobuf.json_format import ParseDict

from ignition.components.paged_batch import BatchInputs
from ignition.components.parallel import TensorParallelShard
from ignition.preparation import contract as c, runtime_package_pb2 as p
from ignition.preparation.execution import GraphSignature
from ignition.recipes.inkling.config import ModelConfig
from ignition.recipes.inkling.contract import prepare
from ignition.recipes.inkling.model import route, mix_routes
from ignition.recipes.inkling.pool import ContinuationPool, continuation_layout, page_bytes
from ignition.recipes.inkling.recipe import RECIPE
from .inkling_helpers import configuration, quantization


def test_checkpoint_alias_and_canonical_geometry():
    from ignition.recipes import resolve
    from ignition.recipes.inkling.bootstrap import PreparationConfig

    assert resolve("inkling-small-nvfp4").recipe is RECIPE
    model = ModelConfig.from_dict(configuration(), quantization())
    assert model.text["mlp_layer_types"] == ["dense", "sparse", "sparse"]
    assert model.text["intermediate_size"] == 256
    assert model.text["moe_intermediate_size"] == 128
    assert [model.attention(i).window for i in range(3)] == [128, -1, -1]
    assert model.quantized_routed_layers == (2,)
    assert model.vocabulary == 250
    assert RECIPE.speculation is None and RECIPE.max_draft_tokens == 0
    assert PreparationConfig(prefill_chunk_size=64).prefill_chunk_size == 64
    with pytest.raises(ValueError):
        PreparationConfig(devices=(0, 1, 2))


@pytest.mark.parametrize("change", [
    {"q_bias": True}, {"use_sconv": False}, {"gate_activation": "softmax"},
    {"sconv_kernel_size": 3}, {"final_logit_softcapping": 30.},
])
def test_rejects_changed_arithmetic(change):
    raw = configuration()
    raw["text_config"].update(change)
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw, quantization())
    raw = configuration()
    raw["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    with pytest.raises(ValueError, match="hf_quant_config"):
        ModelConfig.from_dict(raw, quantization())


def test_checkpoint_quantization_requires_both_calibration_and_exclusions(tmp_path):
    import json

    raw, quant = configuration(), quantization()
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(FileNotFoundError, match="hf_quant_config"):
        ModelConfig.load(tmp_path)
    for key, value in (("group_size", 32), ("quant_algo", "W4A16_NVFP4"), ("kv_cache_quant_algo", "FP8")):
        with pytest.raises(ValueError):
            ModelConfig.from_dict(raw, dict(quant, **{key: value}))
    quant["modelopt_quant_config"]["quant_cfg"]["*input_quantizer"]["enable"] = False
    with pytest.raises(ValueError, match="weight and input"):
        ModelConfig.from_dict(raw, quant)
    quant = quantization()
    quant["exclude_modules"].remove("model.llm.layers.1.mlp.shared_experts")
    with pytest.raises(ValueError, match="shared experts"):
        ModelConfig.from_dict(raw, quant)
    quant = quantization()
    quant["exclude_modules"].remove("model.llm.layers.1.mlp.experts")
    assert ModelConfig.from_dict(raw, quant).quantized_routed_layers == (1, 2)


def test_fp32_router_keeps_shared_sinks_and_unrounded_checkpoint_scale():
    torch.manual_seed(718)
    logits = torch.randn(13, 6, dtype=torch.float32)
    bias = torch.randn(4) * .01
    scale = torch.tensor([.14234226942062378])
    ids, weights, gammas = route(logits, bias, scale, top_k=2, shared_experts=2, route_scale=8.)
    expected_ids = torch.topk(logits[:, :4].sigmoid() + bias, 2, sorted=False).indices
    selected = torch.cat((logits[:, :4].gather(1, expected_ids), logits[:, 4:]), -1)
    expected = selected.sigmoid()
    expected = expected / expected.sum(-1, keepdim=True) * 8. * scale
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(torch.cat((weights, gammas), -1), expected)
    torch.testing.assert_close(weights.sum(-1) + gammas.sum(-1), (8. * scale).expand(13))
    assert not torch.equal(scale, scale.bfloat16().float())
    assert weights.dtype == gammas.dtype == torch.float32


def test_routed_sum_uses_checkpoint_expert_order_and_bf16_boundaries():
    ids = torch.tensor([[3, 0, 2], [2, 1, 0]])
    outputs = torch.tensor([[[1., 2.], [256., 256.], [-256., -256.]],
                            [[.5, .25], [128., 128.], [-128., -128.]]], dtype=torch.bfloat16)
    weights = torch.ones((2, 3), dtype=torch.float32)
    expected = torch.zeros((2, 2), dtype=torch.bfloat16)
    for expert in range(4):
        for row in range(2):
            selected = (ids[row] == expert).nonzero()
            if selected.numel():
                expected[row] += outputs[row, selected[0, 0]]
    assert torch.equal(mix_routes(outputs, weights, ids), expected)


def test_contract_snapshots_every_history_and_retains_bf16_pages():
    config, shard = ModelConfig.from_dict(configuration(), quantization()), TensorParallelShard(0, 1)
    pool = ContinuationPool.allocate(config, shard, 4, 3, "cpu")
    assert page_bytes(config, shard) == 4 * 128 * 128 * 2
    for layer in range(config.layers):
        for kind, channels in (("key", 128), ("value", 128), ("attention", 256), ("mlp", 256)):
            view = pool.convolution(layer, kind)
            assert view.shape == (4, channels, 3) and view.dtype == torch.float32
            assert view.stride(0) * 4 == pool.layout.row_bytes
    pool.backing[1].random_(0, 255)
    pool.backing[2].copy_(pool.backing[1])
    for region in pool.layout.regions:
        assert torch.equal(pool.region(region.name)[1].view(torch.uint8), pool.region(region.name)[2].view(torch.uint8))
    buffers = BatchInputs.allocate(128, 2, "cpu", capacity=4).buffers()
    signatures = {kind: GraphSignature(kind, 1 if "prefill" in kind else 4,
        128 if "prefill" in kind else 1, tuple(c.text_inputs(kind)))
        for kind in ("init", "restore", "snapshot", "prefill", "scored_prefill", "decode", "resume")}
    contract = ParseDict(prepare(signatures, buffers, pool, vocabulary=250), p.ModelContract())
    assert contract.frontend.tools.kind == p.TML_JSON
    assert contract.frontend.reasoning.kind == p.CONTENT_HEADERS
    histories = [r for r in contract.cache.regions if r.kind == p.PAGED_KEY_VALUE]
    assert len(histories) == 4 and all(r.storage.stride == 32768 for r in histories)
    assert all(any(state.view.nbytes == pool.layout.row_bytes for state in graph.state) for graph in contract.graphs)
    assert all(not graph.HasField("draft_cursor") for graph in contract.graphs)
