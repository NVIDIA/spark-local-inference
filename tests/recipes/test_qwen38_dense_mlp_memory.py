"""Dense MLP scratch follows grouped projection and canonical activation requirements."""

from types import SimpleNamespace
import weakref

import pytest
import torch
from b12x.gemm.blockscaled import NVFP4LinearWeight
from transformers.activations import SiLUActivation

from ignition.components import sm103_projection
from ignition.recipes.qwen38_dense.config import ModelConfig, TEXT_PREFIX
from ignition.recipes.qwen38_dense.decoder import Scratch, _allocate_mlp_intermediate


@pytest.fixture
def bf16_projection_calls(monkeypatch):
    from ignition.components import linear

    plan = SimpleNamespace(request=lambda **kwargs: object())
    monkeypatch.setattr(linear.bf16_gemv, "query_from_call", lambda *args, **kwargs: object())
    monkeypatch.setattr(linear.bf16_gemv, "plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(linear, "declare", lambda *args, **kwargs: None)

    def project(source, weight, *, out, plan, bias=None):
        out.copy_(torch.nn.functional.linear(source, weight, bias))
        return out

    monkeypatch.setattr(linear.bf16_gemv, "mm", project)


@torch.no_grad()
def test_grouped_mlp_drops_preparation_buffers_and_returns_local_output(bf16_projection_calls):
    from ignition.recipes.qwen38_dense.decoder import GroupedMLP

    source = torch.randn((4, 8), dtype=torch.bfloat16)
    gate_up_weight = torch.randn((32, 8), dtype=torch.bfloat16)
    down_weight = torch.randn((8, 16), dtype=torch.bfloat16)
    candidates = (torch.empty((4, 32), dtype=torch.bfloat16),
                  torch.empty((4, 16), dtype=torch.bfloat16), torch.empty_like(source))
    references = tuple(weakref.ref(tensor) for tensor in candidates)
    operation = GroupedMLP(SimpleNamespace(act_fn=torch.nn.SiLU()), source, gate_up_weight, down_weight,
        *candidates, session=None, name="mlp", workspace=None)
    del candidates
    assert all(reference() is None for reference in references)
    for factor in (.5, 1., 1.5):
        source.normal_(0, factor)
        source[-1].zero_()
        gate, up = torch.nn.functional.linear(source, gate_up_weight).chunk(2, -1)
        expected = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_weight)
        actual = operation(source)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.data_ptr() != source.data_ptr()
    assert operation.gate_up.output is None and operation.down.output is None


def test_grouped_mlp_trace_has_local_intermediates(bf16_projection_calls):
    from ignition.preparation.compilation import ForwardTracer, _lower_storage
    from ignition.recipes.qwen38_dense.decoder import GroupedMLP

    source = torch.randn((4, 8), dtype=torch.bfloat16)
    operation = GroupedMLP(SimpleNamespace(act_fn=torch.nn.SiLU()), source,
        torch.randn((32, 8), dtype=torch.bfloat16), torch.randn((8, 16), dtype=torch.bfloat16),
        torch.empty((4, 32), dtype=torch.bfloat16), torch.empty((4, 16), dtype=torch.bfloat16),
        torch.empty_like(source), session=None, name="mlp", workspace=None)
    module = ForwardTracer()(lambda: operation(source))
    _lower_storage(module)
    assert any(node.target == torch.ops.aten.empty_strided.default for node in module.graph.nodes)
    assert all(tuple(value.shape) not in ((4, 32), (4, 16)) for value in module.buffers())


def _weights(*, activation=SiLUActivation):
    hidden, intermediate = 5120, 17408
    down = NVFP4LinearWeight(
        torch.empty((hidden, intermediate // 2), dtype=torch.uint8, device="meta"),
        torch.empty((hidden * intermediate // 16,), dtype=torch.float8_e4m3fn, device="meta"),
        torch.ones((1,), device="meta"), "multiplier", intermediate, hidden)
    layers = [SimpleNamespace(mlp=SimpleNamespace(act_fn=activation())) for _ in range(64)]
    projections = {f"layers.{i}.mlp.{field}": down for i in range(len(layers))
                   for field in ("gate_up", "down_proj")}
    return SimpleNamespace(tree=SimpleNamespace(layers=layers), projections=projections,
        config=SimpleNamespace(text=dict(intermediate_size=intermediate)))


@pytest.mark.parametrize("device,rows", [("cpu", 8), ("meta", 4096)])
@pytest.mark.parametrize("activation", [SiLUActivation, torch.nn.SiLU])
def test_fused_layers_allocate_no_mlp_intermediate(device, rows, activation):
    weights = _weights(activation=activation)
    source = torch.empty((rows, 5120), dtype=torch.bfloat16, device=device)
    assert _allocate_mlp_intermediate(weights, source) is None


@pytest.mark.parametrize("device,rows", [("cpu", 8), ("meta", 4096)])
@pytest.mark.parametrize("fallback", ["ungrouped", "bf16_down", "activation", "inplace_silu", "sm103"])
def test_mixed_layers_allocate_one_shared_bf16_intermediate(device, rows, fallback, monkeypatch):
    weights = _weights()
    for index in (1, 63):
        prefix = f"layers.{index}.mlp."
        if fallback == "ungrouped":
            del weights.projections[prefix + "gate_up"]
        elif fallback == "bf16_down":
            weights.projections[prefix + "down_proj"] = torch.empty((5120, 17408),
                dtype=torch.bfloat16, device="meta")
        elif fallback == "activation":
            weights.tree.layers[index].mlp.act_fn = torch.nn.GELU()
        elif fallback == "inplace_silu":
            weights.tree.layers[index].mlp.act_fn = torch.nn.SiLU(inplace=True)
        else:
            monkeypatch.setattr(sm103_projection, "is_sm103", lambda device: True)
    source = torch.empty((rows, 5120), dtype=torch.bfloat16, device=device)
    intermediate = _allocate_mlp_intermediate(weights, source)
    assert intermediate.shape == (rows, 17408)
    assert intermediate.dtype == torch.bfloat16 and intermediate.device == source.device
    assert intermediate.nbytes == rows * 17408 * 2
    if device == "cpu":
        assert torch.count_nonzero(intermediate) == 0


def test_fallback_allocates_bf16_intermediate():
    weights = _weights(activation=torch.nn.GELU)
    source = torch.empty((8, 5120), dtype=torch.bfloat16)
    intermediate = _allocate_mlp_intermediate(weights, source)
    assert intermediate.shape == (8, 17408) and intermediate.dtype == torch.bfloat16
    assert torch.count_nonzero(intermediate) == 0


@pytest.mark.parametrize("formats,expected_bytes", [
    (("MXFP8", "MXFP8"), 0), (("FP8", "FP8"), 24 << 20),
    (("MXFP8", "FP8"), 24 << 20), ((None, None), 0),
])
def test_projection_formats_determine_tensor_fp8_staging_allocation(formats, expected_bytes):
    text = dict(hidden_size=5120, vocab_size=248320, num_attention_heads=24, head_dim=256,
        layer_types=["linear_attention", "full_attention"])
    quantization = {f"{TEXT_PREFIX}.layers.{index}.mlp.down_proj": dict(quant_algo="NVFP4", group_size=16)
                    for index in range(2)}
    for suffix, algorithm in zip(("layers.0.linear_attn.out_proj", "layers.1.self_attn.o_proj"), formats, strict=True):
        if algorithm is not None:
            quantization[f"{TEXT_PREFIX}.{suffix}"] = dict(quant_algo=algorithm,
                group_size=32 if algorithm == "MXFP8" else None)
    config = ModelConfig(text, quantization)
    scratch = Scratch.allocate(4096, 1, config, "meta")
    assert scratch.quantized.dtype == torch.uint8 and scratch.quantized.device.type == "meta"
    assert scratch.quantized.shape == (expected_bytes,)
    assert scratch.rows(128, 1).quantized is scratch.quantized
