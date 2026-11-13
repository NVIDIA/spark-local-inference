# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 attention preserves the prepared SM103 projection path."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.deepseek41.attention import AttentionScratch, PreparedAttention


@pytest.mark.parametrize("sm103", (True, False))
def test_attention_selects_architecture_compatible_mxfp8_preparation(monkeypatch, sm103):
    from ignition.recipes.deepseek41 import attention

    class PackedMXFP8:
        pass

    monkeypatch.setattr(attention.mxfp8, "MXFP8Rows", PackedMXFP8)
    monkeypatch.setattr(attention.sm103_projection, "is_sm103", lambda device: sm103)
    monkeypatch.setattr(attention.mla_norm_mxfp8, "is_supported", lambda device: not sm103)

    calls = {}

    class Component(SimpleNamespace):
        def functional(self, **kwargs):
            return self

    def projection(source, weight, output, **kwargs):
        calls[kwargs["name"].rsplit(".", 1)[-1]] = kwargs
        return Component(plan=object(), output=output)

    class FusedNorm(Component):
        def __init__(self, *args, **kwargs):
            self.encoded = object()
            self.plan = object()

    monkeypatch.setattr(attention, "QuantizedProjection", projection)
    monkeypatch.setattr(attention, "PreparedMLANormQuant", FusedNorm)
    for name in ("PreparedRMSNorm", "PreparedGroupedOutputProjection",
                 "PreparedCompressedCacheWriter", "PreparedCompressedMLA"):
        monkeypatch.setattr(attention, name,
                            lambda *args, **kwargs: Component(plan=object()))

    metadata = Metadata.allocate(1, 1, 1, "cpu", max_query_tokens=1)
    scratch = AttentionScratch.allocate(1, 4, "cpu")
    config = SimpleNamespace(text={"compress_ratios": [0], "rms_norm_eps": 1e-20, "candidate_source_layer_id": 0},
                             source=lambda layer: None)
    pool = SimpleNamespace(config=config,
                           window=lambda *args, **kwargs: torch.zeros((2, 128 * 528), dtype=torch.uint8))
    weights = {"qkv_a": SimpleNamespace(weight=PackedMXFP8()),
               "q_b": SimpleNamespace(weight=PackedMXFP8()),
               "q_norm": torch.ones(1280, dtype=torch.bfloat16),
               "kv_norm": torch.ones(512, dtype=torch.bfloat16),
               "output": object(), "sink": object()}
    prepared = PreparedAttention(0, weights, torch.zeros((1, 5120), dtype=torch.bfloat16),
                                 scratch, metadata, pool, None, mode="decode", world=4,
                                 session=None, name="attention", workspace=None)

    assert set(calls) == {"qkv_a", "q_b"}
    qkv, query = calls["qkv_a"], calls["q_b"]
    assert (prepared.norm_quant is not None) is not sm103
    assert qkv["mxfp8_min_amax"] == (None if sm103 else 1e-4)
    assert query["mxfp8_min_amax"] is None
    assert (query["encoded"] is None) is sm103
    for options in (qkv, query):
        assert options["rhs_prefetch_ready"] is not sm103
        assert options["allow_bf16_atomic_reduction"] is sm103


def test_sm103_mxfp8_projection_reaches_existing_native_plan(monkeypatch):
    from b12x.gemm.blockscaled import _a16
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from ignition.components import linear

    source = torch.zeros((1, 1280), dtype=torch.bfloat16)
    output = torch.empty((1, 8192), dtype=torch.bfloat16)
    values = torch.zeros((8192, 1280), dtype=torch.float8_e4m3fn)
    scales = torch.zeros((8192, 1280 // 32), dtype=torch.uint8)
    weight = MXFP8LinearWeight(
        weight=SimpleNamespace(scale_rows=scales, values_tiled=values),
        in_features=1280, padded_in_features=1280, out_features=8192)
    prepared = SimpleNamespace(request=lambda **kwargs: SimpleNamespace(**kwargs))
    selected = []
    monkeypatch.setattr(linear.sm103_projection, "is_sm103", lambda device: True)
    monkeypatch.setattr(linear.sm103_projection, "query_for",
                        lambda *args, **kwargs: selected.append("query") or SimpleNamespace(mode="quantized"))
    monkeypatch.setattr(linear.sm103_projection, "plan",
                        lambda *args, **kwargs: selected.append("plan") or prepared)
    monkeypatch.setattr(_a16, "_weight_parts", lambda value: (values, scales, None, False))
    monkeypatch.setattr(linear.blockscaled, "query_from_call",
                        lambda *args, **kwargs: pytest.fail("SM103 must keep its native projection query"))
    monkeypatch.setattr(linear, "record_numeric_tensor", lambda tensor: None)
    monkeypatch.setattr(linear, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear, "register_finalizer", lambda *args, **kwargs: None)

    projection = linear.QuantizedProjection(source, weight, output,
        session=SimpleNamespace(native_dense_activations=False), name="query",
        workspace=torch.empty(256, dtype=torch.uint8), activation_mode="auto")
    assert projection.sm103 and projection.plan is prepared
    assert selected == ["query", "plan"]
