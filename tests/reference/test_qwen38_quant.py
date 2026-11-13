# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import json
import os
from pathlib import Path

import pytest
import torch

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.qwen38 import PLE_PREFIX
from ignition.reference.qwen38_quant import (
    CheckpointTensors,
    Nvfp4Projection,
    decode_csf_scales,
    decode_mxfp8,
    decode_nvfp4,
    mxfp8_linear_reference,
    nvfp4_expert_reference,
    nvfp4_linear_reference,
    ple_hash_reference,
    quantize_mxfp8_activation,
    quantize_nvfp4_activation,
    read_nvfp4_projection,
    read_ple_rows,
    round_e2m1,
)


def test_nvfp4_nibble_order_and_global_scale():
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)
    scales = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)
    decoded = decode_nvfp4(packed, scales, torch.tensor(0.25))
    expected = torch.tensor([[0, .25, .5, .75, 1, 1.5, 2, 3, -0., -.25, -.5, -.75, -1, -1.5, -2, -3]])
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    assert decoded[0, 8].signbit()


def test_mxfp8_scale_bytes_are_exponents():
    weights = torch.ones((1, 96)).to(torch.float8_e4m3fn)
    scales = torch.tensor([[126, 127, 128]], dtype=torch.uint8)
    expected = torch.tensor([[0.5] * 32 + [1.] * 32 + [2.] * 32])
    torch.testing.assert_close(decode_mxfp8(weights, scales), expected, rtol=0, atol=0)
    scales[0, 0] = 0
    assert decode_mxfp8(weights, scales)[0, 0].item() == 2.0 ** -127
    scales[0, 0] = 255
    with pytest.raises(ValueError, match="NaN"):
        decode_mxfp8(weights, scales)


def test_csf_decodes_base_offsets_and_exception_positions():
    bases = torch.arange(32, dtype=torch.uint8)
    codes = torch.full((64,), 0x21, dtype=torch.uint8)
    pages = torch.zeros(1, dtype=torch.uint32)
    offsets = torch.tensor([0, 1, 2], dtype=torch.uint16)
    exceptions = torch.tensor([(19 << 8) | 0, (30 << 8) | 63], dtype=torch.uint16)
    expected = bases[:, None] + torch.tensor([[1, 2, 1, 2]], dtype=torch.uint8)
    expected[0, 0] = 19
    expected[-1, -1] = 30
    assert torch.equal(decode_csf_scales(bases, codes, pages, offsets, exceptions, rows=32, columns=4), expected)


@pytest.mark.parametrize("positions", [[0, 0], [10, 3], [64]])
def test_csf_rejects_invalid_exception_positions(positions):
    bases = torch.zeros(32, dtype=torch.uint8)
    codes = torch.zeros(64, dtype=torch.uint8)
    pages = torch.zeros(1, dtype=torch.uint32)
    offsets = torch.tensor([0, len(positions), len(positions)], dtype=torch.uint16)
    exceptions = torch.tensor(positions, dtype=torch.uint16)
    with pytest.raises(ValueError, match="sorted, unique, and in range"):
        decode_csf_scales(bases, codes, pages, offsets, exceptions, rows=32, columns=4)


def test_e2m1_ties_round_to_even_and_saturate():
    inputs = torch.tensor([0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, 100])
    expected = torch.tensor([0, 0, 1, 1, 2, 2, 4, 4, 6], dtype=torch.float32)
    assert torch.equal(round_e2m1(inputs), expected)
    assert torch.equal(round_e2m1(-inputs), -expected)


def test_nvfp4_activation_uses_stored_dequantization_scale():
    values = torch.tensor([[0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5]])
    expected = torch.tensor([[0, 0, 1, 1, 2, 2, 4, 4, 6, 0, -1, -1, -2, -2, -4, -4]], dtype=torch.float32)
    assert torch.equal(quantize_nvfp4_activation(values, torch.tensor(0.5)), expected)


def test_nvfp4_activation_zero_underflow_and_finite_saturation():
    values = torch.stack((torch.zeros(16), torch.full((16,), 1e-7), torch.full((16,), 10000.)))
    expected = torch.stack((torch.zeros(16), torch.zeros(16), torch.full((16,), 2688.)))
    assert torch.equal(quantize_nvfp4_activation(values, torch.tensor(1.)), expected)


def test_nvfp4_midpoint_uses_rounded_reciprocal_metadata():
    from ignition.reference.qwen38_quant import _nvfp4_activation_unscaled

    values = torch.zeros((1, 16), dtype=torch.bfloat16)
    values[0, 0] = -.02685546875
    values[0, 1] = .2041015625
    stored_scale = torch.tensor(.0035807292442768812)
    actual = _nvfp4_activation_unscaled(values, stored_scale)
    assert actual[0, 0].item() == -10
    assert actual[0, 1].item() == 60


def test_mxfp8_activation_uses_ceiling_power_of_two():
    values = torch.zeros((1, 64))
    values[0, 0:3] = torch.tensor([448., 1.03, -1.03])
    values[0, 32:35] = torch.tensor([449., 1.03, -1.03])
    actual = quantize_mxfp8_activation(values)
    assert actual[0, 0].item() == 448.
    assert actual[0, 32].item() == 448.
    assert actual[0, 1].item() == 1.
    assert actual[0, 33].item() == 1.


@pytest.mark.parametrize("scale", [0., -1., float("inf"), float("nan")])
def test_invalid_activation_scales_rejected(scale):
    with pytest.raises(ValueError, match="input_scale"):
        quantize_nvfp4_activation(torch.ones((1, 16)), torch.tensor(scale))


def test_w4a4_projection_is_not_bf16_dequantized_linear():
    projection = Nvfp4Projection(
        torch.full((1, 8), 0x22, dtype=torch.uint8),
        torch.ones((1, 1)).to(torch.float8_e4m3fn),
        torch.tensor(0.5), torch.tensor(0.5),
    )
    values = torch.full((1, 16), 0.3)
    values[0, 0] = 6.
    actual = nvfp4_linear_reference(values, projection)
    assert actual.item() == 6.75
    assert actual.item() != (values @ decode_nvfp4(projection.packed, projection.scales, projection.weight_scale_2).T).item()


@pytest.fixture(scope="module")
def manifest():
    default = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    configured = os.environ.get("FLARE_QWEN38_SNAPSHOT")
    snapshot = Path(configured) if configured else default
    if not snapshot.is_dir():
        if configured:
            pytest.fail(f"FLARE_QWEN38_SNAPSHOT does not exist: {snapshot}")
        pytest.skip("exact Qwen3.8 checkpoint is not cached")
    return inspect_qwen38_checkpoint(snapshot)


def _e4m3_scalar(code):
    sign = -1 if code & 128 else 1
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return float("nan")
    return sign * math.ldexp(mantissa / 8 if exponent == 0 else 1 + mantissa / 8, -6 if exponent == 0 else exponent - 7)


def test_actual_csf_expert_and_activation_projection(manifest):
    prefix = "model.language_model.layers.0.mlp.experts.0"
    with CheckpointTensors(manifest, byte_budget=4 << 20) as reader:
        gate = read_nvfp4_projection(reader, f"{prefix}.gate_proj")
        up = read_nvfp4_projection(reader, f"{prefix}.up_proj")
        down = read_nvfp4_projection(reader, f"{prefix}.down_proj")
        components = {field: reader.read(f"{prefix}.gate_proj.weight_scale.csf32_{field}")
                      for field in ("bases", "codes", "page_bases", "offsets", "exceptions")}
        assert reader.bytes_read < 4 << 20
    assert gate.weight_scale_2.item() == 6.648472481174394e-05
    assert gate.input_scale.item() == 0.0035807292442768812
    assert torch.equal(gate.weight_scale_2, up.weight_scale_2)
    assert torch.equal(gate.input_scale, up.input_scale)
    assert down.input_scale.item() == 0.00730096735060215
    for row, column in [(0, 0), (1, 31), (15, 159), (16, 0), (639, 159)]:
        group = (row // 128) * 40 + column // 4
        block = group * 8 + (column % 4 // 2) * 4 + row % 128 // 32
        page = group // 32
        origin = components["page_bases"][page].item() // 2
        begin, end = [origin + components["offsets"][block + page + i].item() for i in (0, 1)]
        exception_map = {word & 255: word >> 8 for word in components["exceptions"][begin:end].tolist()}
        pair = components["codes"][block * 32 + row % 32].item()
        code = (pair >> (4 * (column % 2))) & 15
        expected = exception_map.get(2 * (row % 32) + column % 2, components["bases"][row].item() + code)
        assert gate.scales.view(torch.uint8)[row, column].item() == expected
    values = torch.linspace(-2, 2, 2560).reshape(1, -1).to(torch.bfloat16)
    trace = nvfp4_expert_reference(values, gate, up, down)
    assert trace.gate.shape == (1, 640)
    assert trace.intermediate.dtype == torch.bfloat16
    assert trace.down.shape == (1, 2560)
    assert torch.isfinite(trace.down).all()
    assert trace.down.abs().max() > 0


def test_actual_mxfp8_selected_rows(manifest):
    prefix = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    with CheckpointTensors(manifest, byte_budget=10000) as reader:
        weight = reader.read(f"{prefix}.weight", rows=(0, 3))
        scales = reader.read(f"{prefix}.weight_scale", rows=(0, 3))
        assert reader.bytes_read == 3 * (2560 + 80)
    decoded = decode_mxfp8(weight, scales)
    for row, column in [(0, 0), (0, 31), (1, 32), (2, 2559)]:
        expected = _e4m3_scalar(weight.view(torch.uint8)[row, column].item()) * 2 ** (scales[row, column // 32].item() - 127)
        assert decoded[row, column].item() == expected
    values = torch.linspace(-1, 1, 2560).reshape(1, -1).to(torch.bfloat16)
    output = mxfp8_linear_reference(values, weight, scales)
    assert output.shape == (1, 3)
    assert torch.isfinite(output).all()


def test_actual_vision_weight_only_nvfp4(manifest):
    with CheckpointTensors(manifest, byte_budget=4 << 20) as reader:
        projection = read_nvfp4_projection(reader, "model.visual.blocks.0.mlp.linear_fc1")
    assert projection.input_scale is None
    values = torch.linspace(-1, 1, 1152).reshape(1, -1).to(torch.bfloat16)
    result = nvfp4_linear_reference(values, projection)
    assert result.shape == (1, 4304)
    assert torch.isfinite(result).all()


def test_actual_ple_reads_selected_rows_across_plane_boundaries(manifest):
    ids = torch.tensor([0, 2500011, 2500012, 320001535, 0], dtype=torch.int64)
    with CheckpointTensors(manifest, byte_budget=1024) as reader:
        result = read_ple_rows(reader, ids)
        assert reader.bytes_read == 4 + 4 * 90
        scale = reader.read(f"{PLE_PREFIX}.ngram_embedding.weight_scale_2").item()
        packed = reader.read(f"{PLE_PREFIX}.ngram_embedding.shard_0.weight", rows=(0, 1))
        scales = reader.read(f"{PLE_PREFIX}.ngram_embedding.shard_0.weight_scale", rows=(0, 1))
    nibble_values = [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]
    expected = []
    for column in range(160):
        code = (packed[0, column // 2].item() >> (4 * (column % 2))) & 15
        block = _e4m3_scalar(scales.view(torch.uint8)[0, column // 16].item())
        expected.append(nibble_values[code] * block * scale)
    assert result.shape == (5, 160)
    assert torch.equal(result[0], torch.tensor(expected).to(torch.bfloat16))
    assert torch.equal(result[0], result[-1])


def test_reader_budget_prevents_full_ple_materialization(manifest):
    name = f"{PLE_PREFIX}.ngram_embedding.shard_0.weight"
    with CheckpointTensors(manifest, byte_budget=128) as reader:
        with pytest.raises(ValueError, match="select fewer rows"):
            reader.read(name)
        assert reader.bytes_read == 0
        assert reader.read(name, rows=(0, 1)).shape == (1, 80)
        with pytest.raises(ValueError, match="select fewer rows"):
            reader.read(name, rows=(1, 2))
    with pytest.raises(RuntimeError, match="closed"):
        reader.read(name, rows=(0, 1))


def test_actual_ple_hash_matches_pinned_transformers_across_chunks_and_eos(manifest):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    config = Qwen4ExpTextConfig(**json.loads((manifest.snapshot / "config.json").read_text())["text_config"])
    with torch.device("meta"):
        canonical = Qwen4ExpTextNGramEmbedding(config, 2560, layer_idx=1)

    class RecordRows(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.empty(0))

        def forward(self, ids):
            return ids[..., None]

    canonical.ngram_embedding = RecordRows()
    with CheckpointTensors(manifest, byte_budget=1024) as reader:
        multipliers = reader.read(f"{PLE_PREFIX}.layer_multipliers")
        sizes = reader.read(f"{PLE_PREFIX}.ngram_heads_vocab_sizes")
        offsets = reader.read(f"{PLE_PREFIX}.ngram_heads_offsets")
    canonical.layer_multipliers = multipliers
    canonical.ngram_heads_vocab_sizes = sizes
    canonical.ngram_heads_offsets = offsets
    eos = 248044
    tokens = torch.tensor([[11, 12, eos, 13, 14, 15], [eos, 22, 23, 24, eos, eos]])
    expected = canonical(tokens, past_key_values=None)
    actual, history = ple_hash_reference(tokens, multipliers, sizes, offsets)
    assert torch.equal(actual, expected)
    assert torch.equal(history, tokens[:, -2:])
    for split in range(1, tokens.shape[1]):
        first, prefix_history = ple_hash_reference(tokens[:, :split], multipliers, sizes, offsets)
        second, final_history = ple_hash_reference(
            tokens[:, split:], multipliers, sizes, offsets, history=prefix_history
        )
        assert torch.equal(torch.cat((first, second), dim=1), expected)
        assert torch.equal(final_history, history)
    empty, unchanged = ple_hash_reference(tokens[:, :0], multipliers, sizes, offsets, history=history)
    assert empty.shape == (2, 0, 16)
    assert torch.equal(unchanged, history)
