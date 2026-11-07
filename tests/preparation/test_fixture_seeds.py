# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Private tuning operands retain their generated samples and storage layouts."""

import pytest
import torch

from ignition.preparation.fixtures.projection import Inputs
from ignition.preparation.fixtures.varlen_attention import attention_benchmark_inputs


@pytest.mark.parametrize("strided", [False, True])
def test_projection_preserves_samples_strides_and_repeatable_restoration(strided):
    source = torch.zeros(5, 64, dtype=torch.bfloat16)
    if strided:
        source = source[:, :32]
    output = torch.zeros(5, 24)[:, :16]
    inputs = Inputs(source, output)
    generator = torch.Generator(device=source.device).manual_seed(0xB12)
    expected = torch.randn(source.shape, dtype=source.dtype, generator=generator).mul_(.125)
    assert inputs.seed.device.type == "cpu"
    assert inputs.source.stride() == source.stride()
    assert inputs.output.stride() == output.stride()
    for _ in range(2):
        inputs.source.fill_(float("nan"))
        inputs.produce()
        assert torch.equal(inputs.source, expected)
    assert not source.count_nonzero() and not output.count_nonzero()


@pytest.mark.parametrize("strided", [False, True])
def test_attention_preserves_device_generator_order_and_restoration(strided):
    source = torch.zeros(9, 4, 8, dtype=torch.bfloat16)
    if strided:
        source = source[:, :2]
    inputs = attention_benchmark_inputs(dict(q=source, k=source, v=source,
        cu_seqlens_q=torch.zeros(3, dtype=torch.int32), max_seqlen_q=5, max_seqlen_k=5))
    generator = torch.Generator(device=source.device).manual_seed(89017)
    expected = {}
    for name in ("q", "k", "v"):
        expected[name] = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype).normal_(generator=generator)
        assert getattr(inputs, f"seed_{name}").device.type == "cpu"
        assert getattr(inputs, name).stride() == source.stride()
        getattr(inputs, name).fill_(float("nan"))
    inputs.produce()
    for name in expected:
        assert torch.equal(getattr(inputs, name), expected[name])
    assert inputs.cu_seqlens_q.tolist() == [0, 5, 9]
    assert not source.count_nonzero()
