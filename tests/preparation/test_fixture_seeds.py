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
    sources = {name: torch.zeros(9, 4, 8, dtype=torch.bfloat16) for name in ("q", "k", "v")}
    if strided:
        sources = {name: source[:, :2] for name, source in sources.items()}
    source = sources["q"]
    inputs = attention_benchmark_inputs(dict(**sources,
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
    assert all(not source.count_nonzero() for source in sources.values())
    assert len({getattr(inputs, name).untyped_storage()._cdata for name in sources}) == 3


@pytest.mark.parametrize("rows", [9, 256])
def test_attention_preserves_packed_qkv_storage_and_bounded_span(rows):
    storage = torch.zeros(rows * 3 * 12 * 128 + 2048, dtype=torch.bfloat16)
    packed = storage[1024:-1024].view(rows, 3, 12, 128)
    sources = dict(zip(("q", "k", "v"), packed.unbind(1), strict=True))
    inputs = attention_benchmark_inputs(dict(**sources,
        cu_seqlens_q=torch.zeros(2, dtype=torch.int32), max_seqlen_q=rows, max_seqlen_k=rows))
    size = packed.numel() * packed.element_size()
    assert inputs.nbytes == 2 * size + inputs.cu_seqlens_q.nbytes
    assert len({getattr(inputs, name).untyped_storage()._cdata for name in sources}) == 1
    assert len({getattr(inputs, "seed_" + name).untyped_storage()._cdata for name in sources}) == 1
    for name, source in sources.items():
        value, seed = getattr(inputs, name), getattr(inputs, "seed_" + name)
        assert value.shape == seed.shape == source.shape
        assert value.stride() == seed.stride() == source.stride()
        assert value.storage_offset() == seed.storage_offset() == source.storage_offset() - 1024
        assert value.untyped_storage()._cdata != source.untyped_storage()._cdata
        assert value.untyped_storage().nbytes() == seed.untyped_storage().nbytes() == size
    expected = {name: getattr(inputs, name).clone() for name in sources}
    inputs.q.fill_(float("nan"))
    assert torch.equal(inputs.k, expected["k"]) and torch.equal(inputs.v, expected["v"])
    inputs.k.fill_(float("nan"))
    inputs.v.fill_(float("nan"))
    inputs.produce()
    for name in sources:
        assert torch.equal(getattr(inputs, name), expected[name])
    assert not storage.count_nonzero()


def test_attention_preserves_overlapping_source_aliases():
    source = torch.zeros(9, 4, 8, dtype=torch.bfloat16)
    inputs = attention_benchmark_inputs(dict(q=source, k=source, v=source,
        cu_seqlens_q=torch.zeros(2, dtype=torch.int32), max_seqlen_q=9, max_seqlen_k=9))
    assert inputs.q.data_ptr() == inputs.k.data_ptr() == inputs.v.data_ptr()
    assert torch.equal(inputs.q, inputs.seed_q)
    inputs.q.fill_(float("nan"))
    inputs.produce()
    assert torch.equal(inputs.q, inputs.seed_q)
    assert torch.equal(inputs.k, inputs.seed_k)
    assert torch.equal(inputs.v, inputs.seed_v)
    assert not source.count_nonzero()


def test_attention_partial_aliasing_keeps_generation_order():
    shared, key = (torch.zeros(9, 4, 8, dtype=torch.bfloat16) for _ in range(2))
    inputs = attention_benchmark_inputs(dict(q=shared, k=key, v=shared,
        cu_seqlens_q=torch.zeros(2, dtype=torch.int32), max_seqlen_q=9, max_seqlen_k=9))
    generator = torch.Generator(device="cpu").manual_seed(89017)
    expected = [torch.empty_like(shared).normal_(generator=generator) for _ in range(3)]
    assert torch.equal(inputs.k, expected[1])
    assert torch.equal(inputs.q, expected[2]) and torch.equal(inputs.v, expected[2])
    assert inputs.q.data_ptr() == inputs.v.data_ptr() != inputs.k.data_ptr()
