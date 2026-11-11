# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Strided fused projection outputs through compiled RMS normalization and replay."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.rms_norm import PreparedRMSNorm

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("with_residual", [False, True])
@torch.no_grad()
def test_functional_normalization_capture(compiled, with_residual):
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    source = torch.randn(16, 5168, device="cuda", dtype=torch.bfloat16)[:, :5120]
    seed = source.clone()
    residual = torch.randn_like(source) if with_residual else None
    weight = torch.randn(5120, device="cuda", dtype=torch.float32) * .1
    output = torch.empty_like(source)
    observed = torch.empty_like(source)
    with RecordedPreparationSession(device=source.device, checkpoint="functional-rms",
            workload_schema="functional-rms", numerical_policy="default") as session:
        bound = PreparedRMSNorm(source, weight, output, eps=1e-6, residual=residual,
            zero_centered=True, session=session, name="rms")
        operation = bound.functional()
        session.tune()
        source.copy_(seed)
        bound(source, residual)
        expected, expected_source = output.clone(), source.clone()

        def forward():
            observed.copy_(operation(source, residual))

        call = compile_forward(forward) if compiled else forward
        call()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            call()
        for _ in range(3):
            source.copy_(seed)
            observed.fill_(float("nan"))
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            torch.testing.assert_close(source, expected_source, rtol=0, atol=0)
        graph.reset()


class NormalizationForward:
    def __init__(self, source, operation):
        self.source, self.operation = source, operation

    def __call__(self):
        return self.operation(self.source)


@torch.no_grad()
def test_rms_norm_compiles_in_metadata_worker(tmp_path):
    from ignition.preparation.compile_pool import CompilePool

    device = torch.device("cuda:0")
    source = torch.randn((4, 1792), dtype=torch.bfloat16, device=device)[:, :1280]
    weight = torch.ones(1280, device=device)
    output = torch.empty((4, 1280), dtype=source.dtype, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        operation = PreparedRMSNorm(source, weight, output, eps=1e-6, session=session,
            name="worker_rms")
        forward = NormalizationForward(source, operation)
        path = tmp_path / "rms.bin"
        with CompilePool(device) as pool:
            pool.submit(forward, path).result()
        with path.open("rb") as file:
            compiled = torch.compiler.load_compiled_function(file)
        compiled(forward)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled(forward)
        source.fill_(2)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.ones_like(output))


@pytest.mark.parametrize("tokens", [1, 4, 17, 4096])
@pytest.mark.parametrize("width", [128, 512, 1280])
@torch.no_grad()
def test_strided_rms_norm_compiled_replay(tokens, width):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(4124 + tokens + width)
    parent = torch.randn((tokens, 1792), dtype=torch.bfloat16, device=device, generator=generator)
    source = parent[:, 1792 - width:]
    output = torch.empty((tokens, width), dtype=source.dtype, device=device)
    weight = torch.linspace(.5, 1.5, width, device=device).bfloat16()
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        operation = PreparedRMSNorm(source, weight, output, eps=1e-6, session=session, name="rms")
        forward = torch.compile(lambda: operation(source), fullgraph=True)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for gain in (1, -.5, 2):
            parent.mul_(gain)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            values = source.float()
            expected = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
                        * weight.float()).bfloat16()
            lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
            upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
            assert ((output >= lower) & (output <= upper)).all()
            assert (output != expected).float().mean() < .001
            del values, expected, lower, upper
        graph.reset()


@pytest.mark.parametrize("tokens", [1, 7, 64])
@pytest.mark.parametrize("update_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_fp32_zero_centered_residual_normalization(tokens, update_dtype):
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    residual = torch.randn(tokens, 4096, device=device)
    seed = residual.clone()
    update = torch.randn(tokens, 4096, device=device).to(update_dtype)
    weight = torch.randn(4096, device=device) * .1
    output = torch.empty_like(residual, dtype=torch.bfloat16)
    with RecordedPreparationSession(device=device, checkpoint="residual-rms", workload_schema="residual-rms",
                                    numerical_policy="default") as session:
        operation = PreparedRMSNorm(residual, weight, output, eps=1e-6, zero_centered=True,
                                   residual=update, session=session, name="residual_rms")
        session.tune()
        forward = compile_forward(lambda: operation(residual, update))
        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        residual.copy_(seed)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected_residual = seed + update.float()
        torch.testing.assert_close(residual, expected_residual, atol=0, rtol=0)
        expected = (expected_residual * torch.rsqrt(expected_residual.square().mean(-1, keepdim=True) + 1e-6)
                    * (1 + weight)).bfloat16()
        lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
        upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
        assert ((output >= lower) & (output <= upper)).all()
        graph.reset()


@torch.no_grad()
def test_paired_normalization_preserves_independent_rounding():
    from ignition.components.rms_norm import PreparedPairedRMSNorm
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    left, right = (torch.randn(7, 4096, device=device).bfloat16() for _ in range(2))
    weights = tuple(torch.randn(4096, device=device) * .1 for _ in range(2))
    out = torch.empty(7, 8192, device=device, dtype=torch.bfloat16)
    with RecordedPreparationSession(device=device, checkpoint="paired-rms", workload_schema="paired-rms",
                                    numerical_policy="default") as session:
        operation = PreparedPairedRMSNorm(left, right, weights, out, eps=1e-6,
                                         zero_centered=True, session=session, name="pair")
        session.tune()
        forward = compile_forward(lambda: operation(left, right))
        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        left.mul_(2)
        right.mul_(-.5)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.cat(tuple((x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                                    * (1 + w)).bfloat16() for x, w in zip((left, right), weights, strict=True)), -1)
        lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
        upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
        assert ((out >= lower) & (out <= upper)).all()
        graph.reset()


@torch.no_grad()
def test_residual_normalization_candidates_share_activation_storage(tmp_path):
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    residual = torch.randn(4096, 4096, device=device)
    update = torch.randn_like(residual, dtype=torch.bfloat16)
    weight = torch.zeros(4096, device=device)
    output = torch.empty_like(update)
    with RecordedPreparationSession(device=device, checkpoint=str(tmp_path), workload_schema="rms-tuning-memory",
                                    numerical_policy="default") as session:
        PreparedRMSNorm(residual, weight, output, eps=1e-6, zero_centered=True,
                        residual=update, session=session, name="residual_rms")
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        report = session.tune()
        torch.cuda.synchronize()
        assert report["benchmarked_candidates"] > 1
        fixture_bytes = 2 * (residual.nbytes + update.nbytes) + output.nbytes
        eviction_bytes = max(2 * torch.cuda.get_device_properties(device).L2_cache_size, 64 << 20)
        assert torch.cuda.max_memory_allocated(device) - before < fixture_bytes + eviction_bytes + (16 << 20)
        assert torch.cuda.memory_allocated(device) - before < 1 << 20


@torch.no_grad()
def test_prefill_residual_norm_retains_sum_on_captured_replay():
    from ignition.preparation.compilation import compile_forward

    source = torch.randn((4096, 5120), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(source)
    weight = torch.zeros(5120, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(source)
    with PreparationSession(device="cuda:0", autotune=False) as session:
        operation = PreparedRMSNorm(source, weight, output, eps=1e-6, zero_centered=True,
            residual=residual, session=session, name="prefill.residual_norm")
        forward = compile_forward(lambda: operation(source, residual))
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            source.normal_()
            residual.normal_()
            preserved = residual.clone()
            expected_sum = source + residual
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert torch.equal(residual, preserved)
            assert torch.equal(source, expected_sum)
            assert torch.isfinite(output).all()
        graph.reset()


@torch.no_grad()
def test_prefill_cross_layer_residual_tap_survives_buffer_reuse():
    from ignition.preparation.compilation import compile_forward

    shape = (4096, 5120)
    projected = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(projected)
    normalized = torch.empty_like(projected)
    attended = torch.empty_like(projected)
    tap = torch.empty((2048, 5120), device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(5120, device="cuda", dtype=torch.bfloat16)
    with PreparationSession(device="cuda:0", autotune=False) as session:
        input_norm = PreparedRMSNorm(projected, weight, normalized, eps=1e-6, zero_centered=True,
            residual=residual, session=session, name="prefill.cross_layer_input")
        post_norm = PreparedRMSNorm(attended, weight, normalized, eps=1e-6, zero_centered=True,
            residual=projected, session=session, name="prefill.cross_layer_post")

        def forward():
            input_norm(projected, residual)
            tap.copy_(projected[-2048:])
            attended.copy_(normalized)
            post_norm(attended, projected)
            projected.zero_()

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            projected.normal_()
            residual.normal_()
            expected_sum = projected + residual
            preserved = residual.clone()
            tap.fill_(float("nan"))
            normalized.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert torch.equal(tap, expected_sum[-2048:])
            assert torch.equal(residual, preserved)
            assert not projected.any()
            assert torch.isfinite(attended).all()
            assert torch.isfinite(normalized).all()
        graph.reset()


@pytest.mark.parametrize("tokens", [1, 16, 17])
@torch.no_grad()
def test_bf16_residual_normalization_preserves_rounded_decoder_sum(tokens):
    from ignition.preparation.compilation import compile_forward

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(9027 + tokens)
    source = torch.randn(tokens, 5120, device=device, generator=generator).bfloat16()
    residual = torch.randn(tokens, 5120, device=device, generator=generator).bfloat16()
    initial_source, initial_residual = source.clone(), residual.clone()
    weight = torch.randn(5120, device=device, generator=generator) * .1
    output = torch.empty_like(source)
    with PreparationSession(device=device, autotune=False) as session:
        operation = PreparedRMSNorm(source, weight, output, eps=1e-6, zero_centered=True,
                                   residual=residual, session=session, name="decoder_residual_rms")
        forward = compile_forward(lambda: operation(source, residual))
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        for gain in (0., .5, 1.5):
            source.copy_(initial_source * gain)
            residual.copy_(initial_residual * (2 - gain))
            expected_source = source + residual
            values = expected_source.float()
            expected = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
                        * (1 + weight)).bfloat16()
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(source, expected_source, atol=0, rtol=0)
            lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
            upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
            assert ((output >= lower) & (output <= upper)).all()
            assert (output != expected).float().mean() < .001
        graph.reset()
