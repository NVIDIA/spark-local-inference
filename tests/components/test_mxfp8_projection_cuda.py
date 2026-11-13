# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared MXFP8 projection owns its encoded rows through native replay."""

import os
from pathlib import Path

import pytest
import torch
from b12x.gemm import blockscaled
from b12x.preparation import PreparedCall
from b12x.quantization import mxfp8

from ignition.components.linear import QuantizedProjection
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, declare
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned GPU")


@pytest.mark.parametrize("atomic", [None, False, True])
@torch.no_grad()
def test_default_tuned_mxfp8_rows_compile_and_replay_without_allocation(atomic):
    device = torch.device("cuda:0")
    rows, width, features = 8, 128, 128
    values = torch.ones((features, width), dtype=torch.float8_e4m3fn, device=device)
    scales = torch.full((features, width // 32), 127, dtype=torch.uint8, device=device)
    weight = blockscaled.pack_weight(values, scales, recipe="mxfp8")
    source = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
    output = torch.empty((rows, features), dtype=torch.bfloat16, device=device)
    arena = ScratchArena(device, overlap=True)
    with RecordedPreparationSession(device=device, checkpoint="mxfp8-projection",
            workload_schema="encoded-rows", numerical_policy="floor1e-4", compile_workers=2) as session:
        session.register_scratch(arena)
        projection = QuantizedProjection(source, weight, output, session=session, name="projection",
            workspace=arena, activation_mode="auto", mxfp8_min_amax=1e-4, rhs_prefetch_ready=True,
            allow_bf16_atomic_reduction=atomic)
        assert projection.plan.query.allow_bf16_atomic_reduction is (atomic is True)
        session.tune()
        forward = compile_forward(lambda: projection(source))
        source.fill_(1)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs={"projection": graph}, buffers={"source": source},
            resources=(source, output, weight, arena, projection, forward, session))
        worker = bundle.start_worker()
        try:
            for fill in (1, 2, 1):
                source.fill_(fill)
                output.fill_(float("nan"))
                arena.tensor.fill_(165)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                ticket = worker.submit(bundle.graph_indices["projection"])
                result = worker.poll(timeout_ms=120000)
                assert result is not None and result.ticket == ticket and result.error is None
                after = torch.cuda.memory_stats()
                for counter in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                    assert before[counter] == after[counter]
                assert output.eq(fill * width).all()
        finally:
            worker.close()
            graph.reset()


@pytest.mark.parametrize("atomic", [None, False, True])
@torch.no_grad()
def test_preencoded_rows_remain_live_through_projection_join(atomic):
    device = torch.device("cuda:0")
    rows, width, features = 8, 128, 128
    weight = blockscaled.pack_weight(
        torch.ones((features, width), dtype=torch.float8_e4m3fn, device=device),
        torch.full((features, width // 32), 127, dtype=torch.uint8, device=device), recipe="mxfp8")
    source = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
    output = torch.empty((rows, features), dtype=torch.bfloat16, device=device)
    independent = torch.randn((rows, width), dtype=torch.bfloat16, device=device)
    normalized = torch.empty_like(independent)
    shapes = mxfp8.dense_rows_storage_shapes(rows, width)
    bases = (torch.empty(shapes[0], dtype=torch.float8_e4m3fn, device=device),
             torch.empty(shapes[1], dtype=torch.uint8, device=device),
             torch.empty(shapes[2], dtype=torch.uint8, device=device))
    encoded = mxfp8.dense_rows_from_bases(*bases)
    arena = ScratchArena(device, overlap=True)
    graphs, owners = {}, []
    with RecordedPreparationSession(device=device, checkpoint="preencoded-mxfp8-projection",
            workload_schema="encoded-rows", numerical_policy="floor1e-4", compile_workers=2) as session:
        session.register_scratch(arena)
        encoder = mxfp8.plan(mxfp8.query_from_call(source, encoded.values, encoded.scale_rows,
            encoded.scale_mma, min_amax=1e-4, release_dependents=True))
        declare(session, (encoder.request(name="producer", prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, encoded.values, encoded.scale_rows, encoded.scale_mma))),))
        projection = QuantizedProjection(source, weight, output, session=session, name="projection",
            workspace=arena, activation_mode="auto", encoded=encoded, rhs_prefetch_ready=True,
            allow_bf16_atomic_reduction=atomic)
        assert projection.plan.query.allow_bf16_atomic_reduction is (atomic is True)
        assert projection.quantizer_plan is None
        norm = PreparedRMSNorm(independent, torch.ones(width, dtype=torch.bfloat16, device=device),
            normalized, eps=1e-6, session=session, name="independent_norm")
        concurrent = ConcurrentProjectionChain((projection,), parallel_plans=(norm.plan,),
            session=session, workspace=arena)
        session.tune()

        def serial():
            mxfp8.quantize_dense_rows(source, *bases, plan=encoder)
            projection(source)
            norm(independent)

        def overlap():
            mxfp8.quantize_dense_rows(source, *bases, plan=encoder)
            concurrent.start(source)
            norm(independent)
            concurrent.finish(source, normalized)

        for name, forward in (("serial", serial), ("overlap", overlap)):
            compiled = compile_forward(forward)
            source.fill_(1)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[name] = graph
            owners.append((compiled, forward))
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"source": source},
            resources=(source, output, independent, normalized, weight, bases, encoded, arena,
                       encoder, projection, norm, concurrent, owners, session))
        worker = bundle.start_worker()
        try:
            for fill in (1, 2, 1):
                source.fill_(fill)
                expected = None
                for name in ("serial", "overlap"):
                    output.fill_(float("nan"))
                    normalized.fill_(float("nan"))
                    arena.tensor.fill_(165)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    ticket = worker.submit(bundle.graph_indices[name])
                    result = worker.poll(timeout_ms=120000)
                    assert result is not None and result.ticket == ticket and result.error is None
                    after = torch.cuda.memory_stats()
                    for counter in ("allocation.all.allocated", "allocation.all.freed",
                                    "segment.all.allocated", "segment.all.freed"):
                        assert before[counter] == after[counter]
                    assert output.eq(fill * width).all()
                    if expected is None:
                        expected = output.clone()
                    else:
                        assert torch.equal(output, expected)
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()


@torch.no_grad()
def test_encoded_projection_join_preserves_mutable_input_and_scratch():
    device = torch.device("cuda:0")
    rows, width, features = 8, 128, 128
    values = torch.ones((features, width), dtype=torch.float8_e4m3fn, device=device)
    scales = torch.full((features, width // 32), 127, dtype=torch.uint8, device=device)
    weight = blockscaled.pack_weight(values, scales, recipe="mxfp8")
    source = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
    output = torch.empty((rows, features), dtype=torch.bfloat16, device=device)
    independent = torch.randn((rows, width), dtype=torch.bfloat16, device=device)
    normalized = torch.empty_like(independent)
    arena = ScratchArena(device, overlap=True)
    with RecordedPreparationSession(device=device, checkpoint="mxfp8-projection-overlap",
            workload_schema="encoded-rows", numerical_policy="floor1e-4", compile_workers=2) as session:
        session.register_scratch(arena)
        projection = QuantizedProjection(source, weight, output, session=session, name="projection",
            workspace=arena, activation_mode="auto", mxfp8_min_amax=1e-4, rhs_prefetch_ready=True)
        norm = PreparedRMSNorm(independent, torch.ones(width, dtype=torch.bfloat16, device=device),
            normalized, eps=1e-6, session=session, name="independent_norm")
        concurrent = ConcurrentProjectionChain((projection,), parallel_plans=(norm.plan,),
            session=session, workspace=arena)
        session.tune()

        def forward():
            concurrent.start(source)
            norm(independent)
            concurrent.finish(source, normalized)

        compiled = compile_forward(forward)
        source.fill_(1)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs={"overlap": graph}, buffers={"source": source},
            resources=(source, output, independent, normalized, weight, arena, projection, norm,
                       concurrent, compiled, session))
        worker = bundle.start_worker()
        try:
            for fill in (1, 2, 1):
                source.fill_(fill)
                output.fill_(float("nan"))
                normalized.fill_(float("nan"))
                arena.tensor.fill_(165)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                ticket = worker.submit(bundle.graph_indices["overlap"])
                result = worker.poll(timeout_ms=120000)
                assert result is not None and result.ticket == ticket and result.error is None
                after = torch.cuda.memory_stats()
                for counter in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                    assert before[counter] == after[counter]
                assert output.eq(fill * width).all()
                assert torch.isfinite(normalized).all()
        finally:
            worker.close()
            graph.reset()


@pytest.mark.parametrize("with_producer", (False, True))
@torch.no_grad()
def test_checkpoint_query_projection_serial_and_overlap_match_exactly(with_producer):
    snapshot = os.environ.get("FLARE_TEST_DS41_CHECKPOINT")
    if snapshot is None:
        pytest.skip("requires DeepSeek V4.1 checkpoint")
    device = torch.device("cuda:0")
    with DeepSeekWeightLoader(inspect_indexed_checkpoint(Path(snapshot)), device,
            shard=TensorParallelShard(0, 4), prefixes=("layers.0.attn.wq_b", "layers.0.attn.q_norm.weight")) as loader:
        weight = loader.load_linear("layers.0.attn.wq_b", rows=(0, 8192))
        norm_weight = loader.load_tensor("layers.0.attn.q_norm.weight")
    projected = torch.empty((8, 1792), dtype=torch.bfloat16, device=device)
    source = torch.empty((8, 1280), dtype=torch.bfloat16, device=device)
    output = torch.empty((8, 8192), dtype=torch.bfloat16, device=device)
    independent = torch.randn((8, 1280), dtype=torch.bfloat16, device=device)
    normalized = torch.empty_like(independent)
    arena = ScratchArena(device, overlap=True)
    graphs, owners = {}, []
    with RecordedPreparationSession(device=device, checkpoint="ds41-real-query-projection",
            workload_schema="encoded-rows", numerical_policy="floor1e-4", compile_workers=2) as session:
        session.register_scratch(arena)
        projection = QuantizedProjection(source, weight, output, session=session, name="q_b",
            workspace=arena, activation_mode="auto", mxfp8_min_amax=1e-4, rhs_prefetch_ready=True)
        producer = (PreparedRMSNorm(projected[:, :1280], norm_weight, source, eps=1e-20,
                    session=session, name="query_norm") if with_producer else None)
        norm = PreparedRMSNorm(independent, torch.ones(1280, dtype=torch.bfloat16, device=device),
            normalized, eps=1e-6, session=session, name="independent_norm")
        concurrent = ConcurrentProjectionChain((projection,), parallel_plans=(norm.plan,),
            session=session, workspace=arena)
        session.tune()
        config = projection.plan.selection.config
        print("selected q_b plan", config, flush=True)

        def serial():
            if producer is not None:
                producer(projected[:, :1280])
            projection(source)
            norm(independent)

        def overlap():
            if producer is not None:
                producer(projected[:, :1280])
            concurrent.start(source)
            norm(independent)
            concurrent.finish(source, normalized)

        for name, forward in (("serial", serial), ("overlap", overlap)):
            compiled = compile_forward(forward)
            source.normal_(std=.1)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[name] = graph
            owners.append((compiled, forward))
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"source": source},
            resources=(projected, source, output, independent, normalized, weight, norm_weight,
                       producer, arena, projection, norm,
                       concurrent, owners, session))
        worker = bundle.start_worker()
        try:
            for step in range(3):
                (projected if with_producer else source).normal_(std=.1 * (step + 1))
                expected = None
                for name in ("serial", "overlap"):
                    output.fill_(float("nan"))
                    arena.tensor.fill_(165)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    ticket = worker.submit(bundle.graph_indices[name])
                    result = worker.poll(timeout_ms=120000)
                    assert result is not None and result.ticket == ticket and result.error is None
                    after = torch.cuda.memory_stats()
                    for counter in ("allocation.all.allocated", "allocation.all.freed",
                                    "segment.all.allocated", "segment.all.freed"):
                        assert before[counter] == after[counter]
                    if expected is None:
                        expected = output.clone()
                    elif not torch.equal(output, expected):
                        difference = (output.float() - expected.float()).abs()
                        raise AssertionError(f"{config}: step {step} max error {difference.max().item()}"
                                             f" RMS error {difference.square().mean().sqrt().item()}")
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()


@pytest.mark.parametrize("atomic", [None, False, True])
@torch.no_grad()
def test_packed_rows_reduction_permission(atomic):
    device = torch.device("cuda:0")
    source = torch.ones((16, 1024), dtype=torch.bfloat16, device=device)
    output = torch.empty((16, 128), dtype=torch.bfloat16, device=device)
    weight = blockscaled.pack_weight(
        torch.ones((128, 1024), dtype=torch.float8_e4m3fn, device=device),
        torch.full((128, 32), 127, dtype=torch.uint8, device=device), recipe="mxfp8")
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="mxfp8-packed-permission",
            workload_schema="packed-rows", numerical_policy="default", autotune_effort="off") as session:
        session.register_scratch(arena)
        projection = QuantizedProjection(source, weight, output, session=session, name="packed",
            workspace=arena, activation_mode="quantized", allow_bf16_atomic_reduction=atomic)
        assert projection.plan.query.allow_bf16_atomic_reduction is (atomic is True)
        session.tune(show_progress=False)
        compiled = compile_forward(lambda: projection(source))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for fill in (1, -0.5, 2):
            source.fill_(fill)
            output.fill_(float("nan"))
            arena.tensor.fill_(165)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for counter in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                assert before[counter] == after[counter]
            assert output.eq(fill * 1024).all()
        graph.reset()
