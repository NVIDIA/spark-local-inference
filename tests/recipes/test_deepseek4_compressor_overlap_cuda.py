# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Overlapped compression preserves cache bytes and input lifetimes during native replay."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ignition.components.compressed_mla import PreparedCompressedCacheWriter
from ignition.components.compressor_overlap import ConcurrentGatedCompression
from ignition.components.fp8_indexer import _write_keys
from ignition.components.gated_compression import PreparedGatedCompressor, pool_gated
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.paged_batch import Metadata
from ignition.components.rotary import PreparedRotaryAngles, _rotate
from ignition.preparation.compile_pool import CompilePool
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek4.config import Compressor
from ignition.recipes.deepseek4.pool import ContinuationPool
from .test_deepseek4_pool import model

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


@torch.library.custom_op("flare_test::poison_overlap_input", mutates_args=("source",))
def poison_overlap_input(source: torch.Tensor) -> None:
    source.fill_(float("nan"))


@poison_overlap_input.register_fake
def fake_poison_overlap_input(source):
    pass


class CompressionForward:
    def __init__(self, source, seed, overlap, main, index, branch, angles, metadata, positions, slots,
                 projection, compressors, writer, indexer, head, outputs):
        self.source, self.seed, self.overlap, self.main = source, seed, overlap, main
        self.index, self.branch, self.angles = index, branch, angles
        self.metadata, self.positions, self.slots = metadata, positions, slots
        self.projection, self.compressors, self.writer = projection, compressors, writer
        self.indexer, self.head, self.outputs = indexer, head, outputs

    def observe(self, compressed, index, branch):
        for name, value in zip(("projected", "pooled0", "normalized0", "pooled1", "normalized1", "head"),
                               compressed, strict=True):
            self.outputs[name].copy_(value)
        self.outputs["index"].copy_(index)
        self.outputs["branch"].copy_(branch)

    def __call__(self):
        self.source.copy_(self.seed)
        offsets = torch.arange(self.source.shape[0], device=self.source.device)
        ends = self.metadata.lengths.cumsum(0)
        begins = ends - self.metadata.lengths
        requests = (offsets[:, None] >= ends[None, :]).sum(1).clamp(max=ends.numel() - 1)
        token_positions = torch.where(self.metadata.state_rows[requests] > 0,
            self.metadata.starts[requests] + offsets - begins[requests], -1)
        metadata = replace(self.metadata, requests=requests, positions=token_positions,
            query_start_loc=torch.cat((torch.zeros(1, dtype=torch.int32, device=offsets.device), ends.int())))
        positions = ((token_positions + 1) // 4 * 4 - 4).clamp_min(0)
        emitted = (token_positions >= 0) & ((token_positions + 1) % 4 == 0)
        slots = torch.where(emitted, (requests + 1) * 64 + (token_positions % 256) // 4, -1)
        angles = self.angles(positions)
        compression = self.overlap.start(self.source, metadata, positions, slots, angles=angles)
        main = self.main(self.source)
        pending = self.index.start(main)
        branch = self.branch(main)
        index = self.index.finish(pending, branch)
        compressed = self.overlap.finish(compression, [main, index, branch])
        self.observe(compressed, index, branch)
        self.source.fill_(float("nan"))
        poison_overlap_input(main)
        poison_overlap_input(angles)
        self.outputs["main_poisoned"].copy_(main)
        self.outputs["angles_poisoned"].copy_(angles)

    def sequential(self):
        self.source.copy_(self.seed)
        projected = self.projection(self.source)
        main = self.main(self.source)
        index = self.index.plan_owners[0](main)
        branch = self.branch(main)
        compressed, offset = [projected], 0
        for ordinal, compressor in enumerate(self.compressors):
            width = compressor.spec.state_shape[1]
            values, scores = projected[:, offset:offset + width], projected[:, offset + width:offset + 2 * width]
            pooled = torch.zeros(compressor.output_shape, dtype=torch.bfloat16, device=values.device)
            pool_gated(values, scores, compressor.ape, compressor.state_values, compressor.state_scores,
                       self.metadata, pooled, ratio=compressor.spec.ratio)
            normalized = compressor.normalize(pooled)
            torch.library.wrap_triton(_rotate)[(self.source.shape[0], 1)](
                normalized, self.positions, self.angles.frequencies, normalized.stride(0), 0,
                normalized.shape[1] - 64, 1, None, False)
            if ordinal == 0:
                self.writer(normalized, self.slots)
            else:
                torch.library.wrap_triton(_write_keys)[(self.source.shape[0],)](
                    normalized, self.indexer.cache.view(torch.float8_e4m3fn), self.indexer.cache.view(torch.float32),
                    self.slots, self.indexer.cache.stride(0), num_warps=4)
            compressed.extend((pooled, normalized))
            offset += 2 * width
        compressed.append(self.head(self.source))
        self.observe(compressed, index, branch)
        self.source.fill_(float("nan"))
        main.fill_(float("nan"))


@pytest.mark.parametrize("requests,inactive,width", [(1, False, 4), (2, False, 4), (2, True, 4), (1, False, 6)])
@torch.no_grad()
def test_compression_overlap_retains_inputs_until_native_join(requests, inactive, width, tmp_path):
    device = torch.device("cuda", 0)
    torch.manual_seed(4621 + requests)
    tokens = requests * width
    config = model()
    config.text["rms_norm_eps"] = 1e-6
    pool = ContinuationPool.allocate(config, requests + 2, 4, device)
    arena = ScratchArena(device)
    metadata = Metadata.allocate(tokens, requests, 2, device)
    metadata.starts.copy_(torch.tensor([127, 255][:requests], device=device))
    metadata.state_rows.copy_(torch.arange(1, requests + 1, device=device))
    metadata.requests.copy_(torch.arange(requests, device=device).repeat_interleave(width))
    metadata.lengths.fill_(width)
    metadata.query_start_loc.copy_(torch.arange(requests + 1, device=device) * width)
    metadata.positions.copy_((metadata.starts[:, None] + torch.arange(width, device=device)).flatten())
    if inactive:
        metadata.state_rows[-1] = 0
        metadata.positions[-width:] = -1
    emitted = (metadata.positions >= 0) & ((metadata.positions + 1) % 4 == 0)
    physical = metadata.requests + 1
    slots = torch.where(emitted, physical * 64 + (metadata.positions % 256) // 4, -1)
    positions = ((metadata.positions + 1) // 4 * 4 - 4).clamp_min(0)
    frequencies = torch.linspace(.01, 1, 32, device=device)
    angles = PreparedRotaryAngles(positions, frequencies)
    hidden, projected_width = (4096, 1536) if width == 6 else (64, 128)
    source = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=device)
    seed = torch.randn_like(source)
    projected = torch.zeros((tokens, 2560), device=device)
    projection_weight = torch.randn((2560, hidden), device=device, dtype=torch.bfloat16) * .1
    head_weights = torch.zeros((tokens, 64), dtype=torch.bfloat16, device=device)
    head_matrix = torch.randn((64, hidden), dtype=torch.bfloat16, device=device) * .1
    main_output = torch.empty((tokens, projected_width), device=device, dtype=torch.bfloat16)
    index_output = torch.empty((tokens, 128), device=device, dtype=torch.bfloat16)
    branch_output = torch.empty_like(index_output)
    main_matrix = torch.randn((projected_width, hidden), device=device, dtype=torch.bfloat16) * .1
    branch_matrix = torch.randn((128, projected_width), device=device, dtype=torch.bfloat16) * .1
    from b12x.gemm.blockscaled import pack_weight
    index_matrix = pack_weight((torch.randn((128, projected_width), device=device) * 16).to(torch.float8_e4m3fn),
        torch.full((128, projected_width // 32), 117, device=device, dtype=torch.uint8), recipe="mxfp8")
    if width == 6:
        main_matrix = pack_weight(main_matrix.to(torch.float8_e4m3fn),
            torch.full((projected_width, hidden // 32), 127, device=device, dtype=torch.uint8), recipe="mxfp8")
    with RecordedPreparationSession(device=device, checkpoint="compression-overlap",
            workload_schema="compression-overlap-v1", numerical_policy="bf16-fp32") as session:
        session.register_scratch(arena)
        projection = UnquantizedProjection(source, projection_weight, projected, session=session, name="projection")
        main = (QuantizedProjection(source, main_matrix, main_output, session=session, name="main", workspace=arena)
                if width == 6 else UnquantizedProjection(source, main_matrix, main_output, session=session, name="main"))
        index_projection = QuantizedProjection(main_output, index_matrix, index_output,
            session=session, name="index", workspace=arena)
        branch = UnquantizedProjection(main_output, branch_matrix, branch_output,
            session=session, name="branch")
        index_chain = ConcurrentProjectionChain((index_projection,), parallel_plans=(branch.plan,),
            session=session, workspace=arena)
        head = UnquantizedProjection(source, head_matrix, head_weights, session=session, name="head")
        compressors, offset, examples = [], 0, {}
        for index, dim in enumerate((512, 128)):
            spec = Compressor(2, 4, dim, index=bool(index))
            projection_width = spec.state_shape[1]
            weights = {"ape": torch.randn((4, projection_width), device=device),
                       "norm": torch.randn(dim, device=device)}
            role = "compression.index" if index else "compression.latent"
            def declare_compressor(name):
                return PreparedGatedCompressor(spec, weights, source, metadata, pool, slots,
                    session=session, name=name, deferred=True, examples=examples, example_role=role,
                    emitted=emitted, emitted_slots=slots,
                    projected=(projected[:, offset:offset + projection_width],
                               projected[:, offset + projection_width:offset + 2 * projection_width]))
            compressor = declare_compressor(f"compress{dim}")
            repeated = declare_compressor(f"compress{dim}.repeated")
            assert compressor.pooled is repeated.pooled
            assert compressor.output is repeated.output
            assert compressor.pooled is not compressor.output
            repeated.functional(session=session)
            compressors.append(compressor)
            offset += 2 * projection_width
            compressors[-1].state_values.normal_()
            compressors[-1].state_scores.normal_()
        writer = PreparedCompressedCacheWriter(compressors[0].output, pool.pages["layer2.latent_page"],
            slots, page_size=64, kind="indexed", cache_format="deepseek_v4", session=session, name="writer")
        indexer = SimpleNamespace(cache=pool.pages["layer2.index_page"])
        overlap = ConcurrentGatedCompression(projection, compressors, writer, indexer=indexer,
            index_weight=head, metadata=metadata, positions=positions, frequencies=frequencies,
            angles=angles.values,
            parallel_plans=(main.plan, branch.plan), session=session, workspace=arena)
        outputs = {"projected": torch.empty_like(projected), "head": torch.empty_like(head_weights),
            "index": torch.empty_like(index_output), "branch": torch.empty_like(branch_output),
            "main_poisoned": torch.empty_like(main_output), "angles_poisoned": torch.empty_like(angles.values)}
        for ordinal, compressor in enumerate(compressors):
            outputs[f"pooled{ordinal}"] = torch.empty_like(compressor.pooled)
            outputs[f"normalized{ordinal}"] = torch.empty_like(compressor.output)
        overlap.functional(session=session)
        index_chain.functional(session=session)
        for operation in (projection, main, branch, head):
            operation.functional(session=session)
        writer.functional(session=session)
        for compressor in compressors:
            compressor.functional(session=session)
        angles.functional()
        session.tune()
        if width == 6:
            assert main.plan.memory_requirements().scratch_nbytes > 0
        examples.clear()
        arena.release_preparation()
        forward = CompressionForward(source, seed, overlap, main, index_chain, branch, angles,
            metadata, positions, slots, projection, compressors, writer, indexer, head, outputs)
        path = tmp_path / "compression.bin"
        with CompilePool(device) as compiler:
            compiler.submit(forward, path).result()
        with path.open("rb") as file:
            compiled = torch.compiler.load_compiled_function(file)
        compiled(forward)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled(forward)
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs={"compression": graph},
            buffers={"source": source, "seed": seed, "output": outputs["main_poisoned"]},
            resources=(overlap, projection, compressors, writer, head, main, index_projection,
                       index_chain, branch, pool, arena, session, compiled, forward))
        worker = bundle.start_worker()
        try:
            observed = [*(value for name, value in outputs.items() if not name.endswith("_poisoned")), *overlap.caches]
            for gain in (1., -.5, 2.):
                seed.mul_(gain)
                for cache in overlap.caches:
                    cache.zero_()
                forward.sequential()
                expected = [value.clone() for value in observed]
                for value in observed:
                    value.zero_()
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                ticket = worker.submit(0, repetitions=3)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                for actual, wanted in zip(observed, expected, strict=True):
                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                assert source.isnan().all()
                assert outputs["main_poisoned"].isnan().all()
                assert outputs["angles_poisoned"].isnan().all()
        finally:
            worker.close()
            graph.reset()
