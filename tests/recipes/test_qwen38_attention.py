# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-checkpoint QSA projection, position, paging, and replay qualification."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import pytest
import torch

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.qwen38 import TEXT_PREFIX
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.recipes.qwen38.attention import (
    PreparedAttention,
    QSAMetadata,
    QSAScratch,
    QSAState,
    RotaryTables,
)
from ignition.recipes.qwen38.weights import QwenWeightLoader
from ignition.reference.qwen38_quant import (
    CheckpointTensors,
    decode_mxfp8,
    quantize_mxfp8_activation,
)

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires explicit GPU assignment and FLARE_RUN_CUDA_TESTS=1")
PREFIX = f"{TEXT_PREFIX}.layers.3.self_attn"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj")
NORMS = ("q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm")


@pytest.fixture(scope="module")
def manifest():
    path = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    path = Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", path))
    if not path.exists():
        pytest.skip("exact checkpoint is unavailable")
    return inspect_qwen38_checkpoint(path)


def _module(manifest):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention
    config = Qwen4ExpTextConfig.from_dict(json.loads((manifest.snapshot / "config.json").read_text())["text_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        module = Qwen4ExpTextAttention(config, layer_idx=0)
    return module.eval().requires_grad_(False)


def test_qsa_metadata_and_state_geometry():
    state = QSAState.allocate(pages=3, slots=2, device=torch.device("cpu"))
    metadata = QSAMetadata.allocate(tokens=8, batch=2, context=128, device=torch.device("cpu"))
    assert state.main_k_cache.shape == (3, 64, 2, 256)
    assert state.main_k_cache.dtype == state.main_v_cache.dtype == torch.float8_e4m3fn
    assert state.compressed_k_cache.dtype == state.raw_k_ring.dtype == torch.bfloat16
    assert state.compressed_k_cache.shape == (3, 16, 128)
    assert state.raw_k_ring.shape == (2, 4, 128)
    assert state.raw_rope_positions.shape == (2, 4, 3)
    assert metadata.main_block_table.shape == (2, 2)
    assert (metadata.request_ids == -1).all()
    assert (metadata.query_start_loc == 0).all()
    assert (state.raw_interval_start_positions == -1).all()


def test_rotary_tables_match_canonical_multimodal_positions(manifest):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextRotaryEmbedding,
    )
    module = _module(manifest)
    tables = RotaryTables.allocate(module.config, 128, torch.device("cpu"))
    positions = torch.tensor([[0, 0, 0], [1, 2, 3], [80, 24, 73], [127, 126, 125]])
    reference = Qwen4ExpTextRotaryEmbedding(module.config)
    expected = reference(torch.empty(1, 4, 256, dtype=torch.bfloat16), positions.T[:, None])
    actual = tables.gather(positions)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want[0], atol=0, rtol=0)


def _reference_interpolation(theta, pairs, factor, original, beta_fast, beta_slow):
    import math

    dims = torch.arange(pairs, dtype=torch.float32)
    frequencies = 1 / theta ** (dims / pairs)

    def boundary(rotations):
        return (2 * pairs) * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(boundary(beta_fast)), 0)
    high = min(math.ceil(boundary(beta_slow)), pairs - 1)
    smooth = 1 - ((dims - low) / max(high - low, 1e-3)).clamp(0, 1)
    return frequencies / factor * (1 - smooth) + frequencies * smooth


def _text_config(max_position_embeddings=262144):
    from types import SimpleNamespace

    return SimpleNamespace(head_dim=256, max_position_embeddings=max_position_embeddings,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": .25,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True})


def test_rotary_tables_interpolate_declared_rope_scaling():
    from ignition.recipes.declaration import RopeScaling

    scaling = RopeScaling(4., 262144)
    config = _text_config()
    tables = RotaryTables.allocate(config, scaling.context, torch.device("cpu"), scaling=scaling)
    inverse = _reference_interpolation(10000000, 32, 4., 262144, 32., 1.)
    positions = torch.arange(262144)
    expected = (positions[:, None].float() * inverse).cos()
    torch.testing.assert_close(tables.cos[positions, 0], expected[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(tables.sin[positions, 0], (positions[:, None].float() * inverse).sin()[:, 0],
                               atol=0, rtol=0)
    assert scaling is tables.scaling and tables.axes.shape == (32,)

    # The interpolation anchors at the checkpoint's declared context and covers the table.
    with pytest.raises(ValueError, match="extend the checkpoint's declared context"):
        RotaryTables.allocate(_text_config(131072), scaling.context, torch.device("cpu"), scaling=scaling)
    with pytest.raises(ValueError, match="extend the checkpoint's declared context"):
        RotaryTables.allocate(config, 262144, torch.device("cpu"), scaling=scaling)


class ReferenceProjection(torch.nn.Module):
    def __init__(self, weight, scales):
        super().__init__()
        self.register_buffer("weight", decode_mxfp8(weight, scales))
        self.activation_modes = ()

    def forward(self, source):
        value = quantize_mxfp8_activation(source)
        if self.activation_modes:
            a16 = torch.tensor([mode == "a16" for mode in self.activation_modes])
            value = torch.where(a16[:, None], source.float(), value)
        return (value @ self.weight.T).bfloat16()


class FP8ReferenceCache:
    def update_indexer(self, keys, layer):
        return keys

    def update(self, key, value, layer):
        return tuple(tensor.to(torch.float8_e4m3fn).to(tensor.dtype) for tensor in (key, value))


def _load_reference(manifest):
    module = _module(manifest)
    with CheckpointTensors(manifest, byte_budget=128 << 20) as reader:
        for name in NORMS:
            module.get_submodule(name).weight = torch.nn.Parameter(reader.read(f"{PREFIX}.{name}.weight"), requires_grad=False)
        for name in PROJECTIONS:
            parent, _, attribute = name.rpartition(".")
            reference = ReferenceProjection(reader.read(f"{PREFIX}.{name}.weight"),
                                            reader.read(f"{PREFIX}.{name}.weight_scale"))
            setattr(module.get_submodule(parent) if parent else module, attribute, reference)
    return module


def _set_metadata(metadata, starts, lengths, positions, *, prefill):
    boundaries = [0, lengths[0], sum(lengths)]
    count = sum(lengths)
    metadata.request_ids.fill_(-1)
    metadata.query_positions.fill_(-1)
    metadata.rope_positions.fill_(-1)
    metadata.slot_mapping.fill_(-1)
    metadata.query_start_loc.copy_(torch.tensor(boundaries, dtype=torch.int32))
    metadata.sequence_lengths.copy_(torch.tensor([a + b for a, b in zip(starts, lengths)], dtype=torch.int32))
    metadata.raw_state_slot_ids.copy_(torch.tensor([2, 0], dtype=torch.int64))
    metadata.main_block_table.copy_(torch.tensor([[1, 3], [0, 2]], dtype=torch.int32))
    metadata.compressed_block_table.copy_(metadata.main_block_table)
    metadata.num_accepted_tokens.fill_(1)
    metadata.is_prefilling.fill_(prefill)
    for request, (start, length) in enumerate(zip(starts, lengths)):
        span = slice(boundaries[request], boundaries[request + 1])
        logical = torch.arange(start, start + length)
        pages = torch.tensor([1, 3] if request == 0 else [0, 2])[logical // 64]
        metadata.request_ids[span].fill_(request)
        metadata.query_positions[span].copy_(logical)
        metadata.rope_positions[span].copy_(positions[request][start:start + length])
        metadata.slot_mapping[span].copy_(pages * 64 + logical % 64)
    return count


@CUDA
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_checkpoint_qsa_chunked_prefill_native_decode(manifest, compiled):
    from b12x.preparation import PreparationSession
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextRotaryEmbedding,
    )
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_capability(0) in ((12, 0), (12, 1))
    module = _module(manifest)
    with QwenWeightLoader(manifest, device, prefixes=(PREFIX + ".",)) as loader:
        for name in NORMS:
            module.get_submodule(name).weight = torch.nn.Parameter(loader.load_tensor(f"{PREFIX}.{name}.weight"), requires_grad=False)
        weights = loader.load_qsa_projections(PREFIX)
    reference = _load_reference(manifest)
    reference_rotary = Qwen4ExpTextRotaryEmbedding(module.config)
    rotary = RotaryTables.allocate(module.config, 128, device)
    state = QSAState.allocate(pages=4, slots=3, device=device)
    graphs, sources, scratch, metadata, prepared, calls = {}, {}, {}, {}, {}, {}
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream(device=device)
    arena = ScratchArena(device)
    buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        for label, capacity in (("prefill", 64), ("decode", 2)):
            sources[label] = torch.zeros((capacity, 2560), device=device, dtype=torch.bfloat16)
            scratch[label] = QSAScratch.allocate(capacity, device, buffers=buffers.cursor())
            metadata[label] = QSAMetadata.allocate(tokens=capacity, batch=2, context=128, device=device)
            prepared[label] = PreparedAttention(module, weights, sources[label], scratch[label], state,
                metadata[label], rotary, session=session, name=label,
                workspace=arena, projection_workspace=arena)
        arena.finalize()
        arena.freeze()
        session.freeze()
        for label, attention in prepared.items():
            call = partial(attention, sources[label])
            if compiled:
                from ignition.preparation.compilation import compile_forward

                call = compile_forward(call)
            calls[label] = call
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                calls[label]()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                calls[label]()
            graphs[label] = graph
        session.freeze()
        assert (state.raw_logical_positions == -1).all()
        assert (state.raw_interval_start_positions == -1).all()
        worker = PreparedGraphBundle(device=0, graphs=graphs,
            buffers={label: value.output for label, value in scratch.items()},
            resources=(module, prepared, calls, state, sources, metadata, session, stream, rotary)).start_worker()
        generator = torch.Generator().manual_seed(9021)
        histories = [(torch.randn(65, 2560, generator=generator) * .15).bfloat16(),
                     (torch.randn(23, 2560, generator=generator) * .15).bfloat16()]
        positions = [torch.arange(65)[:, None].repeat(1, 3), torch.arange(23)[:, None].repeat(1, 3)]
        positions[0][4:12, 1] -= 2
        positions[0][4:12, 2] += 3
        consumed = [0, 0]
        projection_modes = {name: [[], []] for name in PROJECTIONS}
        try:
            for label, lengths in (("prefill", (17, 9)), ("decode", (1, 1)),
                                   ("prefill", (46, 12)), ("decode", (1, 1))):
                count = _set_metadata(metadata[label], consumed, lengths, positions, prefill=label == "prefill")
                sources[label].zero_()
                sources[label][:count].copy_(torch.cat([h[start:start + length] for h, start, length in
                                                       zip(histories, consumed, lengths)]))
                for chunks in buffers.chunks.values():
                    for tensor in chunks:
                        tensor.fill_(float("nan") if tensor.is_floating_point() else -37)
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                ticket = worker.submit(0 if label == "prefill" else 1)
                result = worker.poll(timeout_ms=30000)
                assert result is not None and result.ticket == ticket and result.error is None
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                offset = 0
                for request, length in enumerate(lengths):
                    total = consumed[request] + length
                    for name in PROJECTIONS:
                        packed = "qkv_proj" if name in ("q_proj", "k_proj", "v_proj") else name
                        mode = prepared[label].projections[packed].plan.selection.config.mode
                        modes = projection_modes[name][request]
                        modes.extend([mode] * length)
                        reference.get_submodule(name).activation_modes = modes
                    value = histories[request][:total][None]
                    rope = reference_rotary(value, positions[request][:total].T[:, None])
                    mask = torch.full((total, total), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16).triu(1)[None, None]
                    unquantized, _ = reference(value, rope, mask)
                    expected, _ = reference(value, rope, mask, past_key_values=FP8ReferenceCache())
                    expected = expected[0, -length:].float()
                    actual = scratch[label].output[offset:offset + length].float().cpu()
                    baseline = unquantized[0, -length:].float()
                    assert (actual - baseline).norm() / baseline.norm().clamp_min(1e-12) < .08
                    relative = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                    assert relative < .02, f"{label} request {request} at {total}: relative L2 {relative}"
                    torch.testing.assert_close(actual, expected, rtol=.035, atol=float(expected.abs().max()) * .02)
                    consumed[request] = total
                    offset += length
                assert torch.count_nonzero(scratch[label].output[count:]) == 0
                assert (state.raw_logical_positions[1] == -1).all()
                assert state.raw_interval_start_positions[1] == -1
        finally:
            worker.close()
