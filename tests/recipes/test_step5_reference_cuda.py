# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared Step-5 modules against the released inference implementation and weights."""

import importlib
import os
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.step5.weights import Step5WeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")

CACHE = Path.home() / ".cache/huggingface/hub"
SNAPSHOT = Path(os.environ.get("FLARE_STEP5_SNAPSHOT", CACHE / "models--stepfun-ai--Step-5-Preview-NVFP4/snapshots/bc246cc2848e17b37f377eda8fd825c5b3018d1f"))
REFERENCE = Path(os.environ.get("FLARE_STEP5_REFERENCE", CACHE / "models--stepfun-ai--Step-5-Preview/snapshots/364b3b03fed2477aa0953fe855a2c4c8c48ee3cc/inference"))


@pytest.fixture(scope="module")
def checkpoint():
    if not (SNAPSHOT / "config.json").exists() or not (REFERENCE / "model.py").exists():
        pytest.skip("Step-5 checkpoint and released inference source are required")
    sys.path.insert(0, str(REFERENCE))
    reference = importlib.import_module("model")
    # Match the prepared GEMMs' FP32 accumulation with BF16 inputs and outputs.
    reduced_precision = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield inspect_indexed_checkpoint(SNAPSHOT), reference
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_precision
        sys.path.remove(str(REFERENCE))


def load_reference(module, prefix, manifest):
    state = {}
    for name in module.state_dict():
        tensor = manifest.tensors[prefix + name]
        with safe_open(str(manifest.snapshot / tensor.shard), framework="pt", device="cuda:0") as source:
            state[name] = source.get_tensor(prefix + name)
    module.load_state_dict(state, assign=True)
    return module.eval()


def relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def load_mxfp8_reference_mlp(module, weight):
    from ignition.reference.qwen38_quant import decode_mxfp8

    packed = weight.weight
    gate, up = decode_mxfp8(packed.values, packed.scale_rows[0].view(torch.uint8)).bfloat16().chunk(2)
    module.gate_proj.weight.copy_(gate)
    module.up_proj.weight.copy_(up)


def serial_fp8_window_attention(query, key, value, window):
    """Reference serial decode: quantized history and one BF16 live candidate."""
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1).transpose(0, 1).float()
    value = value.repeat_interleave(repeats, dim=1).transpose(0, 1).float()
    q = query.transpose(0, 1).float()
    history_key = key.to(torch.float8_e4m3fn).float()
    history_value = value.to(torch.float8_e4m3fn).float()
    logits = q @ history_key.transpose(1, 2)
    logits.diagonal(dim1=1, dim2=2).copy_((q * key).sum(-1))
    positions = torch.arange(query.shape[0], device=query.device)
    distance = positions[:, None] - positions[None, :]
    logits = (logits * query.shape[-1] ** -.5).masked_fill((distance < 0) | (distance >= window), -torch.inf)
    probabilities = logits.softmax(-1)
    output = probabilities @ history_value
    output += probabilities.diagonal(dim1=1, dim2=2)[..., None] * (value - history_value)
    return output.transpose(0, 1).bfloat16()


@torch.no_grad()
@pytest.mark.parametrize("index", [0, 91])
def test_dense_decoder_layer(checkpoint, index, monkeypatch, record_property):
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.step5.attention import PreparedAttention
    from ignition.recipes.step5.decoder import PreparedFeedForward, Scratch
    from ignition.recipes.step5.model import DecoderLayer
    from ignition.recipes.step5.pool import ContinuationPool
    from ignition.reference.qwen38_quant import quantize_mxfp8_activation

    manifest, reference = checkpoint
    # Eight regions fit the SM120 shared-memory limit with 192-dimensional heads.
    monkeypatch.setattr(reference, "sparse_attention_prefill", partial(reference.sparse_attention_prefill, block_regions=8))
    device, shard = torch.device("cuda:0"), TensorParallelShard(0, 1)
    with Step5WeightLoader(manifest, device, shard=shard, prefixes=(f"model.layers.{index}.",)) as loader:
        weights, config = loader.load_layer(index), loader.config
    ref_config = reference.Step5Config(**config.text)
    with torch.device("meta"):
        ref = reference.Step5DecoderLayer(ref_config, index, dtype=torch.bfloat16)
    ref = load_reference(ref, f"model.layers.{index}.", manifest)
    ref.self_attn.build_rope(max_position=128, device=device)
    cache = reference.Step5Cache(ref_config, batch=1, max_tokens=128, device=device, dtype=torch.bfloat16)
    pool = ContinuationPool.for_model(config, shard, 2, 1, device, mtp=False)
    scratch = Scratch.allocate(config, shard, 128, 1, 128, device)
    metadata = Metadata.allocate(128, 1, 1, device)
    metadata.positions.copy_(torch.arange(128, device=device))
    metadata.state_rows.fill_(1)
    metadata.lengths.fill_(128)
    metadata.query_start_loc[1] = 128
    metadata.page_table.zero_()
    torch.manual_seed(617)
    residual = torch.randn(128, config.hidden, device=device, dtype=torch.float32) * .3
    checkpoint_output = ref(residual, metadata.positions, cache, metadata.lengths, metadata.starts, is_decode=False)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="step5-reference", workload_schema="decoder-layer",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        common = {"runtime": None, "coordinator": None, "session": session, "workspace": arena}
        attention = PreparedAttention(index, weights, config, shard, scratch.normalized, scratch.attention,
            metadata, pool, mode="prefill", name="attention", **common)
        feed_forward = PreparedFeedForward(index, weights, config, scratch, name="mlp", overlap=False, **common)
        layer = DecoderLayer(weights, attention, feed_forward, config.text["rms_norm_eps"])
        session.tune()
        load_mxfp8_reference_mlp(ref.mlp, weights.mlp.gate_up)
        if ref.self_attn.is_sparse:
            project = ref.self_attn._project

            def fp8_paged_projection(*args, **kwargs):
                query, key, value = project(*args, **kwargs)
                return query, key.to(torch.float8_e4m3fn).bfloat16(), value.to(torch.float8_e4m3fn).bfloat16()

            monkeypatch.setattr(ref.self_attn, "_project", fp8_paged_projection)
        if feed_forward.mlp.up_projection.plan.selection.config.mode == "quantized":
            ref.mlp.register_forward_pre_hook(
                lambda module, args: (quantize_mxfp8_activation(args[0].float()).bfloat16(),))
        expected = ref(residual, metadata.positions, cache, metadata.lengths, metadata.starts, is_decode=False)
        hidden = ref.input_layernorm(residual).bfloat16()
        torch.testing.assert_close(layer.input_norm(residual).bfloat16(), hidden, rtol=0, atol=0)
        reference_attention = ref.self_attn(hidden, metadata.positions, cache, metadata.lengths,
                                            metadata.starts, is_decode=False)
        attended = attention(hidden).clone()
        tolerance = .008 if index == 91 else .006
        assert relative_error(attended, reference_attention) < tolerance
        normalized = ref.post_attention_layernorm(residual + attended.float()).bfloat16()
        assert relative_error(feed_forward(normalized), ref._feed_forward(normalized)) < tolerance
        actual = layer(residual)
        record_property("bf16_checkpoint_relative_error", relative_error(actual, checkpoint_output))
        assert relative_error(actual, expected) < tolerance
        compiled = compile_forward(lambda: layer(residual))
        actual = compiled()
        assert relative_error(actual, expected) < tolerance
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled()
        graph.replay()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
        graph.reset()


@torch.no_grad()
def test_mtp_feedback_and_proposals(checkpoint, monkeypatch, record_property):

    from ignition.components.paged_batch import Metadata
    from ignition.recipes.step5.decoder import Scratch
    from ignition.recipes.step5.model import RMSNorm
    from ignition.recipes.step5.mtp import (
        PreparedAcceptedContext,
        PreparedFeedback,
        PreparedProposals,
    )
    from ignition.recipes.step5.pool import ContinuationPool

    manifest, reference = checkpoint
    device, shard = torch.device("cuda:0"), TensorParallelShard(0, 1)
    prefixes = ("model.embed_tokens.", "model.layers.92.", "model.layers.93.", "model.layers.94.")
    with Step5WeightLoader(manifest, device, shard=shard, prefixes=prefixes) as loader:
        config = loader.config
        weights = SimpleNamespace(config=config, shard=shard, embedding=loader.load_tensor("model.embed_tokens.weight"),
                                  draft=tuple(loader.load_predictor(i) for i in range(3)))
    ref_config = reference.Step5Config(**config.text)
    refs = []
    for index in range(3):
        with torch.device("meta"):
            ref = reference.Step5DecoderLayer(ref_config, config.layers + index, dtype=torch.bfloat16)
        ref = load_reference(ref, f"model.layers.{config.layers + index}.", manifest)
        ref.self_attn.build_rope(max_position=1024, device=device)
        refs.append(ref)
        monkeypatch.setattr(ref.self_attn, "_sliding_prefill_chunked", serial_fp8_window_attention)
    pool = ContinuationPool.for_model(config, shard, 2, 1, device)
    scratch = Scratch.allocate(config, shard, 128, 128, 1024, device)
    metadata = Metadata.allocate(128, 1, 1, device)
    auxiliary = torch.zeros(128, config.hidden, dtype=torch.bfloat16, device=device)
    counts = torch.zeros(1, dtype=torch.int32, device=device)
    torch.manual_seed(26)
    tokens = torch.randint(config.vocabulary, (532,), device=device)
    hidden = torch.randn(532, config.hidden, dtype=torch.bfloat16, device=device) * .3
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="step5-reference", workload_schema="mtp",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        common = {"runtime": None, "embedding_runtime": None, "vocabulary_runtime": None,
                      "coordinator": None, "session": session, "workspace": arena}
        buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        feedback = PreparedFeedback(weights, auxiliary, metadata, pool, scratch=scratch, name="feedback",
                                    buffers=buffers.cursor(), **common)
        draft = PreparedProposals(weights, pool, scratch.rows(1, 1), 1, 3, name="draft",
                                  buffers=buffers.cursor(), **common)
        session.tune()
        compiled = compile_forward(lambda: feedback(auxiliary, counts))
        start = 0
        batches = []
        for offered, accepted in ((1, 1), (2, 2), (128, 128), (128, 128), (128, 128), (128, 128), (5, 5), (4, 2)):
            staged = torch.zeros(1, 128, dtype=torch.int64, device=device)
            staged[0, :offered].copy_(tokens[start:start + offered])
            metadata.set(torch.tensor([start], device=device), torch.tensor([offered], device=device),
                         torch.ones(1, dtype=torch.int64, device=device), metadata.page_table, staged,
                         torch.ones(1, dtype=torch.int32, device=device))
            auxiliary.zero_()
            auxiliary[:offered].copy_(hidden[start:start + offered])
            counts.fill_(accepted)
            batches.append((start, offered, accepted))
            compiled()
            start += accepted
        assert start == 522
        compiled_state = pool.backing.clone()
        pool.backing.zero_()
        for begin, offered, accepted in batches:
            staged.zero_()
            staged[0, :offered].copy_(tokens[begin:begin + offered])
            metadata.set(torch.tensor([begin], device=device), torch.tensor([offered], device=device),
                         torch.ones(1, dtype=torch.int64, device=device), metadata.page_table, staged,
                         torch.ones(1, dtype=torch.int32, device=device))
            auxiliary.zero_()
            auxiliary[:offered].copy_(hidden[begin:begin + offered])
            counts.fill_(accepted)
            feedback(auxiliary, counts)
        torch.testing.assert_close(pool.backing, compiled_state, atol=0, rtol=0)
        pool.backing.zero_()
        for begin in range(start):
            staged.zero_()
            staged[0, 0] = tokens[begin]
            metadata.set(torch.tensor([begin], device=device), torch.ones(1, dtype=torch.int32, device=device),
                         torch.ones(1, dtype=torch.int64, device=device), metadata.page_table, staged,
                         torch.ones(1, dtype=torch.int32, device=device))
            auxiliary.zero_()
            auxiliary[0].copy_(hidden[begin])
            counts.fill_(1)
            compiled()
        torch.testing.assert_close(pool.backing[0], compiled_state[0], atol=0, rtol=0)
        for region in pool.layout.regions:
            actual = pool.backing[:, region.offset:region.offset + region.nbytes].view(region.dtype)
            expected = compiled_state[:, region.offset:region.offset + region.nbytes].view(region.dtype)
            if region.name.startswith("draft") and region.dtype in (torch.bfloat16, torch.float8_e4m3fn):
                if bool(expected.float().any()):
                    # Chunked execution retains BF16 candidates until the history commit.
                    tolerance = .04 if region.dtype == torch.float8_e4m3fn else (.02 if region.name.endswith("hidden") else .002)
                    error = relative_error(actual, expected)
                    record_property(region.name + ".chunk_relative_error", error)
                    assert error < tolerance, (region.name, error)
                else:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            else:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        previous = hidden[:start + 3]
        expected_logits = []
        errors = {}
        for i, (ref, w) in enumerate(zip(refs, weights.draft, strict=True)):
            first, end = i + 1, start + 3
            load_mxfp8_reference_mlp(ref.mlp, w["layer"].mlp.gate_up)
            for predictor in (feedback.predictors[i], feedback.flush.predictors[i][1], draft.predictors[i][1]):
                assert predictor.layer.feed_forward.mlp.up_projection.plan.selection.config.mode == "a16"
            norm = lambda key, w=w: RMSNorm(w[key], config.text["rms_norm_eps"])
            joined = torch.cat((norm("enorm")(weights.embedding[tokens[first:end]]),
                                norm("hnorm")(previous[:-1])), -1)
            residual = torch.nn.functional.linear(joined, w["projection"]).float()
            cache = reference.DenseLayerCache(batch=1, window=512, num_kv_groups=4, head_dim=192,
                                              device=device, dtype=torch.bfloat16)
            caches = SimpleNamespace(layers=[None] * (config.layers + i) + [cache])
            positions = torch.arange(first, end, device=device)
            previous = ref(residual, positions, caches, torch.tensor([end - first], device=device),
                           torch.zeros(1, dtype=torch.int64, device=device), is_decode=False).bfloat16()
            if i + 1 < len(weights.draft):
                actual_hidden = pool.reserved_region(f"draft{i}.hidden")[1]
                errors[f"draft{i}.hidden"] = relative_error(actual_hidden, previous[start - first - 1])
            expected_logits.append(torch.nn.functional.linear(norm("norm")(previous[start + i - first]), w["head"]))
            # The reference ring uses relative indices; Ignition stores absolute positions.
            live_positions = torch.arange(max(first, start - 512), start, device=device)
            prefix_cache = reference.DenseLayerCache(batch=1, window=512, num_kv_groups=4, head_dim=192,
                                                     device=device, dtype=torch.bfloat16)
            prefix = residual[:start - first]
            ref(prefix, positions[:start - first], SimpleNamespace(layers=[None] * (config.layers + i) + [prefix_cache]),
                torch.tensor([start - first], device=device), torch.zeros(1, dtype=torch.int64, device=device), is_decode=False)
            for kind in ("key", "value"):
                actual = pool.cache(config.layers + i, kind)[1, live_positions % 512]
                expected = getattr(prefix_cache, kind)[0, (live_positions - first) % 512].to(actual.dtype)
                errors[f"draft{i}.{kind}"] = relative_error(actual, expected)

        state_before = pool.backing.clone()
        draft.begin(torch.ones(1, dtype=torch.int64, device=device))
        for i in range(3):
            logits = draft.step(i, tokens[start + i:start + i + 1], torch.tensor([start], device=device),
                                torch.ones(1, dtype=torch.int64, device=device), torch.ones(1, dtype=torch.bool, device=device))
            errors[f"draft{i}.logits"] = relative_error(logits[0], expected_logits[i])
        torch.testing.assert_close(pool.backing, state_before, atol=0, rtol=0)
        # FP8 bin crossings amplify small projection differences in stored K/V.
        for name, error in errors.items():
            record_property(name + ".reference_relative_error", error)
            tolerance = .04 if name.endswith((".key", ".value")) else .03
            assert error < tolerance, (name, error)

        record = PreparedAcceptedContext(weights, metadata, pool)
        embedded = torch.zeros_like(auxiliary)
        before = torch.tensor([start + 1], device=device)
        next_tokens = torch.zeros(3, device=device, dtype=torch.int64)
        states = torch.ones(1, device=device, dtype=torch.int64)
        active = torch.ones(1, device=device, dtype=torch.bool)
        actual_logits = torch.empty(3, config.vocabulary, device=device, dtype=torch.bfloat16)

        def deferred_round():
            record(auxiliary, embedded, counts)
            draft.begin(states)
            for i in range(3):
                actual_logits[i].copy_(draft.step(i, next_tokens[i:i + 1], before, states, active)[0])

        execute = compile_forward(deferred_round)
        graph = None
        for accepted in (1, 2, 3, 4, 1):
            staged.zero_()
            staged[0, :4].copy_(tokens[start:start + 4])
            metadata.set(torch.tensor([start], device=device), torch.tensor([4], device=device), states,
                         metadata.page_table, staged, torch.ones(1, device=device, dtype=torch.int32))
            auxiliary.zero_()
            auxiliary[:4].copy_(hidden[start:start + 4])
            embedded[:4].copy_(weights.embedding[tokens[start:start + 4]])
            counts.fill_(accepted)
            before.fill_(start + accepted)
            next_tokens.copy_(tokens[start + accepted:start + accepted + 3])
            pool.backing.copy_(state_before)
            feedback(auxiliary, counts)
            expected_state = pool.backing.clone()
            draft.begin(states)
            expected_logits = [draft.step(i, next_tokens[i:i + 1], before, states, active).clone()
                               for i in range(3)]
            pool.backing.copy_(state_before)
            if graph is None:
                execute()
                pool.backing.copy_(state_before)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    execute()
            pool.backing.copy_(state_before)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            for i in range(3):
                assert relative_error(actual_logits[i], expected_logits[i][0]) < .03
            torch.testing.assert_close(pool.backing[0], state_before[0], atol=0, rtol=0)
            for region in pool.layout.regions:
                if region.name.startswith("draft") and not region.name.startswith("draft.pending"):
                    actual = pool.backing[:, region.offset:region.offset + region.nbytes].view(region.dtype)
                    expected = expected_state[:, region.offset:region.offset + region.nbytes].view(region.dtype)
                    if bool(expected.float().any()):
                        tolerance = .04 if region.dtype == torch.float8_e4m3fn else .03
                        assert relative_error(actual, expected) < tolerance, region.name
                    else:
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert not pool.reserved_region("draft.pending_count")[1].any()
        graph.reset()


@torch.no_grad()
def test_vision_encoder(checkpoint):
    from ignition.recipes.step5.vision import (
        PreparedVision,
        VisionInputs,
        VisionScratch,
    )
    from ignition.recipes.step5.vision_weights import VisionWeights

    manifest, reference = checkpoint
    device, shard = torch.device("cuda:0"), TensorParallelShard(0, 1)
    with Step5WeightLoader(manifest, device, shard=shard, prefixes=("vision_model.", "vit_large_projector.")) as loader:
        weights = VisionWeights.load(loader)
        projector = loader.load_tensor("vit_large_projector.weight")
    with torch.device("meta"):
        ref = reference.Step5VisionEncoder(reference.Step5VisionConfig(**weights.config))
    ref = load_reference(ref, "vision_model.", manifest)
    buffers = GraphBufferPool(device, chunk_bytes=2 << 20).cursor()
    scratch = VisionScratch.allocate(4096, shard, device, buffers=buffers)
    arena = ScratchArena(device)
    torch.manual_seed(912)
    images = [torch.randn(1, 3, side, side, dtype=torch.bfloat16, device=device) for side in (504, 728)]
    expected = []
    for image in images:
        side = image.shape[-1] // 14
        encoded = ref(image).transpose(1, 2).view(1, 1536, side, side)
        downsampled = ref.vit_downsampler2(ref.vit_downsampler1(encoded))
        expected.append(torch.nn.functional.linear(downsampled.flatten(2).transpose(1, 2)[0], projector))
    expected = torch.cat(expected)
    with RecordedPreparationSession(device=device, checkpoint="step5-reference", workload_schema="vision",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        encoder = PreparedVision(weights, scratch, VisionInputs.allocate(device), runtime=None, coordinator=None,
                                 session=session, workspace=arena, name="vision", buffers=buffers)
        session.tune()
        base = first_base = second_base = 0
        cu = [0]
        encoder.first_indices.fill_(-1)
        encoder.second_indices.fill_(-1)
        for image in images:
            side = image.shape[-1] // 14
            patches = side * side
            scratch.patches[base:base + patches].copy_(image[0].reshape(3, side, 14, side, 14)
                .permute(1, 3, 0, 2, 4).reshape(patches, 588))
            y, x = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij")
            positions = torch.stack((y.flatten(), x.flatten()), -1)
            encoder.positions[base:base + patches].copy_(positions)
            coordinate = ((52 / side) * (positions.float() + .5) - .5).clamp_min(0)
            lo, hi = coordinate.long(), (coordinate.long() + 1).clamp_max(51)
            fraction = coordinate - lo
            indices, probabilities = [], []
            for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
                indices.append((hi if i else lo)[:, 0] * 52 + (hi if j else lo)[:, 1])
                probabilities.append((fraction if i else 1 - fraction)[:, 0] * (fraction if j else 1 - fraction)[:, 1])
            encoder.interpolation_indices[base:base + patches].copy_(torch.stack(indices, -1))
            encoder.interpolation_weights[base:base + patches].copy_(torch.stack(probabilities, -1))
            for extent, source_base, output_base, buffer in (
                (side, base, first_base, encoder.first_indices),
                (side // 2, first_base, second_base, encoder.second_indices),
            ):
                neighbors = []
                for row in range(extent // 2):
                    for col in range(extent // 2):
                        neighbors.append([source_base + (row * 2 + y - 1) * extent + col * 2 + x - 1
                            if 0 <= row * 2 + y - 1 < extent and 0 <= col * 2 + x - 1 < extent else -1
                            for y in range(3) for x in range(3)])
                buffer[output_base:output_base + len(neighbors)].copy_(torch.tensor(neighbors, device=device))
            base += patches
            first_base += patches // 4
            second_base += patches // 16
            cu.append(base)
        encoder.cu.fill_(base)
        encoder.cu[:len(cu)].copy_(torch.tensor(cu, device=device))
        actual = encoder()[:expected.shape[0]].clone()
        assert relative_error(actual, expected) < .03
        compiled = compile_forward(encoder)
        actual = compiled()[:expected.shape[0]].clone()
        assert relative_error(actual, expected) < .03
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        baseline = scratch.output.clone()
        graph.replay()
        torch.testing.assert_close(scratch.output, baseline, atol=0, rtol=0)
        graph.reset()
