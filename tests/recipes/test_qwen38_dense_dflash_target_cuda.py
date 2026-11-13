# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual Qwen target residual taps and packed DFlash2 state lifetimes."""

import os
from functools import partial
from pathlib import Path

import pytest
import torch

from ignition.recipes import resolve

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU")


@pytest.mark.parametrize("checkpoint,online_mxfp8", [
    ("qwen3.8-27b", False), ("qwen3.8-27b-qad", False), ("qwen3.8-27b-qad", True), ("hybrid", True),
])
@pytest.mark.parametrize("width", [8, 16])
@torch.no_grad()
def test_checkpoint_compiled_packed_round_preserves_taps_and_inactive_state(checkpoint, online_mxfp8, width):
    from b12x.preparation import PreparationSession
    from ignition.components.completion import CompletionOutputs
    from ignition.components.paged_batch import BatchInputs, Metadata
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38_dense.decoder import Scratch
    from ignition.recipes.qwen38_dense.dflash import SpeculativePreparation
    from ignition.recipes.qwen38_dense.pool import ContinuationPool
    from ignition.recipes.qwen38_dense.weights import ModelWeights

    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    target = (Path(os.environ.get("FLARE_QWEN38_DENSE_CHECKPOINT",
        "/home/luke/models/Qwen3.8-27B-NVIDIA-NVFP4-QAD-MXFP8")) if checkpoint == "hybrid"
        else resolve(checkpoint).snapshot(hub))
    draft = hub / "models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5"
    assert (target / "config.json").is_file() and (draft / "model.safetensors").is_file()
    weights = ModelWeights.load(inspect_indexed_checkpoint(target), torch.device("cuda:0"), mtp=True, draft_snapshot=draft, online_mxfp8=online_mxfp8)
    if checkpoint in ("qwen3.8-27b-qad", "hybrid"):
        assert sum(name.endswith(".mlp.gate_up") for name in weights.projections) == 64
        assert not any(name.endswith((".mlp.gate_proj", ".mlp.up_proj")) for name in weights.projections)
        if online_mxfp8:
            from ignition.components.weights import DirectWeightLoader, quantize_bf16_linear_as_mxfp8

            packed_names = [name for name in weights.projections if name.endswith(".self_attn.qgkv_proj")]
            assert len(packed_names) == 16
            assert not any(name.endswith((".self_attn.q_proj", ".self_attn.kv_proj")) for name in weights.projections)
            packed_weight = weights.projections["layers.3.self_attn.qgkv_proj"]
            prefix = "model.language_model.layers.3.self_attn."
            with DirectWeightLoader(inspect_indexed_checkpoint(target), "cuda:0", prefixes=(prefix,)) as loader:
                original_q = loader.load_linear(prefix + "q_proj")
                original_kv = loader.load_linear_group((prefix + "k_proj", prefix + "v_proj"))
            original = (quantize_bf16_linear_as_mxfp8(original_q), quantize_bf16_linear_as_mxfp8(original_kv))
            order = torch.cat((torch.arange(12288, device="cuda").view(24, 2, 256).transpose(0, 1).reshape(-1),
                               torch.arange(12288, 14336, device="cuda")))
            for field in ("values", "scale_rows"):
                rows = torch.cat([getattr(part.weight, field).view(torch.uint8).reshape(part.out_features, -1)
                                  for part in original])
                actual = getattr(packed_weight.weight, field).view(torch.uint8).reshape(14336, -1)
                assert torch.equal(rows.index_select(0, order), actual)
            del original, original_q, original_kv, rows, actual
        from ignition.recipes.qwen38_dense.config import TEXT_PREFIX
        from ignition.reference.qwen38_quant import CheckpointTensors

        with CheckpointTensors(inspect_indexed_checkpoint(target), byte_budget=2 << 20) as reader:
            for name in ("layers.0.input_layernorm.weight", "layers.0.post_attention_layernorm.weight", "norm.weight"):
                actual = weights.tree.get_parameter(name)
                original = reader.read(TEXT_PREFIX + "." + name)
                expected_dtype = torch.bfloat16 if checkpoint == "hybrid" else torch.float32
                assert actual.dtype == original.dtype == expected_dtype
                assert torch.equal(actual.cpu(), original)
    pool = ContinuationPool.for_model(weights.config, 3, 2, "cuda:0", mtp=True, verification_width=width)
    requests = DeviceRequestState.allocate("cuda:0", vocab_size=weights.config.vocabulary, capacity=2)
    inputs = BatchInputs.allocate(2 * width, 1, "cuda:0", capacity=2)
    inputs.live_rows.fill_(2)
    inputs.row_slot_ids.copy_(torch.tensor([1, 0], device="cuda:0", dtype=torch.int32))
    inputs.row_generations.fill_(1)
    inputs.request_state_rows[0].fill_(1)
    inputs.request_state_rows[1].fill_(2)
    inputs.page_tables.copy_(torch.tensor([[0], [1]], device="cuda:0", dtype=torch.int32))
    inputs.query_lengths.copy_(torch.tensor([width, 1], device="cuda:0", dtype=torch.int32))
    completion = CompletionOutputs.allocate((2, 2, 32), device="cuda:0")
    frame = torch.zeros(1, device="cuda:0", dtype=torch.int32)
    sequence = torch.zeros(1, device="cuda:0", dtype=torch.int64)

    def reset():
        pool.clear_for_preparation()
        for slot in range(2):
            requests.admit(slot, RequestAdmission(generation=1, seed=2718 + slot, max_new_tokens=16,
                top_k=20, top_p=.95), [])
            requests.pending_token[slot] = 11 + slot
        completion.records.zero_()

    reset()
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0")
        scratch = Scratch.allocate(width + 1, 2 * width, weights.config, "cuda:0")
        owner = SpeculativePreparation(weights, pool, requests, inputs, scratch, completion=completion,
            frame=frame, sequence=sequence, session=session, workspace=arena,
            verification_width=width)
        operation = owner.round(2, width, "qwen.actual.packed", target_token_capacity=width + 1)
        decoder = operation.decoder
        if checkpoint in ("qwen3.8-27b-qad", "hybrid"):
            gdn = [layer.layer.linear_attn.operation for layer in decoder.layers
                   if layer.layer.block_type == "linear_attention"]
            assert len(gdn) == 48
            keys = ("in_proj_qkvz", "in_proj_ba", "out_proj") if checkpoint == "hybrid" else ("in_proj_qkvzba", "out_proj")
            assert all(set(attention.projections) == set(keys) for attention in gdn)
            assert all(attention.projections[key].plan.query.allow_bf16_atomic_reduction is False
                       for attention in gdn for key in keys)
        if checkpoint == "hybrid":
            from b12x.gemm.blockscaled import NVFP4LinearWeight

            assert isinstance(weights.head, NVFP4LinearWeight) and weights.draft_head is None
            assert not scratch.quantized.numel()
            assert all(layer.layer.mlp.gate_up.output is None and layer.layer.mlp.down.output is None
                       and not hasattr(layer.layer.mlp, "intermediate") for layer in decoder.layers)
        if online_mxfp8:
            attentions = [layer.layer.self_attn for layer in decoder.layers
                          if layer.layer.block_type == "full_attention"]
            assert len(attentions) == 16
            for attention in attentions:
                assert attention.input_projections == ("qgkv_proj",)
                shape, stride, dtype = attention.projections["qgkv_proj"].output_layout
                packed = torch.empty_strided(shape, stride, dtype=dtype, device="meta")
                query, gate, key, value = packed.split((6144, 6144, 1024, 1024), dim=-1)
                assert query.view(-1, 24, 256).stride() == gate.view(-1, 24, 256).stride() == (14336, 256, 1)
                assert key.stride() == value.stride() == (14336, 1)
        narrow = owner.round(2, 1, "qwen.actual.width1")
        assert operation.operations[2].dummy is None and narrow.operations[2].dummy is None
        owner.release_examples()
        arena.finalize()
        observed_auxiliary = torch.empty((width + 1, 25600), dtype=torch.bfloat16, device="cuda:0")
        def observed_target(metadata):
            result = decoder(metadata)
            hidden, logits, auxiliary, records = result
            assert auxiliary.shape == (width + 1, 25600)
            assert hidden.shape == (2 * width, 5120)
            assert logits.shape == (2 * width, 248320)
            observed_auxiliary.copy_(auxiliary)
            return result
        observed_target.commit = decoder.commit
        operation.decoder = observed_target
        assert decoder.scratch is None and decoder.auxiliary is None
        observed_counts = torch.empty(2, dtype=torch.int32, device="cuda:0")
        observed_candidates = torch.empty((2, width), dtype=torch.int64, device="cuda:0")
        target_metadata = Metadata.allocate(width + 1, 2, 1, "cuda:0", samples=2 * width, max_query_tokens=width)
        metadata_buffers = {name: getattr(target_metadata, name) for name in target_metadata.__dataclass_fields__
                            if isinstance(getattr(target_metadata, name), torch.Tensor)}
        def observed_round():
            proposal, (metadata, acceptance, committed) = operation._round()
            observed_counts.copy_(committed.emitted_count)
            observed_candidates.copy_(proposal.candidates)
            for name, destination in metadata_buffers.items():
                destination.copy_(getattr(metadata, name))
        compiled = compile_forward(observed_round)
        reset()
        compiled()
        reset()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        reset()
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert not requests.error_code.count_nonzero(), requests.error_code.tolist()
        assert observed_counts.gt(0).all()
        accepted = observed_counts.clone()
        auxiliary = observed_auxiliary.clone()
        candidates = observed_candidates.clone()
        target_rows = target_metadata.state_rows.clone()
        normalized = pool.last_hidden[target_rows].clone()
        target_pages = target_metadata.page_table.clone()
        pool.clear_for_preparation()
        tapped = {}
        handles = [decoder.layers[index].register_forward_hook(
            lambda module, args, output, index=index: tapped.__setitem__(index, output[0].clone()))
            for index in weights.config.draft.target_layers]
        eager_hidden, _, eager_auxiliary, _ = decoder(target_metadata)
        torch.cuda.synchronize()
        for handle in handles:
            handle.remove()
        expected = torch.cat([tapped[index] for index in weights.config.draft.target_layers], -1)
        relative = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
        tap_eager_error = relative(auxiliary, expected)
        eager_selected = eager_hidden.view(2, width, weights.config.hidden)[
            torch.arange(2, device="cuda:0"), (accepted - 1).long()]
        hidden_eager_error = relative(normalized, eager_selected)
        assert torch.equal(eager_auxiliary, expected)
        eager_hidden = eager_hidden.clone()
        observed = [torch.empty_like(auxiliary[:, :5120]) for _ in range(5)]
        def observe(destination, module, args, output):
            destination.copy_(output[0])

        handles = [decoder.layers[index].register_forward_hook(partial(observe, destination))
            for index, destination in zip(weights.config.draft.target_layers, observed, strict=True)]
        def standalone_target():
            return decoder(target_metadata)
        standalone = compile_forward(standalone_target)
        for handle in handles:
            handle.remove()
        pool.clear_for_preparation()
        standalone_hidden, _, standalone_auxiliary, _ = standalone()
        torch.cuda.synchronize()
        assert torch.equal(standalone_auxiliary, torch.cat(observed, -1))
        standalone_error = relative(standalone_auxiliary, auxiliary)
        feature_hidden = standalone_hidden.clone()
        saved_capture, decoder.context_capture = decoder.context_capture, None
        target_without_features = compile_forward(standalone_target)
        decoder.context_capture = saved_capture
        pool.clear_for_preparation()
        without_hidden, _, _, _ = target_without_features()
        torch.cuda.synchronize()
        baseline_eager_error = relative(without_hidden, eager_hidden)
        feature_extraction_error = relative(without_hidden, feature_hidden)
        full_lengths = torch.tensor([width, 0], device="cuda:0", dtype=torch.int32)
        for count in range(1, width + 1):
            counts = torch.tensor([count, 0], device="cuda:0", dtype=torch.int32)
            pool.clear_for_preparation()
            inactive = pool.backing[target_rows[1]].clone()
            target_metadata.set(torch.zeros_like(target_rows), full_lengths,
                target_rows, target_pages, candidates, inputs.live_rows)
            _, _, _, records = standalone()
            committed_counts = decoder.commit(counts, records, target_metadata)
            torch.cuda.synchronize()
            assert committed_counts.tolist() == [count, 0]
            assert torch.equal(pool.backing[target_rows[1]], inactive)

        inputs.query_lengths.fill_(1)
        narrow_counts = torch.empty(2, dtype=torch.int32, device="cuda:0")
        def observed_narrow():
            _, (_, _, committed) = narrow._round()
            narrow_counts.copy_(committed.emitted_count)
        compiled_narrow = compile_forward(observed_narrow)
        reset()
        compiled_narrow()
        reset()
        narrow_graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(narrow_graph):
            compiled_narrow()
        reset()
        allocated = torch.cuda.memory_allocated()
        narrow_graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert narrow_counts.tolist() == [1, 1]
        assert requests.consumed_tokens.tolist() == [1, 1]
        for layer in range(5):
            for kind in ("key", "value"):
                assert pool.reserved_region(f"draft{layer}.{kind}")[1:, 0].float().count_nonzero()
        inactive = pool.backing[2].clone()
        inactive_pages = {name: value[1].clone() for name, value in pool.pages.items()}
        inputs.row_slot_ids.copy_(torch.tensor([0, 1], device="cuda:0", dtype=torch.int32))
        inputs.query_lengths.copy_(full_lengths)
        inputs.live_rows.fill_(1)
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert not requests.error_code.count_nonzero(), requests.error_code.tolist()
        assert observed_counts[0] > 0
        assert observed_counts[1] == 0
        assert torch.equal(pool.backing[2], inactive)
        for name, page in inactive_pages.items():
            assert torch.equal(pool.pages[name][1], page), name
        print(dict(physical_tokens=width + 1, logical_samples=2 * width, learned_acceptance=accepted.tolist(),
            tap_eager_relative_l2=tap_eager_error, target_hidden_eager_relative_l2=hidden_eager_error,
            target_without_features_eager_relative_l2=baseline_eager_error,
            round_standalone_relative_l2=standalone_error, feature_extraction_relative_l2=feature_extraction_error,
            accepted_depths=list(range(1, width + 1)), width_transition=[1, width], inactive_rows_preserved=True), flush=True)
        narrow_graph.reset()
        graph.reset()
