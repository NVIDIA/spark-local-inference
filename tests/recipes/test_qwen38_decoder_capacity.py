# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder capacity and public b12x workspace planning without CUDA execution."""

from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace

import pytest
import torch

from ignition.recipes.qwen38.decoder import DecoderScratch
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.decoder_state import DecoderInputs, PreparedDecoderState


@pytest.mark.parametrize("tokens", [3072, 4096, 8192])
def test_large_input_capacity_preserves_inactive_metadata(tokens):
    inputs = DecoderInputs.allocate("cpu", context=262144, tokens=tokens)
    assert inputs.token_ids.shape == (tokens,)
    assert inputs.prefill_positions.shape == (tokens, 3)
    assert inputs.vision_token_indices.shape == (tokens,)
    assert inputs.page_tables.shape == (4, 4096)
    assert torch.all(inputs.token_ids == 248044)
    assert torch.all(inputs.prefill_positions == 0)
    assert torch.all(inputs.vision_token_indices == -1)
    assert torch.all(inputs.page_tables == -1)
    assert torch.all(inputs.row_slot_ids == -1)
    assert inputs.live_rows.item() == 0
    assert torch.all(inputs.query_lengths == 0)
    assert torch.all(inputs.sample_enabled == 0)


def tensors(value):
    for field in fields(value):
        member = getattr(value, field.name)
        if is_dataclass(member):
            yield from tensors(member)
        elif member is not None:
            yield member


@pytest.mark.parametrize("lowrank", [320, 160, 80])
def test_mixer_and_layer_temporaries_preserve_live_residual_and_output(lowrank):
    scratch = DecoderScratch.allocate(128, "cpu", sample_rows=4, hyperconnection_lowrank=lowrank)
    hc = scratch.hyperconnection
    assert scratch.gdn.output.data_ptr() == scratch.qsa.output.data_ptr() == scratch.moe.output.data_ptr()
    live = (scratch.residual, hc.block_input, hc.down_and_injection, scratch.gdn.output)
    for index, tensor in enumerate(live, 1):
        tensor.fill_(index)
    for transient in (hc.normalized, hc.bottleneck, hc.gate_workspace,
                      scratch.gdn.projected_qkvz, scratch.gdn.convolved_qkv, scratch.gdn.gated,
                      scratch.qsa.projected_qkv, scratch.qsa.attended, scratch.moe.routed_output,
                      *((hc.gate_logits,) if hc.gate_logits is not None else ())):
        transient.fill_(-37)
        for index, tensor in enumerate(live, 1):
            assert tensor.eq(index).all()


@pytest.mark.parametrize("tokens", [3072, 4096, 8192])
def test_large_scratch_shapes_and_smaller_views_share_storage(tokens):
    scratch = DecoderScratch.allocate(tokens, "meta")
    assert scratch.residual.shape == (tokens, 10240)
    assert scratch.gdn.projected_qkv.shape == (tokens, 10240)
    assert scratch.qsa.selected_positions.shape == (tokens, 2051)
    assert scratch.ple.key.shape == (tokens, 10240)
    assert scratch.moe.router.ids.shape == (tokens, 10)
    assert scratch.moe.router.selected_logits.shape == (tokens, 10)
    assert scratch.sample_hidden.shape == (4, 2560)
    assert scratch.logits.shape == (4, 248320)
    for rows, sequences in ((tokens, 1), (512, 1), (16, 1), (4, 4), (1, 1)):
        view = scratch.rows(rows, sequences)
        assert view.embedding.shape == (rows, 2560)
        assert view.logits.shape == (sequences, 248320)
        for original, sliced in zip(tensors(scratch), tensors(view), strict=True):
            assert sliced._base is (original if original._base is None else original._base)
            assert sliced.storage_offset() == original.storage_offset()
            assert sliced.stride() == original.stride()
            assert sliced.dtype == original.dtype
    with pytest.raises(ValueError, match="capacity"):
        scratch.rows(tokens + 1, 1)


@pytest.mark.parametrize("tokens", [0, -1, True, 4096.0])
def test_invalid_capacities_reject_before_allocating(tokens):
    with pytest.raises(ValueError, match="capacity"):
        DecoderScratch.allocate(tokens, "meta")
    with pytest.raises(ValueError, match="capacities"):
        DecoderInputs.allocate("cpu", context=262144, tokens=tokens)


@pytest.mark.parametrize("field", ["token_ids", "prefill_positions", "vision_token_indices"])
def test_state_rejects_input_buffer_shorter_than_graph_before_preparation(field):
    inputs = DecoderInputs.allocate("cpu", context=262144, tokens=4096)
    inputs = replace(inputs, **{field: getattr(inputs, field)[:3072]})
    with pytest.raises(ValueError, match="input buffers"):
        PreparedDecoderState(inputs, None, None, tokens=4096, rows=1, mode="prefill")


@pytest.fixture
def metadata_device(monkeypatch):
    from b12x.preparation import device
    from b12x.preparation.types import DeviceIdentity

    detected = device.DetectedDevice(ordinal=0, identity=DeviceIdentity(
        vendor="nvidia", compute_capability=(12, 0), sm_count=170,
        product_name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition"))
    monkeypatch.setattr(device, "detect_device", lambda *args: detected)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())

    def forbidden(*args, **kwargs):
        pytest.fail("metadata-only workspace planning must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


@pytest.mark.parametrize("tokens", [512, 3072, 4096, 8192, 16384])
def test_one_arena_follows_actual_public_plan_extents(metadata_device, tokens):
    from b12x.gemm import blockscaled
    from b12x.preparation import FrozenMapping
    from b12x.sequence import gdn_prefill

    recurrence = gdn_prefill.plan(gdn_prefill.Caps(device="cuda:0", max_tokens=tokens,
        max_seqs=1, max_state_slots=46, key_heads=16, value_heads=48,
        checkpoint_export=True, null_state_index=0, recurrence_policy="sequential_k1"),
        invocation=FrozenMapping({"a_log_dtype": "float32", "dt_bias_dtype": "bfloat16",
                                  "state_indices_dtype": "int64"}))
    plans = [recurrence]
    for input_width, output_width in ((2560, 48), (2560, 512), (2560, 640),
            (2560, 1280), (2560, 6144), (2560, 10240), (2560, 12288),
            (2560, 96), (2560, 13312), (2560, 16384),
            (6144, 2560), (640, 2560)):
        query = blockscaled.BlockscaledQuery(recipe="mxfp8", num_tokens=tokens,
            in_features=input_width, padded_in_features=input_width, out_features=output_width,
            activation_mode="quantized", output_mode="provided", workspace_form="provided",
            workspace_nbytes=None)
        plans.append(blockscaled.plan(query))
    required = [plan.memory_requirements().scratch_nbytes for plan in plans]
    arena = ScratchArena("meta")
    for nbytes in required:
        arena.reserve(nbytes)
    assert arena.nbytes == (max(required) + 255) // 256 * 256
    assert arena.nbytes < sum((n + 255) // 256 * 256 for n in required)
    if tokens >= 3072:
        assert required[0] > 64 << 20
    arena.freeze()
    for nbytes in required:
        assert arena.reserve(nbytes) is arena.tensor


@pytest.mark.parametrize("world", (2, 4))
def test_rank_scratch_keeps_residuals_replicated_and_projects_local_heads(world):
    scratch = DecoderScratch.allocate(128, "meta", sample_rows=8, world_size=world)
    view = scratch.rows(7, 2)
    assert view.residual.shape == (7, 10240)
    assert view.hidden.shape == (7, 2560)
    assert view.gdn.recurrence.shape == (7, 6144 // world)
    assert view.qsa.query.shape == (7, 24 // world, 256)
    assert view.qsa.key.shape == (7, 1, 256)
    assert view.logits.shape == (2, 248320)
    assert view.local_logits.shape == (2, 248320 // world)
    assert view.moe.shared.intermediate.shape == (7, 640 // world)
