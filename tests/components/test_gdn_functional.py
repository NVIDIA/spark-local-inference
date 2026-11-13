# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiler-owned GDN operands preserve prepared launches and persistent state."""

import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from ignition.components import gated_delta, gated_norm, gdn_prefill_sm103, gdn_recovery


def test_functional_gated_norm_releases_bound_operands_and_returns_local_output(monkeypatch):
    monkeypatch.setattr(gated_norm, "warmup_triton", lambda *args, **kwargs: object())
    hidden = torch.ones((2, 3, 128), dtype=torch.bfloat16)
    gate = torch.ones_like(hidden)
    output = torch.zeros_like(hidden)
    weight, live = torch.ones(128), torch.tensor([2], dtype=torch.int32)
    norm = gated_norm.PreparedGatedRMSNorm(hidden, gate, weight, output, live, eps=1e-6)
    selected = norm.compiled
    references = [weakref.ref(tensor) for tensor in (hidden, gate, output)]
    assert norm.functional() is norm
    del hidden, gate, output
    gc.collect()
    assert all(reference() is None for reference in references)
    assert norm.weight is weight and norm.live is live and norm.compiled is selected

    def launch(kernel, compiled, args, grid, **options):
        assert compiled is selected
        args[3].copy_(args[0] + args[1])

    monkeypatch.setattr(gated_norm, "launch_triton", launch)
    hidden = torch.ones((2, 3, 128), dtype=torch.bfloat16)
    gate = torch.full_like(hidden, 2)
    first, second = norm(hidden, gate), norm(hidden * 2, gate)
    assert torch.equal(first, hidden + gate)
    assert torch.equal(second, hidden * 2 + gate)
    assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("fused", [False, True])
def test_functional_recovery_preserves_record_and_convolution_operands(monkeypatch, fused):
    owner = object.__new__(gdn_recovery.PreparedRecoveredRecurrence)
    tokens, sequences, keys, values, width = 4, 2, 1, 3, 2
    mixed = torch.empty((tokens, (2 * keys + values) * 128), dtype=torch.bfloat16)
    a, b = (torch.empty((tokens, values), dtype=torch.bfloat16) for _ in range(2))
    A_log, dt_bias = torch.zeros(values), torch.ones(values)
    state = torch.empty((3, values, 128, 128))
    starts, slots, live = torch.tensor([0, 2, 4], dtype=torch.int32), torch.tensor([1, 2]), torch.tensor([2])
    output = torch.empty((tokens, values * 128), dtype=torch.bfloat16)
    delta, recorded_keys, decay = torch.empty((sequences, values, width, 128)), torch.empty((sequences, keys, width, 128)), torch.empty((sequences, values, width))
    queries, beta = torch.empty_like(recorded_keys), torch.empty_like(decay, dtype=torch.bfloat16)
    owner.args = (mixed, a, b, A_log, dt_bias, state, starts, slots, live, output, delta, recorded_keys,
                  decay, queries, beta, mixed.stride(0), a.stride(0), b.stride(0), state.stride(0), 1)
    owner.prepare_args = (mixed, a, b, A_log, dt_bias, starts, slots, live, queries, recorded_keys, decay,
                          beta, mixed.stride(0), a.stride(0), b.stride(0), 1)
    owner.fused_convolution = fused
    convolution = (torch.empty((mixed.shape[1], 4), dtype=torch.bfloat16),
                   torch.empty((3, mixed.shape[1], 4), dtype=torch.bfloat16), slots,
                   torch.zeros(sequences, dtype=torch.bool), torch.ones(sequences, dtype=torch.int32), live)
    if fused:
        owner.prepare_args += (mixed, *convolution)
    owner.options, owner.prepare_options = {"WINDOW": width, "IN_PLACE": False}, {"WINDOW": width}
    owner.grid, owner.prepare_grid = (1, sequences * values, 1), (width, sequences * keys, 1)
    owner.compiled, owner.prepare_compiled = object(), object()
    launches = []

    def launch(kernel, compiled, args, grid, **options):
        launches.append((kernel, compiled, args, grid, options))
        if kernel is gdn_recovery._verify:
            args[9].fill_(7)

    monkeypatch.setattr(gdn_recovery, "launch_triton", launch)
    owner.functional()
    assert owner.args is None and owner.prepare_args is None
    result = owner(mixed, a, b, projected=mixed,
                   records=gdn_recovery.GDNRecords(delta, recorded_keys, decay))
    assert torch.equal(result, torch.full_like(output, 7))
    prepare, verify = launches
    assert prepare[1] is owner.prepare_compiled and verify[1] is owner.compiled
    assert verify[2][5] is state and verify[2][10] is delta and verify[2][11] is recorded_keys
    assert verify[2][12] is decay
    assert verify[2][13] is prepare[2][8] and verify[2][14] is prepare[2][11]
    assert verify[2][13] is not queries and verify[2][14] is not beta
    if fused:
        assert prepare[0] is gdn_recovery._prepare_convolved_inputs
        assert prepare[2][16] is mixed
        assert all(actual is expected for actual, expected in zip(prepare[2][17:], convolution, strict=True))


@pytest.mark.parametrize("kind", ["combined", "parallel", "separate"])
def test_functional_gdn_routes_projection_slices_without_bound_scratch(kind):
    owner = object.__new__(gated_delta.PreparedGDN)
    owner._functional_requested = owner._functional_ready = True
    owner._has_partitions = False
    owner.world_size = 1
    owner.fused_qkvzba, owner.fused_qkvz = kind == "combined", kind == "parallel"
    owner.tokens = 2
    owner.scratch = None
    owner.partitions = None
    source = torch.zeros((2, 128), dtype=torch.bfloat16)
    qkvz = torch.cat((torch.full((2, 10240), 1), torch.full((2, 6144), 2)), dim=1).to(torch.bfloat16)
    ba = torch.cat((torch.full((2, 48), 3), torch.full((2, 48), 4)), dim=1).to(torch.bfloat16)
    order = []

    def projection(name, output):
        def call(value, **kwargs):
            order.append(name)
            return output
        return call

    owner.projections = {
        "in_proj_qkvzba": projection("combined", torch.cat((qkvz, ba), dim=1)),
        "in_proj_qkv": projection("qkv", qkvz[:, :10240]),
        "in_proj_z": projection("z", qkvz[:, 10240:]),
        "in_proj_ba": projection("ba", ba),
        "out_proj": lambda value: value + 1}
    owner.parallel_projections = (lambda value: (qkvz, ba)) if kind == "parallel" else None

    def sequence(projected, z, a, b, *, records=None):
        assert records is None
        assert torch.equal(projected, qkvz[:, :10240])
        assert torch.equal(z, qkvz[:, 10240:])
        assert torch.equal(a, ba[:, 48:]) and torch.equal(b, ba[:, :48])
        return torch.zeros_like(source), None

    owner._functional_sequence = sequence
    assert torch.equal(owner(source), torch.ones_like(source))
    assert order == {"combined": ["combined"], "parallel": [], "separate": ["qkv", "z", "ba"]}[kind]


def test_functional_gdn_uses_runtime_partition_metadata_and_records():
    owner = object.__new__(gated_delta.PreparedGDN)
    owner._functional_requested = owner._functional_ready = owner._has_partitions = True
    owner.world_size, owner.fused_qkvzba, owner.fused_qkvz = 1, True, True
    owner.partitions = None
    owner.projections = {"in_proj_qkvzba": lambda source: torch.zeros((2, 16480), dtype=torch.bfloat16),
                         "out_proj": lambda value: value}
    seen = []

    def sequence(projected, z, a, b, *, records, metadata):
        seen.append((metadata, records))
        return torch.full((1, 2), float(len(seen))), None

    owner.sequences_by_role = (SimpleNamespace(_functional_sequence=sequence),
                               SimpleNamespace(_functional_sequence=sequence))
    prompt_metadata, decode_metadata, records = object(), object(), object()
    parts = (SimpleNamespace(mode="prefill", metadata=prompt_metadata, gather=lambda value: value[:1]),
             SimpleNamespace(mode="decode", metadata=decode_metadata, gather=lambda value: value[1:]))
    partitions = SimpleNamespace(parts=parts, combine=lambda outputs: torch.cat(outputs))
    result = owner(torch.zeros((2, 2560), dtype=torch.bfloat16), records=records, partitions=partitions)
    assert seen == [(prompt_metadata, None), (decode_metadata, records)]
    assert result.tolist() == [[1.0, 1.0], [2.0, 2.0]]


def test_gdn_finalization_drops_template_scratch_and_bindings():
    owner = object.__new__(gated_delta.PreparedGDN)
    owner.mode, owner.recovered = "prefill", None
    owner.partitions = owner.chunked = None
    scratch = torch.empty(64, dtype=torch.uint8)
    owner.conv_binding = SimpleNamespace(gathered_state=torch.empty((2, 8, 3), dtype=torch.bfloat16),
                                        request_ids=torch.empty(4, dtype=torch.int32))
    owner.recurrent_binding = SimpleNamespace(scratch=scratch)
    owner.scratch, owner.workspace, owner._binding_workspace, owner.output_encoding = object(), scratch, object(), object()
    owner._finalize_functional()
    assert owner.conv_binding is owner.recurrent_binding is owner.scratch is owner.workspace is None
    assert owner._binding_workspace is owner.output_encoding is None
    assert owner.recurrence_scratch == ((64,), (1,), torch.uint8)
    assert owner._functional_ready


def test_functional_sm103_prefill_preserves_gates_and_selected_operation(monkeypatch):
    owner = object.__new__(gdn_prefill_sm103.PreparedGdnPrefill)
    tokens, owner.key_heads, owner.value_heads = 2, 1, 3
    mixed = torch.arange(tokens * 5 * 128, dtype=torch.float32).reshape(tokens, -1).to(torch.bfloat16) / 1024
    owner.mixed, owner.a, owner.b = mixed, torch.full((tokens, 3), -.5, dtype=torch.bfloat16), torch.full((tokens, 3), .7, dtype=torch.bfloat16)
    owner.A_log, owner.dt_bias = torch.zeros(3), torch.full((3,), .125)
    owner.state = torch.empty((2, 3, 128, 128))
    owner.metadata = SimpleNamespace(query_start_loc=torch.tensor([0, 2], dtype=torch.int32))
    owner.indices, owner.plan = torch.tensor([1]), SimpleNamespace(_handle=73)
    owner.output = torch.zeros((tokens, 3, 128), dtype=torch.bfloat16)
    owner.sources = tuple(part.view(tokens, heads, 128) for part, heads in zip(
        mixed.split((128, 128, 384), dim=-1), (1, 1, 3), strict=True))
    owner.q, owner.k = torch.empty((tokens, 1, 128), dtype=torch.bfloat16), torch.empty((tokens, 1, 128), dtype=torch.bfloat16)
    owner.v = torch.empty_like(owner.output)
    owner.g, owner.beta = torch.empty((tokens, 3)), torch.empty((tokens, 3))
    owner.zero_args, owner.zero_grid, owner.zero_compiled = (), (1, 3, 1), object()
    calls = []

    def operation(q, k, v, g, beta, state, starts, indices, output, handle):
        calls.append((q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), state, starts, indices, handle))
        output.copy_(v)

    monkeypatch.setattr(gdn_prefill_sm103, "_tracing", lambda: True)
    monkeypatch.setattr(gdn_prefill_sm103, "launch_triton", lambda *args, **kwargs: None)
    monkeypatch.setattr(gdn_prefill_sm103, "register_finalizer", lambda session, callback: callback())
    monkeypatch.setattr(torch.ops.flare, "sm103_gdn_prefill", operation)
    expected = owner().clone()
    a, b = owner.a, owner.b
    owner.functional(session=None)
    assert owner.sources == () and owner.output is owner.q is owner.k is owner.v is owner.g is owner.beta is None
    actual = owner(mixed, a, b)
    assert torch.equal(actual, expected)
    assert all(torch.equal(first, second) for first, second in zip(calls[0][:5], calls[1][:5], strict=True))
    assert all(first is second for first, second in zip(calls[0][5:8], calls[1][5:8], strict=True))
    assert calls[0][8] == calls[1][8] == 73


def test_functional_state_partitions_preserve_row_maps_without_scratch_owners():
    from ignition.recipes.qwen38.mixed_state import StatePartitions

    packing = SimpleNamespace(prompt_indices=torch.tensor([0, 1]), decode_indices=torch.tensor([2, 3]),
        prompt_valid=torch.tensor([True, False]), decode_valid=torch.tensor([False, True]),
        is_prompt=torch.tensor([True, False, True, False]), prompt_source=torch.tensor([0, 0, 1, 0]),
        decode_source=torch.tensor([0, 0, 0, 1]))
    partitions = StatePartitions(packing, object(), object(), object(), object()).functional()
    source = torch.arange(12).reshape(4, 3)
    prompt, decode = (part.gather(source) for part in partitions.parts)
    assert torch.equal(prompt, torch.stack((source[0], torch.zeros_like(source[0]))))
    assert torch.equal(decode, torch.stack((torch.zeros_like(source[0]), source[3])))
    assert torch.equal(partitions.combine((prompt, decode)), torch.stack((source[0], torch.zeros_like(source[0]),
        torch.zeros_like(source[0]), source[3])))
    assert all(not hasattr(part, "scratch") and not hasattr(part, "packing") for part in partitions.parts)


def test_functional_gated_norm_uses_each_runtime_live_count(monkeypatch):
    monkeypatch.setattr(gated_norm, "warmup_triton", lambda *args, **kwargs: object())
    hidden = torch.ones((4, 2, 128), dtype=torch.bfloat16)
    gate = torch.ones_like(hidden)
    template_live = torch.zeros(1, dtype=torch.int32)
    reference = weakref.ref(template_live)
    norm = gated_norm.PreparedGatedRMSNorm(hidden, gate, torch.ones(128),
        torch.empty_like(hidden), template_live, eps=1e-6).functional(dynamic_live=True)
    del template_live
    gc.collect()
    assert reference() is None and norm.live is None

    def launch(kernel, compiled, args, grid, **options):
        live = int(args[4].item())
        args[3].copy_(args[0])
        args[3][live:].zero_()

    monkeypatch.setattr(gated_norm, "launch_triton", launch)
    for count in (4, 1, 0, 3):
        output = norm(hidden, gate, live=torch.tensor([count], dtype=torch.int32))
        assert torch.equal(output[:count], hidden[:count])
        assert not output[count:].any()
