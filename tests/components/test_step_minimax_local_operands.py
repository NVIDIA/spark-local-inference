# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Invocation operands and output storage for paged attention and normalization."""

from types import SimpleNamespace

import torch

from ignition.components import paged_attention, paged_schedule, rms_norm


def test_kv_writer_accepts_runtime_cache_views_without_retaining_examples(monkeypatch):
    example = torch.full((3, 2, 1, 4), -7.)
    prepared = paged_attention.PreparedPagedKVWriter.__new__(paged_attention.PreparedPagedKVWriter)
    prepared.plan = object()
    prepared.arguments = dict(key=torch.zeros(2, 1, 4), value=torch.zeros(2, 1, 4),
        k_cache=example, v_cache=example, slot_mapping=torch.zeros(2, dtype=torch.int64),
        k_scale=None, v_scale=None, dummy_scale=torch.zeros(1))
    operation = prepared.functional(k_cache=None, v_cache=None)
    assert operation.arguments == dict(k_cache=None, v_cache=None, k_scale=None, v_scale=None)
    seen = []

    def write(plan, *, key, value, k_cache, v_cache, slot_mapping, dummy_scale, **kwargs):
        assert plan is prepared.plan and dummy_scale.shape == (1,)
        seen.append((k_cache, v_cache, slot_mapping))
        k_cache.flatten(0, 1).index_copy_(0, slot_mapping, key)
        v_cache.flatten(0, 1).index_copy_(0, slot_mapping, value)

    monkeypatch.setattr(paged_attention.paged_kv_write, "run", write)
    key, value = torch.arange(8.).view(2, 1, 4), torch.arange(8., 16.).view(2, 1, 4)
    keys, values = torch.zeros_like(example), torch.zeros_like(example)
    slots = torch.tensor([0, 3])
    operation(key, value, slots, k_cache=keys, v_cache=values)
    assert seen[0][0] is keys and seen[0][1] is values and seen[0][2] is slots
    torch.testing.assert_close(keys.flatten(0, 1)[slots], key)
    torch.testing.assert_close(values.flatten(0, 1)[slots], value)
    assert torch.all(example == -7)
    bound = prepared.functional()
    assert bound.arguments["k_cache"] is example


def test_shared_schedules_release_examples_after_selected_slots_are_bound(monkeypatch):
    finalizers, calls = [], []
    monkeypatch.setattr(paged_schedule, "register_finalizer", lambda session, fn: finalizers.append(fn))
    monkeypatch.setattr(paged_schedule, "schedule_counts", lambda caps: (2, 4))
    monkeypatch.setattr(paged_schedule, "schedule_nbytes", lambda caps: 32)
    monkeypatch.setattr(paged_schedule, "_schedule", lambda plan:
        SimpleNamespace(key="same", views=lambda storage: None))
    lengths, cu = torch.zeros(2, dtype=torch.int32), torch.zeros(3, dtype=torch.int32)
    schedules = paged_schedule.PreparedPagedSchedules(lengths, cu, 2)
    components = [SimpleNamespace(plan=SimpleNamespace(handle=index + 11),
        arguments=dict(cache_seqlens=lengths, cu_seqlens_q=cu)) for index in range(2)]
    for component in components:
        schedules.register(component, None)
    schedules.finalize(None)
    schedules.functional(session=None)
    for finalize in finalizers:
        finalize()
    assert [component.schedule_slot for component in components] == [0, 0]
    assert all(component.schedule_storage is None for component in components)
    assert schedules.storage is schedules.lengths is schedules.cu is None
    assert schedules.components == [] and schedules.groups == ()
    monkeypatch.setattr(paged_schedule, "_update_schedules", lambda lengths, cu, storage, handles:
        calls.append((lengths, cu, storage, handles)))
    runtime_lengths, runtime_cu = lengths + 5, cu + 1
    first, second = schedules(runtime_lengths, runtime_cu), schedules(runtime_lengths, runtime_cu)
    assert first.shape == (2, 32) and first.dtype == torch.uint8 and not first.any()
    assert first.data_ptr() != second.data_ptr()
    assert calls[0][0] is runtime_lengths and calls[0][1] is runtime_cu
    assert calls[0][3] == [11, 12]


def test_compact_attention_uses_runtime_selected_blocks_and_schedules(monkeypatch):
    query = torch.arange(12.).view(3, 1, 4)
    selected = torch.arange(48, dtype=torch.int32).view(1, 3, 16)
    cu = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
    schedules = torch.zeros(1, 32, dtype=torch.uint8)
    prepared = SimpleNamespace(plan=SimpleNamespace(handle=71), tma=None, decode=False, window=-1,
        arguments=dict(q=query, output=torch.zeros_like(query), k_cache=torch.zeros(1), v_cache=torch.zeros(1),
            page_table=torch.zeros(3, 1, dtype=torch.int32), cache_seqlens=None, cu_seqlens_q=cu,
            attention_sink_bias=None, k_descale=None, v_descale=None),
        selected_blocks=selected, compact_decode=(query, query, cu, selected),
        schedule_storage=None, schedule_slot=0, output_gate=None)
    attention = paged_attention._FunctionalPagedAttention(prepared)
    attention.compact_decode = (None, None)
    attention.selected_blocks = None
    attention.scratch_layout = ()
    seen = []

    def attend(q, key, value, pages, lengths, compact_cu, sink, output, scratch,
               k_descale, v_descale, blocks, handle, decode, window, storage, slot, gate, seeds, live, lookup):
        assert seeds == live == [] and lookup is None
        seen.append((q.clone(), blocks.clone(), storage, compact_cu))
        output.copy_(q)

    monkeypatch.setattr(paged_attention, "_attend", attend)
    output = attention(query, cache_seqlens=torch.tensor([1, 0, 1], dtype=torch.int32),
        cu_seqlens_q=cu, selected_blocks=selected, schedule_storage=schedules)
    assert seen[0][2] is schedules and seen[0][3] is cu
    assert torch.equal(seen[0][0], torch.stack((query[0], torch.zeros_like(query[0]), query[1])))
    assert torch.equal(seen[0][1][0, 1], torch.full((16,), -1, dtype=torch.int32))
    assert torch.equal(seen[0][1][0, 2], selected[0, 1])
    assert torch.equal(output[:2], query[:2]) and not output[2].any()


def test_indexed_and_paired_norm_adapters_pass_runtime_rows_to_distinct_outputs(monkeypatch):
    source = torch.arange(24.).view(3, 8)[:, :4].bfloat16()
    weight = torch.arange(1., 5.)
    eps, plan = 1e-6, SimpleNamespace(handle=73)
    seen = []

    def normalize(source, weight):
        values = source.float()
        return (values * (values.square().mean(-1, keepdim=True) + eps).rsqrt() * weight).bfloat16()

    def indexed(source, weight, indices, output, epsilon, handle):
        seen.append((source, indices, handle))
        assert epsilon == eps
        output.copy_(normalize(source[indices], weight))

    def paired(left, right, weights, output, epsilon, handle):
        assert epsilon == eps and handle == plan.handle
        output.copy_(torch.cat((normalize(left, weights[:4]), normalize(right, weights[4:])), dim=1))

    monkeypatch.setattr(torch.ops.b12x, "hyperconnection_indexed_rmsnorm", indexed)
    monkeypatch.setattr(torch.ops.b12x, "hyperconnection_paired_rmsnorm", paired)
    indices = torch.tensor([2, 0])
    operation = rms_norm._FunctionalIndexedRMSNorm(weight, plan, eps)
    first, second = operation(source, indices), operation(source, indices)
    assert seen[0][0] is source and seen[0][1] is indices and seen[0][2] == plan.handle
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, normalize(source[indices], weight), rtol=0, atol=0)
    operation = rms_norm._FunctionalPairedRMSNorm(torch.cat((weight, weight + 1)), plan, eps)
    first, second = operation(source, source + 2), operation(source, source + 2)
    torch.testing.assert_close(first, torch.cat((normalize(source, weight), normalize(source + 2, weight + 1)), 1), rtol=0, atol=0)
    assert first.data_ptr() != second.data_ptr()
