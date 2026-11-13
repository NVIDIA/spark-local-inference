# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Active tuning workloads preserve serving inputs and recurrent layouts."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.fixtures import recurrent as gdn_fixtures
from ignition.preparation.fixtures.fused_moe import ExpertBenchmarkInputs
from ignition.preparation.fixtures.recurrent import (
    BenchmarkStateArena,
    RecurrentBenchmarkInputs,
)
from ignition.components.moe import RouterScratch


def _recurrent_args(mode, tokens, sequences):
    state = torch.empty_strided((6, 48, 128, 128),
        (48 * 128 * 128 + 128, 128 * 128, 128, 1), dtype=torch.float32).fill_(13)
    qkv = torch.full((tokens, 10240), 7, dtype=torch.bfloat16)
    args = dict(A_log=torch.arange(48, dtype=torch.float32) / 48,
        dt_bias=torch.zeros(48, dtype=torch.bfloat16), recurrent_state=state,
        num_seqs=torch.zeros(1, dtype=torch.int32), num_tokens=torch.zeros(1, dtype=torch.int32),
        output=torch.zeros(tokens, 48, 128, dtype=torch.bfloat16),
        a=torch.zeros(tokens, 48, dtype=torch.bfloat16), b=torch.zeros(tokens, 48, dtype=torch.bfloat16))
    bounds = torch.zeros(sequences + 1, dtype=torch.int32)
    if mode == 'prefill':
        q, k, v = qkv.split((2048, 2048, 6144), -1)
        args.update(q=q.view(tokens, 16, 128), k=k.view(tokens, 16, 128), v=v.view(tokens, 48, 128),
            cu_seqlens=bounds, initial_state_indices=torch.zeros(sequences, dtype=torch.int64),
            final_state_indices=torch.zeros(sequences, dtype=torch.int64),
            checkpoint_state_indices=torch.zeros(sequences, dtype=torch.int64),
            checkpoint_offsets=torch.zeros(sequences, dtype=torch.int32))
    else:
        args.update(mixed_qkv=qkv, z=torch.zeros(tokens, 48, 128, dtype=torch.bfloat16),
            norm_weight=torch.ones(128, dtype=torch.bfloat16), query_start_loc=bounds,
            num_accepted_tokens=torch.ones(sequences, dtype=torch.int32),
            state_indices=torch.zeros(sequences, 1, dtype=torch.int64))
    return args


@pytest.mark.parametrize('tokens,boundary', [(16, 0), (64, 64), (65, 64)])
def test_prefill_benchmark_has_live_fresh_and_continued_checkpoint_cases(tokens, boundary):
    original = _recurrent_args('prefill', tokens, 1)
    arena = BenchmarkStateArena(original['recurrent_state'])
    fixture = RecurrentBenchmarkInputs(original, mode='prefill', arena=arena)
    fixture.reset()
    fixture.producers[0]()
    assert fixture.args['num_tokens'].item() == tokens
    assert fixture.args['num_seqs'].item() == 1
    assert fixture.args['cu_seqlens'].tolist() == [0, tokens]
    assert fixture.args['initial_state_indices'].tolist() == [1]
    assert fixture.args['final_state_indices'].tolist() == [1]
    assert fixture.args['checkpoint_offsets'].tolist() == [boundary]
    assert fixture.args['checkpoint_state_indices'].tolist() == [2 if boundary else 0]
    assert torch.count_nonzero(fixture.state[1]) > 0
    assert not torch.count_nonzero(fixture.state[0])
    for name in ('q', 'k', 'v', 'recurrent_state'):
        assert fixture.args[name].stride() == original[name].stride()
        assert fixture.args[name].data_ptr() != original[name].data_ptr()
    assert fixture.args['q'].stride(0) == 10240
    qkv = [fixture.args[name] for name in ('q', 'k', 'v')]
    assert len({tensor.untyped_storage().data_ptr() for tensor in qkv}) == 1
    assert [tensor.storage_offset() for tensor in qkv] == [0, 2048, 4096]
    for name in ('k', 'v'):
        assert (fixture.args[name].data_ptr() - fixture.args['q'].data_ptr()
            == original[name].data_ptr() - original['q'].data_ptr())
    packed = fixture.packed_source.view(tokens, 10240)
    assert torch.equal(packed, fixture.packed_fixture.view_as(packed))
    assert torch.equal(fixture.args['q'].flatten(1), packed[:, :2048])
    assert torch.equal(fixture.args['k'].flatten(1), packed[:, 2048:4096])
    assert torch.equal(fixture.args['v'].flatten(1), packed[:, 4096:])
    assert fixture.args['A_log'] is original['A_log']
    fixture.state[1].fill_(float('nan'))
    fixture.state[2].fill_(123)
    fixture.reset()
    fixture.producers[1]()
    assert fixture.args['initial_state_indices'].tolist() == [0]
    assert torch.equal(fixture.state[:fixture.state_rows], fixture.initial_state)
    assert original['num_tokens'].item() == original['num_seqs'].item() == 0
    assert torch.all(original['recurrent_state'] == 13)
    assert torch.all(original['q'] == 7)


def test_decode_benchmark_distinct_live_slots_and_repeatable_state():
    original = _recurrent_args('decode', 4, 4)
    fixture = RecurrentBenchmarkInputs(original, mode='decode', arena=BenchmarkStateArena(original['recurrent_state']))
    fixture.reset()
    fixture.producers[0]()
    assert fixture.args['query_start_loc'].tolist() == [0, 1, 2, 3, 4]
    assert fixture.args['state_indices'].flatten().tolist() == [1, 2, 3, 4]
    assert fixture.args['num_accepted_tokens'].tolist() == [1] * 4
    assert fixture.args['num_seqs'].item() == fixture.args['num_tokens'].item() == 4
    before = fixture.state[:5].clone()
    fixture.state[:5].add_(42)
    fixture.reset()
    assert torch.equal(fixture.state[:5], before)
    assert original['state_indices'].count_nonzero() == 0


def test_convolution_and_recurrence_share_benchmark_arena_with_exact_byte_strides():
    recurrent = _recurrent_args('decode', 1, 1)['recurrent_state']
    slot_bytes = recurrent.stride(0) * recurrent.element_size()
    template = torch.empty_strided((6, 10240, 3), (slot_bytes // 2, 3, 1), dtype=torch.bfloat16).fill_(9)
    arena = BenchmarkStateArena(recurrent)
    args = dict(x=torch.zeros(16, 10240, dtype=torch.bfloat16),
        out=torch.zeros(16, 10240, dtype=torch.bfloat16),
        conv_weight=torch.zeros(10240, 4, dtype=torch.bfloat16), conv_state=template,
        query_start_loc=torch.zeros(2, dtype=torch.int32), state_slot_ids=torch.full((1,), -1, dtype=torch.int64),
        state_is_fresh=torch.zeros(1, dtype=torch.bool), num_accepted_tokens=torch.ones(1, dtype=torch.int32),
        num_seqs=torch.zeros(1, dtype=torch.int32), num_tokens=torch.zeros(1, dtype=torch.int32))
    fixture = RecurrentBenchmarkInputs(args, mode='prefill', arena=arena)
    assert fixture.state.data_ptr() == arena.view(recurrent).data_ptr()
    assert fixture.state.stride() == template.stride()
    assert arena.nbytes == 6 * slot_bytes
    fixture.reset()
    fixture.producers[0]()
    assert fixture.args['state_slot_ids'].tolist() == [1]
    assert fixture.args['state_is_fresh'].tolist() == [False]
    fixture.producers[1]()
    assert fixture.args['state_is_fresh'].tolist() == [True]
    assert torch.all(template == 9)
    storage = torch.empty(arena.nbytes, dtype=torch.uint8)
    borrowed = RecurrentBenchmarkInputs(args, mode='prefill',
        arena=BenchmarkStateArena(template, storage=storage))
    for left_producer, right_producer in zip(fixture.producers, borrowed.producers, strict=True):
        fixture.reset()
        borrowed.reset()
        left_producer()
        right_producer()
        for name, left in fixture.args.items():
            right = borrowed.args[name]
            assert left.shape == right.shape and left.stride() == right.stride()
            if name == 'conv_state':
                left, right = left[:fixture.state_rows], right[:borrowed.state_rows]
            assert torch.equal(left, right), name


@pytest.mark.parametrize('mode,tokens,sequences', [('prefill', 65, 1), ('decode', 4, 4)])
def test_borrowed_arena_preserves_exact_benchmark_operands_and_reset(mode, tokens, sequences):
    original = _recurrent_args(mode, tokens, sequences)
    template = original['recurrent_state']
    private = BenchmarkStateArena(template)
    storage = torch.full((private.nbytes,), 123, dtype=torch.uint8)
    borrowed = BenchmarkStateArena(template, storage=storage)
    assert borrowed.allocated_nbytes == 0 and borrowed.borrowed_nbytes == private.nbytes
    assert private.allocated_nbytes == private.nbytes and private.borrowed_nbytes == 0
    assert borrowed.view(template).data_ptr() == storage.data_ptr()
    left = RecurrentBenchmarkInputs(original, mode=mode, arena=private)
    right = RecurrentBenchmarkInputs(original, mode=mode, arena=borrowed)
    assert torch.equal(left.initial_state, right.initial_state)
    for left_producer, right_producer in zip(left.producers, right.producers, strict=True):
        left.reset()
        right.state[:right.state_rows].fill_(float('nan'))
        right.reset()
        left_producer()
        right_producer()
        for name in left.args:
            a, b = left.args[name], right.args[name]
            assert a.shape == b.shape and a.stride() == b.stride() and a.dtype == b.dtype
            if name == 'recurrent_state':
                a, b = a[:left.state_rows], b[:right.state_rows]
            assert torch.equal(a, b), name
    assert torch.all(original['recurrent_state'] == 13)


@pytest.mark.parametrize('kind', ['dtype', 'extent', 'stride', 'device'])
def test_borrowed_arena_rejects_storage_contract_mismatch(kind):
    template = torch.empty_strided((2, 4), (8, 1))
    size = 2 * 8 * 4
    storage = {'dtype': lambda: torch.empty(size // 4),
        'extent': lambda: torch.empty(size - 1, dtype=torch.uint8),
        'stride': lambda: torch.empty(size * 2, dtype=torch.uint8)[::2],
        'device': lambda: torch.empty(size, dtype=torch.uint8, device='meta')}[kind]()
    with pytest.raises(ValueError, match='exact extent and device'):
        BenchmarkStateArena(template, storage=storage)


def test_gdn_candidate_cohort_shares_fixture_and_reserved_scratch_with_per_trial_reset(monkeypatch):
    from ignition.components import recurrent

    args = _recurrent_args('prefill', 65, 1)
    resource_cache = {}
    callbacks = []
    workspace = torch.empty(256, dtype=torch.uint8)
    session = SimpleNamespace(prepare=lambda requests: callbacks.extend(requests))
    monkeypatch.setattr(recurrent, 'register_finalizer', lambda *args, **kwargs: None)

    def resource(session, key, factory):
        if key not in resource_cache:
            resource_cache[key] = factory()
        return resource_cache[key]

    monkeypatch.setattr(gdn_fixtures, 'benchmark_resource', resource)
    specs = lambda size: (SimpleNamespace(name='scratch', shape=(size,), nbytes=size, dtype=torch.uint8,
                                          device=torch.device('cpu')),)
    plan = SimpleNamespace(scratch_specs=lambda: specs(16))
    plan.request = lambda **kwargs: SimpleNamespace(**kwargs)
    recurrent.prepare_sequence_binding(plan, args, session=session, name='prefill65.layer0',
        public_bind=lambda plan, **kwargs: SimpleNamespace(**kwargs), mode='prefill',
        set_binding=lambda value: None, workspace=workspace)
    runs = []

    def candidate(index, size):
        def run(binding):
            runs.append(index)
            binding.recurrent_state[1].add_(index + 1)
            binding.output.fill_(index + 1)
            binding.scratch.fill_(index + 1)
        return SimpleNamespace(layout=SimpleNamespace(scratch_specs=lambda: specs(size)),
            bind=lambda **kwargs: SimpleNamespace(**kwargs), run=run)

    calls = [callbacks[0].benchmark_call(candidate(index, size))
             for index, size in enumerate((32, 128, 256))]
    operands = calls[0].owners[0]
    assert len(resource_cache) == 2
    assert resource_cache[('delta_inputs', 'prefill65.layer0')] is operands
    assert all(call.owners[0] is operands and call.owners[1] is workspace for call in calls)
    assert all(call.output is operands.args['output'] for call in calls)
    assert all(call.owners[2].q is operands.args['q'] for call in calls)
    assert all(call.owners[2].scratch is workspace for call in calls)
    for index in (0, 1, 2, 0):
        call = calls[index]
        for produce in call.benchmark_producers:
            call.reset()
            produce()
            assert torch.equal(operands.state[:operands.state_rows], operands.initial_state)
            assert not call.output.count_nonzero()
            assert torch.equal(operands.packed_source, operands.packed_fixture)
            call.run()
            assert torch.all(call.output == index + 1)
        call.restore()
        assert torch.equal(operands.state[:operands.state_rows], operands.initial_state)
    assert runs == [0, 0, 1, 1, 2, 2, 0, 0]
    assert torch.all(args['recurrent_state'] == 13) and torch.all(args['q'] == 7)

    owned = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
             for value in operands._owned_tensors if value is not None}
    assert operands.nbytes == sum(owned.values())
    assert operands.arena.storage.data_ptr() not in owned
    assert all(args[name].untyped_storage().data_ptr() not in owned for name in ('A_log', 'dt_bias'))
    assert sum(value is operands for value in resource_cache.values()) == 1


@pytest.mark.parametrize("tokens", [1, 4, 16, 128])
@pytest.mark.parametrize("scaling_factor", [1.0, 2.5])
def test_moe_benchmark_uses_shared_b12x_routes_without_mutating_serving_operands(tokens, scaling_factor):
    from b12x.moe.fused_moe.workloads import make_tuning_routes

    source = torch.zeros(tokens, 32, dtype=torch.bfloat16)
    plan = SimpleNamespace(query=SimpleNamespace(routed_scaling_factor=scaling_factor))
    fixture = ExpertBenchmarkInputs(source, experts=512, top_k=10,
        allocate_router=RouterScratch.allocate, router_plan=plan)
    expected = make_tuning_routes(tokens, 10, 512, device="cpu")
    expected_weights = torch.softmax(torch.arange(10, dtype=torch.float32) / 8, -1).bfloat16().float() * scaling_factor
    assert len(fixture.producers) == len(expected) == 6
    for produce, routes in zip(fixture.producers, expected, strict=True):
        produce()
        ids = fixture.router.ids
        assert torch.equal(ids, routes)
        assert bool(((ids >= 0) & (ids < 512)).all())
        assert all(row.unique().numel() == 10 for row in ids)
        assert torch.count_nonzero(fixture.source) > 0
        assert torch.equal(fixture.router.weights, expected_weights.expand_as(fixture.router.weights))
    assert not source.count_nonzero()
    assert fixture.source.data_ptr() != source.data_ptr()
