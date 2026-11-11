# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Functional wrappers expose per-call metadata, outputs, and asynchronous lifetimes."""

from types import SimpleNamespace
import weakref

import pytest
import torch


def plan():
    return SimpleNamespace(handle=7, scratch_specs=lambda: (SimpleNamespace(shape=(64,), dtype=torch.uint8, nbytes=64, device=torch.device("cpu")),))


def test_varlen_releases_templates_and_preserves_strided_output(monkeypatch):
    from ignition.components import varlen_attention as module
    op = object.__new__(module.PreparedVarlenAttention)
    template = torch.zeros(4, 2, 3)
    reference = weakref.ref(template)
    op.q = op.k = op.v = template
    op.cu = torch.tensor([0, 4], dtype=torch.int32)
    op.out = torch.empty_strided((4, 2, 3), (8, 3, 1))
    op.workspace = op.scratch = torch.zeros(64, dtype=torch.uint8)
    op.plan, op.sink, op.max_seqlen = plan(), None, 4
    op.functional(session=None)
    del template
    assert reference() is None
    seen = []

    def execute(q, k, v, cu, *, out, scratch, **kwargs):
        seen.append((cu.clone(), scratch))
        out[:2].copy_((q + k + v)[:2])

    monkeypatch.setattr(module.varlen, "run_out", execute)
    q = torch.ones(4, 2, 3)
    first = op(q, q * 2, q * 3, torch.tensor([0, 2], dtype=torch.int32))
    second = op(q * 4, q * 2, q, torch.tensor([0, 1], dtype=torch.int32))
    assert first.stride() == second.stride() == (8, 3, 1)
    assert torch.equal(first[:2], torch.full((2, 2, 3), 6.))
    assert torch.equal(second[:2], torch.full((2, 2, 3), 7.))
    assert not first[2:].count_nonzero() and not second[2:].count_nonzero()
    assert first.data_ptr() != second.data_ptr()
    assert seen[0][1].data_ptr() != seen[1][1].data_ptr()
    op._finalize_functional()
    assert op.q is op.out is op.scratch is None


def test_dense_attention_runtime_cache_lengths_and_lse(monkeypatch):
    from ignition.components import dense_mla as module
    op = object.__new__(module.PreparedDenseMLA)
    template = torch.zeros(3, 2, 4)
    op.arguments = dict(q=template, kv_cache=template, output=template,
                        page_table=template, cache_seqlens=template, cu_seqlens_q=template,
                        kv_scale=None, q_scale=None)
    op.output, op.lse, op.plan, op.scale = template, torch.zeros(3, 2), plan(), .5
    op.functional(session=None, cache=None)
    assert op.arguments == dict(kv_cache=None, kv_scale=None, q_scale=None)
    calls = []

    def attend(q, cache, output, pages, lengths, starts, scratch, handle, scale, lse, *scales):
        calls.append((cache, lengths, starts, scratch))
        output[0].copy_(q[0] + cache[0])
        lse[0].fill_(lengths[0])

    monkeypatch.setattr(module, "_attend", attend)
    q, cache = torch.ones(3, 2, 4), torch.full((3, 2, 4), 2.)
    lengths, starts = torch.tensor([5], dtype=torch.int32), torch.tensor([0, 1], dtype=torch.int32)
    out, lse = op(q, torch.zeros(1, 1, dtype=torch.int32), lengths, starts, cache=cache)
    assert torch.equal(out[0], torch.full((2, 4), 3.))
    assert not out[1:].count_nonzero()
    assert torch.equal(lse[0], torch.full((2,), 5.))
    assert calls[0][:3] == (cache, lengths, starts)
    op._finalize_functional()
    assert op.output is op.lse is op.scratch is None


def test_window_attention_consumes_runtime_metadata_and_proposals(monkeypatch):
    from ignition.components import window_attention as module
    op = object.__new__(module.PreparedWindowAttention)
    q = torch.ones(2, 1, 4)
    history = torch.zeros(3, 8, 1, 4)
    op.arguments = (q, q, q, history, history, None, None, None, None, None, q)
    op.key = op.value = op.proposal = op.output_gate = None
    op.plan, op.descale = plan(), torch.ones(1)
    op.functional(session=None)
    m = SimpleNamespace(requests=torch.tensor([1, 0]), positions=torch.tensor([3, 7]),
                        starts=torch.tensor([5, 2]), query_start_loc=torch.tensor([0, 1, 2]),
                        state_rows=torch.tensor([2, 1]))
    proposal = torch.zeros(2, 3, 1, 4), torch.zeros(2, 3, 1, 4), 1
    seen = []

    def execute(*args):
        seen.append(args)
        args[10].copy_(args[0])

    monkeypatch.setattr(module, "_execute", execute)
    out = op(q, q * 2, q * 3, m, proposal=proposal)
    torch.testing.assert_close(out, q)
    torch.testing.assert_close(proposal[0][:, 1], q * 2)
    torch.testing.assert_close(proposal[1][:, 1], q * 3)
    assert seen[0][5] is m.requests and seen[0][9] is m.state_rows
    assert seen[0][1].shape == (6, 1, 4)
    assert op.arguments is op.proposal is None


def test_projection_pending_keeps_every_operand_through_join(monkeypatch):
    from ignition.components import projection_overlap as module
    op = object.__new__(module.ConcurrentProjectionChain)
    op.projections = None
    op.output_layouts = (((2, 4), (4, 1), torch.float32),)
    op.workspace_nbytes = 32
    op.encoded_layouts = (((2, 4), torch.float32), ((2,), torch.uint8), ((2,), torch.uint8))
    op.encoded_weights = torch.ones(4, 4), torch.ones(1, dtype=torch.uint8)
    op.quantizer_handle, op.dense_handle, op.stream_handle = None, 3, 9
    calls = []
    monkeypatch.setattr(module, "_start_preencoded", lambda *args: calls.append(("start", args)))
    monkeypatch.setattr(module, "_join_encoded", lambda *args: calls.append(("join", args)))
    source, dependency = torch.ones(2, 4), torch.zeros(2, 4)
    bases = torch.ones(2, 4), torch.ones(2, dtype=torch.uint8), torch.ones(2, dtype=torch.uint8)
    pending = op.start(source, encoded_bases=bases)
    output = op.finish(pending, dependency)
    assert calls[1][1][0] is source and calls[1][1][1] is dependency
    assert calls[1][1][2] is output
    assert all(actual is expected for actual, expected in zip(calls[1][1][3:6], bases))
    assert calls[1][1][6] is pending[2]
    assert op.projections is None


@pytest.mark.parametrize("with_angles", [False, True])
def test_compressor_join_carries_runtime_metadata_and_angles(monkeypatch, with_angles):
    from ignition.components import compressor_overlap as module
    op = object.__new__(module.ConcurrentGatedCompression)
    op.outputs = None
    op.output_layouts = (((2, 4), (4, 1), torch.float32),)
    op.weights = op.history = op.caches = op.plans = []
    op.frequencies, op.ratio, op.eps, op.stream_handle = torch.ones(4), 4, 1e-6, 9
    m = SimpleNamespace(**{key: torch.ones(2, dtype=torch.int64)
        for key in ("requests", "positions", "starts", "query_start_loc", "state_rows")})
    positions, slots = torch.ones(2), torch.ones(2)
    angles = torch.ones(2, 4) if with_angles else None
    calls = []
    monkeypatch.setattr(module, "_start", lambda *args: calls.append(args))
    monkeypatch.setattr(module, "_join", lambda *args: calls.append(args))
    pending = op.start(torch.ones(2, 4), m, positions, slots, angles=angles)
    output = op.finish(pending, [torch.ones(2, 4)])
    borrowed = calls[0][8]
    assert calls[1][0] is borrowed
    assert calls[1][3] is output
    originals = (calls[0][0], m.requests, m.positions, m.starts,
        m.query_start_loc, m.state_rows, positions, slots)
    if with_angles:
        originals += (angles,)
    for original, retained in zip(originals, borrowed, strict=True):
        assert retained.shape == original.shape and retained.dtype == original.dtype
        assert retained.data_ptr() != original.data_ptr()


def test_varlen_gather_source_alias_is_explicit(monkeypatch):
    from ignition.components import collectives as module
    op = object.__new__(module.PreparedVarlenGather)
    op.source = op.output = op.counts = torch.zeros(1)
    op.local_rank, op.handle = 1, 31
    op.functional(session=None)
    calls = []
    monkeypatch.setattr(module, "_all_gather_varlen", lambda *args: calls.append(args))
    output = torch.arange(24).view(3, 8)
    counts = torch.tensor([16, 32, 16], dtype=torch.int64)
    assert op(output, counts) is output
    assert calls[0][0].data_ptr() == output[1].data_ptr()
    assert calls[0][1] is counts and calls[0][2] is output
    assert op.source is op.counts is op.output is None


@pytest.mark.parametrize("compiled", [False, True])
def test_fused_reduction_runtime_addends_and_residual_mutation(monkeypatch, compiled):
    from ignition.components import collectives as module
    op = object.__new__(module.PreparedAllReduceRMSNorm)
    op.output = torch.empty(2, 4)
    op.residual = op.input_addend = op.group_weights = op.group_ids = torch.zeros(2, 4)
    op.weight, op.handle = torch.ones(4), 13
    op.functional(session=None)

    def reduce(source, residual, weight, output, handle, addend, weights, ids):
        residual.add_(source + addend)
        output.copy_(residual * weight)

    monkeypatch.setattr(module, "_all_reduce_rmsnorm", reduce)
    def invoke(source, residual, addend):
        return op(source, residual, input_addend=addend)
    fn = torch.compile(invoke, backend="eager", fullgraph=True) if compiled else invoke
    residual = torch.ones(2, 4)
    output = fn(torch.full((2, 4), 2.), residual, torch.full((2, 4), 3.))
    torch.testing.assert_close(output, torch.full((2, 4), 6.))
    torch.testing.assert_close(residual, output)
    assert output.data_ptr() != residual.data_ptr()


def test_selected_attention_preserves_noncontiguous_output_layout(monkeypatch):
    from ignition.components import selected_attention as module
    op = object.__new__(module.PreparedSelectedAttention)
    q = torch.ones(3, 2, 4)
    output = torch.empty_strided(q.shape, (24, 4, 1))
    op.arguments = (q, q, q, None, None, None, None, output)
    op.output_gate, op.descale, op.plan = None, torch.ones(1), plan()
    op.functional(session=None)
    monkeypatch.setattr(module, "_execute", lambda *args: args[7].copy_(args[0]))
    result = op(q, torch.zeros(1, 1), torch.zeros(3), torch.zeros(3), torch.zeros(3, 1))
    assert result.stride() == output.stride()
    torch.testing.assert_close(result, q)


def test_dense_declaration_survives_repeated_functional_finalization(monkeypatch):
    from ignition.components import dense_mla as module
    callbacks, requests = [], []
    selected = plan()
    selected.request = lambda **kwargs: requests.append(kwargs) or kwargs
    monkeypatch.setattr(module.dense_mla, "plan", lambda *args, **kwargs: selected)
    monkeypatch.setattr(module.dense_mla, "invocation_from_tensors", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "register_finalizer", lambda session, callback, **kwargs: callbacks.append(callback))
    monkeypatch.setattr(module.dense_mla, "bind", lambda *args, **kwargs: None)
    query = torch.ones(3, 2, 4)
    output = torch.empty_strided(query.shape, (24, 4, 1))
    cache = torch.zeros(2, 4, 4)
    pages, lengths, starts = torch.zeros(1, 1, dtype=torch.int32), torch.tensor([3], dtype=torch.int32), torch.tensor([0, 3], dtype=torch.int32)
    op = module.PreparedDenseMLA(query, cache, output, pages, lengths, starts,
        caps=SimpleNamespace(max_cache_tokens=8), scale=.5, session=None, name="attention", workspace=None)
    op.functional(session=None, cache=None)
    for _ in range(2):
        for callback in callbacks:
            callback()
    observed = []
    state = SimpleNamespace(scratch_specs=selected.scratch_specs,
        bind=lambda **kwargs: observed.append(kwargs) or kwargs,
        prime=lambda _: None, run=lambda binding: None)
    requests[0]["prepare_call"](state).run()
    assert observed[0]["q"] is query and observed[0]["kv_cache"] is cache
    assert op.arguments["kv_cache"] is None and op.output is None
    assert op.output_layout[1] == (24, 4, 1)


@pytest.mark.parametrize("encoded", [False, True])
def test_projection_chain_rebinds_contextual_plan_before_runtime_launch(monkeypatch, encoded):
    from types import SimpleNamespace
    from ignition.components import projection_overlap as module

    def plan(handle, size):
        return SimpleNamespace(handle=handle, memory_requirements=lambda: SimpleNamespace(scratch_nbytes=size))

    owner = SimpleNamespace(plan=plan(3, 32), quantizer_plan=None)
    chain = object.__new__(module.ConcurrentProjectionChain)
    chain.plan_owners = (owner,)
    chain.output_layouts = (((2, 4), (4, 1), torch.float32),)
    chain.encoded_layouts = (((2, 4), torch.float32), ((2,), torch.uint8), ((2,), torch.uint8)) if encoded else None
    chain.encoded_weights = torch.ones(4, 4), torch.ones(1, dtype=torch.uint8)
    chain.weights, chain.scales, chain.gains, chain.activation_scales, chain.scale_kinds = [], [], [], [], []
    chain.stream_handle = 9
    calls = []
    monkeypatch.setattr(module, "_start_preencoded", lambda *args: calls.append((args[-2], args[-3].nbytes)))
    monkeypatch.setattr(module, "_start", lambda *args: calls.append((args[-2][0], args[-3].nbytes)))
    source = torch.ones(2, 4)
    bases = torch.ones(2, 4), torch.ones(2, dtype=torch.uint8), torch.ones(2, dtype=torch.uint8)
    for handle, size in ((3, 32), (7, 96)):
        owner.plan = plan(handle, size)
        chain._finalize_functional()
        chain.start(source, encoded_bases=bases if encoded else None)
        assert calls[-1] == (handle, size)
