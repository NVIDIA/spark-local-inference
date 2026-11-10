# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for metadata packing and prepared token operand routing."""

from dataclasses import fields
from types import SimpleNamespace
import weakref

import pytest
import torch

from ignition.components import paged_batch, token_state
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.sampling import DeviceRequestState, ERROR, RUNNING
from ignition.preparation.triton import PreparedTriton


def _assert_metadata_equal(actual, expected):
    for field in fields(Metadata):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert left == right


@pytest.mark.parametrize("lengths,capacity", [
    ([3], 5), ([0], 5), ([3, 1], 7), ([0, 4], 7),
    ([1, 0, 3, 2], 9), ([0, 0, 0, 0], 9),
])
def test_pack_matches_bound_metadata(lengths, capacity):
    rows, width = len(lengths), 4
    starts = torch.arange(rows, dtype=torch.int64) * 128 + 126
    lengths = torch.tensor(lengths, dtype=torch.int32)
    state_rows = torch.arange(1, rows + 1)
    pages = torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2)
    tokens = torch.arange(rows * width).reshape(rows, width)
    live_rows = torch.tensor([rows], dtype=torch.int32)
    expected = Metadata.allocate(capacity, rows, 2, "cpu", samples=rows * width, max_query_tokens=width)
    expected.set(starts, lengths, state_rows, pages, tokens, live_rows)
    actual = Metadata.pack(starts, lengths, state_rows, pages, tokens, live_rows,
                           token_capacity=capacity, sample_count=rows * width, max_query_tokens=width)
    _assert_metadata_equal(actual, expected)
    starts.zero_()
    lengths.zero_()
    pages.fill_(-1)
    _assert_metadata_equal(actual, expected)
    if rows == 1:
        assert actual.token_page_table.untyped_storage() is actual.page_table.untyped_storage()
        assert actual.token_page_table.stride(0) == 0


@pytest.fixture
def cpu_token_kernels(monkeypatch):
    """Replace CUDA launches while exercising prepared operand ownership and routing."""
    calls = []

    class Compiled:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            def launch(*args):
                calls.append((self.kernel, args))
                if self.kernel is not token_state._gather:
                    return
                (slots, epochs, live, generation, status, pending, consumed, history,
                 packed_pending, positions, packed_history, active, rows, *_) = args
                safe = slots.long().clamp(0, generation.numel() - 1)
                valid = (torch.arange(rows) < live) & (slots >= 0) & (slots < generation.numel())
                valid &= (generation[safe] == epochs) & (status[safe] == RUNNING)
                packed_pending.copy_(torch.where(valid, pending[safe], -1))
                positions.copy_(torch.where(valid, consumed[safe], 0))
                active.copy_(valid)
                if history is not None:
                    packed_history.copy_(torch.where(valid[:, None], history[safe], args[-2]))
            return launch

    def compile_kernel(kernel, args, grid):
        return PreparedTriton(kernel, Compiled(kernel), args, grid, {})

    def make_state(state, tokens, cu, slots, epochs, live, *, history=None, history_pad_id=0, buffers=None):
        owner = token_state.PreparedTokenState.__new__(token_state.PreparedTokenState)
        rows = slots.numel()
        owner.state, owner._rows, owner._token_capacity = state, rows, tokens.numel()
        owner.pending_tokens = torch.empty(rows, dtype=torch.int64)
        owner.positions = torch.empty_like(owner.pending_tokens)
        owner.active = torch.empty(rows, dtype=torch.int32)
        owner.history = None if history is None else torch.empty((rows, 2), dtype=torch.int64)
        owner._gather_args = (slots, epochs, live, state.generation, state.status,
            state.pending_token, state.consumed_tokens, history, owner.pending_tokens, owner.positions,
            owner.history, owner.active, rows, state.consumed_tokens.stride(0),
            0 if history is None else history.stride(0), history is not None, history_pad_id, state.capacity)
        owner._consume_args = (tokens, cu, slots, epochs, live, state.generation, state.status,
            state.error_code, state.pending_token, state.consumed_tokens, history, rows, tokens.numel(),
            state.consumed_tokens.stride(0), 0 if history is None else history.stride(0),
            history is not None, state.capacity)
        owner._prepared = False
        return owner

    monkeypatch.setattr(token_state, "_compile", compile_kernel)
    monkeypatch.setattr(paged_batch, "PreparedTokenState", make_state)
    return make_state, calls


def _make_batch(mode):
    requests = DeviceRequestState.allocate("cpu", vocab_size=32, capacity=4)
    requests.generation.fill_(7)
    requests.status.fill_(RUNNING)
    requests.consumed_tokens.copy_(torch.tensor([0, 127, 255, 3]))
    requests.pending_token.copy_(torch.tensor([11, 12, 13, 14]) if mode == "decode"
                                 else torch.full((4,), -1))
    rows, tokens = (4, 4) if mode == "decode" else (1, 6)
    inputs = BatchInputs.allocate(tokens, 4, "cpu", capacity=4)
    inputs.row_slot_ids[:rows].copy_(torch.tensor([1, 3, 0, 2])[:rows])
    inputs.row_generations[:rows].fill_(7)
    inputs.live_rows.fill_(rows)
    inputs.request_state_rows[:, 0].copy_(torch.arange(1, 5))
    inputs.page_tables.copy_(torch.arange(16, dtype=torch.int32).reshape(4, 4))
    inputs.query_lengths.fill_(3 if mode == "prefill" else 1)
    inputs.token_ids.copy_(torch.arange(tokens) + 1)
    metadata = Metadata.allocate(tokens, rows, 4, "cpu")
    return PreparedBatch(inputs, requests, metadata, mode=mode,
                         shard=SimpleNamespace(world_size=2),
                         pool=SimpleNamespace(memory_capacity=torch.tensor([5, 16])))


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("fault", [None, "pending", "duplicate", "stale", "missing_rank", "pages"])
def test_batch_validation_and_packing_preserve_slot_contract(cpu_token_kernels, mode, fault):
    batch = _make_batch(mode)
    if fault == "pending":
        batch.requests.pending_token[1] = 99
    elif fault == "duplicate" and mode == "decode":
        batch.inputs.row_slot_ids[1] = 1
    elif fault == "stale":
        batch.inputs.row_generations[0] = 6
    elif fault == "pages":
        batch.inputs.page_tables[1, 0] = -1
    template = weakref.ref(batch.metadata)
    batch.functional()
    assert template() is None
    assert batch.validation_input is batch.validation_sum is None
    assert batch.token_state.pending_tokens is batch.token_state.positions is None
    assert batch.token_state._gather.args is batch.token_state._consume.args is None

    slots = [1, 3, 0, 2] if mode == "decode" else [1]
    if fault == "duplicate" and mode == "decode":
        slots[1] = 1
    resident = [fault != "stale" or index != 0 for index in range(len(slots))]
    invalid = fault in ("pending", "pages") or (fault == "duplicate" and mode == "decode")
    expected_present = torch.tensor([any(live and slot == index for slot, live in zip(slots, resident))
                                     for index in range(4)])
    expected_bad = torch.tensor([False, invalid, False, False])
    expected_validation = torch.cat((expected_bad, expected_present)).to(torch.bfloat16).view(1, 8)
    validation, present = batch.validate()
    torch.testing.assert_close(validation, expected_validation, rtol=0, atol=0)
    torch.testing.assert_close(present, expected_present, rtol=0, atol=0)
    totals = validation * 2
    if fault == "missing_rank":
        totals[0, batch.requests.capacity + 1] = 1
    actual = batch.begin(totals, present=present)
    failed = invalid or fault == "missing_rank"
    expected_status = torch.full((4,), RUNNING, dtype=batch.requests.status.dtype)
    expected_errors = torch.zeros(4, dtype=batch.requests.error_code.dtype)
    if failed:
        expected_status[1], expected_errors[1] = ERROR, 32
    torch.testing.assert_close(batch.requests.status, expected_status, rtol=0, atol=0)
    torch.testing.assert_close(batch.requests.error_code, expected_errors, rtol=0, atol=0)

    active = torch.tensor([live and not (failed and slot == 1) for slot, live in zip(slots, resident)])
    slot_ids = torch.tensor(slots)
    starts = torch.tensor([0, 127, 255, 3])[slot_ids].masked_fill(~active, 0)
    lengths = active.to(torch.int32) * (3 if mode == "prefill" else 1)
    states = (slot_ids + 1).masked_fill(~active, 0)
    pages = batch.inputs.page_tables[slot_ids].masked_fill(~active[:, None], -1)
    tokens = (batch.inputs.token_ids.view(1, -1) if mode == "prefill"
              else batch.requests.pending_token[slot_ids].view(-1, 1))
    expected = Metadata.allocate(batch.tokens, batch.rows, 4, "cpu")
    expected.set(starts, lengths, states, pages, tokens, batch.inputs.live_rows)
    expected.sample_indices.copy_((lengths.cumsum(0) - 1).clamp_min(0))
    _assert_metadata_equal(actual, expected)
    batch.finish(actual)
    kernel, args = cpu_token_kernels[1][-1]
    assert kernel is token_state._consume
    assert args[0] is actual.token_ids and args[1] is actual.query_start_loc


@pytest.mark.parametrize("with_history", [False, True])
def test_functional_token_state_returns_fresh_operands(cpu_token_kernels, with_history):
    factory, calls = cpu_token_kernels
    requests = DeviceRequestState.allocate("cpu", vocab_size=32, capacity=4)
    requests.generation.fill_(2)
    requests.status.fill_(RUNNING)
    requests.pending_token.copy_(torch.tensor([11, 12, 13, 14]))
    requests.consumed_tokens.copy_(torch.tensor([1, 2, 3, 4]))
    history = torch.arange(16, dtype=torch.int64).reshape(4, 4)[:, :2] if with_history else None
    owner = factory(requests, torch.zeros(2, dtype=torch.int64), torch.zeros(3, dtype=torch.int32),
                    torch.tensor([3, 1], dtype=torch.int32), torch.tensor([2, 2]),
                    torch.tensor([2], dtype=torch.int32), history=history)
    owner.prepare()
    owner.functional()
    first, second = owner.gather(), owner.gather()
    assert first[0].tolist() == [14, 12]
    assert first[1].tolist() == [4, 2]
    assert first[3].tolist() == [1, 1]
    if with_history:
        assert first[2].tolist() == [[12, 13], [4, 5]]
    else:
        assert first[2] is None
    for left, right in zip(first, second):
        if left is not None:
            assert left.data_ptr() != right.data_ptr()
    boundaries = torch.tensor([0, 1, 2], dtype=torch.int32)
    owner.consume(second[0], boundaries)
    assert calls[-1][1][0] is second[0] and calls[-1][1][1] is boundaries
    assert calls[-1][1][8] is requests.pending_token
    with pytest.raises(ValueError, match="requires tokens"):
        owner.consume()
    with pytest.raises(ValueError, match="cu_seqlens"):
        owner.consume(second[0], boundaries.to(torch.int64))
