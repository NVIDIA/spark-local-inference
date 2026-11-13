# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""FP32 vocabulary words retain their bits through BF16 transport lanes."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import collectives


def _words():
    return (
        torch.tensor(
            [
                0x3F800001,
                0x7FC01234,
                0x80000000,
                0x00000001,
                0xBF800001,
                0x7F800000,
                0xFF800000,
                0x00000000,
            ],
            dtype=torch.int64,
        )
        .to(torch.int32)
        .view(2, 4)
    )


def test_fp32_transport_preserves_nan_payloads_subnormals_and_signed_zero():
    words = _words()
    source = words.view(torch.float32)
    transport = collectives._gather_transport_view(source)
    assert transport.dtype == torch.bfloat16
    assert transport.shape == (2, 8)
    assert transport.data_ptr() == source.data_ptr()
    torch.testing.assert_close(transport.view(torch.int32), words, rtol=0, atol=0)
    other = words.flip(0).contiguous().view(torch.float32)
    gathered = torch.cat((transport, collectives._gather_transport_view(other)), dim=1)
    torch.testing.assert_close(
        gathered.view(torch.int32),
        torch.cat((words, words.flip(0)), dim=1),
        rtol=0,
        atol=0,
    )


def test_fp32_native_gather_declares_and_replays_byte_geometry(monkeypatch):
    from b12x.comm import pcie

    observed = {}
    source = torch.randn(2, 128, dtype=torch.float32)
    output = torch.empty(2, 512, dtype=torch.float32)
    runtime = SimpleNamespace(
        world_size=4,
        device=source.device,
        max_batch_size=8,
        heads_per_rank=4,
        query_head_dim=64,
        total_heads=16,
        exchange_group=object(),
    )

    def query(owner, *, surface, call):
        assert owner is runtime
        observed["declaration"] = call
        return object()

    plan = SimpleNamespace(request=lambda **kwargs: object())

    def run(value, *, out, plan):
        observed["replay"] = value, out

    runtime.all_gather_heads = run
    monkeypatch.setattr(pcie, "query_from_runtime", query)
    monkeypatch.setattr(pcie, "plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: False)
    monkeypatch.setattr(collectives, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: [0, 1, 2, 3]
    )
    operation = collectives.PreparedAllGather(
        source,
        output,
        runtime=runtime,
        session=None,
        coordinator=None,
        name="fp32-vocabulary",
    )
    call = observed["declaration"]
    assert call["local_input"].shape == (2, 4, 64)
    assert call["out"].shape == (2, 16, 64)
    assert call["local_input"].dtype == call["out"].dtype == torch.bfloat16
    assert call["local_input"].data_ptr() == source.data_ptr()
    collectives._gather_columns(source, output, operation.handle)
    sent, received = observed["replay"]
    assert sent.dtype == received.dtype == torch.bfloat16
    assert sent.data_ptr() == source.data_ptr()
    assert received.data_ptr() == output.data_ptr()
    torch.testing.assert_close(
        sent.view(torch.float32).reshape_as(source), source, rtol=0, atol=0
    )


def test_fp32_gather_rejects_a_channel_sized_for_bf16(monkeypatch):
    source = torch.zeros(2, 128, dtype=torch.float32)
    output = torch.zeros(2, 512, dtype=torch.float32)
    runtime = SimpleNamespace(
        world_size=4,
        device=source.device,
        max_batch_size=8,
        heads_per_rank=2,
        query_head_dim=64,
        total_heads=8,
    )
    monkeypatch.setattr(collectives, "_roce", lambda runtime: False)
    with pytest.raises(ValueError, match="channel"):
        collectives.PreparedAllGather(
            source,
            output,
            runtime=runtime,
            session=None,
            coordinator=None,
            name="fp32-vocabulary",
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_roce_gather_uses_the_same_byte_views_during_preparation_and_replay(monkeypatch, dtype):
    source = _words().view(dtype)
    output = torch.empty((2, source.shape[1] * 2), dtype=dtype)
    calls, declarations = [], []

    def gather(value, *, out, **kwargs):
        calls.append((value, out))
        out.copy_(torch.cat((value, value), dim=-1))

    runtime = SimpleNamespace(world_size=2, max_gather_bytes=source.nbytes,
        _group=object(), _run_prepared_all_gather=gather, all_gather=gather)
    plan = SimpleNamespace(request=lambda **kwargs: declarations.append(kwargs) or object())
    monkeypatch.setattr(collectives, "_roce", lambda runtime: True)
    monkeypatch.setattr(collectives, "_roce_plan", lambda *args: plan)
    monkeypatch.setattr(collectives, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    operation = collectives.PreparedAllGather(source, output, runtime=runtime,
        session=None, coordinator=None, name="roce-vocabulary")
    prepared = declarations[0]["prepare_call"](object())
    prepared.run()
    collectives._gather_columns(source, output, operation.handle)
    for sent, received in calls:
        assert sent.dtype == received.dtype == torch.bfloat16
        assert sent.data_ptr() == source.data_ptr()
        assert received.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output.view(torch.int32),
        torch.cat((_words(), _words()), dim=-1), rtol=0, atol=0)
    runtime.max_gather_bytes -= 16
    with pytest.raises(ValueError, match="byte capacity"):
        collectives.PreparedAllGather(source, output, runtime=runtime,
            session=None, coordinator=None, name="oversized-vocabulary")
