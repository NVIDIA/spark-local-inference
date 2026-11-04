# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learned relative attention and residual convolution preserve packed histories."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.relative_paged_attention import PreparedRelativePagedAttention
from ignition.components.residual_short_conv import PreparedResidualShortConvolution
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an assigned CUDA GPU")


@pytest.mark.parametrize("window,counts,starts,capacity", [
    (-1, (1, 0), (1024, 0), 1),
    (-1, (129, 0, 3), (127, 0, 1030), 144),
    (511, (129, 0, 1), (510, 0, 1024), 144),
    (-1, (4096,), (3,), 4096),
])
@torch.no_grad()
def test_relative_attention_canonical_paged_capture(window, counts, starts, capacity):
    from transformers.models.inkling.modeling_inkling import eager_attention_forward

    torch.manual_seed(835)
    device = torch.device("cuda", 0)
    heads, groups, dim, extent = 4, 1 if capacity == 1 else 2, 128, 1024 if window < 0 else 512
    batches = len(counts)
    blocks = (max(start + count for start, count in zip(starts, counts)) + 127) // 128
    pages = torch.randperm(batches * blocks, device=device).int().reshape(batches, blocks)
    # Noncontiguous page strides exercise the runtime's interleaved layer storage.
    key = torch.randn((batches * blocks, 2, 128, groups, dim), device=device).bfloat16()[:, 0]
    value = torch.randn_like(key)
    query = torch.randn((capacity, heads, dim), device=device).bfloat16()
    relative = (torch.randn((capacity, heads, 16), device=device).bfloat16()
                @ torch.randn((16, extent), device=device).bfloat16()) * .125
    output = torch.empty_like(query)
    lengths = torch.tensor([s + n for s, n in zip(starts, counts)], dtype=torch.int32, device=device)
    cu = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32, device=device)
    with RecordedPreparationSession(device=device, checkpoint="relative-paged-test", workload_schema="relative-paged",
                                    numerical_policy="default", compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        operation = PreparedRelativePagedAttention(query, key, value, pages, lengths, cu, relative, output,
            window=window, softmax_scale=1 / dim, session=session, workspace=arena, name="attention")
        saved = tuple(t.clone() for t in (pages, lengths, cu))
        for tensor in (pages, lengths, cu):
            tensor.fill_(1 << 29)
        session.tune()
        for tensor, original in zip((pages, lengths, cu), saved):
            tensor.copy_(original)
        from ignition.preparation.compilation import compile_forward
        run = compile_forward(operation)
        run()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        session.freeze()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        reference = torch.zeros_like(output)
        for batch, (count, start) in enumerate(zip(counts, starts)):
            if count == 0:
                continue
            begin = int(cu[batch])
            positions = torch.arange(start, start + count, device=device)
            offsets = torch.arange(start + count, device=device)
            k = key[pages[batch, offsets // 128], offsets % 128]
            v = value[pages[batch, offsets // 128], offsets % 128]
            distance = positions[:, None] - offsets[None, :]
            bias = relative[begin:begin + count].transpose(0, 1).gather(
                -1, distance.clamp(0, extent - 1)[None].expand(heads, -1, -1))
            bias.masked_fill_((distance < 0)[None] | (distance >= extent)[None], 0)
            valid = distance >= 0
            if window >= 0:
                valid &= distance <= window
            mask = torch.where(valid, 0., float("-inf")).bfloat16()[None, None]
            expected, _ = eager_attention_forward(SimpleNamespace(num_key_value_groups=heads // groups, training=False),
                query[begin:begin + count].transpose(0, 1)[None], k.transpose(0, 1)[None],
                v.transpose(0, 1)[None], mask, scaling=1 / dim, position_bias=bias[None])
            reference[begin:begin + count] = expected[0]
        # BF16-rounded matmul/softmax with a different FP32 reduction tree.
        torch.testing.assert_close(output, reference, atol=.001, rtol=.008)
        assert (output.float() - reference.float()).norm() / reference.float().norm() < .0002
        first = output.clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(first, output)
        assert torch.count_nonzero(output[sum(counts):]) == 0
        # Changing the live request set after capture cannot read its stale pages.
        lengths.zero_()
        cu.zero_()
        pages.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(output) == 0
        graph.reset()


@pytest.mark.parametrize("capacity,channels", [(1, 67), (129, 257), (4096, 67)])
@torch.no_grad()
def test_residual_convolution_canonical_chunking_and_state(capacity, channels):
    from transformers.models.inkling.modeling_inkling import InklingShortConvolution

    torch.manual_seed(836)
    device = torch.device("cuda", 0)
    rows = 1 if capacity == 4096 else 3
    source = torch.randn((capacity, channels + 16), device=device).bfloat16()[:, :channels]
    output = torch.empty_like(source)
    weights = torch.randn((channels, 4), device=device) * .1
    backing = torch.zeros((rows + 2, channels * 3 + 32), device=device)
    history = backing[:, :channels * 3].view(rows + 2, channels, 3)
    m = Metadata.allocate(capacity, rows, 1, device)
    with RecordedPreparationSession(device=device, checkpoint="residual-short-conv-test", workload_schema="residual-conv",
                                    numerical_policy="default", compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        operation = PreparedResidualShortConvolution(source, weights, history, m, output,
            session=session, workspace=arena, name="convolution")
        for tensor in (m.state_rows, m.starts, m.lengths, m.query_start_loc):
            tensor.fill_(1 << 29)
        session.tune()
        m.state_rows.zero_()
        m.lengths.zero_()
        m.query_start_loc.zero_()
        from ignition.preparation.compilation import compile_forward
        run = compile_forward(operation)
        run()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        session.freeze()
        backing.zero_()
        all_inputs = [torch.empty((0, channels), device=device, dtype=torch.bfloat16) for _ in range(rows)]
        canonical = InklingShortConvolution(channels, 4, 0, 0).to(device)
        canonical.conv1d.weight.copy_(weights[:, None])
        patterns = ([1, 1, 0, 1] if capacity == 1 else [capacity - 3, 1, 3, 0, capacity - 2])
        for count in patterns:
            source.normal_()
            lengths = [count] if rows == 1 else [max(0, count - 1), 0, min(count, 1)]
            starts = [len(x) for x in all_inputs]
            m.lengths.copy_(torch.tensor(lengths, device=device, dtype=torch.int32))
            m.starts.copy_(torch.tensor(starts, device=device))
            m.state_rows.copy_(torch.arange(1, rows + 1, device=device))
            m.query_start_loc[0] = 0
            m.query_start_loc[1:].copy_(m.lengths.cumsum(0))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            begin = 0
            for row, n in enumerate(lengths):
                if n:
                    all_inputs[row] = torch.cat((all_inputs[row], source[begin:begin + n]))
                    expected = canonical(all_inputs[row][None])[0, -n:]
                    torch.testing.assert_close(output[begin:begin + n], expected, atol=1e-7, rtol=.008)
                    assert (output[begin:begin + n].float() - expected.float()).norm() / expected.float().norm() < .0001
                    recent = all_inputs[row][-3:].float().T
                    expected_history = torch.nn.functional.pad(recent, (3 - recent.shape[1], 0))
                    torch.testing.assert_close(history[row + 1], expected_history, atol=0, rtol=0)
                begin += n
            assert torch.count_nonzero(output[begin:]) == 0
            assert torch.count_nonzero(history[0]) == 0
            assert torch.count_nonzero(history[-1]) == 0
            if rows > 1:
                assert torch.count_nonzero(history[2]) == 0
        # A reused physical row at position zero must ignore the previous request.
        stale = history.clone()
        m.starts.zero_()
        m.lengths.zero_()
        m.lengths[0] = 1
        m.query_start_loc.fill_(1)
        m.query_start_loc[0] = 0
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output[:1], canonical(source[:1][None])[0], atol=1e-7, rtol=.008)
        torch.testing.assert_close(history[1, :, :2], torch.zeros_like(history[1, :, :2]), atol=0, rtol=0)
        torch.testing.assert_close(history[1, :, 2], source[0].float(), atol=0, rtol=0)
        torch.testing.assert_close(history[2:], stale[2:], atol=0, rtol=0)
        graph.reset()


@pytest.mark.parametrize("round_silu", [True, False])
@torch.no_grad()
def test_bf16_experts_silu_rounding_and_capture(round_silu):
    from ignition.components.bf16_experts import PreparedBF16Experts
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool

    torch.manual_seed(837)
    source = torch.randn((17, 128), device="cuda", dtype=torch.bfloat16)
    gate_up = (torch.randn((4, 128, 128), device="cuda") * .05).bfloat16()
    down = (torch.randn((4, 128, 64), device="cuda") * .05).bfloat16()
    ids = torch.rand((17, 4), device="cuda").topk(2, dim=-1).indices.int()
    weights = torch.rand((17, 2), device="cuda")
    valid = torch.tensor([13], device="cuda", dtype=torch.int32)
    output = torch.empty((17, 2, 128), device="cuda", dtype=torch.bfloat16)
    operation = PreparedBF16Experts(source, gate_up, down, ids, weights, output, valid,
        buffers=GraphBufferPool("cuda").cursor(), round_silu=round_silu)
    expected = torch.zeros_like(output)
    intermediate = torch.empty((13, 2, 64), dtype=torch.bfloat16, device="cuda")
    for token in range(13):
        for route in range(2):
            expert = ids[token, route].item()
            gate, up = torch.nn.functional.linear(source[token], gate_up[expert]).chunk(2)
            # BF16 intermediate activation for existing users; Inkling's supported
            # implementation computes SiLU and multiplication in FP32, rounding once.
            activated = (torch.nn.functional.silu(gate) * up if round_silu else
                         (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16())
            intermediate[token, route] = activated
            expected[token, route] = torch.nn.functional.linear(activated, down[expert])
    operation()
    torch.testing.assert_close(operation.intermediate[:26], intermediate.flatten(0, 1), rtol=.008, atol=2e-5)
    torch.testing.assert_close(output, expected, rtol=.008, atol=2e-4)
    run = compile_forward(operation)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    torch.testing.assert_close(output, expected, rtol=.008, atol=2e-4)
    assert torch.count_nonzero(output[13:]) == 0
    valid.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output) == 0
    graph.reset()
