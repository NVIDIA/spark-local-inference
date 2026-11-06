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
from ignition.preparation.compilation import compile_forward
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
        operation = PreparedRelativePagedAttention(query, key, value, pages, lengths, cu, relative, output,
            window=window, softmax_scale=1 / dim, session=session, name="attention")
        saved = tuple(t.clone() for t in (pages, lengths, cu))
        for tensor in (pages, lengths, cu):
            tensor.fill_(1 << 29)
        session.tune()
        for tensor, original in zip((pages, lengths, cu), saved):
            tensor.copy_(original)
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
        operation = PreparedResidualShortConvolution(source, weights, history, m, output,
            session=session, name="convolution")
        for tensor in (m.state_rows, m.starts, m.lengths, m.query_start_loc):
            tensor.fill_(1 << 29)
        session.tune()
        m.state_rows.zero_()
        m.lengths.zero_()
        m.query_start_loc.zero_()
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
