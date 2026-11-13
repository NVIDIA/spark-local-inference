# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Circular attention state survives chunk boundaries and prefix branching."""

import pytest
import torch

from ignition.recipes.mimo_v2.attention import commit_window, stage_window


class PrefillWindowForward:
    def __init__(self):
        self.history = torch.zeros((3, 128, 1, 192), dtype=torch.float8_e4m3fn, device='cuda')
        self.values = torch.randn((4096, 1, 192), device='cuda').bfloat16()
        self.output = torch.empty((33, 128, 1, 192), dtype=self.history.dtype, device='cuda')
        self.row, self.start, self.length = (torch.tensor(value, device='cuda') for value in (1, 0, 4096))

    def __call__(self):
        stage_window(self.history, self.values, self.output, self.row, self.start, self.length)
        commit_window(self.history, self.values, self.row, self.start, self.length)


def test_prefill_window_compiles_from_metadata_and_replays(tmp_path):
    import os
    if os.environ.get('FLARE_RUN_CUDA_TESTS') != '1':
        pytest.skip('requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU')
    from ignition.preparation.compile_pool import CompilePool

    forward = PrefillWindowForward()
    path = tmp_path / 'prefill.bin'
    with CompilePool('cuda:0') as pool:
        pool.submit(forward, path).result()
    with path.open('rb') as file:
        compiled = torch.compiler.load_compiled_function(file)
    compiled(forward)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(forward)
    forward.history.zero_()
    full = torch.empty((0, 1, 192), dtype=torch.bfloat16)
    for count in [127, 1, 3, 4096, 0, 129]:
        forward.start.fill_(len(full))
        forward.length.fill_(count)
        forward.values.normal_()
        values = forward.values.cpu()[:count].to(forward.history.dtype).bfloat16()
        expected = torch.cat((full[-128:], values))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(forward.output.flatten(0, 1)[:len(expected)].cpu().bfloat16(), expected, rtol=0, atol=0)
        full = torch.cat((full, values))
        positions = torch.arange(max(0, len(full) - 128), len(full)) % 128
        torch.testing.assert_close(forward.history.cpu()[1, positions].bfloat16(), full[-128:], rtol=0, atol=0)
        assert forward.history[0].float().count_nonzero() == forward.history[2].float().count_nonzero() == 0
    graph.reset()


@pytest.mark.parametrize("chunks", [(1, 127, 1, 128, 129), (64, 4096, 3), (0, 4, 0, 252)])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_circular_history_matches_full_sequence(chunks, cache_dtype):
    history = torch.zeros((3, 128, 1, 16), dtype=cache_dtype)
    row, start = torch.tensor(1), torch.tensor(0)
    generator = torch.Generator().manual_seed(5241)
    full = []
    for length in chunks:
        values = torch.randn((max(1, length), 1, 16), generator=generator).bfloat16()
        values[0, 0, :2] = torch.tensor([-1024., 1024.])
        output = torch.empty(((length + 128 + 127) // 128, 128, 1, 16), dtype=cache_dtype)
        stage_window(history, values, output, row, start, torch.tensor(length))
        prefix = torch.cat(full)[-128:] if full else values[:0]
        quantized = (values[:length].clamp(-448, 448) if cache_dtype == torch.float8_e4m3fn
                     else values[:length]).to(cache_dtype).bfloat16()
        expected = torch.cat((prefix, quantized))
        torch.testing.assert_close(output.flatten(0, 1)[:expected.shape[0]].bfloat16(), expected, rtol=0, atol=0)
        commit_window(history, values, row, start, torch.tensor(length))
        full.append(quantized)
        start += length
        positions = torch.arange(max(0, int(start) - 128), int(start))
        torch.testing.assert_close(history[row, positions % 128].bfloat16(), torch.cat(full)[-128:], rtol=0, atol=0)
        assert history[0].float().count_nonzero() == history[2].float().count_nonzero() == 0


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_restored_window_can_branch_without_touching_prefix(cache_dtype):
    history = torch.randn((3, 128, 1, 16)).to(cache_dtype)
    snapshot = history[1].clone()
    history[2].copy_(snapshot)
    branch = torch.randn((4, 1, 16)).bfloat16()
    commit_window(history, branch, torch.tensor(2), torch.tensor(255), torch.tensor(4))
    torch.testing.assert_close(history[1].float(), snapshot.float(), rtol=0, atol=0)
    torch.testing.assert_close(history[2, torch.tensor([127, 0, 1, 2])].float(), branch.to(cache_dtype).float(), rtol=0, atol=0)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_packed_candidate_windows_preserve_committed_history(cache_dtype):
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.mimo_v2.attention import stage_windows

    generator = torch.Generator().manual_seed(5021)
    history = torch.randn((5, 128, 1, 16), generator=generator).to(cache_dtype)
    original = history.clone()
    metadata = Metadata.allocate(32, 4, 1, "cpu", samples=32)
    starts = torch.tensor([127, 900, 256, 1])
    lengths = torch.tensor([8, 0, 3, 1], dtype=torch.int32)
    metadata.set(starts, lengths, torch.tensor([1, 2, 3, 4]), torch.zeros((4, 1), dtype=torch.int32),
                 torch.zeros((4, 8), dtype=torch.int64), torch.tensor([4]))
    values = torch.randn((32, 1, 16), generator=generator).bfloat16()
    values[0, 0, :2] = torch.tensor([-1024., 1024.])
    output = torch.empty((8, 128, 1, 16), dtype=cache_dtype)
    stage_windows(history, values, output, metadata)
    for row, state in enumerate([1, 2, 3, 4]):
        staged = output.view(4, 256, 1, 16)[row].bfloat16()
        if lengths[row] == 0:
            assert staged.count_nonzero() == 0
            continue
        prefix = min(128, int(starts[row]))
        positions = torch.arange(int(starts[row]) - prefix, int(starts[row]))
        begin = int(metadata.query_start_loc[row])
        current = values[begin:begin + lengths[row]]
        if cache_dtype == torch.float8_e4m3fn:
            current = current.clamp(-448, 448)
        expected = torch.cat((history[state, positions % 128].bfloat16(), current.to(cache_dtype).bfloat16()))
        torch.testing.assert_close(staged[:expected.shape[0]], expected, rtol=0, atol=0)
        assert staged[expected.shape[0]:].count_nonzero() == 0
    torch.testing.assert_close(history.float(), original.float(), rtol=0, atol=0)


@pytest.mark.parametrize('window,dimension', [(128, 192), (1024, 128)])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_compiled_window_replay_commits_only_accepted_tokens(window, dimension, cache_dtype):
    import os
    if os.environ.get('FLARE_RUN_CUDA_TESTS') != '1':
        pytest.skip('requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU')
    from ignition.components.paged_attention import PreparedPagedKVWriter
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.mimo_v2.attention import stage_windows

    device = torch.device('cuda', 0)
    history = torch.randn((5, window, 1, dimension), device=device).to(cache_dtype)
    values = torch.randn((32, 1, dimension), device=device).bfloat16()
    value_history = torch.randn((5, window, 1, 128), device=device).to(cache_dtype)
    value_rows = torch.randn((32, 1, 128), device=device).bfloat16()
    metadata = Metadata.allocate(32, 4, 1, device, samples=32)
    counts = torch.zeros((4,), dtype=torch.int32, device=device)
    slots = torch.full((32,), -1, dtype=torch.int64, device=device)
    capacity = window + 128
    staged = torch.empty((4 * capacity // 128, 128, 1, dimension), dtype=history.dtype, device=device)
    states = torch.tensor([1, 2, 3, 4], device=device)
    starts = torch.tensor([window - 1, window * 2 - 1, 0, window + 9], device=device)

    def set_rows(lengths, accepted):
        metadata.set(starts, torch.tensor(lengths, dtype=torch.int32, device=device), states,
                     torch.zeros((4, 1), dtype=torch.int32, device=device),
                     torch.zeros((4, 8), dtype=torch.int64, device=device), torch.tensor([4], device=device))
        counts.copy_(torch.tensor(accepted, device=device))

    set_rows([8, 0, 3, 1], [2, 0, 3, 1])
    with RecordedPreparationSession(device=device, checkpoint='circular-state',
            workload_schema='circular-state', numerical_policy='default') as session:
        writer = PreparedPagedKVWriter(values, value_rows, history, value_history, slots,
                                       session=session, name='write')

        def forward():
            stage_windows(history, values, staged, metadata)
            offset = metadata.positions - metadata.starts[metadata.requests]
            live = (metadata.positions >= 0) & (offset < counts[metadata.requests])
            slots.copy_(torch.where(live,
                metadata.state_rows[metadata.requests] * window + metadata.positions % window, -1))
            writer()

        session.tune()
        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        allocated = torch.cuda.memory_allocated(device)
        for lengths, accepted in [([8, 0, 3, 1], [2, 0, 3, 1]), ([1, 1, 0, 1], [1, 1, 0, 1]),
                                  ([8, 8, 8, 0], [8, 3, 1, 0]), ([1, 0, 1, 1], [1, 0, 1, 1])]:
            set_rows(lengths, accepted)
            values.normal_()
            value_rows.normal_()
            expected_key, expected_value = history.cpu(), value_history.cpu()
            source_key, source_value = values.cpu(), value_rows.cpu()
            expected_staged = torch.zeros_like(staged, device='cpu').view(4, capacity, 1, dimension)
            begin = 0
            for row, (length, count) in enumerate(zip(lengths, accepted)):
                start = int(starts[row])
                if length:
                    prefix = min(window, start)
                    positions = torch.arange(start - prefix, start) % window
                    expected_staged[row, :prefix] = expected_key[row + 1, positions]
                    expected_staged[row, prefix:prefix + length] = source_key[begin:begin + length]
                for token in range(count):
                    slot = (start + token) % window
                    expected_key[row + 1, slot] = source_key[begin + token]
                    expected_value[row + 1, slot] = source_value[begin + token]
                begin += length
            staged.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(staged.cpu().view_as(expected_staged).float(), expected_staged.float(), rtol=0, atol=0)
            torch.testing.assert_close(history.cpu().float(), expected_key.float(), rtol=0, atol=0)
            torch.testing.assert_close(value_history.cpu().float(), expected_value.float(), rtol=0, atol=0)
            assert torch.cuda.memory_allocated(device) == allocated
            starts.add_(counts)
        graph.reset()
