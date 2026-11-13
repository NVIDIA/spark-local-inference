"""Ordered selection matches an integer-bit oracle under captured storage reuse."""

import os

import pytest
import torch

from .test_ordered_row_topk import (
    SPECIAL_SCORE_BITS, adversarial_rows, ordered_reference, ordered_key, signed32,
)

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                               reason='requires an explicitly assigned CUDA device')


class GuardedBuffers:
    def __init__(self, device):
        self.device, self.storage = device, []

    def zeros(self, shape, dtype):
        import math

        backing = torch.empty(math.prod(shape) + 32, dtype=dtype, device=self.device)
        backing.view(torch.int32).fill_(0x35abcdef)
        result = backing[16:-16].view(shape)
        result.zero_()
        self.storage.append(backing)
        return result

    def check(self):
        for backing in self.storage:
            bits = backing.view(torch.int32)
            assert torch.all(bits[:16] == 0x35abcdef)
            assert torch.all(bits[-16:] == 0x35abcdef)


class Selection:
    def __init__(self, rows, width, topk, device, session, radix=None):
        from ignition.components.ordered_row_topk import PreparedOrderedRowTopK
        from ignition.components.row_topk import PreparedRowTopK

        self.scores = torch.empty((rows, width), dtype=torch.float32, device=device)
        self.ids = torch.empty((rows, width), dtype=torch.int32, device=device)
        self.lengths = torch.empty(rows, dtype=torch.int32, device=device)
        self.guards = GuardedBuffers(device)
        self.values = self.guards.zeros((rows, topk), torch.float32)
        self.selected = self.guards.zeros((rows, topk), torch.int32)
        self.control_values = torch.empty_like(self.values)
        self.control_ids = torch.empty_like(self.selected)
        self.ordered = PreparedOrderedRowTopK(self.scores, self.lengths, self.ids, self.values, self.selected,
            session=session, name='ordered', buffers=self.guards, radix=radix)
        self.control = PreparedRowTopK(self.scores, self.lengths, self.ids, self.control_values, self.control_ids,
                                      session=session, name='control')

    def __call__(self):
        self.control()
        self.ordered()
        return self.selected


def check(operation, source, lengths, ids, labels):
    operation.guards.check()
    topk = operation.selected.shape[1]
    expected_values, expected_ids, _ = ordered_reference(source, lengths, ids, topk)
    observed_values, observed_ids = operation.values.view(torch.int32).cpu(), operation.selected.cpu()
    cutoffs = operation.ordered.cutoffs.cpu().long() & 0xffffffff
    greater = operation.ordered.greater.cpu().sum(1)
    ties = operation.ordered.ties.cpu().sum(1)
    for row, label in enumerate(labels):
        if lengths[row] > topk:
            assert greater[row] <= topk <= greater[row] + ties[row], (label, int(cutoffs[row]), int(greater[row]), int(ties[row]))
            expected_cutoff = min(ordered_key(v) for v in expected_values[row].tolist())
            assert int(cutoffs[row]) == expected_cutoff, (label, int(cutoffs[row]), expected_cutoff)
        assert torch.equal(observed_values[row], expected_values[row]), (label, 'score bits')
        assert torch.equal(observed_ids[row], expected_ids[row]), (label, 'IDs')
    assert torch.equal(operation.scores.view(torch.int32).cpu(), source.view(torch.int32))
    assert torch.equal(operation.ids.cpu(), ids)


@pytest.mark.parametrize('compiled', [False, True])
@torch.no_grad()
def test_ordered_topk_exceptional_bits_and_reuse(compiled):
    from ignition import runtime_patches
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    runtime_patches.apply()
    torch.set_num_threads(2)
    device = torch.device('cuda', 0)
    source, lengths, ids, labels = adversarial_rows()
    with RecordedPreparationSession(device=device, checkpoint='ordered-candidate-bits',
            workload_schema='ordered-row-topk', numerical_policy='ordered-uint32-input-ties', compile_workers=16) as session:
        operation = Selection(source.shape[0], source.shape[1], 2048, device, session)
        operation.scores.view(torch.int32).copy_(source.view(torch.int32))
        operation.lengths.copy_(lengths)
        operation.ids.copy_(ids)
        session.tune(show_progress=False)
        call = compile_forward(operation) if compiled else operation
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            call()
        try:
            for seed in (0, 1, 2):
                source, lengths, ids, labels = adversarial_rows(seed=seed)
                operation.scores.view(torch.int32).copy_(source.view(torch.int32))
                operation.lengths.copy_(lengths)
                operation.ids.copy_(ids)
                operation.values.view(torch.int32).fill_(0x7fc12345)
                operation.selected.fill_(-91)
                operation.ordered.cutoffs.fill_(-71)
                operation.ordered.greater.fill_(-73)
                operation.ordered.ties.fill_(-79)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
                check(operation, source, lengths, ids, labels)
            snapshots = torch.empty((6, *operation.selected.shape), dtype=torch.int32, device=device)
            score_snapshots = torch.empty_like(snapshots)
            fixtures = [adversarial_rows(seed=seed) for seed in (0, 1, 2)]
            for replay in range(6):
                source, lengths, ids, labels = fixtures[replay % 3]
                operation.scores.view(torch.int32).copy_(source.view(torch.int32))
                operation.lengths.copy_(lengths)
                operation.ids.copy_(ids)
                operation.values.view(torch.int32).fill_(0x7fc12345)
                operation.selected.fill_(-91)
                graph.replay()
                snapshots[replay].copy_(operation.selected)
                score_snapshots[replay].copy_(operation.values.view(torch.int32))
            torch.cuda.synchronize()
            operation.guards.check()
            for replay in range(6):
                source, lengths, ids, labels = fixtures[replay % 3]
                expected_values, expected_ids, _ = ordered_reference(source, lengths, ids, 2048)
                assert torch.equal(snapshots[replay].cpu(), expected_ids)
                assert torch.equal(score_snapshots[replay].cpu(), expected_values)
        finally:
            graph.reset()


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('width,topk,seed', [(8192, 2048, 3), (3000, 1024, 4)])
@torch.no_grad()
def test_radix_ordered_topk_matches_integer_oracle(compiled, width, topk, seed):
    from ignition import runtime_patches
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    runtime_patches.apply()
    torch.set_num_threads(2)
    device = torch.device('cuda', 0)
    source, lengths, ids, labels = adversarial_rows(width=width, topk=topk, seed=seed)
    with RecordedPreparationSession(device=device, checkpoint='radix-ordered-candidate-bits',
            workload_schema='radix-ordered-row-topk', numerical_policy='ordered-uint32-input-ties') as session:
        operation = Selection(source.shape[0], width, topk, device, session, radix=True)
        operation.scores.view(torch.int32).copy_(source.view(torch.int32))
        operation.lengths.copy_(lengths)
        operation.ids.copy_(ids)
        session.tune(show_progress=False)
        call = compile_forward(operation) if compiled else operation
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            call()
        try:
            for variation in range(3):
                source, lengths, ids, labels = adversarial_rows(width=width, topk=topk, seed=seed + variation)
                operation.scores.view(torch.int32).copy_(source.view(torch.int32))
                operation.lengths.copy_(lengths)
                operation.ids.copy_(ids)
                operation.values.view(torch.int32).fill_(0x7fc12345)
                operation.selected.fill_(-91)
                operation.ordered.cutoffs.fill_(-71)
                operation.ordered.greater.fill_(-73)
                operation.ordered.ties.fill_(-79)
                graph.replay()
                torch.cuda.synchronize()
                check(operation, source, lengths, ids, labels)
        finally:
            graph.reset()


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('width', [1, 64, 511])
@torch.no_grad()
def test_radix_underfull_widths_preserve_bits_padding_and_guarded_replay(compiled, width):
    from ignition import runtime_patches
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    runtime_patches.apply()
    device = torch.device('cuda', 0)
    lengths = torch.tensor([0, 1, width // 2, width], dtype=torch.int32)
    labels = ('empty', 'one', 'partial', 'full_width')
    ids = torch.arange(4 * width, dtype=torch.int32).view(4, width)
    ids[:, ::3] = -2
    with RecordedPreparationSession(device=device, checkpoint='underfull-ordered-candidate-bits',
            workload_schema='radix-ordered-row-topk', numerical_policy='ordered-uint32-input-ties') as session:
        operation = Selection(4, width, 512, device, session, radix=True)

        def inputs(variation):
            words = [signed32(SPECIAL_SCORE_BITS[(index + variation) % len(SPECIAL_SCORE_BITS)])
                     for index in range(4 * width)]
            source = torch.tensor(words, dtype=torch.int32).view(4, width).view(torch.float32)
            operation.scores.view(torch.int32).copy_(source.view(torch.int32))
            operation.lengths.copy_(lengths)
            operation.ids.copy_(ids)
            return source

        inputs(0)
        session.tune(show_progress=False)
        call = compile_forward(operation) if compiled else operation
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            call()
        try:
            for variation in range(3):
                source = inputs(variation)
                operation.values.view(torch.int32).fill_(0x7fc12345)
                operation.selected.fill_(-91)
                operation.ordered.cutoffs.fill_(-71)
                operation.ordered.greater.fill_(-73)
                operation.ordered.ties.fill_(-79)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
                check(operation, source, lengths, ids, labels)
        finally:
            graph.reset()


@torch.no_grad()
def test_negative_nan_below_zero_without_coarse_overflow():
    from ignition import runtime_patches
    from ignition.preparation.tuning import RecordedPreparationSession

    runtime_patches.apply()
    device = torch.device('cuda', 0)
    source = torch.zeros((1, 4096), dtype=torch.float32)
    source.view(torch.int32)[0, 0] = 0xffc12345 - 2**32
    lengths = torch.tensor([4096], dtype=torch.int32)
    ids = torch.arange(4096, dtype=torch.int32).view(1, -1)
    with RecordedPreparationSession(device=device, checkpoint='negative-nan-coarse-bin',
            workload_schema='ordered-row-topk', numerical_policy='ordered-uint32-input-ties') as session:
        operation = Selection(1, 4096, 2048, device, session)
        operation.scores.view(torch.int32).copy_(source.view(torch.int32))
        operation.lengths.copy_(lengths)
        operation.ids.copy_(ids)
        session.tune(show_progress=False)
        operation()
        torch.cuda.synchronize()
        operation.guards.check()
        print('Non-overflow NaN case: canaries intact; cutoff/G/E=',
              int(operation.ordered.cutoffs.item()) & 0xffffffff,
              int(operation.ordered.greater.sum()), int(operation.ordered.ties.sum()), flush=True)
        print('Base selector retained negative NaN=', bool((operation.control_values.view(torch.int32) == (0xffc12345 - 2**32)).any()), flush=True)
        check(operation, source, lengths, ids, ('negative_nan_nonoverflow',))
