"""Exact paired-column packing and bounded shared-buffer contracts on the host."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import column_pair_gather


def fields(rows, rank, variation):
    first = (torch.arange(rows * 224, dtype=torch.int32) + 257 * rank + variation).short().view(rows, 224)
    second = (torch.arange(rows * 56, dtype=torch.int32) * 1048583 + 1069547520 + 193 * rank + variation).view(rows, 56)
    first[0, :4] = torch.tensor([0, -32768, 32704, -64], dtype=torch.int16)
    second[0, :4] = torch.tensor([0, -2147483648, 2143289345, -4194303], dtype=torch.int32)
    return first.view(torch.bfloat16), second.view(torch.float32)


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("rows", [16, 128, 1025, 4096])
def test_paired_columns_preserve_bits_and_partial_chunks(monkeypatch, rows, functional):
    runtime = SimpleNamespace(world_size=16, heads_per_rank=1, query_head_dim=336, max_batch_size=1024,
                              offset=0, variation=0)

    class Gather:
        def __init__(self, source, output, **kwargs):
            self.output = output

        def functional(self, *, session):
            self.output = None

        def __call__(self, source):
            count = source.shape[0]
            parts = []
            for rank in range(16):
                first, second = fields(rows, rank, runtime.variation)
                parts.append(torch.cat((first, second.view(torch.bfloat16)), dim=1)[runtime.offset:runtime.offset + count])
            assert torch.equal(source.view(torch.int16), parts[0].view(torch.int16))
            result = torch.cat(parts, dim=1)
            if self.output is not None:
                self.output.copy_(result)
                result = self.output
            runtime.offset += count
            return result

    monkeypatch.setattr(column_pair_gather, 'PreparedAllGather', Gather)
    first, second = fields(rows, 0, 0)
    output_first = torch.empty((rows, 3584), dtype=torch.bfloat16)
    output_second = torch.empty((rows, 896), dtype=torch.float32)
    buffers = {}
    operation = column_pair_gather.PreparedColumnPairGather(first, second, output_first, output_second,
        runtime=runtime, session=None, coordinator=None, name='columns', buffers=buffers)
    assert operation.send.shape == (min(rows, 1024), 336)
    assert operation.received.shape == (min(rows, 1024), 5376)
    if functional:
        operation.functional(session=None)
    for variation in range(3):
        runtime.variation, runtime.offset = variation, 0
        left, right = fields(rows, 0, variation)
        first.copy_(left)
        second.copy_(right)
        if not functional:
            operation.send.fill_(torch.nan)
            operation.received.fill_(torch.nan)
        output_first, output_second = operation(first, second) if functional else operation()
        expected_first = torch.cat([fields(rows, rank, variation)[0] for rank in range(16)], dim=1)
        expected_second = torch.cat([fields(rows, rank, variation)[1] for rank in range(16)], dim=1)
        assert torch.equal(output_first.view(torch.int16), expected_first.view(torch.int16))
        assert torch.equal(output_second.view(torch.int32), expected_second.view(torch.int32))
        assert runtime.offset == rows


@pytest.mark.parametrize("columns,max_rows", [(335, 1024), (336, 0), (344, 1024)])
def test_paired_columns_reject_channel_mismatch(columns, max_rows):
    first, second = fields(16, 0, 0)
    with pytest.raises(ValueError):
        column_pair_gather.validate_columns(first, second, torch.empty((16, 3584), dtype=torch.bfloat16),
            torch.empty((16, 896)), world=16, columns=columns, max_rows=max_rows)


def test_paired_columns_reject_strided_transport_inputs():
    first = torch.empty((16, 225), dtype=torch.bfloat16)[:, :224]
    second = torch.empty((16, 56), dtype=torch.float32)
    with pytest.raises(ValueError, match='contiguous'):
        column_pair_gather.validate_columns(first, second, torch.empty((16, 3584), dtype=torch.bfloat16),
            torch.empty((16, 896)), world=16, columns=336, max_rows=1024)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_paired_columns_reject_live_aliases_and_allow_disjoint_storage_views(device):
    first = torch.empty((16, 224), dtype=torch.bfloat16, device=device)
    second = torch.empty((16, 56), dtype=torch.float32, device=device)
    shared = torch.empty(2 * 16 * 3584, dtype=torch.bfloat16, device=device)
    output_first = shared[:16 * 3584].view(16, 3584)
    output_second = torch.empty((16, 896), dtype=torch.float32, device=device)

    def validate(source=first, target=output_second):
        column_pair_gather.validate_columns(source, second, output_first, target,
            world=16, columns=336, max_rows=1024)

    validate()
    validate(shared[16 * 3584:16 * (3584 + 224)].view(16, 224))
    with pytest.raises(ValueError, match='overlaps: first and output_first'):
        validate(shared[:16 * 224].view(16, 224))
    with pytest.raises(ValueError, match='overlaps: output_first and output_second'):
        validate(target=shared[:16 * 896 * 2].view(torch.float32).view(16, 896))


def test_paired_columns_reject_aliased_caller_transport_buffers():
    first, second = fields(16, 0, 0)
    received = torch.empty((16, 5376), dtype=torch.bfloat16)
    send = received.view(-1)[:16 * 336].view(16, 336)
    runtime = SimpleNamespace(world_size=16, heads_per_rank=1, query_head_dim=336, max_batch_size=1024)
    with pytest.raises(ValueError, match='overlaps: send and received'):
        column_pair_gather.PreparedColumnPairGather(first, second,
            torch.empty((16, 3584), dtype=torch.bfloat16), torch.empty((16, 896)),
            runtime=runtime, session=None, coordinator=None, name='columns', buffers=dict(send=send, received=received))


def test_paired_columns_preserve_fake_tensor_storage_identity():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        first = torch.empty((16, 224), dtype=torch.bfloat16)
        second = torch.empty((16, 56), dtype=torch.float32)
        output_first = torch.empty((16, 3584), dtype=torch.bfloat16)
        output_second = torch.empty((16, 896), dtype=torch.float32)
        column_pair_gather.validate_columns(first, second, output_first, output_second,
            world=16, columns=336, max_rows=1024)
        with pytest.raises(ValueError, match='overlaps'):
            column_pair_gather.validate_columns(output_first.view(-1)[:16 * 224].view(16, 224),
                second, output_first, output_second, world=16, columns=336, max_rows=1024)


def test_projection_precision_gate_rejects_accuracy_regression():
    from qualify.kimi_latent_prefill import check_projection_gate

    reference = torch.tensor([[1., 2.], [0., 0.]], dtype=torch.bfloat16)
    control = reference.clone()
    control[0, 0] += 1 / 128
    result = check_projection_gate(control, reference, reference)
    assert result['row']['relative_l2'] == 0
    with pytest.raises(AssertionError, match='regresses'):
        check_projection_gate(reference, control, reference)
    wrong_zero = reference.clone()
    wrong_zero[1, 0] = 1e-4
    with pytest.raises(AssertionError, match='zero-reference'):
        check_projection_gate(control, wrong_zero, reference)


def test_projection_precision_gate_checks_worst_row_independently():
    from qualify.kimi_latent_prefill import check_projection_gate

    reference = torch.tensor([[1., 1.], [100., 100.]], dtype=torch.bfloat16)
    control = torch.tensor([[1.05, 1.05], [100.9, 100.9]], dtype=torch.bfloat16)
    candidate = torch.tensor([[1.08, 1.08], [100., 100.]], dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match='worst_row_relative_l2'):
        check_projection_gate(control, candidate, reference)
