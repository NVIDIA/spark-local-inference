"""Host validation of interleaved-history storage and mutation dependencies."""

import pytest
import torch

from ignition.components.interleaved_history import validate_history


def operands():
    cache = torch.empty((2, 128, 576), dtype=torch.uint8)
    return [cache, torch.zeros(2, dtype=torch.int32), torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64), torch.empty((256, 576), dtype=torch.uint8),
            torch.zeros(16, dtype=torch.int64), torch.empty((4096, 576), dtype=torch.uint8),
            tuple(torch.zeros((1, 8), dtype=torch.bfloat16) for _ in range(4))]


def test_history_transfer_accepts_independent_caller_storage():
    values = operands()
    validate_history(*values)
    values[0] = values[0].view(torch.float8_e4m3fn)
    values[6] = values[6].view(torch.float8_e4m3fn)
    validate_history(*values)


def test_history_transfer_validates_worker_metadata_aliases():
    from ignition.preparation.forward_metadata import dumps, loads

    values = loads(dumps(operands()))
    validate_history(*values)
    assert all(value.fake_mode is not None for value in values[:7])
    values[4] = values[0].view(256, 576)
    with pytest.raises(ValueError, match="overlap"):
        validate_history(*values)


def test_history_transfer_rejects_misaligned_record_storage():
    values = operands()
    values[0] = torch.empty(values[0].numel() + 1, dtype=torch.uint8)[1:].view_as(values[0])
    with pytest.raises(ValueError, match="word-aligned"):
        validate_history(*values)


@pytest.mark.parametrize("field", [0, 4, 6, 7])
def test_history_transfer_rejects_aliases(field):
    values = operands()
    if field == 0:
        values[4] = values[0].view(256, 576)
    elif field == 4:
        values[4] = values[6][:256]
    elif field == 6:
        values[6] = values[4].expand(16, -1, -1).reshape(4096, 576)
        values[4] = values[6][:256]
    else:
        values[7] = (values[7][0],) * 4
    with pytest.raises(ValueError, match="overlap"):
        validate_history(*values)


@pytest.mark.parametrize("field,value", [(1, torch.zeros(1, dtype=torch.int32)),
    (2, torch.zeros(1, dtype=torch.int32)), (3, torch.zeros(2, dtype=torch.int64)),
    (5, torch.zeros((2, 8), dtype=torch.int64)), (7, (torch.zeros((1, 8)),) * 4)])
def test_history_transfer_rejects_invalid_metadata(field, value):
    values = operands()
    values[field] = value
    with pytest.raises((ValueError, TypeError)):
        validate_history(*values)


def test_dependent_gather_declares_slab_read_and_output_write():
    from ignition.components.peer_record_gather import _dependent_gather

    schema = _dependent_gather._opoverload._schema
    arguments = {argument.name: argument for argument in schema.arguments}
    assert arguments["output"].alias_info.is_write
    assert arguments["dependencies"].alias_info is None
    assert "Tensor" in str(arguments["dependencies"].type)
