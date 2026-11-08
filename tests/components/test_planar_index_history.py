"""Storage validation and shared-page validity for FP8 index history."""

import pytest
import torch

from ignition.components.planar_index_history import global_index_pages, validate_planar_history


def operands():
    return [torch.empty((6, 64 * 132), dtype=torch.uint8), torch.zeros(2, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64),
            torch.empty((4, 64 * 132), dtype=torch.uint8), torch.zeros(16, dtype=torch.int64),
            torch.empty((64, 64 * 132), dtype=torch.uint8),
            tuple(torch.zeros((1, 8), dtype=torch.bfloat16) for _ in range(4))]


def test_planar_history_geometry_survives_worker_serialization():
    from ignition.preparation.forward_metadata import dumps, loads

    values = operands()
    validate_planar_history(*values)
    validate_planar_history(*loads(dumps(values)))


@pytest.mark.parametrize("field", [0, 1, 2, 4, 5, 6, 7])
def test_planar_history_rejects_incompatible_storage(field):
    values = operands()
    if field in (0, 4, 6):
        values[field] = values[field][:, :-4]
    elif field in (1, 2, 5):
        values[field] = values[field].float()
    else:
        values[field] = values[field][:-1]
    with pytest.raises((ValueError, TypeError)):
        validate_planar_history(*values)


@pytest.mark.parametrize("alias", ["stage", "signals", "metadata", "offset"])
def test_planar_history_rejects_overlapping_or_unaligned_storage(alias):
    values = operands()
    if alias == "stage":
        values[4] = values[6][:4]
    elif alias == "signals":
        values[7] = (values[7][0],) * 4
    elif alias == "metadata":
        values[3] = values[2]
    else:
        values[0] = torch.empty(values[0].numel() + 1, dtype=torch.uint8)[1:].view_as(values[0])
    with pytest.raises(ValueError, match="overlap|word-aligned"):
        validate_planar_history(*values)


def test_shared_model_page_validity_covers_all_owner_records():
    model = torch.tensor([5, -1, 0, 7], dtype=torch.int32)
    columns = torch.arange(128, dtype=torch.int32)
    output = torch.empty((1, 128), dtype=torch.int32)
    global_index_pages(model, columns, output, world=16, cache_pages=6)
    assert torch.equal(output[0, :32], columns[:32])
    assert torch.equal(output[0, 64:96], columns[64:96])
    assert (output[0, 32:64] == -1).all() and (output[0, 96:] == -1).all()
    # Remapping valid physical pages cannot alter global token addressing.
    model.copy_(torch.tensor([0, 5, 4, 3]))
    global_index_pages(model, columns, output, world=16, cache_pages=6)
    assert torch.equal(output[0], columns)


@pytest.mark.parametrize("world", [1, 2, 16])
def test_global_page_table_uses_two_subpages_per_owner(world):
    columns = torch.arange(4 * world, dtype=torch.int32)
    output = torch.empty((1, 4 * world), dtype=torch.int32)
    global_index_pages(torch.tensor([-1, 0], dtype=torch.int32), columns, output, world=world, cache_pages=1)
    assert (output[0, :2 * world] == -1).all()
    assert torch.equal(output[0, 2 * world:], columns[2 * world:])
