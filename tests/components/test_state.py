# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed continuation regions preserve alignment, slot isolation, and byte identity."""

import pytest
import torch

from ignition.components.state import StateLayout, StateStorage, save_hidden


def test_mixed_dtype_regions_alias_only_their_slot_bytes():
    layout = StateLayout.from_specs((("count", (1,), torch.int64),
                                    ("history", (3, 5), torch.bfloat16),
                                    ("recurrent", (2, 3, 3), torch.float32)))
    assert [region.offset for region in layout.regions] == [0, 256, 512]
    assert layout.row_bytes == 768
    pool = StateStorage.allocate(layout, 6, "cpu")
    for region in layout.regions:
        view = pool.region(region.name)
        assert view.stride(0) * view.element_size() == layout.row_bytes
        assert view.untyped_storage().data_ptr() == pool.backing.data_ptr()
        view[1].fill_(3)
        assert not view[0].count_nonzero() and not view[2:].count_nonzero()
    pool.backing[5].copy_(pool.backing[1])
    for region in layout.regions:
        torch.testing.assert_close(pool.region(region.name)[5], pool.region(region.name)[1], rtol=0, atol=0)
    assert layout.snapshot_capacity(1535) == 1


@pytest.mark.parametrize("specifications,error", [
    ([], ValueError),
    ([("", (1,), torch.float32)], ValueError),
    ([("x", (1,), torch.float32)] * 2, ValueError),
    ([("x", (0,), torch.float32)], ValueError),
    ([("x", (True,), torch.float32)], ValueError),
    ([("x", (1,), "float32")], TypeError),
])
def test_invalid_region_specifications(specifications, error):
    with pytest.raises(error):
        StateLayout.from_specs(specifications)


def test_hidden_checkpoints_preserve_unselected_physical_slots():
    pool = StateStorage.allocate(StateLayout.from_specs((("hidden", (7,), torch.bfloat16),)), 6, "cpu")
    pool.backing.fill_(0x31)
    before = pool.backing.clone()
    source = torch.arange(28, dtype=torch.bfloat16).reshape(4, 7)
    save_hidden(source, pool.region("hidden")[1:5], torch.tensor([2, -1, 0, 3]), torch.tensor([1, 0, 1, 0]))
    torch.testing.assert_close(pool.region("hidden")[1], source[2])
    torch.testing.assert_close(pool.region("hidden")[3], source[0])
    assert torch.equal(pool.backing[[0, 2, 4, 5]], before[[0, 2, 4, 5]])


def test_declared_state_extent_does_not_depend_on_rank_local_device_memory():
    class Storage:
        def device_allocate(self, name, shape, dtype, *, group, maximum, units):
            assert units in (3, 6)
            assert (name, group, maximum) == ("continuations", "STATE_ROWS", 73)
            return torch.zeros(shape, dtype=dtype)

    layout = StateLayout.from_specs((("hidden", (7,), torch.bfloat16),))
    pool = StateStorage.allocate(layout, 6, "cpu", storage=Storage(), maximum_slots=73)
    assert pool.backing.shape == (6, layout.row_bytes)
    assert pool.memory_capacity.tolist() == [6, 0]
    partial = StateStorage.allocate(layout, 6, "cpu", storage=Storage(), maximum_slots=73, backed_slots=3)
    assert partial.backing.shape == pool.backing.shape
    assert partial.memory_capacity.tolist() == [3, 0]
    with pytest.raises(ValueError, match="partial state backing"):
        StateStorage.allocate(layout, 6, "cpu", backed_slots=3)
    with pytest.raises(ValueError, match="extent"):
        StateStorage.allocate(layout, 6, "cpu", storage=Storage(), maximum_slots=5)


def test_request_row_scatter_and_prefix_gather_preserve_unselected_data():
    from ignition.components.state import gather_rows, scatter_rows

    retained = torch.full((4, 7, 13), -1.0)
    proposals = torch.arange(2 * 3 * 13, dtype=torch.float32).view(2, 3, 13)
    scatter_rows(proposals, torch.tensor([3, 1]), retained, torch.tensor([True, True]))
    assert torch.equal(retained[3, :3], proposals[0])
    assert torch.equal(retained[1, :3], proposals[1])
    assert retained[[0, 2]].eq(-1).all() and retained[:, 3:].eq(-1).all()
    verification = torch.empty((2, 2, 13))
    gather_rows(retained, torch.tensor([1, 3]), verification)
    assert verification.is_contiguous()
    assert torch.equal(verification, proposals[[1, 0], :2])
    scatter_rows(proposals, torch.tensor([-1, 0]), retained, torch.tensor([True, False]))
    assert retained[0].eq(-1).all()
