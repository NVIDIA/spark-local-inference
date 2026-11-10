# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed destinations preserve addresses, layouts, and core ownership."""

from contextlib import nullcontext
import ctypes as C
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.core_storage import CoreStorage, _Properties
from ignition.preparation.persistent import (
    active_core_storage, allocated_tensor_bytes, persistent_cat, persistent_copy, persistent_empty, persistent_zeros,
)
from ignition.preparation.tensor_metadata import NumericStorage


class Memory:
    def __init__(self, capacity=4096):
        self.backing = torch.empty(capacity + 256, dtype=torch.uint8)
        self.address = -(-self.backing.data_ptr() // 256) * 256
        self.capacity = capacity
        self.calls, self.mapped, self.properties = [], [], []
        self.fail = None

    def call(self, name, *args):
        self.calls.append(name)
        if name == self.fail:
            raise RuntimeError(name)
        if name == "cuMemCreate":
            C.cast(args[0], C.POINTER(C.c_uint64))[0] = len(self.calls)
            properties = C.cast(args[2], C.POINTER(_Properties)).contents
            self.properties.append((properties.handles, tuple(properties.flags)))
        elif name == "cuMemMap":
            address, size = args[0].value, args[1].value
            assert self.address <= address < address + size <= self.address + self.capacity
            self.mapped.append((address, size))
        elif name == "cuMemsetD8_v2":
            C.memset(args[0].value, args[1].value, args[2].value)
        elif name == "cuMemUnmap":
            self.mapped.remove((args[0].value, args[1].value))


@pytest.fixture
def packed_storage(monkeypatch):
    from b12x.sequence._shared import disk_table

    original_view = disk_table._tensor_from_pointer
    monkeypatch.setattr(disk_table, "_tensor_from_pointer", lambda pointer, **kwargs:
        original_view(pointer, **dict(kwargs, device=torch.device("cpu"))))
    memory = Memory()
    storage = CoreStorage.__new__(CoreStorage)
    storage.device, storage.rank, storage.world = 0, 0, 1
    storage.cuda, storage.granularity, storage.gpudirect = memory, 128, True
    storage.DEVICE_BASE = memory.address
    storage.RESERVATION_BYTES = memory.capacity * 2
    storage._device_cursor = memory.address
    storage._persistent_total_bytes = memory.capacity
    storage.persistent_regions, storage.dynamic_regions, storage.host_regions = [], [], []
    storage._persistent_arena, storage._persistent_closed = None, False
    storage._persistent_model_owned = False
    storage._context_tokens = []
    storage._scope, storage.numeric_storage = nullcontext(), NumericStorage()
    storage.library = SimpleNamespace(flare_core_error=lambda: None)
    storage._preparation_pools = set()
    storage._weight_arenas = []
    storage._snapshot = lambda: {"mappings": {}}
    storage.memory = memory
    return storage


def test_growing_persistent_mapping_keeps_addresses_and_independent_storage(packed_storage):
    storage = packed_storage
    with storage:
        first = persistent_empty((5, 5), torch.float32, "cuda:0", stride=(7, 1))
        first.fill_(13)
        address = first.data_ptr()
        second = persistent_zeros((64,), torch.uint8, "cuda:0")
        second.fill_(29)
    assert first.data_ptr() == address == storage.memory.address
    assert first.stride() == (7, 1) and first.eq(13).all()
    assert second.data_ptr() == address + 256 and second.eq(29).all()
    assert first.untyped_storage()._cdata != second.untyped_storage()._cdata
    assert storage._persistent_arena.used == 320
    assert storage.persistent_regions[0].nbytes == 384
    assert storage._persistent_arena.unused_ranges() == [(address + 132, address + 256),
                                                       (address + 320, address + 384)]
    assert storage.mappings[address].nbytes == 384
    assert len(storage.memory.mapped) == 2
    assert all(flags == (0, 1, 0, 0, 0, 0, 0, 0) for _, flags in storage.memory.properties)
    assert "cuMemUnmap" not in storage.memory.calls
    storage.close()
    storage.close()
    assert not storage.memory.mapped and not storage.mappings
    assert storage.memory.calls.count("cuCtxSynchronize") == 1


def test_state_transfer_pointer_tables_use_packed_storage_without_numeric_registration(packed_storage):
    from ignition.components.state_transfer import _pointer_table

    storage = packed_storage
    values = [(0x2000000000, 128), (0x3000000000, 256)]
    with storage:
        table = _pointer_table(values, "cuda:0")
        trailing = _pointer_table(values[:1], "cuda:0")
        assert storage.numeric_storage.ranges() == []
    assert table.tolist() == [list(row) for row in values]
    assert trailing.tolist() == [list(values[0])]
    assert table.data_ptr() == storage.memory.address
    assert trailing.data_ptr() == table.data_ptr() + 256
    assert storage._persistent_arena.used == 272
    storage.close()


@pytest.mark.parametrize("failure", ["cuMemMap", "cuMemSetAccess", "cuMemsetD8_v2"])
def test_mapping_failure_preserves_existing_destinations(packed_storage, failure):
    storage = packed_storage
    with storage:
        first = persistent_zeros((64,), torch.uint8, "cuda:0")
        first.fill_(7)
        storage.memory.fail = failure
        with pytest.raises(RuntimeError, match=failure):
            persistent_empty((128,), torch.uint8, "cuda:0")
        assert storage._persistent_arena.used == 64
        assert storage.persistent_regions[0].nbytes == 128
        assert storage.memory.mapped == [(first.data_ptr(), 128)]
        assert first.eq(7).all()
        storage.memory.fail = None
        following = persistent_empty((128,), torch.uint8, "cuda:0")
        assert following.data_ptr() == first.data_ptr() + 256
    storage.close()


def test_persistent_capacity_and_device_mismatches_fail_before_mapping(packed_storage):
    with packed_storage:
        with pytest.raises(ValueError, match="core rank"):
            persistent_empty((8,), torch.float32, "cuda:1")
        with pytest.raises(MemoryError, match="reserved device capacity"):
            persistent_empty((packed_storage.memory.capacity + 1,), torch.uint8, "cuda:0")
    assert not packed_storage.memory.mapped


def test_active_storage_is_restored_after_nested_exception(packed_storage):
    inner = CoreStorage.__new__(CoreStorage)
    inner._scope, inner.numeric_storage = nullcontext(), NumericStorage()
    inner._context_tokens = []
    inner._persistent_arena = None
    inner.library = SimpleNamespace(flare_core_error=lambda: None)
    assert active_core_storage() is None
    with packed_storage:
        assert active_core_storage() is packed_storage
        with pytest.raises(RuntimeError, match="body"):
            with inner:
                assert active_core_storage() is inner
                raise RuntimeError("body")
        assert active_core_storage() is packed_storage
    assert active_core_storage() is None


def test_exception_before_model_ownership_releases_persistent_mappings(packed_storage):
    storage = packed_storage
    failure = RuntimeError("preparation failed")
    with pytest.raises(RuntimeError) as caught:
        with storage:
            persistent_empty((64,), torch.uint8, "cuda:0")
            raise failure
    assert caught.value is failure
    assert active_core_storage() is None
    assert storage._persistent_closed and not storage.memory.mapped
    assert storage.memory.calls.count("cuCtxSynchronize") == 1


def test_failed_exception_drain_retains_mappings_and_primary_error(packed_storage):
    storage = packed_storage
    failure = RuntimeError("preparation failed")
    with pytest.raises(RuntimeError) as caught:
        with storage:
            persistent_empty((64,), torch.uint8, "cuda:0")
            storage.memory.fail = "cuCtxSynchronize"
            raise failure
    assert caught.value is failure
    assert active_core_storage() is None
    assert not storage._persistent_closed and storage.memory.mapped
    assert "cuMemUnmap" not in storage.memory.calls
    assert any("cleanup failed" in note and "cuCtxSynchronize" in note for note in failure.__notes__)
    storage.memory.fail = None
    storage.close()
    assert storage._persistent_closed and not storage.memory.mapped


def test_allocator_error_releases_unowned_persistent_mappings(packed_storage):
    storage = packed_storage
    with pytest.raises(RuntimeError, match="native core allocator failed") as caught:
        with storage:
            persistent_empty((64,), torch.uint8, "cuda:0")
            storage.library.flare_core_error = lambda: b"native allocator failure"
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "native allocator failure"
    assert storage._persistent_closed and not storage.memory.mapped
    assert storage.memory.calls.count("cuCtxSynchronize") == 1
    assert active_core_storage() is None


def test_allocator_error_drain_failure_retains_mappings(packed_storage):
    storage = packed_storage
    with pytest.raises(RuntimeError, match="native core allocator failed") as caught:
        with storage:
            persistent_empty((64,), torch.uint8, "cuda:0")
            storage.library.flare_core_error = lambda: b"native allocator failure"
            storage.memory.fail = "cuCtxSynchronize"
    assert str(caught.value.__cause__) == "native allocator failure"
    assert not storage._persistent_closed and storage.memory.mapped
    assert "cuMemUnmap" not in storage.memory.calls
    assert any("cleanup failed" in note and "cuCtxSynchronize" in note
               for note in caught.value.__notes__)
    assert active_core_storage() is None
    storage.memory.fail = None
    storage.close()
    assert storage._persistent_closed and not storage.memory.mapped


def test_exception_after_model_ownership_defers_to_post_drain_cleanup(packed_storage):
    storage = packed_storage
    with pytest.raises(RuntimeError, match="preparation failed"):
        with storage:
            persistent_empty((64,), torch.uint8, "cuda:0")
            storage._persistent_model_owned = True
            raise RuntimeError("preparation failed")
    assert active_core_storage() is None
    assert not storage._persistent_closed and storage.memory.mapped
    assert "cuCtxSynchronize" not in storage.memory.calls
    storage.close()
    assert storage._persistent_closed and not storage.memory.mapped


def test_ordinary_destinations_copy_strided_values_and_concatenate_without_promotion():
    source = torch.arange(24, dtype=torch.float32).view(4, 6).T
    copied = persistent_copy(source)
    assert copied.stride() == source.stride()
    assert copied.data_ptr() != source.data_ptr()
    torch.testing.assert_close(copied, source, rtol=0, atol=0)
    for dim in (0, 1, -1):
        result = persistent_cat((source, source), dim=dim)
        torch.testing.assert_close(result, torch.cat((source, source), dim=dim), rtol=0, atol=0)
    assert persistent_zeros((2, 0, 3), torch.bfloat16, "cpu").shape == (2, 0, 3)
    assert persistent_zeros((), torch.int32, "cpu").item() == 0
    with pytest.raises(ValueError, match="dtypes"):
        persistent_cat((source, source.double()))


@pytest.mark.parametrize("shape,stride,alignment", [((2, 3), (0, 1), 256),
    ((2, 3), (-3, 1), 256), ((2, 3), (3,), 256), ((-1,), None, 256), ((1,), None, 3)])
def test_persistent_destinations_reject_invalid_layouts(shape, stride, alignment):
    with pytest.raises(ValueError):
        persistent_empty(shape, torch.float32, "cpu", stride=stride, alignment=alignment)


def test_packed_mapping_is_exported_as_fixed_storage(monkeypatch, packed_storage, tmp_path):
    import flare_heliopause
    from ignition.preparation import memory_inventory, runtime_core, tensor_metadata
    from ignition.components import weights

    storage = packed_storage
    with storage:
        tensor = persistent_zeros((64,), torch.uint8, "cuda:0")
    storage.transports, storage.peer_storage = [], None
    storage.graph_pool = SimpleNamespace(snapshot=lambda **kwargs: [])
    storage.local_ranks, storage.integrated = 1, False
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda **kwargs: [])
    monkeypatch.setattr(flare_heliopause, "current_context", lambda: (1, 0))
    monkeypatch.setattr(flare_heliopause, "export_synchronize", lambda *args: None)
    monkeypatch.setattr(tensor_metadata, "live_metadata", lambda *args, **kwargs: ([], {}))
    monkeypatch.setattr(weights, "record_final_fp4_expert_provenance_all", lambda: None)
    monkeypatch.setattr(memory_inventory, "collect_inventory", lambda *args, **kwargs: {})
    captured = []
    def capture(package, request, *args):
        captured.append(request)
        package.external = []
    monkeypatch.setattr(runtime_core.RuntimeCore, "_capture", capture)
    prepared = SimpleNamespace(_consumed=False, export_zero_ranges=(), export_events=(), buffers={},
        graphs={}, core_data=None, server_config={}, export_manifest=lambda **kwargs: {})
    runtime_core.export_model(prepared, SimpleNamespace(allocations={}, native=None), storage,
                                 tmp_path / "package", options={})
    allocation, = captured[0]["allocations"]
    assert allocation["address"] == tensor.data_ptr()
    assert allocation["nbytes"] == 128 and allocation["sizing"] is None
    assert allocation["preserve_address"] and allocation["memory"] == "device"
    assert captured[0]["inactive"] == [(tensor.data_ptr() + 64, tensor.data_ptr() + 128)]
    storage.close()


def test_persistent_physical_backing_and_padding_are_included_in_inventory(monkeypatch, packed_storage):
    import flare_heliopause
    from ignition.preparation.memory_inventory import collect_inventory
    from ignition.preparation.runtime_core import Allocation
    from ignition.preparation.tensor_metadata import record_numeric_tensor

    storage = packed_storage
    with storage:
        tensor = persistent_zeros((64,), torch.uint8, "cuda:0")
        record_numeric_tensor(tensor)
    storage.peer_storage = None
    storage.graph_pool = SimpleNamespace(id=(1, 1))
    prepared = SimpleNamespace(resources=(), export_zero_ranges=(), buffers={}, graphs={})
    package = SimpleNamespace(allocations=[Allocation("persistent", tensor.data_ptr(), 128)],
        module_globals=[], export_counts={"kernels": 0, "modules": 0, "cubin_bytes": 0})
    monkeypatch.setattr(flare_heliopause, "memory_inventory", lambda: {
        "mappings": [], "total_bytes": 10000, "free_bytes": 9000})
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda **kwargs: [])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    report = collect_inventory(prepared, storage, package)
    assert report["tracked_physical_bytes"] == report["fixed_core_bytes"] == 128
    assert report["core_padding_bytes"] == 0
    sizes = {row["category"]: row["nbytes"] for row in report["entries"]}
    assert sizes == {"Weights / numeric constants": 64, "Allocator slack": 64}
    storage.close()


def test_tensor_accounting_includes_packed_destinations_until_close(packed_storage, monkeypatch):
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 512)
    assert allocated_tensor_bytes("cuda:0") == 512
    with packed_storage:
        persistent_empty((19,), torch.uint8, "cuda:0")
        persistent_empty((3,), torch.float32, "cuda:0")
        assert allocated_tensor_bytes("cuda:0") == 512 + 256 + 12
        assert allocated_tensor_bytes("cuda:1") == 512
        packed_storage.close()
        assert allocated_tensor_bytes("cuda:0") == 512
    assert allocated_tensor_bytes("cuda:0") == 512
