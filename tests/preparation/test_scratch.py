# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU ownership and address-stability contracts for serialized operator scratch."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from b12x._lib.scratch import ScratchBufferSpec, scratch_tensor
from b12x.preparation import PreparationSession

from ignition.preparation.scratch import (
    GraphBufferPool, ScratchArena, bind_scratch, buffer_rows, planned_scratch, scratch_storage,
)
from ignition.preparation.tuning import register_finalizer


def spec(nbytes, *, dtype=torch.uint8, shape=None, device="cpu"):
    return ScratchBufferSpec(name="workspace", shape=(nbytes,) if shape is None else shape,
                             dtype=dtype, device=torch.device(device))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_named_row_views_reuse_identity_and_preserve_contents_and_geometry(dtype):
    backing = torch.arange(60, dtype=dtype).reshape(6, 10)[:, 1:9:2]
    values = {"source": backing}
    full = buffer_rows(values, "source", 6)
    prefix = buffer_rows(values, "source", 3)
    assert full is not backing
    assert buffer_rows(values, "source", 6) is full
    assert buffer_rows(values, "source", 3) is prefix
    assert prefix.shape == (3, 4)
    assert prefix.dtype == dtype
    assert prefix.stride() == backing.stride()
    assert prefix.storage_offset() == backing.storage_offset()
    assert prefix.untyped_storage()._cdata == backing.untyped_storage()._cdata
    expected = backing.clone()
    expected[:3].add_(2)
    before = backing._version
    prefix.add_(2)
    assert backing._version == full._version == prefix._version == before + 1
    assert torch.equal(buffer_rows(values, "source", 6), expected)


@pytest.mark.parametrize("replacement_rows", [4, 8])
def test_named_row_views_follow_source_replacement_and_retain_previous_storage(replacement_rows):
    backing = torch.full((4, 3), 7, dtype=torch.int32)
    values = {"source": backing}
    original = buffer_rows(values, "source", 2)
    replacement = torch.full((replacement_rows, 3), 11, dtype=torch.int32)
    values["source"] = replacement
    rebound = buffer_rows(values, "source", 2)
    assert rebound is buffer_rows(values, "source", 2)
    assert rebound is not original
    assert rebound.shape == original.shape == (2, 3)
    assert rebound.untyped_storage()._cdata == replacement.untyped_storage()._cdata
    assert original.untyped_storage()._cdata == backing.untyped_storage()._cdata
    rebound.fill_(13)
    assert (replacement[:2] == 13).all()
    assert (replacement[2:] == 11).all()
    assert (original == 7).all() and (backing == 7).all()


def test_named_row_views_keep_names_distinct_and_remain_exportable_scratch():
    from ignition.preparation.disposable import storage_ranges

    backing = torch.zeros((4, 3))
    values = {"first": backing, "second": backing}
    first = buffer_rows(values, "first", 2)
    second = buffer_rows(values, "second", 2)
    assert first is not second
    assert first is buffer_rows(values, "first", 2)
    assert second is buffer_rows(values, "second", 2)
    assert first.untyped_storage()._cdata == second.untyped_storage()._cdata
    storage = backing.untyped_storage()
    assert storage_ranges(values) == ((storage.data_ptr(), storage.nbytes()),)


def test_named_operands_share_capacity_across_shapes_without_overlapping_within_a_call():
    layout = dict(scores=((4, 6, 128), torch.float32), valid=((4, 6), torch.bool))
    workspace = bind_scratch(layout, device="cpu")
    large = bind_scratch(layout, device="cpu", workspace=workspace)
    small = bind_scratch(dict(scores=((2, 1, 128), torch.float32), valid=((2, 1), torch.bool)),
                         device="cpu", workspace=workspace)
    for operands in (large, small):
        for name, tensor in operands.items():
            assert tensor.is_contiguous()
            assert tensor.data_ptr() == workspace[name].data_ptr()
        operands["scores"].fill_(2)
        operands["valid"].fill_(True)
        assert operands["scores"].eq(2).all()
    assert large["scores"].data_ptr() == small["scores"].data_ptr()
    with pytest.raises(ValueError, match="requires"):
        bind_scratch(dict(scores=((8, 6, 128), torch.float32)), device="cpu", workspace=workspace)
    with pytest.raises(ValueError, match="meta"):
        bind_scratch(layout, device="meta", workspace=workspace)
    with pytest.raises(ValueError, match="torch.int32"):
        bind_scratch(dict(scores=((1,), torch.int32)), device="cpu", workspace=workspace)


def test_rank_scratch_finalization_preserves_shared_offsets(monkeypatch):
    import torch.distributed as dist

    arena = ScratchArena("cpu")
    arena.reserve(512)
    owner = SimpleNamespace()
    def bind():
        owner.main, owner.side = arena.views(((64,), torch.uint8), ((128,), torch.uint8))
    arena.register_finalizer(bind)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    def gather(ranks, value, group):
        ranks[:] = [value, (1024, [(((300,), torch.uint8), ((400,), torch.uint8))])]
    monkeypatch.setattr(dist, "all_gather_object", gather)
    arena.finalize(group=object())
    assert owner.main.shape == (300,) and owner.side.shape == (400,)
    assert owner.side.storage_offset() == 512
    assert owner.main.untyped_storage()._cdata == owner.side.untyped_storage()._cdata == arena.tensor.untyped_storage()._cdata
    assert arena.nbytes == 1024


def test_trial_binding_replaces_parallel_plans_and_retains_finalizers():
    arena = ScratchArena("cpu")
    class Plan:
        def __init__(self, size):
            self.size = size

        def memory_requirements(self):
            return SimpleNamespace(scratch_nbytes=self.size)

    original, replacement, side = Plan(128), Plan(384), Plan(256)
    arena.register_parallel(original, side)
    arena.reserve(1024)
    bound = []
    arena.register_finalizer(lambda: bound.append(arena.views(((64,), torch.uint8))[0]))
    arena.finalize(retain=True)
    assert len(bound) == 1 and bound[0].data_ptr() == arena.tensor.data_ptr()
    arena.replace_parallel_plans({original: replacement})
    assert arena.parallel_nbytes(lambda plan: plan.memory_requirements().scratch_nbytes) == 768
    arena.finalize()
    assert len(bound) == 2 and bound[1].data_ptr() == bound[0].data_ptr()
    assert arena._finalizers == []
    arena.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        arena.replace_parallel_plans({replacement: original})


def test_arena_owns_one_aligned_highwater_storage_across_operator_specs():
    arena = ScratchArena("cpu")
    assert arena.nbytes == 0 and arena.device == torch.device("cpu")
    assert arena.reserve(0) is arena.tensor
    first = planned_scratch((spec(1),), arena)
    original = arena.tensor
    reference = weakref.ref(original)
    assert arena.nbytes == 256 and first.shape == (1,)
    for requirement in (255, 0):
        view = planned_scratch((spec(requirement),), arena)
        assert view.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        assert arena.tensor is original
    del view
    second = planned_scratch((spec(257),), arena)
    assert arena.nbytes == 512 and arena.tensor is not original
    assert second.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    assert original.nbytes == 256
    del first, original
    assert reference() is None
    for requirement in (1, 257, 128, 512):
        view = planned_scratch((spec(requirement),), arena)
        assert view.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
        assert view.shape == (requirement,)
        assert scratch_tensor(view, (spec(requirement),), owner="test").data_ptr() == second.data_ptr()
    assert arena.nbytes == 512


def test_preparation_release_drops_backing_after_binding_finalization():
    arena = ScratchArena("cpu")
    arena.reserve(1024)
    backing = weakref.ref(arena.tensor)
    bound_sizes = []
    arena.register_finalizer(lambda: bound_sizes.append(arena.nbytes))
    arena.freeze()
    arena.finalize()
    assert bound_sizes == [1024]
    arena.release_preparation()
    assert backing() is None
    assert arena.nbytes == 0 and arena.device == torch.device("cpu")
    empty = arena.tensor
    arena.release_preparation()
    assert arena.tensor is empty


def test_preparation_release_rejects_unfinished_binding_without_dropping_storage():
    arena = ScratchArena("cpu")
    backing = arena.reserve(256)
    with pytest.raises(RuntimeError, match="completed binding finalization"):
        arena.release_preparation()
    arena.register_finalizer(lambda: None)
    arena.freeze()
    with pytest.raises(RuntimeError, match="completed binding finalization"):
        arena.release_preparation()
    assert arena.tensor is backing and arena.nbytes == 256
    arena.finalize()
    arena.release_preparation()
    assert arena.nbytes == 0


def test_declaration_owners_release_after_all_bindings_with_backing_alive():
    arena = ScratchArena("cpu")
    arena.reserve(512)
    events = []
    arena.register_finalizer(lambda: events.append(("binding", arena.nbytes)))
    arena.register_preparation_release(lambda: events.append(("owner", arena.nbytes)))
    arena.freeze()
    with pytest.raises(RuntimeError, match="completed binding finalization"):
        arena.release_preparation()
    assert events == []
    arena.finalize()
    arena.release_preparation()
    arena.release_preparation()
    assert events == [("binding", 512), ("owner", 512)]
    assert arena.nbytes == 0


def test_failed_declaration_release_preserves_backing_and_can_retry():
    arena = ScratchArena("cpu")
    backing = arena.reserve(256)
    attempts = []

    def release():
        attempts.append(arena.nbytes)
        if len(attempts) == 1:
            raise RuntimeError("declaration owner is still in use")

    arena.register_preparation_release(release)
    arena.freeze()
    arena.finalize()
    with pytest.raises(RuntimeError, match="owner is still in use"):
        arena.release_preparation()
    assert arena.tensor is backing
    arena.release_preparation()
    assert attempts == [256, 256]
    assert arena.nbytes == 0


def test_preparation_release_registration_rejects_invalid_and_retired_owners():
    arena = ScratchArena("cpu")
    with pytest.raises(TypeError, match="must be callable"):
        arena.register_preparation_release(None)
    arena.freeze()
    arena.finalize()
    arena.release_preparation()
    with pytest.raises(RuntimeError, match="storage has been released"):
        arena.register_preparation_release(lambda: None)


@pytest.mark.parametrize("operation", [
    lambda arena: arena.reserve(0),
    lambda arena: arena.reserve(256),
    lambda arena: arena.views(((0,), torch.uint8)),
    lambda arena: arena.trim(0),
    lambda arena: arena.finalize(),
    lambda arena: arena.auxiliary_stream,
])
def test_released_arena_rejects_storage_and_binding_reuse(operation):
    arena = ScratchArena("cpu")
    arena.reserve(256)
    arena.freeze()
    arena.finalize()
    arena.release_preparation()
    with pytest.raises(RuntimeError, match="storage has been released"):
        operation(arena)
    assert arena.nbytes == 0


def test_serial_arena_admits_no_auxiliary_stream_or_concurrent_plans():
    plan = SimpleNamespace(memory_requirements=lambda: SimpleNamespace(scratch_nbytes=256))
    assert ScratchArena("cpu").overlap
    serial = ScratchArena("cpu", overlap=False)
    assert serial.overlap is False
    with pytest.raises(RuntimeError, match="stream overlap is disabled; compose this branch on the main stream"):
        serial.auxiliary_stream
    with pytest.raises(RuntimeError, match="stream overlap is disabled"):
        serial.register_parallel(plan, plan)
    # Serial work shares one region: nothing reserves concurrent storage.
    assert serial.parallel_nbytes(lambda plan: plan.memory_requirements().scratch_nbytes) == 0
    with pytest.raises(TypeError, match="overlap must be a boolean"):
        ScratchArena("cpu", overlap=1)


@pytest.mark.parametrize("nbytes", [-1, True, False, 1.5, "256", None])
def test_invalid_sizes_reject_before_allocating_or_replacing_owner(nbytes):
    arena = ScratchArena("cpu")
    initial = arena.tensor
    with pytest.raises(ValueError, match="nonnegative integer"):
        arena.reserve(nbytes)
    with pytest.raises(ValueError, match="nonnegative integer"):
        scratch_storage(arena, nbytes=nbytes, device="cpu")
    assert arena.tensor is initial and arena.nbytes == 0


def test_frozen_arena_reuses_capacity_and_rejects_growth_without_mutation():
    arena = ScratchArena("cpu")
    tensor = arena.reserve(257)
    tensor.fill_(19)
    arena.freeze()
    arena.freeze()
    for size in (0, 1, 257, 512):
        assert arena.reserve(size) is tensor
    with pytest.raises(RuntimeError, match="frozen scratch arena"):
        planned_scratch((spec(513),), arena)
    assert arena.tensor is tensor and arena.nbytes == 512
    assert torch.all(tensor == 19)


def test_explicit_raw_storage_keeps_owner_view_and_rejects_undersizing():
    backing = torch.empty(64, dtype=torch.uint8)
    raw = backing[16:48]
    assert scratch_storage(raw, nbytes=32, device="cpu") is raw
    assert planned_scratch((spec(17),), raw) is raw
    assert raw.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert raw.storage_offset() == 16
    with pytest.raises(ValueError, match="requires 33"):
        scratch_storage(raw, nbytes=33, device="cpu")
    assert raw.shape == (32,)


@pytest.mark.parametrize("workspace,kind", [
    (torch.empty(8, dtype=torch.float32), TypeError),
    (torch.empty((2, 4), dtype=torch.uint8), ValueError),
    (torch.empty(16, dtype=torch.uint8)[::2], ValueError),
    (object(), TypeError),
])
def test_raw_storage_validates_dtype_shape_and_contiguity(workspace, kind):
    with pytest.raises(kind):
        scratch_storage(workspace, nbytes=8, device="cpu")


@pytest.mark.parametrize("workspace", [ScratchArena("cpu"), torch.empty(8, dtype=torch.uint8)])
def test_storage_device_mismatch_rejects_without_cuda_initialization(workspace):
    with pytest.raises(ValueError):
        scratch_storage(workspace, nbytes=8, device="meta")


def test_standalone_fallback_allocates_exact_required_byte_capacity():
    first = planned_scratch((spec(19),))
    second = planned_scratch((spec(19),))
    assert first.shape == second.shape == (19,)
    assert first.dtype == second.dtype == torch.uint8
    assert first.device == second.device == torch.device("cpu")
    assert first.data_ptr() != second.data_ptr()
    assert scratch_storage(None, nbytes=0, device="cpu").shape == (0,)


@pytest.mark.parametrize("specs", [(), (spec(8), spec(16)), (spec(8, dtype=torch.float32),),
                                  (spec(8, shape=(2, 4)),)])
def test_explicit_raw_workspace_rejects_unsupported_layouts(specs):
    raw = torch.empty(1024, dtype=torch.uint8)
    with pytest.raises(ValueError):
        planned_scratch(specs, raw)
    assert raw.shape == (1024,)


def test_plain_session_arena_finalizers_rebind_every_alias_and_release_old_storage():
    ordinary = object.__new__(PreparationSession)
    arena = ScratchArena("cpu")
    original = arena.reserve(8)
    reference = weakref.ref(original)
    owners = [SimpleNamespace(scratch=original) for _ in range(2)]
    order = []
    for index, owner in enumerate(owners):
        def finalize(index=index, owner=owner, prior=original):
            order.append(index)
            assert prior.nbytes == 256
            owner.scratch = arena.tensor
        register_finalizer(ordinary, finalize, workspace=arena)
    assert order == [0, 1]
    del finalize, original
    arena.reserve(300)
    assert reference() is not None
    arena.finalize()
    assert order == [0, 1, 0, 1]
    assert all(owner.scratch is arena.tensor for owner in owners)
    assert reference() is None
    arena.finalize()
    assert order == [0, 1, 0, 1]
    arena.freeze()
    with pytest.raises(RuntimeError):
        arena.register_finalizer(lambda: None)


def test_arena_finalizer_failure_keeps_callbacks_and_owners_for_retry():
    arena = ScratchArena("cpu")
    callbacks = []

    class Owner:
        def finalize(self):
            callbacks.append("owner")

    owner = Owner()
    reference = weakref.ref(owner)
    arena.register_finalizer(owner.finalize)
    del owner
    attempts = []

    def fail_once():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("binding refresh failed")

    arena.register_finalizer(fail_once)
    arena.freeze()
    with pytest.raises(RuntimeError, match="binding refresh failed"):
        arena.finalize()
    assert reference() is not None
    arena.finalize()
    assert reference() is None
    assert callbacks == ["owner", "owner"] and len(attempts) == 2


def test_simultaneous_views_have_aligned_offsets_disjoint_contents_and_shared_owner():
    arena = ScratchArena("cpu")
    byte, bf16, fp32 = arena.views(((3,), torch.uint8), ((2, 3), torch.bfloat16),
                                  ((5,), torch.float32))
    assert arena.nbytes == 768
    for value, offset, dtype, shape in ((byte, 0, torch.uint8, (3,)),
            (bf16, 256, torch.bfloat16, (2, 3)), (fp32, 512, torch.float32, (5,))):
        assert value.untyped_storage().data_ptr() == arena.tensor.untyped_storage().data_ptr()
        assert value.storage_offset() * value.element_size() == offset
        assert value.dtype == dtype and value.shape == shape and value.is_contiguous()
    byte.fill_(7)
    bf16.fill_(2)
    fp32.fill_(9)
    assert byte.eq(7).all() and bf16.eq(2).all() and fp32.eq(9).all()
    original = arena.tensor
    another, = arena.views(((129,), torch.float32))
    assert arena.tensor is original and arena.nbytes == 768
    assert another.storage_offset() == 0 and another.shape == (129,)
    another.zero_()
    assert byte.eq(0).all() and bf16.eq(0).all() and fp32[0].item() == 0
    assert fp32[1:].eq(9).all()


def test_empty_simultaneous_views_share_storage_without_reserving_extra_blocks():
    arena = ScratchArena("cpu")
    zero, value, trailing = arena.views(((0,), torch.float64), ((1,), torch.uint8),
                                       ((2, 0, 3), torch.int32))
    assert arena.nbytes == 256 and zero.shape == (0,) and trailing.shape == (2, 0, 3)
    assert zero.storage_offset() == 0 and value.storage_offset() == 0
    assert trailing.storage_offset() * trailing.element_size() == 256
    for tensor in (zero, value, trailing):
        assert tensor.untyped_storage().data_ptr() == arena.tensor.untyped_storage().data_ptr()


def test_planned_simultaneous_typed_specs_use_one_arena_and_reject_mixed_devices():
    arena = ScratchArena("cpu")
    specs = (spec(3), spec(6, shape=(2, 3), dtype=torch.bfloat16),
             spec(5, dtype=torch.float32))
    views = planned_scratch(specs, arena)
    assert isinstance(views, tuple) and len(views) == 3 and arena.nbytes == 768
    assert tuple(view.shape for view in views) == ((3,), (2, 3), (5,))
    original = arena.tensor
    with pytest.raises(ValueError):
        planned_scratch((spec(8), spec(4, device="meta")), arena)
    assert arena.tensor is original
    with pytest.raises(ValueError):
        planned_scratch((), arena)
    assert arena.tensor is original


@pytest.mark.parametrize("shape,dtype", [((-1,), torch.uint8), ((True,), torch.uint8),
    ((1.5,), torch.uint8), (("2",), torch.uint8), ((2,), "float32"), ((2,), None)])
def test_invalid_simultaneous_view_metadata_rejects_before_growth(shape, dtype):
    arena = ScratchArena("cpu")
    original = arena.tensor
    with pytest.raises((TypeError, ValueError)):
        arena.views(((4096,), torch.uint8), (shape, dtype))
    assert arena.tensor is original


def test_frozen_simultaneous_views_reuse_capacity_and_reject_excess_total():
    arena = ScratchArena("cpu")
    arena.reserve(512)
    arena.freeze()
    one, two = arena.views(((128,), torch.uint8), ((64,), torch.float32))
    assert one.storage_offset() == 0 and two.storage_offset() * two.element_size() == 256
    with pytest.raises(RuntimeError):
        arena.views(((257,), torch.uint8), ((1,), torch.uint8))
    assert arena.nbytes == 512


def test_trim_replaces_volatile_storage_with_aligned_selected_capacity():
    arena = ScratchArena("cpu")
    original = arena.reserve(1024)
    original.fill_(19)
    reference = weakref.ref(original)
    arena.trim(257)
    assert arena.nbytes == 512 and arena.tensor is not original
    assert arena.tensor.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    assert original.eq(19).all()
    del original
    assert reference() is None
    selected = arena.tensor
    arena.trim(511)
    assert arena.tensor is selected
    arena.freeze()
    with pytest.raises(RuntimeError):
        arena.trim(1)
    assert arena.tensor is selected and arena.nbytes == 512


def test_trim_to_zero_releases_arena_owner_without_invalidating_existing_views():
    arena = ScratchArena("cpu")
    view, = arena.views(((3,), torch.int64))
    view.fill_(11)
    pointer = view.data_ptr()
    arena.trim(0)
    assert arena.nbytes == 0 and view.eq(11).all()
    assert view.data_ptr() == pointer and view.untyped_storage().nbytes() == 256


def test_trim_releases_unshared_candidate_before_selected_allocation(monkeypatch):
    arena = ScratchArena("cpu")
    arena.reserve(1024)
    original = weakref.ref(arena.tensor)
    allocate = torch.empty
    def inspect(size, **kwargs):
        if size == 512:
            assert original() is None
            assert arena.nbytes == 0
        return allocate(size, **kwargs)
    monkeypatch.setattr(torch, "empty", inspect)
    arena.trim(257)
    assert arena.nbytes == 512


def test_trim_allocation_failure_leaves_empty_unfrozen_storage(monkeypatch):
    arena = ScratchArena("cpu")
    arena.reserve(1024)
    original = weakref.ref(arena.tensor)
    allocate = torch.empty
    def fail(size, **kwargs):
        if size == 512:
            raise torch.OutOfMemoryError("selected allocation unavailable")
        return allocate(size, **kwargs)
    monkeypatch.setattr(torch, "empty", fail)
    with pytest.raises(torch.OutOfMemoryError, match="selected allocation"):
        arena.trim(257)
    assert original() is None and arena.nbytes == 0 and not arena.frozen


@pytest.mark.parametrize("nbytes", [-1, True, False, 1.5, "256", None, 257])
def test_trim_rejects_invalid_or_larger_requirements_without_mutation(nbytes):
    arena = ScratchArena("cpu")
    original = arena.reserve(256)
    with pytest.raises(ValueError):
        arena.trim(nbytes)
    assert arena.tensor is original and arena.nbytes == 256


def test_priming_owners_retain_arena_instead_of_transient_views():
    from ignition.preparation.scratch import prime_owners

    arena = ScratchArena("cpu")
    scratch = arena.reserve(1024)
    binding = SimpleNamespace(scratch=scratch)
    assert prime_owners(arena, scratch, binding) == (arena,)
    assert prime_owners(scratch, scratch, binding) == (scratch, binding)
    assert prime_owners(None, scratch, binding) == (scratch, binding)
    assert prime_owners(None) == ()
