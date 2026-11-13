# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Signed preparation metadata names stored layouts without changing serving payloads."""

from dataclasses import dataclass

import pytest
import torch

from ignition.preparation import runtime_core_pb2 as proto
from ignition.preparation.rebuild_manifest import rebuild_weights_metadata, validate_weights, weight_views
from helios.tests.signing import publisher
from tests.preparation.test_core_snapshot import native, assert_native
from tests.preparation import test_core_protection as fixtures


@dataclass
class PackedWeight:
    values: torch.Tensor
    scales: torch.Tensor

    @property
    def forbidden(self):
        raise AssertionError('cataloging must not evaluate properties')


def test_prepared_views_preserve_names_aliases_strides_and_packing_owner():
    storage = torch.arange(48, dtype=torch.uint8)
    matrix = storage.view(6, 8)
    weights = PackedWeight(matrix[1:4, ::2], matrix[4:, :4])
    roots = {'packed': weights, 'alias': weights.values}
    catalog = rebuild_weights_metadata(roots)
    entries = {t['name']: t for t in catalog['tensors']}
    assert entries['packed.values']['shape'] == [3, 4]
    assert entries['packed.values']['stride'] == [8, 2]
    assert entries['packed.values']['address'] == storage.data_ptr() + 8
    assert entries['packed.values']['address'] == entries['alias']['address']
    assert entries['packed.values']['extent'] == 23
    assert entries['packed.scales']['address'] == storage.data_ptr() + 32
    assert entries['packed.values']['layout_type'].endswith('.PackedWeight')
    assert entries['packed.values']['dtype'] == 'uint8'
    assert not catalog['unavailable']


def test_tensor_metadata_bounds_cycles_and_defers_storage_resolution():
    stored = torch.ones(8)
    disposable = torch.zeros(8)
    roots = {'stored': stored, 'unused': torch.ones(3), 'scratch': disposable,
             'meta': torch.empty(3, device='meta'), 'empty': torch.empty(0)}
    roots['cycle'] = roots
    catalog = rebuild_weights_metadata(roots)
    assert {entry['name'] for entry in catalog['tensors']} == {
        'scratch', 'stored', 'unused', "cycle['scratch']", "cycle['stored']", "cycle['unused']"}
    assert set(catalog['unavailable']) == {'meta', 'empty', "cycle['meta']", "cycle['empty']"}
    assert len(list(weight_views(roots))) < 20


@pytest.mark.parametrize('change', ['version', 'dtype', 'shape', 'duplicate', 'dynamic', 'peer', 'missing'])
def test_invalid_weight_catalog_cannot_be_used_for_preparation(change):
    tensor = torch.ones(4)
    catalog = proto.RebuildWeights(version=1)
    catalog.tensors.add(name='w', dtype='float32', element_bytes=4, shape=[4], stride=[1],
                        storage=proto.Reference(allocation='weights'))
    bundle = proto.BundleIndex(allocations=[proto.Allocation(name='weights', nbytes=tensor.nbytes)],
                              rebuild_weights=catalog)
    entry = bundle.rebuild_weights.tensors[0]
    if change == 'version': bundle.rebuild_weights.version = 99
    elif change == 'dtype': entry.dtype = 'unknown'
    elif change == 'shape': entry.shape.append(3)
    elif change == 'duplicate': bundle.rebuild_weights.tensors.add().CopyFrom(entry)
    elif change == 'dynamic': bundle.allocations[0].sizing.maximum = 1024
    elif change == 'peer': bundle.allocations[0].memory = proto.PEER
    else: entry.storage.allocation = 'absent'
    with pytest.raises(ValueError): validate_weights(bundle)


def test_manifest_is_authenticated_with_shared_and_rank_metadata(native, tmp_path):
    index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])
    index.rebuild.version = 1
    index.rebuild.recipe = 'signed-recipe'
    from helios.tests.signing import publish_legacy as publish
    path = tmp_path / 'model.safetensors'
    publish(path, index, [], publisher=publisher())
    assert_native(native, path)
    raw = path.read_bytes()
    assert raw.count(b'signed-recipe') == 1
    path.write_bytes(raw.replace(b'signed-recipe', b'forged-recipe'))
    result = fixtures.read(native, path, weights=True)
    assert result.returncode != 0


def test_absent_catalog_is_not_inferred_from_allocation_names():
    bundle = proto.BundleIndex.FromString(b'\x12\x0b\x0a\x07weights\x10\x40')
    assert not bundle.HasField('rebuild_weights')
    validate_weights(bundle)
    assert bundle.allocations[0].nbytes == 64


@pytest.mark.parametrize("mode", ["static", "runtime"])
def test_rebuild_manifest_records_collective_selection(mode):
    from dataclasses import replace
    from google.protobuf.json_format import ParseDict
    from ignition.preparation.rebuild_manifest import from_plan
    from ignition.recipes import ResolvedModel
    from tests.cli.test_planning import derive

    plan = replace(derive(), collective_tuning=mode)
    model = ResolvedModel(plan.recipe, plan.checkpoint)
    manifest = ParseDict(from_plan(model, plan, compiled_forwards=True), proto.RebuildManifest())
    restored = proto.RebuildManifest.FromString(manifest.SerializeToString())
    assert restored.collective_tuning == mode
