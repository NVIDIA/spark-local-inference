# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Atomic publication and rank ownership in native model cores."""

from types import SimpleNamespace

import pytest

from helios.tests.signing import publisher
from ignition.preparation import model_core
from ignition.preparation.core_file import combine

from ignition.preparation.core_snapshot import export_options
from tests.preparation.payload_fixture import rank_source, manifest


@pytest.fixture(autouse=True)
def configured_cache(tmp_path):
    with export_options(tmp_path / '.flare-cache'):
        yield


@pytest.mark.parametrize('world', [1, 2])
def test_publication_contains_one_bundle_per_rank(monkeypatch, tmp_path, world):
    import torch
    import torch.distributed as dist

    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(dist, 'barrier', lambda: None)
    exported = []
    closed = []
    printed = []
    def export(prepared, images, storage, path, **kwargs):
        source = rank_source(path.parent / path.stem)
        path.write_bytes(source.read_bytes())
        exported.append(path.name)
        return SimpleNamespace(memory_inventory={'rank': 0})
    monkeypatch.setattr(model_core, 'export_model', export)
    class Control:
        def send(self, command):
            assert command[0] == 'export'
            export(None, None, None, command[1])
        def recv(self):
            return {'peer_memory': [(0x210000000000, 65536)], 'memory_inventory': {'rank': 1}}
    prepared = SimpleNamespace(model=object(), controls=[Control()] if world == 2 else [],
                               close=lambda: closed.append('model'))
    storage = SimpleNamespace(world=world, peer_storage=SimpleNamespace(
        owned={0x200000000000: 65536}, reset=lambda: None),
        host_regions=[SimpleNamespace(close=lambda: closed.append('host'))])
    def assemble(*args, **kwargs):
        assert closed == ['model', 'host']
        assert len(exported) == world
        return combine(*args, **kwargs)
    monkeypatch.setattr(model_core, 'combine', assemble)
    output = tmp_path / 'model'
    def inventory(reports, **kwargs):
        assert (output / "model.safetensors").is_file() and closed == ['model', 'host']
        printed.extend(reports)
    monkeypatch.setattr(model_core, 'print_inventory', inventory)
    model_core.export_core(prepared, None, storage, output, options={}, publisher=publisher())
    envelope = manifest(output)
    assert len(envelope.ranks) == world
    assert sorted(exported) == [f'rank-{rank}.json' for rank in range(world)]
    assert (output / "model.safetensors").is_file()
    assert printed == [{'rank': rank} for rank in range(world)]
    assert {p.name for p in tmp_path.iterdir()} == {output.name, '.flare-cache'}


def test_failed_rank_export_leaves_no_published_model(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError('missing module image')
    monkeypatch.setattr(model_core, 'export_model', fail)
    with pytest.raises(RuntimeError, match='missing module image'):
        model_core.export_core(object(), None, SimpleNamespace(world=1), tmp_path / 'model', options={}, publisher=publisher())
    assert not list(tmp_path.iterdir())
