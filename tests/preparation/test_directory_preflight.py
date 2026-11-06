"""Standalone directory exports reject unsupported publication before capture."""

import sys
from types import SimpleNamespace

import pytest

from ignition.preparation import model_package


def test_directory_preflight_stops_export_before_capture(monkeypatch, tmp_path):
    from ignition.preparation.package_snapshot import _OPTIONS, export_options

    calls = []
    staging = tmp_path / "staging"

    def reject(destination):
        calls.append(destination)
        raise RuntimeError("atomic no-replace directory publication is unsupported")

    def unexpected_capture(_):
        raise AssertionError("rank capture must not start")

    def new_stage(*_):
        staging.mkdir()
        return SimpleNamespace(path=str(staging), rank_path=unexpected_capture, close=staging.rmdir)

    monkeypatch.setitem(sys.modules, "flare_heliopause", SimpleNamespace(
        export_preflight_directory=reject, NativeExportStage=new_stage))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=object()))
    monkeypatch.setitem(sys.modules, "torch.distributed", object())
    destination = tmp_path / "export"
    with export_options(tmp_path / "cache"):
        with pytest.raises(RuntimeError, match="atomic no-replace"):
            model_package.export_package(None, None, SimpleNamespace(world=1), destination,
                                         options={}, publisher=object())
    assert calls == [str(destination)]
    assert not destination.exists()
    assert not staging.exists()
    assert _OPTIONS.get() is None


@pytest.mark.parametrize("destination,form", [(None, "directory"), ("packed.safetensors", "packed")])
def test_cache_only_and_packed_exports_do_not_probe_directory_publication(monkeypatch, destination, form):
    def unexpected(_):
        raise AssertionError("directory publication probe must not run")

    monkeypatch.setitem(sys.modules, "flare_heliopause", SimpleNamespace(export_preflight_directory=unexpected))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=object()))
    monkeypatch.setitem(sys.modules, "torch.distributed", object())
    with pytest.raises(RuntimeError, match="configured package cache"):
        model_package.export_package(None, None, None, destination, options={}, publisher=object(), form=form)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux no-replace directory preflight")
def test_native_directory_preflight_preserves_existing_output(tmp_path):
    import flare_heliopause

    destination = tmp_path / "export"
    flare_heliopause.export_preflight_directory(str(destination))
    assert list(tmp_path.iterdir()) == []
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="destination exists"):
        flare_heliopause.export_preflight_directory(str(destination))
    assert sentinel.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [destination]
