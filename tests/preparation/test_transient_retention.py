# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical backing retention spans compiled preparation and excludes capture."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from ignition.preparation.package_storage import PackageStorage
from ignition.preparation import capture


@pytest.mark.parametrize('fail', [False, True])
def test_retention_scope_completion_and_exception(fail):
    events = []
    def operation(name):
        events.append(name)
        return 0
    storage = PackageStorage.__new__(PackageStorage)
    storage._retention_scope = False
    storage._retention_pool = None
    storage.library = SimpleNamespace(
        flare_package_retention_begin=lambda: operation('begin'),
        flare_package_retention_finish=lambda: operation('finish'),
        flare_package_retention_abort=lambda: operation('abort'),
        flare_package_error=lambda: None)
    try:
        with storage.retain_transient_backing():
            assert storage._retention_scope
            with pytest.raises(RuntimeError, match='nested'):
                with storage.retain_transient_backing():
                    pass
            events.append('body')
            if fail:
                raise ValueError('preparation failed')
    except ValueError:
        assert fail
    assert not storage._retention_scope
    assert events == ['begin', 'body', 'abort' if fail else 'finish']


def test_retention_scope_preserves_sync_error_and_live_pool(monkeypatch):
    import torch

    events = []
    sync_error = RuntimeError('device synchronization failed')
    pool = object()
    storage = PackageStorage.__new__(PackageStorage)
    storage._retention_scope = False
    storage._retention_pool = pool
    storage.device = 0
    storage.library = SimpleNamespace(
        flare_package_retention_begin=lambda: events.append('begin') or 0,
        flare_package_retention_finish=lambda: events.append('finish') or 0,
        flare_package_retention_abort=lambda: events.append('abort'),
        flare_package_error=lambda: None)

    def synchronize(device):
        events.append('sync')
        raise sync_error

    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: events.append('empty_cache'))
    with pytest.raises(RuntimeError, match='device synchronization failed') as raised:
        with storage.retain_transient_backing():
            pass

    assert raised.value is sync_error
    assert storage._retention_pool is pool
    assert not storage._retention_scope
    assert events == ['begin', 'sync', 'sync', 'abort']
    assert any('transient pool synchronization failed' in note for note in raised.value.__notes__)


def test_retention_scope_preserves_body_error_when_pool_cleanup_fails(monkeypatch):
    import torch

    events = []
    body_error = ValueError('preparation failed')
    cleanup_error = RuntimeError('pool cache cleanup failed')
    storage = PackageStorage.__new__(PackageStorage)
    storage._retention_scope = False
    storage._retention_pool = object()
    storage.device = 0
    storage.library = SimpleNamespace(
        flare_package_retention_begin=lambda: events.append('begin') or 0,
        flare_package_retention_finish=lambda: events.append('finish') or 0,
        flare_package_retention_abort=lambda: events.append('abort'),
        flare_package_error=lambda: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: events.append('sync'))

    def empty_cache():
        events.append('empty_cache')
        raise cleanup_error

    monkeypatch.setattr(torch.cuda, 'empty_cache', empty_cache)
    with pytest.raises(ValueError, match='preparation failed') as raised:
        with storage.retain_transient_backing():
            raise body_error

    assert raised.value is body_error
    assert storage._retention_pool is None
    assert not storage._retention_scope
    assert events == ['begin', 'sync', 'empty_cache', 'abort']
    assert any('transient pool cleanup failed' in note for note in raised.value.__notes__)


def test_capture_drains_after_prepare_and_sync_before_capture(monkeypatch):
    events = []
    @contextmanager
    def retention():
        events.append('begin')
        yield
        events.append('finish')
    def forward():
        events.append('run')
    forward.prepare_capture = lambda: events.append('prepare_capture') or 0
    class Cache:
        def __init__(self, **kwargs):
            pass
        def prepare_many(self, functions, signatures, **kwargs):
            events.append('prepare_many')
            return functions
    stream = SimpleNamespace(wait_stream=lambda value: None,
                             synchronize=lambda: events.append('sync'))
    monkeypatch.setattr(capture, 'ForwardCache', Cache)
    monkeypatch.setattr(capture.torch.cuda, 'current_stream', lambda device: stream)
    monkeypatch.setattr(capture.torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(capture.torch.cuda, 'synchronize', lambda device: None)
    monkeypatch.setattr(capture.torch.cuda, 'CUDAGraph', lambda: 'graph')
    monkeypatch.setattr(capture.torch.cuda, 'graph', lambda *args, **kwargs: nullcontext())
    capture.capture_forwards({'speculative': forward},
        {'speculative': SimpleNamespace(kind='speculative')},
        session=SimpleNamespace(namespace=SimpleNamespace(to_dict=lambda: {}), capture=nullcontext),
        device='cuda:0', stream=stream, pool=None,
        warmup_storage=SimpleNamespace(transient=nullcontext, retain_transient_backing=retention))
    assert events[:5] == ['begin', 'prepare_many', 'sync', 'finish', 'prepare_capture']
