# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical backing retention spans compiled preparation and excludes capture."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from ignition.preparation.core_storage import CoreStorage
from ignition.preparation import capture


@pytest.mark.parametrize('fail', [False, True])
def test_retention_scope_completion_and_exception(fail):
    events = []
    def operation(name):
        events.append(name)
        return 0
    storage = CoreStorage.__new__(CoreStorage)
    storage._retention_scope = False
    storage.library = SimpleNamespace(
        flare_core_retention_begin=lambda: operation('begin'),
        flare_core_retention_finish=lambda: operation('finish'),
        flare_core_retention_abort=lambda: operation('abort'),
        flare_core_error=lambda: None)
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
    forward.retire_after_capture = lambda: None
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
