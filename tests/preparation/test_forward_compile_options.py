# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit compilation policies retain their cache, backend and restore identity."""
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from inspect import signature
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from ignition.preparation import aot, compile_pool, inductor_tuning


class _Selections(inductor_tuning.InductorSelections):
    """Compilation-policy fixtures contain no generated GPU kernels."""

    def register(self, bound):
        self.forwards[id(bound)] = SimpleNamespace(_tuning_frozen=False), None, {}

    def freeze(self, bound):
        pass


def _loaded_backend():
    from torch._dynamo.guards import GuardManagerWrapper

    backend = aot.StorageBackend.__new__(aot.StorageBackend)
    backend.artifact = SimpleNamespace(_flare_graph=object())
    backend._serialized_artifact = b"portable backend"
    return SimpleNamespace(compiled_fn=backend, signature=signature(lambda forward: None),
        runtime_env=SimpleNamespace(closure=()), guards_state=b"portable guards",
        guard_manager=GuardManagerWrapper(root=SimpleNamespace(check=lambda arguments: True)))


@pytest.fixture
def caches(monkeypatch, tmp_path):
    aot.runtime_patches.apply()
    monkeypatch.setattr(inductor_tuning, 'InductorSelections', _Selections)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: SimpleNamespace(synchronize=lambda: None))
    core = tmp_path / 'ignition' / 'preparation'
    core.mkdir(parents=True)
    backend = core / 'aot.py'
    backend.write_text('backend = 1\n')
    kernels = tmp_path / 'flashinfer'
    kernels.mkdir()
    (kernels / 'operator.py').write_text('kernel = 1\n')
    dependency = ModuleType('flashinfer')
    dependency.__file__ = str(kernels / '__init__.py')
    monkeypatch.setitem(sys.modules, 'flashinfer', dependency)
    monkeypatch.setattr(aot, '__file__', str(backend))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device:
        SimpleNamespace(name='example', major=12, minor=0, multi_processor_count=96))
    return lambda **kwargs: aot.ForwardCache(namespace={'model': 'example'}, device='cuda:0',
                                            root=tmp_path / 'forwards', **kwargs)


def test_explicit_options_are_an_immutable_snapshot_and_part_of_cache_identity(caches):
    baseline = dict(aot._OPTIONS)
    ordinary = caches()
    supplied = dict(baseline, **{'triton.autotune_at_compile_time': True})
    prepared = caches(options=supplied)
    supplied['triton.autotune_at_compile_time'] = False
    assert prepared.options['triton.autotune_at_compile_time']
    assert prepared.root != ordinary.root
    assert dict(ordinary.options) == baseline == aot._OPTIONS
    assert ordinary.options['triton.autotune_at_compile_time'] is False
    with pytest.raises(TypeError):
        prepared.options['compile_threads'] = 1
    assert caches(options=dict(prepared.options, compile_threads=1)).root == prepared.root
    assert caches(options=dict(baseline)).root == ordinary.root


def test_runtime_patch_identity_invalidates_forward_artifacts(caches, monkeypatch):
    with torch.compiler.config.patch(cache_key_tag=torch.compiler.config.cache_key_tag):
        ordinary = caches()
        monkeypatch.setattr(aot.runtime_patches, 'identity', lambda: 'different-runtime-patches')
        assert caches().root != ordinary.root


def test_aot_forward_snapshots_explicit_options_for_its_backend(monkeypatch):
    baseline = dict(aot._OPTIONS)
    supplied = dict(baseline, **{'triton.autotune_at_compile_time': True})
    observed = []
    def compile(function, *, backend, **kwargs):
        supplied['triton.autotune_at_compile_time'] = False
        observed.append(dict(backend.keywords['options']))
        assert backend.func is aot.StorageBackend
        return SimpleNamespace(aot_compile=lambda inputs: inputs)
    monkeypatch.setattr(torch, 'compile', compile)
    owner = object()
    assert aot.aot_forward(owner, options=supplied) == ((owner,), {})
    assert observed[0]['triton.autotune_at_compile_time']
    assert aot._OPTIONS == baseline


def test_storage_backend_passes_explicit_policy_to_real_lowering(monkeypatch):
    baseline = dict(aot._OPTIONS)
    supplied = dict(baseline, **{'triton.autotune_at_compile_time': True})
    observed = []
    def compile(module, inputs, **kwargs):
        observed.append(kwargs['options']['config_patches'])
        assert kwargs['aot'] and kwargs['donate_graph_module']
        return object()
    monkeypatch.setattr(torch._inductor, 'standalone_compile', compile)
    backend = aot.StorageBackend(lambda x: x + 1, [torch.ones(4)], options=supplied)
    assert observed == [supplied]
    assert observed[0] is not supplied
    assert backend._serialized_artifact is None
    assert aot._OPTIONS == baseline


def test_pool_and_compiler_entry_pass_policy_without_mutating_defaults(monkeypatch, tmp_path):
    from torch._inductor.compile_worker import subproc_pool
    from ignition.preparation.forward_metadata import dumps

    baseline = dict(aot._OPTIONS)
    supplied = dict(baseline, **{'triton.autotune_at_compile_time': True})
    monkeypatch.setattr(compile_pool, 'device_memory_info', lambda device: (40 << 30, 96 << 30))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(is_integrated=False))
    calls = []
    class Subproc:
        def __init__(self, workers, **kwargs):
            pass
        def submit(self, function, *args):
            calls.append(args)
        def shutdown(self):
            pass
    monkeypatch.setattr(subproc_pool, 'SubprocPool', Subproc)
    monkeypatch.setattr(compile_pool, 'ProcessPoolExecutor', Subproc)
    with compile_pool.CompilePool('cuda:0', options=supplied) as pool:
        supplied['triton.autotune_at_compile_time'] = False
        pool.submit(SimpleNamespace(), tmp_path / 'first.bin')
    assert calls[0][2]['triton.autotune_at_compile_time']
    observed = []
    class Artifact:
        def save_compiled_function(self, path):
            Path(path).write_bytes(b'compiled')
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: None)
    monkeypatch.setattr(torch, 'set_grad_enabled', lambda enabled: None)
    monkeypatch.setattr(aot, 'aot_forward', lambda forward, *, options:
        (observed.append(dict(options)), Artifact())[1])
    compile_pool._compile_forward(dumps(SimpleNamespace()), 'cuda:0', calls[0][2],
                                  tmp_path / 'saved.bin', 1, False)
    assert observed == [calls[0][2]]
    assert (tmp_path / 'saved.bin').read_bytes() == b'compiled'
    assert aot._OPTIONS == baseline


def test_prepare_and_binding_rebuild_retain_instance_options(caches, monkeypatch, tmp_path):
    cache = caches(options=dict(aot._OPTIONS, max_autotune=False))
    observed = []
    class Artifact:
        _artifacts = _loaded_backend()
        def __call__(self, owner):
            return owner
        fn = __call__
        def save_compiled_function(self, path):
            Path(path).write_bytes(b'compiled')
    def compile(owner, *, options):
        observed.append(dict(options))
        return Artifact()
    monkeypatch.setattr(aot, 'aot_forward', compile)
    owner = SimpleNamespace()
    bound, path = cache._prepare(owner, name='ordinary', signature=('prefill', 1, 1))
    cache._warm(bound, path, name='ordinary')
    monkeypatch.setattr(cache, '_load', lambda *args, **kwargs: None)
    bound, path = cache._prepare(owner, name='ordinary', signature=('prefill', 1, 1))
    cache._warm(bound, path, name='ordinary')
    assert observed == [dict(cache.options)] * 2
    assert observed[0]['max_autotune'] is False


def test_prepare_many_passes_snapshot_and_preserves_warmup_scope(caches, monkeypatch):
    cache = caches(options=dict(aot._OPTIONS, max_autotune=False))
    policies, executed = [], []
    class Pool:
        workers = 1
        def __init__(self, device, *, options, **kwargs):
            policies.append(dict(options))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, owner, path):
            result = Future()
            result.set_result(None)
            return result
    class Artifact:
        _artifacts = _loaded_backend()
        def __call__(self, owner):
            executed.append('execute')
        fn = __call__
        def save_compiled_function(self, path):
            Path(path).write_bytes(b'compiled')
    @contextmanager
    def warmup():
        executed.append('enter')
        yield
        executed.append('exit')
    monkeypatch.setattr(compile_pool, 'CompilePool', Pool)
    monkeypatch.setattr(aot, '_guard_code_cache', lambda path: nullcontext())
    monkeypatch.setattr(aot.ForwardCache, '_load',
        lambda cache, owner, path, **kwargs: aot.BoundForward(Artifact(), owner, True))
    cache.prepare_many({'example': object()}, {'example': ('prefill', 1, 1)}, warmup=warmup)
    assert policies == [dict(cache.options)]
    assert executed == ['enter', 'execute', 'exit'] * 2


def test_prepare_many_recompiles_a_materialized_forward_that_no_longer_binds(caches, monkeypatch, tmp_path):
    cache = caches(options=dict(aot._OPTIONS, max_autotune=False))
    executed, compiled = [], []
    class Pool:
        workers = 1
        def __init__(self, device, *, options, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, owner, path):
            result = Future()
            result.set_result(None)
            return result
    class Stale:
        def __call__(self, owner):
            raise aot.BindingMismatch('another catalog layout')
    class Artifact:
        _artifacts = _loaded_backend()
        def __call__(self, owner):
            executed.append('execute')
        fn = __call__
        def save_compiled_function(self, path):
            Path(path).write_bytes(b'recompiled')
    def compile(owner, *, options):
        compiled.append(dict(options))
        return Artifact()
    paths = {}
    def load(cache, owner, path, **kwargs):
        paths['example'] = path
        if kwargs.get('required', False):
            raise aot.BindingMismatch('another catalog layout')
        return aot.BoundForward(Stale(), owner, True)
    monkeypatch.setattr(compile_pool, 'CompilePool', Pool)
    monkeypatch.setattr(aot, '_guard_code_cache', lambda path: nullcontext())
    monkeypatch.setattr(aot.ForwardCache, '_load', load)
    monkeypatch.setattr(aot, 'aot_forward', compile)
    monkeypatch.setattr(aot.ForwardCache, '_path', lambda cache, owner, *, name, signature: tmp_path / f'{name}.bin')
    result = cache.prepare_many({'example': object()}, {'example': ('prefill', 1, 1)}, materialize=lambda: None)
    assert compiled == [dict(cache.options)]
    assert executed == ['execute'] * 2
    assert result['example'].cache_hit is False
    assert (tmp_path / 'example.bin').read_bytes() == b'recompiled'
