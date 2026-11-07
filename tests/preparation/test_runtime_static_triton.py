# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Static launch metadata survives cache loading and process boundaries."""

import __future__
import ast
import inspect
import linecache
import pickle
import textwrap
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

import pytest
import torch
from torch._inductor.runtime import static_triton_launcher as launcher
from torch._inductor.runtime import triton_heuristics as heuristics

from ignition import runtime_patches
from ignition.runtime_patches import static_triton


def _install(monkeypatch, replacements):
    for owner, name, replacement in replacements:
        if hasattr(replacement, '__set_name__'):
            replacement.__set_name__(owner, name)
        monkeypatch.setattr(owner, name, replacement, raising=False)


def _clean_source(monkeypatch):
    replace = static_triton._replace

    def original_source(function, edits, **kwargs):
        source = textwrap.dedent(inspect.getsource(function))
        for before, after in edits:
            if after and source.count(after) == 1:
                source = source.replace(after, before)
        tree = ast.parse(source)
        tree.body[0].decorator_list = []
        filename = f'<clean-static-triton:{function.__qualname__}>'
        flags = __future__.annotations.compiler_flag & function.__code__.co_flags
        module = compile(tree, filename, 'exec', flags=flags, dont_inherit=True)
        code, = (value for value in module.co_consts if isinstance(value, CodeType))
        code = code.replace(co_firstlineno=1)
        result = FunctionType(code, function.__globals__, function.__name__, function.__defaults__)
        result.__kwdefaults__ = function.__kwdefaults__
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        return result

    monkeypatch.setattr(static_triton, '_replace', original_source)
    originals = static_triton.replacements()
    monkeypatch.setattr(static_triton, '_replace', replace)
    additions = {'_cooperative_launcher', '_run_cooperative', '__getstate__'}
    _install(monkeypatch, [(owner, name, replacement) for owner, name, replacement in originals
                           if not (owner is launcher.StaticallyLaunchedCudaKernel and name in additions)])
    for name in additions:
        if name in launcher.StaticallyLaunchedCudaKernel.__dict__:
            monkeypatch.delattr(launcher.StaticallyLaunchedCudaKernel, name)


@pytest.fixture(params=['clean', 'source-patched'])
def patched(monkeypatch, request):
    if request.param == 'clean':
        _clean_source(monkeypatch)
    _install(monkeypatch, static_triton.replacements())
    return request.param


def _kernel(*, name='kernel', cooperative=True, pdl=False, signature=None, constants=None):
    signature = signature or {0: '*fp32', 1: 'i32', 2: 'constexpr', 3: 'fp32'}
    fn = SimpleNamespace(__name__=name, arg_names=['pointer', 'count', 'width', 'scale'],
                         params=[SimpleNamespace(num=2, is_constexpr=True)])
    metadata = SimpleNamespace(num_warps=4, shared=256, num_ctas=1, global_scratch_size=0,
                               profile_scratch_size=0, launch_cooperative_grid=cooperative,
                               launch_pdl=pdl, tensordesc_meta=None)
    src = SimpleNamespace(fn=fn, signature=signature, constants=constants or {})
    return SimpleNamespace(src=src, metadata=metadata, hash='kernel-hash',
                           _cubin_path='/tmp/kernel.cubin', asm={'cubin': b'cubin'})


@pytest.mark.parametrize('signature,constants', [
    ({0: '*fp32', 1: 'i32', 2: 'constexpr', 3: 'fp32'}, {2: 16, 3: 1.0}),
    ({'pointer': '*fp32', 'count': 'i32', 'width': 'constexpr', 'scale': 'fp32'}, {'width': 16, 'scale': 1.0}),
    ({(0,): '*fp32', (1,): 'i32', (2,): 'constexpr', (3,): 'fp32'}, {(2,): 16, (3,): 1.0}),
])
def test_constructor_preserves_abi_and_filters_all_constant_key_forms(patched, signature, constants):
    kernel = _kernel(signature=signature, constants=constants)
    result = launcher.StaticallyLaunchedCudaKernel(kernel)
    parameters = inspect.signature(launcher.StaticallyLaunchedCudaKernel).parameters
    assert tuple(parameters) == ('kernel',)
    assert parameters['kernel'].annotation is launcher.CompiledKernel
    assert parameters['kernel'].default is inspect.Parameter.empty
    assert result.launch_cooperative_grid
    assert result.runtime_signature == ('*fp32', 'i32')
    assert result.arg_tys == 'Oi'
    assert result.full_constexprs == [2, 3]
    assert result.has_global_scratch and result.has_profile_scratch
    base = launcher.StaticallyLaunchedTritonKernel(kernel)
    assert base.launch_cooperative_grid
    assert base.runtime_signature == ('*fp32', 'i32')


def test_cooperative_run_packs_exact_triton_abi_without_duplicate_scratch(patched):
    result = launcher.StaticallyLaunchedCudaKernel(_kernel(constants={2: 16, 3: 1.0}))
    calls = []
    annotations, signature = object(), object()
    result.__dict__['_cooperative_launcher'] = (lambda *args: calls.append(args), annotations, signature)
    result.function = 333
    result.C_impl = SimpleNamespace(_launch_kernel=lambda *args: pytest.fail('cooperative launch used static ABI'))
    pointer = object()
    result.run(2, 3, 5, 777, pointer, 19)
    assert calls == [(2, 3, 5, 777, 333, True, False, (4, 1, 256),
                      None, None, None, None, None, annotations, signature, (pointer, 19))]
    with pytest.raises(AssertionError, match='Expected 2 args, got 1'):
        result.run(2, 3, 5, 777, pointer)
    assert len(calls) == 1


def test_device_agnostic_cooperative_run_uses_current_device_function(patched):
    result = launcher.StaticallyLaunchedCudaKernel(_kernel(constants={2: 16, 3: 1.0}))
    result.device_agnostic = True
    result._current_device = lambda: 7
    result.functions = {7: 555}
    result.C_impl = SimpleNamespace(_launch_kernel=lambda *args: pytest.fail('cooperative launch used static ABI'))
    calls = []
    result.__dict__['_cooperative_launcher'] = (lambda *args: calls.append(args), None, None)
    result.run(2, 1, 1, 777, object(), 19)
    assert calls[0][4] == 555


def test_regular_launch_keeps_static_abi_and_null_scratch(patched):
    result = launcher.StaticallyLaunchedCudaKernel(_kernel(cooperative=False, constants={2: 16, 3: 1.0}))
    calls = []
    result.C_impl = SimpleNamespace(_launch_kernel=lambda *args: calls.append(args))
    result.function = 333
    pointer = object()
    result.run(2, 3, 5, 777, pointer, 19)
    assert calls == [(333, 2, 3, 5, 4, 256, 'OiOO', (pointer, 19, None, None), 777)]


def test_pickle_drops_process_launcher_and_recreates_cached_property(patched, monkeypatch):
    from triton.backends.nvidia import driver as cuda_driver
    from triton.runtime import driver

    monkeypatch.setattr(cuda_driver, 'PyKernelArg', SimpleNamespace)
    monkeypatch.setattr(cuda_driver, 'ARG_KERNEL', 1)
    monkeypatch.setattr(cuda_driver, 'ARG_CONSTEXPR', 2)
    monkeypatch.setattr(cuda_driver, 'ARG_TUPLE', 3)
    result = launcher.StaticallyLaunchedCudaKernel(_kernel(constants={2: 16, 3: 1.0}))
    metadata_calls, launches = [], []
    utils = SimpleNamespace(launch=lambda *args: launches.append(args),
        build_signature_metadata=lambda signature: metadata_calls.append(signature) or tuple(signature))
    monkeypatch.setattr(driver, '_active', SimpleNamespace(utils=utils))
    first = result._cooperative_launcher
    assert result._cooperative_launcher is first
    assert metadata_calls == [['*fp32', 'i32']]
    try:
        result.function, result.module = 333, 444
        result.functions, result.modules = {0: 333}, {0: 444}
        restored = pickle.loads(pickle.dumps(result))
    finally:
        result.module = None
        result.modules = {}
    assert '_cooperative_launcher' not in restored.__dict__
    assert restored.function is None and restored.module is None
    assert restored.functions == {} and restored.modules == {}
    assert restored.cubin_path is None
    assert restored.cubin_raw == b'cubin'
    second = restored._cooperative_launcher
    assert second is restored._cooperative_launcher
    assert second is not first
    assert metadata_calls == [['*fp32', 'i32'], ['*fp32', 'i32']]
    restored.function = 888
    restored.run(1, 1, 1, 777, object(), 19)
    assert len(launches) == 1 and launches[0][4] == 888


def test_cooperative_fast_launcher_is_rejected_before_function_binding(patched):
    result = launcher.StaticallyLaunchedCudaKernel(_kernel())
    scope = {'runner': result.run}
    exec('def run(*args):\n    runner(*args)\n', scope)
    run = scope['run']
    run._is_static = True
    tuner = SimpleNamespace(inductor_meta={'use_fast_triton_launcher': True},
                            device_props=SimpleNamespace(type='cuda'))
    assert heuristics.CachingAutotuner._build_fast_launcher(tuner, run) is None


def test_cache_filename_truncation_survives_static_selection_and_raw_reload(patched, tmp_path, monkeypatch):
    monkeypatch.setattr(heuristics, 'triton_cache_dir', lambda device: str(tmp_path))
    monkeypatch.setattr(heuristics, 'triton_hash_to_path_key', lambda value: 'hash-key')
    kernel = _kernel(name='generated_' + 'x' * 300)
    target = tmp_path / 'hash-key' / (kernel.src.fn.__name__[:150] + '.cubin')
    target.parent.mkdir()
    target.write_bytes(b'compiled')
    with torch._inductor.config.patch(use_static_triton_launcher=True, strict_static_triton_launcher=True):
        result = heuristics.StaticTritonCompileResult.can_statically_launch(
            kernel, {}, {'device_type': 'cuda', 'device': 0}, heuristics.HeuristicType.POINTWISE)
    assert isinstance(result, launcher.StaticallyLaunchedCudaKernel)
    assert result.cubin_path == str(target)
    target.unlink()
    result.cubin_path = None
    wrapper = SimpleNamespace(kernel=result, compile_meta={'device_type': 'cuda', 'device': 0})
    heuristics.StaticTritonCompileResult.reload_cubin_path(wrapper)
    assert result.cubin_path == str(target)
    assert target.read_bytes() == b'cubin'
    assert len(target.name) == 156


@pytest.mark.parametrize('device_type,hip,pdl,error', [
    ('cuda', None, True, 'launch_pdl'),
    ('hip', None, False, 'static cooperative launch requires CUDA'),
    ('xpu', None, False, 'static cooperative launch requires CUDA'),
    ('cuda', '7.0', False, 'static cooperative launch requires CUDA'),
])
def test_unsupported_launch_attributes_are_rejected(patched, monkeypatch, device_type, hip, pdl, error):
    kernel = _kernel(pdl=pdl)
    monkeypatch.setattr(torch.version, 'hip', hip)
    monkeypatch.setattr(heuristics, 'XPU_KERNEL_FORMAT', 'zebin')
    with torch._inductor.config.patch(use_static_triton_launcher=True, strict_static_triton_launcher=True):
        with pytest.raises(heuristics.CannotStaticallyLaunchKernel, match=error):
            heuristics.StaticTritonCompileResult.can_statically_launch(
                kernel, {}, {'device_type': device_type, 'device': 0}, heuristics.HeuristicType.POINTWISE)
    assert kernel._cubin_path == '/tmp/kernel.cubin'


def test_installation_does_not_write_dependency_sources(monkeypatch):
    import builtins
    import io

    dependency = Path(torch.__file__).resolve().parent.parent
    original_open, original_io_open = builtins.open, io.open

    def checked_open(opener):
        def open_file(file, mode='r', *args, **kwargs):
            if isinstance(file, (str, bytes, Path)) and any(flag in mode for flag in 'wax+'):
                path = Path(file).resolve()
                assert not path.is_relative_to(dependency), f'dependency write: {path}'
            return opener(file, mode, *args, **kwargs)
        return open_file

    monkeypatch.setattr(builtins, 'open', checked_open(original_open))
    monkeypatch.setattr(io, 'open', checked_open(original_io_open))
    _install(monkeypatch, static_triton.replacements())
    assert not torch.cuda.is_initialized()


def test_clean_source_loses_cooperative_attributes_and_cannot_find_truncated_filename(monkeypatch, tmp_path):
    _clean_source(monkeypatch)
    result = launcher.StaticallyLaunchedCudaKernel(_kernel(constants={2: 16, 3: 1.0}))
    assert not hasattr(result, 'runtime_signature')
    calls = []
    result.C_impl = SimpleNamespace(_launch_kernel=lambda *args: calls.append(args))
    result.function = 333
    result.run(1, 1, 1, 777, object(), 19)
    assert len(calls) == 1
    assert not hasattr(result, '_cooperative_launcher')
    monkeypatch.setattr(heuristics, 'triton_cache_dir', lambda device: str(tmp_path))
    monkeypatch.setattr(heuristics, 'triton_hash_to_path_key', lambda value: 'hash-key')
    kernel = _kernel(name='generated_' + 'x' * 300, cooperative=False)
    target = tmp_path / 'hash-key' / (kernel.src.fn.__name__[:150] + '.cubin')
    target.parent.mkdir()
    target.write_bytes(b'compiled')
    with torch._inductor.config.patch(use_static_triton_launcher=True, strict_static_triton_launcher=True):
        with pytest.raises(heuristics.CannotStaticallyLaunchKernel, match='Cubin path not found'):
            heuristics.StaticTritonCompileResult.can_statically_launch(
                kernel, {}, {'device_type': 'cuda', 'device': 0}, heuristics.HeuristicType.POINTWISE)
